# %% [markdown]
# # 10b — How should the high-resolution reader M8 be read at test time? (out-of-fold check on the gold studies)
# **Settings:** Accelerator = GPU T4 (a few minutes) · Internet = OFF
# **Inputs:** competition data · 03d cache · 03e Preprocess HiRes A + B · 04h Train Round2 Reader A (M7 folds 0,1) ·
# 04i Train HiRes Reader A (M8 folds 0,1)
#
# M8 trained on random 224 px crops of the 320 px 03e cache and was validated on the full 320 px image: OOF GT 0.903
# (n=26) vs M7 0.929 on the same studies. Is the train/test view mismatch the reason? Variants for M8:
# `train` (full 320 px, as validated) · `c224` (centre 224 crop = the training view) · `5crop` (centre + 4 shifted
# 224 crops, probabilities averaged) · and each with dense windows. Then M7 (dense) + M8 rank ensembles.

# %%
import os, re, sys, json, time, importlib.util
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

T0 = time.time()
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
WORK = "/kaggle/working"
FROM_04 = lambda p: re.search(r"04[c-z]-train", p) is not None

def find_inputs(match):
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        hits += [os.path.join(root, f) for f in files if match(os.path.join(root, f))]
    return sorted(hits)

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

DATA = os.path.dirname(find_inputs(lambda p: p.endswith("/train_series.csv"))[0])
train = pd.read_csv(f"{DATA}/train.csv")
GT = train[train[LABELS].notna().all(axis=1)].set_index("StudyInstanceUID")[LABELS].astype(int)
GOLD = set(GT.index)

FAMILIES = {}
for p in find_inputs(lambda p: re.search(r"/config_[^/]+\.json$", p) is not None and FROM_04(p)):
    c = json.load(open(p))
    if c.get("ARCH") == "reader":
        c["PREP"], c["_DIR"] = c.get("PREP", "03c"), os.path.dirname(p)
        FAMILIES.setdefault(c["TAG"], c)
PREP_FILES = {}
for p in find_inputs(lambda p: p.endswith("/knee_prep.py")):
    hit = re.search(r"(03[a-z])-preprocess", p)
    if hit:
        PREP_FILES.setdefault(hit.group(1), p)
PREPS = {n: load_module(f"knee_prep_{n}", PREP_FILES[n]) for n in sorted({c["PREP"] for c in FAMILIES.values()})}
print("families:", {t: (c["PREP"], c["BACKBONE"], c["T_PER_SLOT"]) for t, c in FAMILIES.items()})

# %% [markdown]
# ## Gold volumes from the caches, fold of every gold study, fold models

# %%
def load_gold(prep):
    vol, pres = {}, {}
    for sh in find_inputs(lambda p: "/cache/shard_" in p and p.endswith(".npz") and f"{prep}-preprocess" in p):
        with np.load(sh) as z:
            for key in z.files:
                uid, what = key.split("/")
                if uid in GOLD:
                    (vol if what == "vol" else pres)[uid] = z[key]
    return vol, pres

CACHE = {}
for n in PREPS:
    CACHE[n] = load_gold(n)
    print(f"cache {n}: {len(CACHE[n][0])} gold volumes | {(time.time() - T0) / 60:.1f} min", flush=True)

def layout(kp, t_per_slot, shift=0.0):
    """window slice indices (N, 3), slot of each window, relative position — knee_reader.windows with shifted centres"""
    idx, slot, pos = [], [], []
    for k, (t, d) in enumerate(zip(t_per_slot, kp.DS)):
        a, b = 1.0, d - 2.0
        c = np.linspace(a, b, t)
        if shift and t > 1:
            c = c + shift * (b - a) / (t - 1)
        c = np.clip(np.round(c).astype(int), 1, d - 2)
        idx.append(kp.OFF[k] + np.stack([c - 1, c, c + 1], 1))
        slot += [k] * t
        pos.append(c / max(d - 1, 1))
    return np.concatenate(idx), np.array(slot), np.concatenate(pos).astype(np.float32)

def variants(kp, t):
    """name → list of reads (windows per slot, phase shift, crop (oy, ox, size) or None); probabilities averaged"""
    dense = [d - 2 for d in kp.DS]
    if kp.S <= 224:
        return {"train": [(t, 0.0, None)], "dense": [(dense, 0.0, None)]}
    m = kp.S - 224
    c = (m // 2, m // 2, 224)
    five = [c] + [(m // 2 + dy, m // 2 + dx, 224) for dy, dx in ((-m // 4, -m // 4), (-m // 4, m // 4), (m // 4, -m // 4), (m // 4, m // 4))]
    return {"train": [(t, 0.0, None)], "dense": [(dense, 0.0, None)],
            "c224": [(t, 0.0, c)], "5crop": [(t, 0.0, f) for f in five],
            "dense c224": [(dense, 0.0, c)], "dense 5crop": [(dense, 0.0, f) for f in five]}

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1).cuda()
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1).cuda()

@torch.no_grad()
def predict(model, kp, uids, vol, pres, t_per_slot, shift, batch=4, crop=None):
    trip, slot, pos = layout(kp, t_per_slot, shift)
    model.slot = torch.as_tensor(slot, dtype=torch.long).cuda()          # slot of each window for this layout
    out = []
    for s in range(0, len(uids), batch):
        part = uids[s:s + batch]
        x = torch.from_numpy(np.stack([vol[u][trip] for u in part])).cuda().float().div_(255)
        if crop is not None:
            oy, ox, cs = crop
            x = x[..., oy:oy + cs, ox:ox + cs]
        m = torch.from_numpy(np.stack([pres[u][slot] for u in part])).cuda()
        p = torch.from_numpy(np.tile(pos, (len(part), 1))).cuda()
        out.append(torch.sigmoid(model((x - MEAN) / STD, m, p)).float().cpu().numpy())
    return np.concatenate(out)

PRED = {}                                           # (tag, variant) → DataFrame gold × 12 (out-of-fold)
for tag, c in sorted(FAMILIES.items()):
    kp = PREPS[c["PREP"]]
    vol, pres = CACHE[c["PREP"]]
    folds = pd.read_csv(os.path.join(c["_DIR"], "folds.csv")).set_index("StudyInstanceUID").fold
    own = os.path.join(c["_DIR"], "knee_reader.py")
    kr = sys.modules.get("knee_reader_" + tag) or load_module("knee_reader_" + tag, own)
    files = {int(re.search(r"_fold(\d)\.pt$", p).group(1)): p for p in find_inputs(
        lambda p: re.search(rf"/models/{tag}_fold\d\.pt$", p) is not None and FROM_04(p))}
    var = variants(kp, c["T_PER_SLOT"])
    parts = {v: [] for v in var}
    for k, f in sorted(files.items()):
        uids = [u for u in GT.index if u in vol and folds.get(u, -1) == k]
        if not uids:
            continue
        _, slot0, _ = layout(kp, c["T_PER_SLOT"])
        extra = {"enc_kwargs": c["ENC_KWARGS"]} if c.get("ENC_KWARGS") else {}
        model = kr.ReaderNet(c["BACKBONE"], slot0, kp.N_SLOT, pretrained=False, dim=c["DIM"], depth=c["DEPTH"], **extra)
        model.load_state_dict(torch.load(f, map_location="cpu"))
        model = model.cuda().eval()
        for v, runs in var.items():
            p = np.mean([predict(model, kp, uids, vol, pres, t, sh, crop=cr) for t, sh, cr in runs], axis=0)
            parts[v].append(pd.DataFrame(p, index=uids, columns=LABELS))
        del model
        torch.cuda.empty_cache()
    for v in var:
        if parts[v]:
            PRED[(tag, v)] = pd.concat(parts[v])
    n = len(PRED[(tag, "train")]) if (tag, "train") in PRED else 0
    print(f"{tag}: folds {sorted(files)} | {n} gold studies predicted | {(time.time() - T0) / 60:.1f} min", flush=True)

# %% [markdown]
# ## Results

# %%
def aucs(p):
    y = GT.loc[p.index]
    return [roc_auc_score(y[L], p[L]) if 0 < y[L].sum() < len(y) else np.nan for L in LABELS]

VARS = ["train", "dense", "c224", "5crop", "dense c224", "dense 5crop"]
TAGS = sorted({t for t, _ in PRED})
rows, notes = {}, []
for tag in TAGS:
    # sanity: the training layout must reproduce the out-of-fold predictions saved by the training notebooks
    oof = [pd.read_csv(p, index_col=0) for p in find_inputs(lambda p: f"/oof_{tag}_folds" in p and p.endswith(".csv") and FROM_04(p))]
    if oof:
        o = pd.concat(oof)
        o = o[~o.index.duplicated()]
        common = PRED[(tag, "train")].index.intersection(o.index)
        d = float(np.abs(PRED[(tag, "train")].loc[common, LABELS].values - o.loc[common, LABELS].values).max())
        notes.append(f"{tag}: max |train layout - saved OOF| = {d:.4f} on {len(common)} gold studies")
    for v in [v for v in VARS if (tag, v) in PRED]:
        rows[(tag, v)] = [len(PRED[(tag, v)])] + [np.nanmean(aucs(PRED[(tag, v)]))] + aucs(PRED[(tag, v)])
rank = lambda d: d.rank(pct=True)
if len(TAGS) == 2:
    t7, t8 = sorted(TAGS, key=lambda t: not t.startswith("m7"))      # m7 first
    idx = sorted(set(PRED[(t7, "train")].index) & set(PRED[(t8, "train")].index))
    for v8 in [v for v in VARS if (t8, v) in PRED]:
        for w8 in (0.33, 0.5):
            e = (1 - w8) * rank(PRED[(t7, "dense")].loc[idx]) + w8 * rank(PRED[(t8, v8)].loc[idx])
            rows[(f"m7 dense + {w8:.2f} m8", v8)] = [len(idx)] + [np.nanmean(aucs(e))] + aucs(e)
tab = pd.DataFrame(rows, index=["n", "macro"] + LABELS).T
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 400)
print(tab.round(3).to_string())
tab.round(4).to_csv(f"{WORK}/hires_read_check.csv")

s = ["=== SUMMARY 10b hi-res read check ===", *notes]
for (name, v), r in rows.items():
    s.append(f"{name} | {v}: n={r[0]} macro {r[1]:.4f} | " + " ".join(f"{L[:6]} {a:.3f}" for L, a in zip(LABELS, r[2:])))
s += [f"total {(time.time() - T0) / 60:.1f} min", "=== END ==="]
print("\n".join(s))
