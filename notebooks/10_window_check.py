# %% [markdown]
# # 10 — Do more slice windows at test time help? (out-of-fold check on the gold studies)
# **Settings:** Accelerator = GPU T4 (a few minutes) · Internet = OFF
# **Inputs:** competition data · 03c and 03d caches · every 04e-04h training notebook (fold models, folds.csv, config)
#
# Why: our readers see 32 three-slice windows per study (8, 8, 8, 4, 4 per series slot) in training and at test time.
# The authors of the public RAPTOR model report 0.940 → 0.934 when they cut 94 windows to 24 at inference, so the
# number of windows matters for them. Our model takes any number of windows (slot and position embeddings, attention
# pooling), so denser reading needs no retraining. This notebook predicts every gold study with the model of the fold
# that did **not** train on it, with several window layouts, and compares the macro AUC:
# * `train`  : the training layout (must reproduce the saved out-of-fold predictions);
# * `dense`  : every slice as a window centre (18, 18, 18, 8, 8 = 70 windows);
# * `double` : twice the training count (16, 16, 16, 8, 8);
# * `2 phases` / `3 phases` : the training count, read 2 or 3 times at shifted centres, probabilities averaged.

# %%
import os, re, sys, json, time, importlib.util, itertools
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
    for sh in find_inputs(lambda p: "/cache/shard_" in p and p.endswith(".npz") and prep in p):
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
    dense = [d - 2 for d in kp.DS]
    return {"train": [(t, 0.0)], "dense": [(dense, 0.0)], "double": [([2 * x for x in t], 0.0)],
            "2 phases": [(t, 0.0), (t, 0.5)], "3 phases": [(t, 0.0), (t, 1 / 3), (t, 2 / 3)]}

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1).cuda()
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1).cuda()

@torch.no_grad()
def predict(model, kp, uids, vol, pres, t_per_slot, shift, batch=4):
    trip, slot, pos = layout(kp, t_per_slot, shift)
    model.slot = torch.as_tensor(slot, dtype=torch.long).cuda()          # slot of each window for this layout
    out = []
    for s in range(0, len(uids), batch):
        part = uids[s:s + batch]
        x = torch.from_numpy(np.stack([vol[u][trip] for u in part])).cuda().float().div_(255)
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
            p = np.mean([predict(model, kp, uids, vol, pres, t, sh) for t, sh in runs], axis=0)
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

VARS = ["train", "dense", "double", "2 phases", "3 phases"]
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
    for v in VARS:
        rows[(tag, v)] = [len(PRED[(tag, v)])] + [np.nanmean(aucs(PRED[(tag, v)]))] + aucs(PRED[(tag, v)])
rank = lambda d: d.rank(pct=True)
ENS = {}
for r in range(2, len(TAGS) + 1):
    for combo in itertools.combinations(TAGS, r):
        idx = sorted(set.intersection(*[set(PRED[(t, "train")].index) for t in combo]))
        for v in VARS:
            e = sum(rank(PRED[(t, v)].loc[idx]) for t in combo) / len(combo)
            ENS[("+".join(t[:2] for t in combo), v)] = e
            rows[("+".join(t[:2] for t in combo), v)] = [len(idx)] + [np.nanmean(aucs(e))] + aucs(e)
tab = pd.DataFrame(rows, index=["n", "macro"] + LABELS).T
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 400)
print(tab.round(3).to_string())
tab.round(4).to_csv(f"{WORK}/window_check.csv")

# paired bootstrap (same resampled studies for both layouts): is a layout better than the training layout?
rng = np.random.default_rng(0)
def paired(pa, pb, n=2000):
    y = GT.loc[pa.index].values
    a, b = pa[LABELS].values, pb[LABELS].values
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        ok = [j for j in range(12) if 0 < y[i, j].sum() < len(i)]
        d.append(np.mean([roc_auc_score(y[i, j], b[i, j]) - roc_auc_score(y[i, j], a[i, j]) for j in ok]))
    d = np.array(d)
    return float(d.mean()), float((d > 0).mean())

s = ["=== SUMMARY 10 window check ===", *notes]
names = TAGS + sorted({k for k, _ in ENS}, key=len)
for name in names:
    get = (lambda v: PRED[(name, v)]) if name in TAGS else (lambda v: ENS[(name, v)])
    line = f"{name} (n={len(get('train'))}): train {np.nanmean(aucs(get('train'))):.4f}"
    for v in VARS[1:]:
        m, frac = paired(get("train"), get(v))
        line += f" | {v} {np.nanmean(aucs(get(v))):.4f} ({m:+.4f}, better in {frac:.0%})"
    s.append(line)
s += [f"total {(time.time() - T0) / 60:.1f} min", "=== END ==="]
print("\n".join(s))
