# %% [markdown]
# # 05c — Submission for our models (M2 ResNet-34, M4 / M5 / M6 readers) and the blend script
# **Settings:** Accelerator = GPU T4 x2 · **Internet = OFF** (required for submission)
# **Inputs to attach:** competition data · "RSNA Knee 00 Wheels" · "RSNA Knee 03c Preprocess M2" · "RSNA Knee 03d Preprocess Zoom" (when M5 is attached) ·
# "RSNA Knee 04c Train M2 A/B/C" · "RSNA Knee 04e Train Reader A/B/C/D" · "RSNA Knee 04f Train Zoom Reader A/B/C" · "RSNA Knee 04g Train EffNet Reader A/B/C"
# (M3 / 04d is ignored: it underfit)
#
# The test-time pipeline lives in **`ours_blend.py`** (written to this notebook's output). It
# 1. installs the DICOM decoders offline, imports `knee_prep.py` (03c), `knee_model.py` (04c) and `knee_reader.py` (04e),
# 2. preprocesses the test studies in chunks (low RAM), runs every fold model of every family,
# 3. rank-averages the families → our prediction, and
# 4. if a `submission.csv` already exists (the public blend in notebook 07 wrote it), blends
#    `final = (1 − W)·rank(public) + W·rank(ours)`; otherwise writes ours alone.
# Notebook 07 (fork of the public blend) runs it with one line at the end: `!python <path>/ours_blend.py 0.3`
#
# This notebook: self-check (preprocessing == 03c cache), runs the script (ours alone → `submission.csv`),
# and reports the 5-fold out-of-fold AUC on the 58 ground-truth studies per family and for the ensemble.

# %%
%%writefile /kaggle/working/ours_blend.py
"""Our test-time pipeline (M2 / M4-reader families) + optional rank blend with an existing submission.csv.
usage: python ours_blend.py [W] [FAMILIES] [WINDOWS]      (each family is preprocessed by the knee_prep.py of its own cache: 03c or 03d)
  W        = weight of our models in the blend (default 0.3)
  FAMILIES = comma-separated TAG prefixes to use, e.g. "m4" or "m2,m4" (default: every family except m3, which underfit);
             "m7:2" gives a family weight 2 in the rank average of the families
  WINDOWS  = "train" (the 32 windows of training) or "dense" (every slice is a window centre: +0.004-0.006 OOF AUC
             on the gold studies for every family, notebook 10 Window Check)"""
import os, sys, re, json, time, subprocess
T0 = time.time()
W = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
_FAM = [f for f in (sys.argv[2] if len(sys.argv) > 2 else "").split(",") if f]
KEEP = [f.split(":")[0] for f in _FAM]
FAM_W = {f.split(":")[0]: float(f.split(":")[1]) for f in _FAM if ":" in f}
WINDOWS = sys.argv[3] if len(sys.argv) > 3 else "train"
assert WINDOWS in ("train", "dense"), WINDOWS
OUT = "/kaggle/working"

def find_inputs(match):
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series", "cache")]
        hits += [os.path.join(root, f) for f in files if match(os.path.join(root, f))]
    return sorted(hits)

import importlib.util
have = lambda: {m: importlib.util.find_spec(m) is not None for m in ("pylibjpeg", "libjpeg", "openjpeg", "gdcm")}
decoders_ok = lambda h: (h["pylibjpeg"] and h["libjpeg"] and h["openjpeg"]) or h["gdcm"]
# Only folders that really hold the decoder wheels (07b lesson: the public notebook ships its own "wheels" folder
# without them; taking the first folder found left the run with no JPEG decoder at all).
is_decoder = lambda f: f.endswith(".whl") and f.lower().startswith(("pylibjpeg", "python_gdcm", "python-gdcm"))
wheel_dirs = sorted({os.path.dirname(p) for p in find_inputs(lambda p: is_decoder(os.path.basename(p)))})
print("[ours] decoders before install:", have(), "| wheel dirs", wheel_dirs, flush=True)
if wheel_dirs and not all(have().values()):
    pip = [sys.executable, "-m", "pip", "install", "-q", "--no-index"]
    links = [a for d in wheel_dirs for a in ("--find-links", d)]
    r = subprocess.run(pip + links + ["pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "python-gdcm"],
                       capture_output=True, text=True)
    print("[ours] decoders install rc", r.returncode, flush=True)
    if r.returncode != 0:      # e.g. another notebook pinned packages: install the decoder wheels one by one, no dependencies
        print("[ours] pip stderr:", r.stderr[-1200:], flush=True)
        for d in wheel_dirs:
            for whl in sorted(f for f in os.listdir(d) if is_decoder(f)):
                r = subprocess.run(pip + ["--no-deps", os.path.join(d, whl)], capture_output=True, text=True)
                print(f"[ours]   {whl}: rc {r.returncode} {r.stderr[-300:] if r.returncode else ''}", flush=True)
    importlib.invalidate_caches()
print("[ours] decoders after install:", have(), flush=True)
if not decoders_ok(have()):
    raise RuntimeError("no JPEG / JPEG-2000 DICOM decoder available: compressed series would be read as blank images")

from multiprocessing import Pool
import numpy as np
import pandas as pd
import torch

FROM_04 = lambda p: re.search(r"04[c-z]-train", p) is not None

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# one knee_prep.py per preprocessing notebook: "03c" (130 mm, 240 px: M2, M4) and "03d" (100 mm, 224 px: M5)
PREP_FILES = {}
for p in find_inputs(lambda p: p.endswith("/knee_prep.py")):
    hit = re.search(r"(03[a-z])-preprocess", p)
    if hit:
        PREP_FILES.setdefault(hit.group(1), p)
for mod in ("knee_model.py", "knee_reader.py"):
    hit = find_inputs(lambda p: p.endswith("/" + mod) and FROM_04(p))
    if hit:
        sys.path.insert(0, os.path.dirname(hit[0]))

DATA = os.path.dirname(find_inputs(lambda p: p.endswith("/test_series.csv"))[0])
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
uids = pd.read_csv(f"{DATA}/test.csv").StudyInstanceUID.tolist()
sample = pd.read_csv(f"{DATA}/sample_submission.csv")

FAMILIES = {}
for p in find_inputs(lambda p: re.search(r"/config_[^/]+\.json$", p) is not None and FROM_04(p)):
    c = json.load(open(p))
    keep = any(c["TAG"].startswith(f) for f in KEEP) if KEEP else not c["TAG"].startswith("m3")
    if not keep:
        continue
    c["PREP"] = c.get("PREP", "03c")
    assert c["PREP"] in PREP_FILES, f"{p}: preprocessing notebook {c['PREP']} is not attached"
    c["_DIR"] = os.path.dirname(p)                        # the notebook output this family was trained in
    FAMILIES.setdefault(c["TAG"], c)
PREPS = {n: load_module(f"knee_prep_{n}", PREP_FILES[n]) for n in sorted({c["PREP"] for c in FAMILIES.values()})}
test_series = pd.read_csv(f"{DATA}/test_series.csv")
for kp in PREPS.values():
    kp.setup(DATA, {"test": test_series})
for c in FAMILIES.values():
    kp = PREPS[c["PREP"]]
    assert c["DS"] == kp.DS and c["S"] == kp.S, f"{c['TAG']}: trained on a different cache layout"
KP0 = next(iter(PREPS.values()))                         # slots / slices are the same in every preprocessing
print("[ours] preprocessing:", {n: (kp.S, kp.CROP_MM) for n, kp in PREPS.items()}, flush=True)

def build(c):
    """→ (model on the GPU, window indices (N, 3), slot of each window (N,), positions (N,) or None)"""
    kp = PREPS[c["PREP"]]
    if c.get("ARCH") == "reader":
        # each family uses the knee_reader.py written by its own training notebook (later families add options)
        own = os.path.join(c["_DIR"], "knee_reader.py")
        if os.path.exists(own):
            kr = sys.modules.get("knee_reader_" + c["TAG"]) or load_module("knee_reader_" + c["TAG"], own)
        else:
            import knee_reader as kr
        t_eval = [d - 2 for d in kp.DS] if WINDOWS == "dense" else c["T_PER_SLOT"]
        trip, slot_of_tok, pos = kr.windows(t_eval, kp.DS, kp.OFF)
        extra = {"enc_kwargs": c["ENC_KWARGS"]} if c.get("ENC_KWARGS") else {}
        m = kr.ReaderNet(c["BACKBONE"], slot_of_tok, kp.N_SLOT, pretrained=False, dim=c["DIM"], depth=c["DEPTH"], **extra)
        return m, trip, slot_of_tok, pos
    import knee_model as km
    trip, slot_of_tok = km.triplets(c["T_PER_SLOT"], kp.DS, kp.OFF)
    m = km.KneeNet2(c["BACKBONE"], c["N_TOK"], pretrained=False, n_layers=c.get("N_LAYERS", 0), img=c.get("IMG"))
    return m, trip, slot_of_tok, None

MODELS, LAYOUT = {}, {}                                  # TAG → models of all folds (on the GPU), window layout
for tag, c in FAMILIES.items():
    files = sorted({os.path.basename(f): f for f in find_inputs(
        lambda p: f"/models/{tag}_fold" in p and p.endswith(".pt") and FROM_04(p))}.values())
    for f in files:
        m, trip, slot_of_tok, pos = build(c)
        m.load_state_dict(torch.load(f, map_location="cpu"))
        MODELS.setdefault(tag, []).append(m.cuda().eval())
        LAYOUT[tag] = (trip, slot_of_tok, pos)
print(f"[ours] {len(uids)} test studies | " + ", ".join(f"{t}: {len(v)} folds" for t, v in MODELS.items()), flush=True)
assert MODELS, "no model files found for the requested families"

def work(uid):
    """→ uid, {prep: (volume, present flags)}, study-level error, series-level errors (a series that failed to load),
    number of series. Each preprocessing reads the study itself: exactly the code that built its training cache."""
    out, err, s_err, n_ser = {}, "", [], 0
    for i, (name, kp) in enumerate(PREPS.items()):
        try:
            _, vol, present, meta, _ = kp.process_study(uid, "test")
            out[name] = (vol, present)
            if i == 0:
                s_err, n_ser = [m["error"] for m in meta if m.get("error")], len(meta)
        except Exception as e:
            out[name] = (None, np.zeros(kp.N_SLOT, bool))
            err = f"{type(e).__name__}: {e}"[:200]
    return uid, out, err, s_err, n_ser

@torch.no_grad()
def predict(vols, presents, batch=8):
    """vols {prep: (n, N_SLICES, S, S) uint8} → {tag: (n, 12) mean probability over that family's folds}"""
    batch = 4 if WINDOWS == "dense" else batch            # 70 windows per study instead of 32
    out = {}
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 1, 3, 1, 1)
    for tag, models in MODELS.items():
        vol, present = vols[FAMILIES[tag]["PREP"]], presents[FAMILIES[tag]["PREP"]]
        trip, slot_of_tok, pos = LAYOUT[tag]
        mask = present[:, slot_of_tok]
        ps = []
        for s in range(0, len(vol), batch):
            x = (torch.from_numpy(np.ascontiguousarray(vol[s:s + batch][:, trip])).cuda().float().div_(255) - mean) / std
            m = torch.from_numpy(mask[s:s + batch]).cuda()
            extra = () if pos is None else (torch.from_numpy(np.tile(pos, (len(x), 1))).cuda(),)
            ps.append(np.mean([torch.sigmoid(md(x, m, *extra)).float().cpu().numpy() for md in models], axis=0))
        out[tag] = np.concatenate(ps)
    return out

CHUNK = 96                                               # studies per chunk: ~0.5 GB of pixels per preprocessing
PRED = {tag: np.zeros((len(uids), 12), np.float32) for tag in MODELS}
PRESENT = np.zeros((len(uids), KP0.N_SLOT), bool)
ERRORS, SERIES_ERRORS, N_SERIES = {}, [], 0
with Pool(os.cpu_count()) as pool:
    for c0 in range(0, len(uids), CHUNK):
        part = uids[c0:c0 + CHUNK]
        vols = {n: np.zeros((len(part), kp.N_SLICES, kp.S, kp.S), np.uint8) for n, kp in PREPS.items()}
        press = {n: np.zeros((len(part), kp.N_SLOT), bool) for n, kp in PREPS.items()}
        for uid, out, err, s_err, n_ser in pool.imap_unordered(work, part, chunksize=2):
            j = part.index(uid)
            for n, (v, p) in out.items():
                if v is not None:
                    vols[n][j], press[n][j] = v, p
            if err:
                ERRORS[uid] = err
            SERIES_ERRORS += s_err
            N_SERIES += n_ser
        for tag, p in predict(vols, press).items():
            PRED[tag][c0:c0 + len(part)] = p
        PRESENT[c0:c0 + len(part)] = next(iter(press.values()))
        print(f"[ours] {min(c0 + CHUNK, len(uids))}/{len(uids)} studies  {(time.time() - T0) / 60:.1f} min", flush=True)
print(f"[ours] series loaded {N_SERIES - len(SERIES_ERRORS)}/{N_SERIES} | series errors {len(SERIES_ERRORS)}"
      + (f" | first: {SERIES_ERRORS[0]}" if SERIES_ERRORS else ""), flush=True)
if len(ERRORS) > 0.03 * len(uids) or len(SERIES_ERRORS) > 0.03 * max(N_SERIES, 1):
    raise RuntimeError(f"too many unreadable inputs: {len(ERRORS)} studies, {len(SERIES_ERRORS)} of {N_SERIES} series "
                       f"({(SERIES_ERRORS or list(ERRORS.values()))[0]})")

rank01 = lambda a: pd.DataFrame(a).rank(pct=True).to_numpy()
empty = ~PRESENT.any(1)
for tag in PRED:                                         # no image at all → the family's mean prediction
    if empty.any() and (~empty).any():
        PRED[tag][empty] = PRED[tag][~empty].mean(0)
    pd.DataFrame(PRED[tag], columns=LABELS).assign(StudyInstanceUID=uids).to_csv(f"{OUT}/pred_{tag}.csv", index=False)
fam_w = [next((w for f, w in FAM_W.items() if tag.startswith(f)), 1.0) for tag in PRED]
ours = next(iter(PRED.values())) if len(PRED) == 1 else np.average([rank01(p) for p in PRED.values()], axis=0, weights=fam_w)

cols = [c for c in sample.columns if c != "StudyInstanceUID"]
ours_df = pd.DataFrame(ours, columns=LABELS).assign(StudyInstanceUID=uids).set_index("StudyInstanceUID")[cols]
ours_df.to_csv(f"{OUT}/submission_ours.csv")
public_path = f"{OUT}/submission.csv"
if os.path.exists(public_path) and W < 1:
    pub = pd.read_csv(public_path).set_index("StudyInstanceUID").reindex(ours_df.index)[cols]
    pub.to_csv(f"{OUT}/submission_public.csv")
    final = (1 - W) * pub.rank(pct=True).fillna(0.5) + W * ours_df.rank(pct=True)
    mode = f"blend W={W} (public {1 - W:.2f})"
    corr = float(np.mean([pub[c].rank().corr(ours_df[c].rank()) for c in cols]))
else:
    final, mode, corr = ours_df, "ours only", float("nan")
final = sample[["StudyInstanceUID"]].merge(final.reset_index(), on="StudyInstanceUID", how="left")[sample.columns]
final.iloc[:, 1:] = final.iloc[:, 1:].fillna(0.5).clip(0, 1)
final.to_csv(public_path, index=False)
s = (f"=== OURS | {mode} | families {list(PRED)} | windows {WINDOWS} | family weights {fam_w} | studies {len(uids)} | errors {len(ERRORS)} | empty {int(empty.sum())} | "
     f"series errors {len(SERIES_ERRORS)}/{N_SERIES} | "
     f"rank corr public~ours {corr:.3f} | {(time.time() - T0) / 60:.1f} min | shape {final.shape} NaN {int(final.isna().sum().sum())} ===")
print(s, flush=True)
open(f"{OUT}/ours_summary.txt", "w").write(s + ("\nfirst error: " + next(iter(ERRORS.values())) if ERRORS else ""))

# %% [markdown]
# ## Self-check: preprocessing at test time == the 03c training cache

# %%
import os, sys, re, json, subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def find_inputs(match):
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        hits += [os.path.join(root, f) for f in files if match(os.path.join(root, f))]
    return sorted(hits)

FROM_04 = lambda p: re.search(r"04[c-z]-train", p) is not None
RUN_FAMILIES = "m7,m9:2"     # families in this notebook's own submission.csv ("" = every family except m3); notebook 14: M7 + 2 x M9 = 0.9315 OOF on 58 gold
RUN_WINDOWS = "dense"        # "train" or "dense" (every slice a window centre, see notebook 10)
import importlib.util
DATA = os.path.dirname(find_inputs(lambda p: p.endswith("/test_series.csv"))[0])
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
wheel_dirs = sorted({os.path.dirname(p) for p in find_inputs(lambda p: "/wheels/" in p and p.endswith(".whl"))})
if wheel_dirs:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", wheel_dirs[0],
                    "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "python-gdcm"])
checks = []
for prep_file in find_inputs(lambda p: p.endswith("/knee_prep.py") and re.search(r"03[a-z]-preprocess", p) is not None):
    prep = re.search(r"(03[a-z])-preprocess", prep_file).group(1)
    spec = importlib.util.spec_from_file_location(f"knee_prep_{prep}", prep_file)
    kp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kp)
    shards = find_inputs(lambda p: "/cache/shard_" in p and p.endswith(".npz") and f"{prep}-preprocess" in p)
    if not (shards and os.path.exists(f"{DATA}/train_series.csv")):
        checks.append(f"{prep}: skipped (cache or train data not attached)")
        continue
    kp.setup(DATA, {"train": pd.read_csv(f"{DATA}/train_series.csv")})
    with np.load(shards[0]) as z:
        uids_chk = [k.split("/")[0] for k in z.files if k.endswith("/vol")][:3]
        diffs = []
        for uid in uids_chk:
            _, vol, present, _, _ = kp.process_study(uid, "train")
            diffs.append(int(np.abs(vol.astype(int) - z[f"{uid}/vol"].astype(int)).max()))
            diffs.append(int((present != z[f"{uid}/present"]).sum()))
    checks.append(f"{prep} (S {kp.S}, {kp.CROP_MM:.0f} mm): {len(uids_chk)} studies, max |diff| = {max(diffs)}")
check = " | ".join(checks) if checks else "skipped (no preprocessing notebook attached)"
print("preprocessing self-check:", check)

# %% [markdown]
# ## Run our pipeline (no public submission.csv in this notebook → our models alone)

# %%
if os.path.exists("/kaggle/working/submission.csv"):
    os.remove("/kaggle/working/submission.csv")
!python /kaggle/working/ours_blend.py 1.0 {RUN_FAMILIES} {RUN_WINDOWS}

# %% [markdown]
# ## Out-of-fold AUC on the 58 ground-truth studies (per family and rank ensemble)

# %%
rank01 = lambda a: pd.DataFrame(a).rank(pct=True).to_numpy()

def gt_auc_report(oof, name):
    tr = pd.read_csv(f"{DATA}/train.csv").set_index("StudyInstanceUID")
    gt = tr[tr[LABELS].notna().all(axis=1)][LABELS].astype(int)
    common = gt.index.intersection(oof.index)
    aucs = [roc_auc_score(gt.loc[common, L], oof.loc[common, L]) for L in LABELS]
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(300):                                 # 300 resamples: enough for a 95% interval, 7x faster
        b = rng.choice(len(common), len(common))
        yb, pb = gt.loc[common].values[b], oof.loc[common].values[b]
        boot.append(np.nanmean([roc_auc_score(yb[:, j], pb[:, j]) for j in range(12) if 0 < yb[:, j].sum() < len(b)]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return (f"{name}: n={len(common)} GT, {len(oof)} OOF rows, macro AUC {np.mean(aucs):.4f} [95% CI {lo:.3f}, {hi:.3f}] | "
            + ", ".join(f"{L} {a:.2f}" for L, a in zip(LABELS, aucs)))

notes, OOFS = [], {}
if os.path.exists(f"{DATA}/train.csv"):
    tags = {json.load(open(p))["TAG"] for p in find_inputs(lambda p: re.search(r"/config_[^/]+\.json$", p) is not None and FROM_04(p))}
    run = [f.split(":")[0] for f in RUN_FAMILIES.split(",") if f]
    for tag in sorted(t for t in tags if not t.startswith("m3") and (not run or any(t.startswith(f + "_") for f in run))):
        parts = [pd.read_csv(p, index_col=0) for p in find_inputs(
            lambda p: f"/oof_{tag}_folds" in p and p.endswith(".csv") and FROM_04(p))]
        if parts:
            o = pd.concat(parts)
            OOFS[tag] = o[~o.index.duplicated()]
            notes.append(gt_auc_report(OOFS[tag], tag))
    import itertools
    if len(OOFS) > 1:                                    # the submitted ensemble only (all combinations: notebook 14)
        wts = {f.split(":")[0]: float(f.split(":")[1]) if ":" in f else 1.0 for f in RUN_FAMILIES.split(",") if f}
        combo = sorted(OOFS)
        w = [next((v for k, v in wts.items() if t.startswith(k + "_")), 1.0) for t in combo]
        idx = sorted(set.intersection(*[set(OOFS[t].index) for t in combo]))
        ens = pd.DataFrame(np.average([rank01(OOFS[t].loc[idx].values) for t in combo], axis=0, weights=w), index=idx, columns=LABELS)
        notes.append(gt_auc_report(ens, "rank-ensemble " + " + ".join(f"{wi:g}x{t}" for t, wi in zip(combo, w))))

# %% [markdown]
# ## Summary — copy this block to Claude

# %%
ours = open("/kaggle/working/ours_summary.txt").read() if os.path.exists("/kaggle/working/ours_summary.txt") else "ours_blend.py did not finish"
print("\n".join(["=== SUMMARY 05c ===", f"preprocessing self-check: {check}", ours, "OOF GT:", *notes, "=== END ==="]))
