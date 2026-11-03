# %% [markdown]
# # 05 — Submission: test DICOMs → 3-plane volumes → fold models → submission.csv
# **Settings:** Accelerator = GPU T4 x2 · **Internet = OFF** (required for submission)
# **Inputs to attach:** competition data · notebook "RSNA Knee 04 Train" (models/, config.json) ·
# notebook "RSNA Knee 00 Wheels" (wheels/) · notebook "RSNA Knee 03 Preprocess" (only for the preprocessing self-check)
#
# 1. installs the DICOM decoders offline from the wheels
# 2. preprocesses every test study with a **verbatim copy** of the 03 preprocessing functions
#    (self-check: 3 training studies are re-processed and compared pixel-by-pixel with the 03 cache)
# 3. runs every fold model found in 04's `models/` (deterministic centre triplets, as in validation) and averages
# 4. writes `submission.csv` (a study that fails anywhere gets the mean prediction instead of crashing the run)
#
# The last cell prints a block starting with `=== SUMMARY` — copy that to Claude.

# %%
import os, glob, re, time, math, json, collections, subprocess, sys
T0 = time.time()

def find_inputs(match):
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        hits += [os.path.join(root, f) for f in files if match(os.path.join(root, f))]
    return sorted(hits)

wheel_dirs = sorted({os.path.dirname(p) for p in find_inputs(lambda p: "/wheels/" in p and p.endswith(".whl"))})
if wheel_dirs:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", wheel_dirs[0],
                        "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "python-gdcm"],
                       capture_output=True, text=True)
    print("decoders install rc", r.returncode, r.stderr[-500:])
else:
    print("!! no wheels attached — JPEG / JPEG-2000 DICOMs may fail to decode")

from multiprocessing import Pool
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
import torch.nn as nn
import timm

N_WORKERS = os.cpu_count()
DATA = os.path.dirname(find_inputs(lambda p: p.endswith("/test_series.csv"))[0])
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
test = pd.read_csv(f"{DATA}/test.csv")
test_series = pd.read_csv(f"{DATA}/test_series.csv")
sample = pd.read_csv(f"{DATA}/sample_submission.csv")
CFG = json.load(open(find_inputs(lambda p: p.endswith("/config.json") and "04-train" in p)[0]))
D, S, T, BACKBONE = CFG["D"], CFG["S"], CFG["T"], CFG["BACKBONE"]
print(f"{len(test)} test studies, {len(test_series)} series | D={D} S={S} T={T} {BACKBONE} | {N_WORKERS} CPUs")

# %% [markdown]
# ## Preprocessing — verbatim copy of 03_preprocess (do not edit one without the other)

# %%
AX = {"x": np.array([1., 0, 0]), "y": np.array([0, 1., 0]), "z": np.array([0, 0, 1.])}
TARGET = {"sag": (AX["y"], -AX["z"], AX["x"]),
          "cor": (AX["x"], -AX["z"], AX["y"]),
          "ax":  (AX["x"], AX["y"], -AX["z"])}
PLANE_OF_CSV = {"Sagittal": "sag", "Coronal": "cor", "Axial": "ax"}
FLIP_AXIS_FOR_LEFT = {"sag": 0, "cor": 2, "ax": 2}

RIGHT_RE = re.compile(r"\b(right|rechts|derecha|dcha|droite|desno|desna|dx|rt)\b", re.I)
LEFT_RE = re.compile(r"\b(left|links|izquierda|izq|gauche|lijevo|lijeva|sx|lt)\b", re.I)

def side_from_tags(d):
    for tag in ("Laterality", "ImageLaterality"):
        v = str(d.get(tag, "") or "").strip().upper()
        if v in ("R", "L"):
            return v, tag
    for tag in ("SeriesDescription", "ProtocolName", "BodyPartExamined", "StudyDescription"):
        v = str(d.get(tag, "") or "")
        r, l = bool(RIGHT_RE.search(v)), bool(LEFT_RE.search(v))
        if r != l:
            return ("R" if r else "L"), tag
    return None, None

def plane_from_normal(n):
    return ["sag", "cor", "ax"][int(np.argmax(np.abs(n)))]

def load_series(path):
    ds = [pydicom.dcmread(f) for f in glob.glob(f"{path}/*.dcm")]
    ds = [d for d in ds if "ImageOrientationPatient" in d and "ImagePositionPatient" in d] or ds
    shape = collections.Counter((int(d.Rows), int(d.Columns)) for d in ds).most_common(1)[0][0]
    ds = [d for d in ds if (int(d.Rows), int(d.Columns)) == shape]
    iop = np.array(ds[0].ImageOrientationPatient, float)
    r, c = iop[:3], iop[3:]
    n = np.cross(r, c)
    pos = np.array([np.dot(n, np.array(d.ImagePositionPatient, float)) for d in ds])
    order = np.argsort(pos)
    ds = [ds[i] for i in order]
    vol = np.stack([d.pixel_array.astype(np.float32) for d in ds])
    if str(ds[0].get("PhotometricInterpretation", "")) == "MONOCHROME1":
        vol = vol.max() - vol
    ps = [float(v) for v in ds[0].get("PixelSpacing", [1, 1])]
    mid = ds[len(ds) // 2]
    centre = (np.array(mid.ImagePositionPatient, float)
              + r * shape[1] * ps[1] / 2 + c * shape[0] * ps[0] / 2)
    side, side_src = side_from_tags(ds[0])
    geo = dict(r=r, c=c, n=n, n_slices=len(ds), x_centre=float(centre[0]), side_tag=side)
    return vol, geo

def canonical(vol, geo, plane):
    h_t, v_t, s_t = TARGET[plane]
    r, c, n = geo["r"], geo["c"], geo["n"]
    if abs(np.dot(c, h_t)) > abs(np.dot(r, h_t)):
        vol = vol.transpose(0, 2, 1)
        r, c = c, r
    if np.dot(r, h_t) < 0:
        vol = vol[:, :, ::-1]
    if np.dot(c, v_t) < 0:
        vol = vol[:, ::-1, :]
    if np.dot(n, s_t) < 0:
        vol = vol[::-1]
    return vol

def resize(vol):
    vol = np.stack([cv2.resize(s, (S, S), interpolation=cv2.INTER_AREA) for s in vol])
    src = np.linspace(0, len(vol) - 1, D)
    lo, frac = np.floor(src).astype(int), (src - np.floor(src))[:, None, None]
    hi = np.minimum(lo + 1, len(vol) - 1)
    vol = vol[lo] * (1 - frac) + vol[hi] * frac
    a, b = np.percentile(vol, [0.5, 99.5])
    return (np.clip((vol - a) / max(b - a, 1e-6), 0, 1) * 255).astype(np.uint8)

def series_rank(row):
    n = row["n_files"]
    shape_ok = 1 if 12 <= n <= 80 else 0
    return (row["Fluid_Sensitive"], row["Fat_Suppression"], shape_ok, -abs(n - 30))

def process_study(args):
    """(uid, split) → (uid, {plane: (D,S,S) uint8}, error string)"""
    uid, split = args
    try:
        series = SERIES[split]
        root = f"{DATA}/{split}_series/{uid}"
        ss = series[series.StudyInstanceUID == uid].copy()
        ss["n_files"] = [len(os.listdir(f"{root}/{s}")) for s in ss.SeriesInstanceUID]
        out, geos, errs = {}, [], []
        for csv_plane, plane in PLANE_OF_CSV.items():
            cand = ss[ss.Anatomical_Plane == csv_plane]
            if cand.empty:
                continue
            best = max(cand.to_dict("records"), key=series_rank)
            try:
                vol, geo = load_series(f"{root}/{best['SeriesInstanceUID']}")
                out[plane] = canonical(vol, geo, plane_from_normal(geo["n"]))
                geos.append(geo)
            except Exception as e:
                errs.append(f"{plane}: {type(e).__name__}: {e}"[:150])
        votes = [g["side_tag"] for g in geos if g.get("side_tag")]
        xs = [g["x_centre"] for g in geos]
        side_x = ("R" if np.median(xs) < 0 else "L") if xs else None
        side = collections.Counter(votes).most_common(1)[0][0] if votes else side_x
        vols = {}
        for plane, vol in out.items():
            if side == "L":
                vol = np.flip(vol, FLIP_AXIS_FOR_LEFT[plane])
            vols[plane] = resize(np.ascontiguousarray(vol))
        return uid, vols, "; ".join(errs)
    except Exception as e:
        return uid, {}, f"study: {type(e).__name__}: {e}"[:200]

SERIES = {"test": test_series}
if os.path.exists(f"{DATA}/train_series.csv"):
    SERIES["train"] = pd.read_csv(f"{DATA}/train_series.csv")

# %% [markdown]
# ## Self-check: preprocessing here == preprocessing used for training

# %%
check = "skipped (03 output not attached)"
shards = find_inputs(lambda p: "/cache/shard_" in p and p.endswith(".npz"))
if shards and "train" in SERIES:
    with np.load(shards[0]) as z:
        keys = z.files[:9]
        uids_chk = list(dict.fromkeys(k.split("/")[0] for k in keys))[:3]
        diffs = []
        for uid in uids_chk:
            _, vols, err = process_study((uid, "train"))
            for plane, v in vols.items():
                key = f"{uid}/{plane}"
                if key in z.files:
                    diffs.append(int(np.abs(v.astype(int) - z[key].astype(int)).max()))
    check = f"{len(diffs)} volumes, max |diff| = {max(diffs) if diffs else 'n/a'}"
print("preprocessing self-check:", check)

# %% [markdown]
# ## Preprocess the test studies (parallel)

# %%
t0 = time.time()
PLANE = {"sag": 0, "cor": 1, "ax": 2}
uids = test.StudyInstanceUID.tolist()
VOL = np.zeros((len(uids), 3, D, S, S), np.uint8)
PRESENT = np.zeros((len(uids), 3), bool)
ERRORS = {}
with Pool(N_WORKERS) as pool:
    for k, (uid, vols, err) in enumerate(pool.imap(process_study, [(u, "test") for u in uids], chunksize=4)):
        i = uids.index(uid)
        for plane, v in vols.items():
            VOL[i, PLANE[plane]] = v
            PRESENT[i, PLANE[plane]] = True
        if err:
            ERRORS[uid] = err
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(uids)} studies  {(time.time() - t0) / 60:.1f} min", flush=True)
t_pre = (time.time() - t0) / 60
print(f"preprocessed {len(uids)} studies in {t_pre:.1f} min | planes present {PRESENT.mean(0).round(3).tolist()} | "
      f"studies with errors {len(ERRORS)}")

# %% [markdown]
# ## Model (verbatim copy of 04's KneeNet) and inference

# %%
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3, 1, 1)

class KneeNet(nn.Module):
    def __init__(self, backbone, pretrained=True, n_out=12):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        C = self.enc.num_features
        self.plane_emb = nn.Parameter(torch.zeros(1, 3, 1, C))
        self.slice_emb = nn.Parameter(torch.zeros(1, 1, T, C))
        self.att = nn.Sequential(nn.Linear(C, 256), nn.Tanh(), nn.Linear(256, n_out))
        self.cls_w = nn.Parameter(torch.randn(n_out, C) * 0.01)
        self.cls_b = nn.Parameter(torch.zeros(n_out))
        self.drop = nn.Dropout(0.2)
    def forward(self, x, present):
        B, P, Tn = x.shape[:3]
        with torch.autocast("cuda", dtype=torch.float16):
            f = self.enc(x.flatten(0, 2)).float()
        f = f.view(B, P, Tn, -1) + self.plane_emb + self.slice_emb[:, :, :Tn]
        f = f.flatten(1, 2)
        a = self.att(f)
        mask = present[:, :, None].expand(B, P, Tn).flatten(1)
        a = a.float().masked_fill(~mask[..., None], -1e4).softmax(1)
        pooled = torch.einsum("bnk,bnc->bkc", a, f.float())
        return (self.drop(pooled) * self.cls_w).sum(-1) + self.cls_b

c = np.clip(np.round(np.linspace(1, D - 2, T)).astype(int), 1, D - 2)
TRIPLETS = np.stack([c - 1, c, c + 1], 1)                        # same centres as validation in 04

@torch.no_grad()
def predict(model, batch=16):
    out = []
    for s in range(0, len(uids), batch):
        x = torch.from_numpy(np.ascontiguousarray(VOL[s:s + batch][:, :, TRIPLETS])).cuda().float().div_(255)
        x = (x - MEAN.cuda()) / STD.cuda()
        m = torch.from_numpy(PRESENT[s:s + batch]).cuda()
        out.append(torch.sigmoid(model(x, m)).float().cpu().numpy())
    return np.concatenate(out)

t0 = time.time()
model_files = find_inputs(lambda p: "/models/" in p and p.endswith(".pt") and "04-train" in p)
preds = []
for f in model_files:
    model = KneeNet(BACKBONE, pretrained=False).cuda().eval()
    model.load_state_dict(torch.load(f, map_location="cuda"))
    preds.append(predict(model))
    print(f"  {os.path.basename(f)} done", flush=True)
P = np.mean(preds, axis=0)
t_inf = (time.time() - t0) / 60

# studies with no usable image at all get the mean prediction (a constant) instead of garbage
empty = ~PRESENT.any(1)
if empty.any():
    P[empty] = P[~empty].mean(0) if (~empty).any() else 0.5

# %% [markdown]
# ## submission.csv

# %%
sub = pd.DataFrame(P, columns=LABELS)
sub.insert(0, "StudyInstanceUID", uids)
sub = sample[["StudyInstanceUID"]].merge(sub, on="StudyInstanceUID", how="left") if set(sample.StudyInstanceUID) == set(uids) else sub
sub = sub[["StudyInstanceUID"] + [c for c in sample.columns if c != "StudyInstanceUID"]]
sub.iloc[:, 1:] = sub.iloc[:, 1:].fillna(0.5).clip(0, 1)
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(sub.head())

s = [f"=== SUMMARY submit | {BACKBONE} | {len(model_files)} fold model(s) ===",
     f"test studies {len(uids)} | preprocessing {t_pre:.1f} min | inference {t_inf:.1f} min | total {(time.time() - T0) / 60:.1f} min",
     f"preprocessing self-check: {check}",
     f"planes present {PRESENT.mean(0).round(3).tolist()} | studies with errors {len(ERRORS)} | empty studies {int(empty.sum())}",
     "first error: " + (next(iter(ERRORS.values())) if ERRORS else "none"),
     f"submission shape {sub.shape} | columns ok {list(sub.columns) == list(sample.columns)} | NaN {int(sub.isna().sum().sum())}",
     "=== END ==="]
print("\n".join(s))
