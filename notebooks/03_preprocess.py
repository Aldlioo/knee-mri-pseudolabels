# %% [markdown]
# # 03 — Preprocess: DICOM → canonical 3-plane volumes (uint8 cache)
# **Settings:** Accelerator = None (CPU) · Internet = ON (installs JPEG/JPEG-2000 DICOM decoders)
#
# For every study:
# 1. pick one series per plane (sagittal / coronal / axial): fluid-sensitive first, then fat-suppressed,
#    then the one closest to a normal 2D stack (not a localizer, not a huge 3D block)
# 2. sort slices by position along the slice normal (not by InstanceNumber)
# 3. re-orient to a fixed patient frame from ImageOrientationPatient (same view for every vendor)
# 4. knee side (L/R) from DICOM tags, else from the knee's position left/right of the scanner centre;
#    **left knees are mirrored** so that in every volume the *lateral* side comes first
#    (sagittal: slice 0 = lateral; coronal/axial: image-left = lateral) → medial/lateral labels line up
# 5. resize to `D × S × S`, normalise intensities per volume (0.5–99.5 percentile) → uint8
#
# `PREVIEW = True`  → 48 studies, QC pictures, runtime/size projection (a few minutes)
# `PREVIEW = False` → all train studies (the log prints progress)
#
# Output: `cache/shard_XXX.npz` (keys `<StudyInstanceUID>/<sag|cor|ax>`), `series_meta.csv`, `study_meta.csv`.
# The last cell prints a block starting with `=== SUMMARY` — copy that to Claude.

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "python-gdcm"])

import os, glob, re, time, math, collections
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pydicom
import cv2
import matplotlib.pyplot as plt

PREVIEW = False
D, S = 32, 224                  # slices, in-plane size
STUDIES_PER_SHARD = 200
N_WORKERS = os.cpu_count()
WORK = "/kaggle/working"
CACHE = f"{WORK}/cache"
os.makedirs(CACHE, exist_ok=True)

DATA = os.path.dirname((glob.glob("/kaggle/input/*/train_series.csv") +
                        glob.glob("/kaggle/input/*/*/train_series.csv"))[0])
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
train = pd.read_csv(f"{DATA}/train.csv")
series = pd.read_csv(f"{DATA}/train_series.csv")
print(len(train), "studies,", len(series), "series,", N_WORKERS, "CPU workers")

# %% [markdown]
# ## Geometry helpers
# DICOM patient frame is LPS: +x = patient left, +y = posterior, +z = superior.
# Target display per plane (horizontal axis, vertical axis, slice-stacking axis):
# - sagittal: anterior→posterior, superior→inferior, stack right→left
# - coronal: right→left, superior→inferior, stack anterior→posterior
# - axial: right→left, anterior→posterior, stack superior→inferior

# %%
AX = {"x": np.array([1., 0, 0]), "y": np.array([0, 1., 0]), "z": np.array([0, 0, 1.])}
TARGET = {"sag": (AX["y"], -AX["z"], AX["x"]),
          "cor": (AX["x"], -AX["z"], AX["y"]),
          "ax":  (AX["x"], AX["y"], -AX["z"])}
PLANE_OF_CSV = {"Sagittal": "sag", "Coronal": "cor", "Axial": "ax"}
FLIP_AXIS_FOR_LEFT = {"sag": 0, "cor": 2, "ax": 2}   # the patient x axis in each canonical volume

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
    """Read one series → (float32 volume sorted along the normal, geometry dict)."""
    ds = [pydicom.dcmread(f) for f in glob.glob(f"{path}/*.dcm")]
    ds = [d for d in ds if "ImageOrientationPatient" in d and "ImagePositionPatient" in d] or ds
    shape = collections.Counter((int(d.Rows), int(d.Columns)) for d in ds).most_common(1)[0][0]
    ds = [d for d in ds if (int(d.Rows), int(d.Columns)) == shape]
    iop = np.array(ds[0].ImageOrientationPatient, float)
    r, c = iop[:3], iop[3:]                      # r: direction of increasing column, c: increasing row
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
    geo = dict(r=r, c=c, n=n, n_slices=len(ds), rows=shape[0], cols=shape[1],
               px_row=ps[0], px_col=ps[1], thick=float(ds[0].get("SliceThickness", 0) or 0),
               x_centre=float(centre[0]), side_tag=side, side_src=side_src,
               maker=str(ds[0].get("Manufacturer", "")), field=float(ds[0].get("MagneticFieldStrength", 0) or 0),
               desc=str(ds[0].get("SeriesDescription", "")))
    return vol, geo

def canonical(vol, geo, plane):
    """Reorient to the target display frame of `plane` (see table above)."""
    h_t, v_t, s_t = TARGET[plane]
    r, c, n = geo["r"], geo["c"], geo["n"]
    if abs(np.dot(c, h_t)) > abs(np.dot(r, h_t)):        # rows/cols swapped relative to the target
        vol = vol.transpose(0, 2, 1)
        r, c = c, r
    if np.dot(r, h_t) < 0:
        vol = vol[:, :, ::-1]
    if np.dot(c, v_t) < 0:
        vol = vol[:, ::-1, :]
    if np.dot(n, s_t) < 0:                                 # slices were sorted along +n
        vol = vol[::-1]
    return vol

def resize(vol):
    """(n, h, w) float → (D, S, S) uint8 with per-volume percentile normalisation."""
    vol = np.stack([cv2.resize(s, (S, S), interpolation=cv2.INTER_AREA) for s in vol])
    src = np.linspace(0, len(vol) - 1, D)
    lo, frac = np.floor(src).astype(int), (src - np.floor(src))[:, None, None]
    hi = np.minimum(lo + 1, len(vol) - 1)
    vol = vol[lo] * (1 - frac) + vol[hi] * frac
    a, b = np.percentile(vol, [0.5, 99.5])
    return (np.clip((vol - a) / max(b - a, 1e-6), 0, 1) * 255).astype(np.uint8)

# %% [markdown]
# ## Series selection + per-study processing

# %%
def series_rank(row):
    n = row["n_files"]
    shape_ok = 1 if 12 <= n <= 80 else 0                  # typical 2D stack, not a localizer or huge 3D block
    return (row["Fluid_Sensitive"], row["Fat_Suppression"], shape_ok, -abs(n - 30))

def process_study(uid, split="train"):
    root = f"{DATA}/{split}_series/{uid}"
    ss = series[series.StudyInstanceUID == uid].copy()
    ss["n_files"] = [len(os.listdir(f"{root}/{s}")) for s in ss.SeriesInstanceUID]
    out, meta = {}, []
    for csv_plane, plane in PLANE_OF_CSV.items():
        cand = ss[ss.Anatomical_Plane == csv_plane]
        if cand.empty:
            continue
        best = max(cand.to_dict("records"), key=series_rank)
        try:
            vol, geo = load_series(f"{root}/{best['SeriesInstanceUID']}")
            geo_plane = plane_from_normal(geo["n"])
            vol = canonical(vol, geo, geo_plane)
            out[plane] = vol
            meta.append(dict(StudyInstanceUID=uid, plane=plane, geo_plane=geo_plane,
                             SeriesInstanceUID=best["SeriesInstanceUID"], n_candidates=len(cand),
                             fluid=best["Fluid_Sensitive"], fatsat=best["Fat_Suppression"],
                             **{k: v for k, v in geo.items() if k not in ("r", "c", "n")}, error=""))
        except Exception as e:
            meta.append(dict(StudyInstanceUID=uid, plane=plane, SeriesInstanceUID=best["SeriesInstanceUID"],
                             error=f"{type(e).__name__}: {e}"[:200]))
    # knee side: majority of tag votes, else the sign of the knee centre's x (LPS: x < 0 = patient right)
    ok = [m for m in meta if not m["error"]]
    votes = [m["side_tag"] for m in ok if m.get("side_tag")]
    xs = [m["x_centre"] for m in ok]
    side_x = ("R" if np.median(xs) < 0 else "L") if xs else None
    if votes:
        side, side_src = collections.Counter(votes).most_common(1)[0][0], "tag"
    else:
        side, side_src = side_x, "x_centre"
    vols = {}
    for plane, vol in out.items():
        if side == "L":
            vol = np.flip(vol, FLIP_AXIS_FOR_LEFT[plane])     # lateral first / image-left = lateral
        vols[plane] = resize(np.ascontiguousarray(vol))
    study = dict(StudyInstanceUID=uid, side=side, side_src=side_src, side_x=side_x,
                 side_tag=collections.Counter(votes).most_common(1)[0][0] if votes else None,
                 planes="".join(p[0] for p in ("sag", "cor", "ax") if p in vols), n_series=len(ss))
    return vols, meta, study

def process_shard(args):
    shard_id, uids, split = args
    t0, arrays, metas, studies = time.time(), {}, [], []
    for uid in uids:
        try:
            vols, meta, study = process_study(uid, split)
        except Exception as e:
            vols, meta, study = {}, [], dict(StudyInstanceUID=uid, error=f"{type(e).__name__}: {e}"[:200])
        for plane, v in vols.items():
            arrays[f"{uid}/{plane}"] = v
        metas += meta
        studies.append(study)
    np.savez_compressed(f"{CACHE}/shard_{shard_id:03d}.npz", **arrays)
    for s in studies:
        s["shard"] = shard_id
    return metas, studies, time.time() - t0

# %% [markdown]
# ## Run

# %%
gt = train[LABELS].notna().all(axis=1)
uids = train.StudyInstanceUID.tolist()
if PREVIEW:   # 24 ground-truth + 24 other studies
    uids = train[gt].StudyInstanceUID.head(24).tolist() + train[~gt].StudyInstanceUID.sample(24, random_state=0).tolist()
    per_shard = 12
else:
    per_shard = STUDIES_PER_SHARD
jobs = [(i, uids[j:j + per_shard], "train") for i, j in enumerate(range(0, len(uids), per_shard))]

t0, all_meta, all_studies, shard_secs = time.time(), [], [], []
with Pool(N_WORKERS) as pool:
    for k, (metas, studies, secs) in enumerate(pool.imap_unordered(process_shard, jobs)):
        all_meta += metas
        all_studies += studies
        shard_secs.append(secs)
        done = len(all_studies)
        rate = done / (time.time() - t0)
        print(f"shards {k + 1}/{len(jobs)}  studies {done}/{len(uids)}  {rate:.2f} studies/s  "
              f"eta {(len(uids) - done) / rate / 60:.1f} min", flush=True)
elapsed = time.time() - t0

series_meta = pd.DataFrame(all_meta)
study_meta = pd.DataFrame(all_studies)
series_meta.to_csv(f"{WORK}/series_meta.csv", index=False)
study_meta.to_csv(f"{WORK}/study_meta.csv", index=False)
cache_bytes = sum(os.path.getsize(f) for f in glob.glob(f"{CACHE}/*.npz"))
print(f"done in {elapsed / 60:.1f} min, cache {cache_bytes / 1e9:.2f} GB")

# %% [markdown]
# ## Quality control

# %%
ok = series_meta[series_meta["error"] == ""]
errs = series_meta[series_meta["error"] != ""]
print("series errors:", len(errs))
print(errs["error"].str.slice(0, 80).value_counts().head(10).to_string())
print("\nplane label (CSV) vs geometry plane mismatches:", int((ok.plane != ok.geo_plane).sum()))
print("\nplanes present per study:\n", study_meta["planes"].value_counts().to_string())
print("\nside source:\n", study_meta["side_src"].value_counts(dropna=False).to_string())
both = study_meta.dropna(subset=["side_tag", "side_x"])
side_agree = (both.side_tag == both.side_x).mean() if len(both) else float("nan")
print(f"\nside: tag vs x-centre agreement on {len(both)} studies: {side_agree:.3f}")
print("\nchosen series: fluid-sensitive rate by plane:\n", ok.groupby("plane")["fluid"].mean().round(3).to_string())
print("\nchosen series: fat-sat rate by plane:\n", ok.groupby("plane")["fatsat"].mean().round(3).to_string())
print("\noriginal slices per chosen series:\n", ok.groupby("plane")["n_slices"].describe().round(0).to_string())

# %%
# Visual check: middle slice of each plane for 6 studies (3 right knees, 3 left knees).
# After canonicalisation every knee should look the same way round:
#  sagittal: anterior (patella) on the LEFT of the image; slice shown = middle
#  coronal / axial: LATERAL side (fibula in coronal) on the LEFT of the image
z = np.load(glob.glob(f"{CACHE}/*.npz")[0])
show = (study_meta[study_meta.side == "R"].head(3).StudyInstanceUID.tolist() +
        study_meta[study_meta.side == "L"].head(3).StudyInstanceUID.tolist())
fig, axes = plt.subplots(len(show), 4, figsize=(14, 3.4 * len(show)))
for row, uid in zip(np.atleast_2d(axes), show):
    shard = int(study_meta.loc[study_meta.StudyInstanceUID == uid, "shard"].iloc[0])
    z = np.load(f"{CACHE}/shard_{shard:03d}.npz")
    side = study_meta.loc[study_meta.StudyInstanceUID == uid, "side"].iloc[0]
    for a, (plane, idx) in zip(row, [("sag", D // 4), ("sag", D // 2), ("cor", D // 2), ("ax", D // 2)]):
        key = f"{uid}/{plane}"
        if key in z.files:
            a.imshow(z[key][idx], cmap="gray")
        a.set_title(f"{uid[-6:]} {side} {plane}[{idx}]", fontsize=9)
        a.axis("off")
plt.tight_layout()
plt.savefig(f"{WORK}/qc_orientation.png", dpi=70)
plt.show()

# %% [markdown]
# ## Summary — copy this block to Claude

# %%
n_all = len(train)
s = [f"=== SUMMARY preprocess | PREVIEW={PREVIEW} | D={D} S={S} ===",
     f"studies: {len(study_meta)} | minutes: {elapsed / 60:.1f} | studies/s: {len(study_meta) / elapsed:.2f} | cache GB: {cache_bytes / 1e9:.2f}",
     f"projected for all {n_all}: {n_all / (len(study_meta) / elapsed) / 60:.0f} min, {cache_bytes / len(study_meta) * n_all / 1e9:.1f} GB",
     f"series errors: {len(errs)} " + (("| top: " + errs['error'].iloc[0][:120]) if len(errs) else ""),
     f"plane mismatches (CSV vs geometry): {int((ok.plane != ok.geo_plane).sum())}",
     "planes per study: " + ", ".join(f"{k} {v}" for k, v in study_meta['planes'].value_counts().items()),
     "side source: " + ", ".join(f"{k} {v}" for k, v in study_meta['side_src'].value_counts(dropna=False).items()),
     "side counts: " + ", ".join(f"{k} {v}" for k, v in study_meta['side'].value_counts(dropna=False).items()),
     f"side tag vs x-centre agreement: {side_agree:.3f} (n={len(both)})",
     "fluid-sensitive chosen: " + ", ".join(f"{k} {v:.2f}" for k, v in ok.groupby('plane')['fluid'].mean().items()),
     "fat-sat chosen: " + ", ".join(f"{k} {v:.2f}" for k, v in ok.groupby('plane')['fatsat'].mean().items()),
     "median original slices: " + ", ".join(f"{k} {v:.0f}" for k, v in ok.groupby('plane')['n_slices'].median().items()),
     "field strength: " + ", ".join(f"{k}T {v}" for k, v in ok.drop_duplicates('StudyInstanceUID')['field'].round(1).value_counts().items()),
     "=== END ==="]
s = "\n".join(s)
print(s)
with open(f"{WORK}/preprocess_summary.txt", "w") as f:
    f.write(s)
