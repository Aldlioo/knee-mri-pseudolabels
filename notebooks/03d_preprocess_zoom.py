# %% [markdown]
# # 03d — Preprocess for model M5 (zoomed reader): 5 series slots, 100 mm physical crop, 224 px
# **Settings:** Accelerator = None (CPU) · Internet = ON (installs JPEG/JPEG-2000 DICOM decoders)
#
# Same pipeline as 03c (M2 / M4 cache: 130 mm crop at 240 px = 0.54 mm per pixel) with a **zoomed view**:
# - **5 series per study**: sagittal / coronal / axial fluid-sensitive fat-sat (20 slices each) +
#   sagittal *without* fat-sat (PD/T2, else T1) and coronal T1 (10 slices each). The non-fat-sat images show
#   anatomy (menisci, ligaments, cortex, cartilage outline) that fat-sat images hide.
# - **100 mm physical crop** around the image centre at 224 px (≈0.45 mm per pixel; 256 px would need 21 GB, over the 20 GB output limit): menisci, cartilage and ligaments
#   get 1.2× finer sampling. The public 0.954 / 0.955 second models also use ~100 mm knee-centred local views.
# - Orientation, sorting and left-knee mirroring are the same as in 03 (lateral first).
#
# The preprocessing code lives in **`knee_prep.py`**, written to this notebook's output. Training (04e with PREP = "03d")
# and the submission script *import* that file, so training and test images are made by the same code.
#
# Output: `cache/shard_XXX.npz` (keys `<uid>/vol` = (80, 224, 224) uint8, `<uid>/present` = 5 flags),
# `knee_prep.py`, `series_meta.csv`, `study_meta.csv`, `qc_slots.png`. Last cell: `=== SUMMARY` block.

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "python-gdcm"])

# %%
%%writefile /kaggle/working/knee_prep.py
"""M5 (zoomed) preprocessing, shared by 03d (train cache), 04e (layout constants) and 05c (test). Same code as the
03c knee_prep.py except S and CROP_MM. Import it, never copy it."""
import os, glob, re, collections
import numpy as np
import pandas as pd
import pydicom
import cv2

S = 224                      # in-plane size (pixels)
CROP_MM = 100.0              # physical square crop around the image centre (03c / M2 / M4: 130 mm at 240 px)
# slot name, CSV plane, canonical plane, kind, slices kept
SLOTS = [("sag_fs", "Sagittal", "sag", "fs", 20),
         ("cor_fs", "Coronal", "cor", "fs", 20),
         ("ax_fs", "Axial", "ax", "fs", 20),
         ("sag_nfs", "Sagittal", "sag", "nfs", 10),
         ("cor_t1", "Coronal", "cor", "t1", 10)]
N_SLOT = len(SLOTS)
DS = [s[4] for s in SLOTS]
OFF = [int(v) for v in np.concatenate([[0], np.cumsum(DS)])]   # slot k = slices OFF[k]:OFF[k+1]
N_SLICES = OFF[-1]

# DICOM patient frame is LPS (+x patient left, +y posterior, +z superior).
# Display per plane (horizontal, vertical, stacking): sag A→P, S→I, R→L · cor R→L, S→I, A→P · ax R→L, A→P, S→I
AX = {"x": np.array([1., 0, 0]), "y": np.array([0, 1., 0]), "z": np.array([0, 0, 1.])}
TARGET = {"sag": (AX["y"], -AX["z"], AX["x"]),
          "cor": (AX["x"], -AX["z"], AX["y"]),
          "ax":  (AX["x"], AX["y"], -AX["z"])}
FLIP_AXIS_FOR_LEFT = {"sag": 0, "cor": 2, "ax": 2}   # the patient x axis in each canonical volume

RIGHT_RE = re.compile(r"\b(right|rechts|derecha|dcha|droite|desno|desna|dx|rt)\b", re.I)
LEFT_RE = re.compile(r"\b(left|links|izquierda|izq|gauche|lijevo|lijeva|sx|lt)\b", re.I)

DATA, SERIES = None, {}

def setup(data_root, series_by_split):
    """Call once in the parent process before creating a Pool (workers inherit it via fork)."""
    global DATA, SERIES
    DATA = data_root
    SERIES = {k: v.assign(Fluid_Sensitive=v.Fluid_Sensitive.fillna(0).astype(int),
                          Fat_Suppression=v.Fat_Suppression.fillna(0).astype(int))
              for k, v in series_by_split.items()}

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
    """One series → (float32 volume sorted along the slice normal, geometry dict)."""
    ds = [pydicom.dcmread(f) for f in glob.glob(f"{path}/*.dcm")]
    ds = [d for d in ds if "ImageOrientationPatient" in d and "ImagePositionPatient" in d] or ds
    shape = collections.Counter((int(d.Rows), int(d.Columns)) for d in ds).most_common(1)[0][0]
    ds = [d for d in ds if (int(d.Rows), int(d.Columns)) == shape]
    iop = np.array(ds[0].ImageOrientationPatient, float)
    r, c = iop[:3], iop[3:]                      # r: direction of increasing column, c: increasing row
    n = np.cross(r, c)
    pos = np.array([np.dot(n, np.array(d.ImagePositionPatient, float)) for d in ds])
    ds = [ds[i] for i in np.argsort(pos)]
    vol = np.stack([d.pixel_array.astype(np.float32) for d in ds])
    if str(ds[0].get("PhotometricInterpretation", "")) == "MONOCHROME1":
        vol = vol.max() - vol
    ps = [float(v) for v in ds[0].get("PixelSpacing", [0, 0])]
    mid = ds[len(ds) // 2]
    centre = (np.array(mid.ImagePositionPatient, float)
              + r * shape[1] * (ps[1] or 1) / 2 + c * shape[0] * (ps[0] or 1) / 2)
    side, side_src = side_from_tags(ds[0])
    geo = dict(r=r, c=c, n=n, n_slices=len(ds), rows=shape[0], cols=shape[1],
               px_row=ps[0], px_col=ps[1], thick=float(ds[0].get("SliceThickness", 0) or 0),
               x_centre=float(centre[0]), side_tag=side, side_src=side_src,
               maker=str(ds[0].get("Manufacturer", "")), field=float(ds[0].get("MagneticFieldStrength", 0) or 0),
               desc=str(ds[0].get("SeriesDescription", "")))
    return vol, geo

def canonical(vol, geo, plane):
    """Reorient to the display frame of `plane`; returns (volume, (vertical, horizontal) pixel spacing in mm)."""
    h_t, v_t, s_t = TARGET[plane]
    r, c, n = geo["r"], geo["c"], geo["n"]
    ps_v, ps_h = geo["px_row"], geo["px_col"]        # DICOM PixelSpacing = (between rows, between columns)
    if abs(np.dot(c, h_t)) > abs(np.dot(r, h_t)):    # rows/cols swapped relative to the target
        vol = vol.transpose(0, 2, 1)
        r, c = c, r
        ps_v, ps_h = ps_h, ps_v
    if np.dot(r, h_t) < 0:
        vol = vol[:, :, ::-1]
    if np.dot(c, v_t) < 0:
        vol = vol[:, ::-1, :]
    if np.dot(n, s_t) < 0:                           # slices were sorted along +n
        vol = vol[::-1]
    return vol, (ps_v, ps_h)

def crop_resize(vol, ps, D):
    """(n, h, w) float → (D, S, S) uint8: CROP_MM square around the centre (black padding outside the scan),
    per-volume 0.5–99.5 percentile normalisation (on the scanned part only), slices resampled to D."""
    n, h, w = vol.shape
    ok = 0.05 < ps[0] < 5 and 0.05 < ps[1] < 5
    ch, cw = (CROP_MM / ps[0], CROP_MM / ps[1]) if ok else (float(h), float(w))   # no spacing → whole image
    ch, cw = max(int(round(ch)), 8), max(int(round(cw)), 8)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    inner = vol[:, max(y0, 0):min(y0 + ch, h), max(x0, 0):min(x0 + cw, w)]
    a, b = np.percentile(inner, [0.5, 99.5])
    inner = np.clip((inner - a) / max(b - a, 1e-6), 0, 1).astype(np.float32)
    canvas = np.zeros((n, ch, cw), np.float32)
    py, px = max(0, -y0), max(0, -x0)
    canvas[:, py:py + inner.shape[1], px:px + inner.shape[2]] = inner
    canvas = np.stack([cv2.resize(s, (S, S), interpolation=cv2.INTER_AREA) for s in canvas])
    src = np.linspace(0, n - 1, D)
    lo, frac = np.floor(src).astype(int), (src - np.floor(src))[:, None, None]
    hi = np.minimum(lo + 1, n - 1)
    out = canvas[lo] * (1 - frac) + canvas[hi] * frac
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)

def shape_ok(n):
    return 1 if 12 <= n <= 80 else 0                  # typical 2D stack, not a localizer or a huge 3D block

def slot_key(kind, r):
    n, f, fs = r["n_files"], r["Fluid_Sensitive"], r["Fat_Suppression"]
    if kind == "fs":
        return (f, fs, shape_ok(n), -abs(n - 30))
    if kind == "nfs":                                  # PD/T2 without fat-sat, else T1
        return (f, shape_ok(n), -abs(n - 30))
    return (1 - f, shape_ok(n), -abs(n - 30))          # "t1": non-fluid first

def choose_series(ss):
    """{slot index: series row}. Fat-sat slots first; the other slots only take unused series without fat-sat."""
    chosen, used = {}, set()
    for k, (name, csv_plane, plane, kind, D) in enumerate(SLOTS):
        cand = [r for r in ss.to_dict("records") if r["Anatomical_Plane"] == csv_plane
                and r["SeriesInstanceUID"] not in used and (kind == "fs" or r["Fat_Suppression"] == 0)]
        if cand:
            best = max(cand, key=lambda r: slot_key(kind, r))
            chosen[k] = best
            used.add(best["SeriesInstanceUID"])
    return chosen

def process_study(uid, split="train"):
    """→ (uid, vol (N_SLICES, S, S) uint8, present (N_SLOT,) bool, series meta list, study dict)."""
    root = f"{DATA}/{split}_series/{uid}"
    series = SERIES[split]
    ss = series[series.StudyInstanceUID == uid].copy()
    ss["n_files"] = [len(os.listdir(f"{root}/{s}")) if os.path.isdir(f"{root}/{s}") else 0
                     for s in ss.SeriesInstanceUID]
    ss = ss[ss.n_files > 0]
    loaded, meta = {}, []
    for k, row in choose_series(ss).items():
        name, _, _, kind, D = SLOTS[k]
        try:
            vol, geo = load_series(f"{root}/{row['SeriesInstanceUID']}")
            geo_plane = plane_from_normal(geo["n"])
            vol, ps = canonical(vol, geo, geo_plane)
            loaded[k] = (vol, ps, geo_plane)
            meta.append(dict(StudyInstanceUID=uid, slot=name, geo_plane=geo_plane,
                             SeriesInstanceUID=row["SeriesInstanceUID"], fluid=row["Fluid_Sensitive"],
                             fatsat=row["Fat_Suppression"], fov_v=ps[0] * vol.shape[1], fov_h=ps[1] * vol.shape[2],
                             **{a: b for a, b in geo.items() if a not in ("r", "c", "n")}, error=""))
        except Exception as e:
            meta.append(dict(StudyInstanceUID=uid, slot=name, SeriesInstanceUID=row["SeriesInstanceUID"],
                             error=f"{type(e).__name__}: {e}"[:200]))
    ok = [m for m in meta if not m["error"]]
    votes = [m["side_tag"] for m in ok if m.get("side_tag")]
    xs = [m["x_centre"] for m in ok]
    side_x = ("R" if np.median(xs) < 0 else "L") if xs else None
    side, side_src = (collections.Counter(votes).most_common(1)[0][0], "tag") if votes else (side_x, "x_centre")
    out = np.zeros((N_SLICES, S, S), np.uint8)
    present = np.zeros(N_SLOT, bool)
    for k, (vol, ps, geo_plane) in loaded.items():
        if side == "L":
            vol = np.flip(vol, FLIP_AXIS_FOR_LEFT[geo_plane])   # lateral first / image-left = lateral
        out[OFF[k]:OFF[k + 1]] = crop_resize(np.ascontiguousarray(vol), ps, DS[k])
        present[k] = True
    study = dict(StudyInstanceUID=uid, side=side, side_src=side_src, side_x=side_x,
                 present="".join("1" if p else "0" for p in present), n_series=len(ss))
    return uid, out, present, meta, study

# %%
import os, glob, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, "/kaggle/working")
import knee_prep as kp

PREVIEW = False
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
kp.setup(DATA, {"train": pd.read_csv(f"{DATA}/train_series.csv")})
print(len(train), "studies |", N_WORKERS, "CPU workers | slots", [s[0] for s in kp.SLOTS], "| slices", kp.N_SLICES, "| S", kp.S)

def process_shard(args):
    shard_id, uids = args
    t0, arrays, metas, studies = time.time(), {}, [], []
    for uid in uids:
        try:
            _, vol, present, meta, study = kp.process_study(uid, "train")
            arrays[f"{uid}/vol"], arrays[f"{uid}/present"] = vol, present
        except Exception as e:
            meta, study = [], dict(StudyInstanceUID=uid, present="00000", error=f"{type(e).__name__}: {e}"[:200])
        metas += meta
        studies.append(dict(study, shard=shard_id))
    np.savez_compressed(f"{CACHE}/shard_{shard_id:03d}.npz", **arrays)
    return metas, studies, time.time() - t0

# %%
gt = train[LABELS].notna().all(axis=1)
uids = train.StudyInstanceUID.tolist()
if PREVIEW:
    uids = train[gt].StudyInstanceUID.head(12).tolist() + train[~gt].StudyInstanceUID.sample(12, random_state=0).tolist()
per_shard = 6 if PREVIEW else STUDIES_PER_SHARD
jobs = [(i, uids[j:j + per_shard]) for i, j in enumerate(range(0, len(uids), per_shard))]

t0, all_meta, all_studies = time.time(), [], []
with Pool(N_WORKERS) as pool:
    for k, (metas, studies, secs) in enumerate(pool.imap_unordered(process_shard, jobs)):
        all_meta += metas
        all_studies += studies
        done = len(all_studies)
        rate = done / (time.time() - t0)
        print(f"shards {k + 1}/{len(jobs)}  studies {done}/{len(uids)}  {rate:.2f} studies/s  "
              f"eta {(len(uids) - done) / rate / 60:.1f} min", flush=True)
        if k == 2:   # stop early if the finished cache would not fit in the 20 GB notebook output
            proj = sum(os.path.getsize(f) for f in glob.glob(f"{CACHE}/*.npz")) / done * len(uids) / 1e9
            print(f"projected cache size {proj:.1f} GB", flush=True)
            if proj > 19.0:
                raise RuntimeError(f"projected cache {proj:.1f} GB > 19 GB: lower S in knee_prep.py")
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
ok = series_meta[series_meta["error"].fillna("") == ""]
errs = series_meta[series_meta["error"].fillna("") != ""]
pres = np.array([[c == "1" for c in p] for p in study_meta.present.fillna("00000")])
slot_names = [s[0] for s in kp.SLOTS]
print("slot present rate:", dict(zip(slot_names, pres.mean(0).round(3))))
print("series errors:", len(errs)); print(errs["error"].str.slice(0, 80).value_counts().head(8).to_string())
print(ok.groupby("slot")[["fluid", "fatsat"]].mean().round(3).to_string())
print("field of view (mm):\n", ok.groupby("slot")[["fov_v", "fov_h"]].describe().round(0).T.to_string())
small_fov = float(((ok.fov_v < kp.CROP_MM) | (ok.fov_h < kp.CROP_MM)).mean())
print(f"series with FOV < {kp.CROP_MM:.0f} mm (padded): {small_fov:.3f}")

# All 5 slots (middle slice) for 2 right and 2 left knees: lateral must be on the image-left in coronal/axial,
# anterior (patella) on the left in sagittal.
show = (study_meta[study_meta.side == "R"].head(2).StudyInstanceUID.tolist() +
        study_meta[study_meta.side == "L"].head(2).StudyInstanceUID.tolist())
fig, axes = plt.subplots(len(show), kp.N_SLOT, figsize=(3 * kp.N_SLOT, 3.2 * len(show)))
for row, uid in zip(np.atleast_2d(axes), show):
    shard = int(study_meta.loc[study_meta.StudyInstanceUID == uid, "shard"].iloc[0])
    with np.load(f"{CACHE}/shard_{shard:03d}.npz") as z:
        vol = z[f"{uid}/vol"]
    side = study_meta.loc[study_meta.StudyInstanceUID == uid, "side"].iloc[0]
    for k, a in enumerate(row):
        a.imshow(vol[(kp.OFF[k] + kp.OFF[k + 1]) // 2], cmap="gray")
        a.set_title(f"{uid[-6:]} {side} {slot_names[k]}", fontsize=8)
        a.axis("off")
plt.tight_layout()
plt.savefig(f"{WORK}/qc_slots.png", dpi=60)
plt.show()

# %% [markdown]
# ## Summary — copy this block to Claude

# %%
s = [f"=== SUMMARY preprocess M5 zoom | PREVIEW={PREVIEW} | slices {kp.DS} S={kp.S} crop {kp.CROP_MM:.0f} mm ===",
     f"studies: {len(study_meta)} | minutes: {elapsed / 60:.1f} | cache GB: {cache_bytes / 1e9:.2f} | "
     f"projected all: {len(train) / (len(study_meta) / elapsed) / 60:.0f} min, {cache_bytes / len(study_meta) * len(train) / 1e9:.1f} GB",
     "slot present: " + ", ".join(f"{n} {v:.3f}" for n, v in zip(slot_names, pres.mean(0))),
     f"series errors: {len(errs)} " + (("| top: " + errs['error'].iloc[0][:120]) if len(errs) else ""),
     "fat-sat rate per slot: " + ", ".join(f"{k} {v:.2f}" for k, v in ok.groupby('slot')['fatsat'].mean().items()),
     "fluid rate per slot: " + ", ".join(f"{k} {v:.2f}" for k, v in ok.groupby('slot')['fluid'].mean().items()),
     f"median FOV mm: {ok.fov_v.median():.0f} x {ok.fov_h.median():.0f} | padded share {small_fov:.3f}",
     "side source: " + ", ".join(f"{k} {v}" for k, v in study_meta['side_src'].value_counts(dropna=False).items()),
     "=== END ==="]
s = "\n".join(s)
print(s)
open(f"{WORK}/preprocess_summary.txt", "w").write(s)
