# %% [markdown]
# # 01 — EDA: RSNA Knee Abnormality Detection
# Run as a Kaggle notebook attached to the competition data (Internet ON is fine for EDA).
# Paste the printed outputs back to Claude / into EXPERIMENTS.md "Data facts".

# %%
import os, glob, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom

# Kaggle's mount point for competition data has changed over time -> locate it instead of hardcoding
DATA = os.path.dirname((glob.glob("/kaggle/input/*/train_series.csv") +
                        glob.glob("/kaggle/input/*/*/train_series.csv"))[0])
print("DATA =", DATA)
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

print(sorted(os.listdir(DATA)))
train = pd.read_csv(f"{DATA}/train.csv")
series = pd.read_csv(f"{DATA}/train_series.csv")
print(train.shape, series.shape)
print(train.columns.tolist())
print(train.head(3).T)

# %% [markdown]
# ## Labeled vs unlabeled studies

# %%
n_lab = train[LABELS].notna().sum(axis=1)
print("labels present per study:\n", n_lab.value_counts().sort_index())
labeled = train[n_lab == len(LABELS)]
print(f"\nfully labeled: {len(labeled)} / {len(train)}")
print("\nprevalence in labeled subset:")
print(labeled[LABELS].mean().sort_values(ascending=False).round(3))

# %% [markdown]
# ## Label co-occurrence (labeled subset)

# %%
corr = labeled[LABELS].astype(float).corr()
plt.figure(figsize=(9, 7))
plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
plt.xticks(range(12), LABELS, rotation=90); plt.yticks(range(12), LABELS)
plt.colorbar(); plt.title("Label correlation"); plt.tight_layout(); plt.show()

# %% [markdown]
# ## Reports: length and language

# %%
train["report_len"] = train["Report"].fillna("").str.len()
print(train["report_len"].describe())

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "langdetect"])  # needs Internet ON
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    def lang(t):
        try: return detect(t[:1000])
        except Exception: return "unk"
    train["lang"] = train["Report"].fillna("").map(lang)
    print(train["lang"].value_counts())
    print("\nlabeled-subset languages:\n", train.loc[labeled.index, "lang"].value_counts())
except ImportError:
    print("pip install langdetect (Internet ON) to get language counts")

for i in random.Random(0).sample(range(len(labeled)), 2):
    r = labeled.iloc[i]
    print("=" * 80); print(r["Report"][:1500])
    print({k: int(r[k]) for k in LABELS})

# %% [markdown]
# ## Series structure

# %%
per_study = series.groupby("StudyInstanceUID").size()
print("series per study:\n", per_study.describe())
print(series["Anatomical_Plane"].value_counts())
print(pd.crosstab(series["Anatomical_Plane"], [series["Fluid_Sensitive"], series["Fat_Suppression"]]))

# Does every study have a fluid-sensitive series in each plane?
fs = series[series["Fluid_Sensitive"] == 1]
cov = fs.groupby("StudyInstanceUID")["Anatomical_Plane"].agg(set)
for p in ["Sagittal", "Coronal", "Axial"]:
    print(f"studies with fluid-sensitive {p}: {cov.map(lambda s: p in s).mean():.3f}")

# %% [markdown]
# ## DICOM sample: slices, spacing, size, tags

# %%
def load_series(study, ser):
    files = glob.glob(f"{DATA}/train_series/{study}/{ser}/*.dcm")
    ds = [pydicom.dcmread(f) for f in files]
    ds.sort(key=lambda d: float(getattr(d, "InstanceNumber", 0)))
    return ds

rows = []
for _, s in series.sample(40, random_state=0).iterrows():
    ds = load_series(s.StudyInstanceUID, s.SeriesInstanceUID)
    d0 = ds[0]
    rows.append(dict(plane=s.Anatomical_Plane, n=len(ds), shape=d0.pixel_array.shape,
                     px=tuple(round(float(x), 2) for x in getattr(d0, "PixelSpacing", [0, 0])),
                     thick=getattr(d0, "SliceThickness", None),
                     maker=getattr(d0, "Manufacturer", None)))
print(pd.DataFrame(rows).to_string())
print("\nAvailable tags:", [e.keyword for e in d0 if e.keyword][:90])

# %%
s = series[(series.Fluid_Sensitive == 1)].groupby("Anatomical_Plane").head(1)
fig, ax = plt.subplots(1, len(s), figsize=(5 * len(s), 5))
for a, (_, r) in zip(np.atleast_1d(ax), s.iterrows()):
    ds = load_series(r.StudyInstanceUID, r.SeriesInstanceUID)
    a.imshow(ds[len(ds) // 2].pixel_array, cmap="gray"); a.set_title(f"{r.Anatomical_Plane} ({len(ds)} slices)"); a.axis("off")
plt.show()
