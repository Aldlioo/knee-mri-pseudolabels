# %% [markdown]
# # 16 — MRI example figure for the paper (CPU only)
# **Settings:** Accelerator = None · Internet = OFF
# **Inputs:** competition data · notebook "RSNA Knee 03d Preprocess Zoom" (cache shards, study_meta.csv, knee_prep.py).
#
# Draws one expert-labelled study exactly as our readers see it (03d cache: 5 slots, 100 mm crop, 224 px, left knees
# mirrored so the lateral side comes first) and shows how three neighbouring slices become one 3-channel window.
# Output: `paper_fig_mri.png` (no identifiers or DICOM text in the image) and `paper_fig_mri.json` (labels of the study).

# %%
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def find_inputs(match):
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series", "train_images", "test_images")]
        hits += [os.path.join(root, f) for f in files if match(os.path.join(root, f))]
    return sorted(hits)

PREP_DIR = os.path.dirname(find_inputs(lambda p: p.endswith("/knee_prep.py") and "03d" in p)[0])
sys.path.insert(0, PREP_DIR)
import knee_prep as kp
META = pd.read_csv(find_inputs(lambda p: p.endswith("/study_meta.csv") and "03d" in p)[0])
SHARDS = {int(os.path.basename(p)[6:9]): p for p in find_inputs(lambda p: "/cache/shard_" in p and p.endswith(".npz") and "03d" in p)}
DATA = os.path.dirname(find_inputs(lambda p: p.endswith("/train_series.csv"))[0])
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
train = pd.read_csv(f"{DATA}/train.csv")
gold = train[train[LABELS].notna().all(axis=1)].set_index("StudyInstanceUID")[LABELS].astype(int)
print("slots", [s[0] for s in kp.SLOTS], "| slices", kp.DS, "| S", kp.S, "| crop mm", kp.CROP_MM, "| shards", len(SHARDS))

# %%
# One expert-labelled study with all five slots, preferring an ACL tear with contusion (visible on the middle
# sagittal slices) and then the most positive findings.
m = META.set_index("StudyInstanceUID")
cand = gold[gold.index.isin(m.index) & (m.reindex(gold.index).present.astype(str) == "11111")].copy()
cand["score"] = cand["ACL"] * 4 + cand["Contusion"] * 2 + cand["Effusion"] + cand.sum(axis=1) * 0.1
uid = cand.sort_values("score", ascending=False).index[0]
row = m.loc[uid]
with np.load(SHARDS[int(row["shard"])]) as z:
    vol = z[f"{uid}/vol"]
pos = [L for L in LABELS if gold.loc[uid, L] == 1]
print("chosen study: side", row["side"], "| positive findings:", pos)

# %%
NAMES = {"sag_fs": "sagittal FS", "cor_fs": "coronal FS", "ax_fs": "axial FS",
         "sag_nfs": "sagittal non-FS", "cor_t1": "coronal T1"}
names = [s[0] for s in kp.SLOTS]
mid = [(kp.OFF[k] + kp.OFF[k + 1]) // 2 for k in range(kp.N_SLOT)]

fig = plt.figure(figsize=(7.2, 3.35), dpi=300)
gs = fig.add_gridspec(2, 5, hspace=0.28, wspace=0.06, height_ratios=[1, 1])
for k in range(kp.N_SLOT):
    a = fig.add_subplot(gs[0, k])
    a.imshow(vol[mid[k]], cmap="gray", vmin=0, vmax=255)
    a.set_title(NAMES[names[k]], fontsize=6.2, pad=2)
    a.axis("off")
fig.axes[0].text(-0.08, 1.12, "a", transform=fig.axes[0].transAxes, fontsize=9, weight="bold", ha="right", va="bottom")

# b: three neighbouring sagittal slices -> one 3-channel window (shown as an RGB image)
c = mid[0]
for j, d in enumerate((-1, 0, 1)):
    a = fig.add_subplot(gs[1, j])
    a.imshow(vol[c + d], cmap="gray", vmin=0, vmax=255)
    a.set_title(["slice i − 1", "slice i", "slice i + 1"][j] + ["  (R)", "  (G)", "  (B)"][j], fontsize=6.2, pad=2)
    a.axis("off")
a = fig.add_subplot(gs[1, 3])
a.axis("off")
a.text(0.5, 0.55, "→", fontsize=22, ha="center", va="center")
a.text(0.5, 0.25, "stacked as the 3 colour\nchannels of one 2D image", fontsize=5.8, ha="center", va="center")
a = fig.add_subplot(gs[1, 4])
a.imshow(np.stack([vol[c - 1], vol[c], vol[c + 1]], -1))
a.set_title("one window (backbone input)", fontsize=6.2, pad=2)
a.axis("off")
fig.axes[5].text(-0.08, 1.12, "b", transform=fig.axes[5].transAxes, fontsize=9, weight="bold", ha="right", va="bottom")

fig.savefig("/kaggle/working/paper_fig_mri.png", bbox_inches="tight", facecolor="white", dpi=300)
json.dump({"side": str(row["side"]), "positive_findings": pos, "slots": names, "middle_slices": mid,
           "slices_per_slot": kp.DS, "crop_mm": kp.CROP_MM, "size_px": kp.S},
          open("/kaggle/working/paper_fig_mri.json", "w"), indent=1)
# The raw panels (uint8, 224 x 224) so the figure can be redrawn elsewhere in the journal's font, as a PDF.
np.savez_compressed("/kaggle/working/paper_fig_panels.npz",
                    mids=np.stack([vol[i] for i in mid]), window=np.stack([vol[c - 1], vol[c], vol[c + 1]]))
print("=== SUMMARY 16 === side", row["side"], "| positives", pos, "| file", os.path.getsize("/kaggle/working/paper_fig_mri.png"),
      "bytes | panels", os.path.getsize("/kaggle/working/paper_fig_panels.npz"), "bytes")
