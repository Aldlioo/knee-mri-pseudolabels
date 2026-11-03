"""Figure 2 of the journal version: one expert-labelled study as the readers see it (run: py -3 paper/make_mri_figure_tubitak.py).

Input: paper/mri_panels.npz from Kaggle notebook "RSNA Knee 16 Paper MRI Figure" (mids = middle slice of the 5 slots,
window = sagittal slices i-1, i, i+1; uint8, 224 x 224, 03d cache). Output: paper/latex/figures/fig2_mri_example.pdf,
16 cm wide, Times New Roman, no line above 1 pt. Each 224 px panel is upsampled 3x (bicubic) so the embedded images
exceed the journal's 118 pixels/cm at print size; the upsampling adds no information."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "mri_panels.npz"), allow_pickle=False)
mids, win = z["mids"], z["window"]
plt.rcParams.update({"font.family": "Times New Roman", "pdf.fonttype": 42})


def up(a, k=3):
    return np.asarray(Image.fromarray(a).resize((a.shape[1] * k, a.shape[0] * k), Image.BICUBIC))


TITLES = ["Sagittal FS", "Coronal FS", "Axial FS", "Sagittal non-FS", "Coronal T1"]
fig = plt.figure(figsize=(16 / 2.54, 7.0 / 2.54))
gs = fig.add_gridspec(2, 5, hspace=0.32, wspace=0.06, left=0.03, right=0.995, top=0.92, bottom=0.02)
axes_a = [fig.add_subplot(gs[0, k]) for k in range(5)]
for k, a in enumerate(axes_a):
    a.imshow(up(mids[k]), cmap="gray", vmin=0, vmax=255, interpolation="none")
    a.set_title(TITLES[k], fontsize=8, pad=2)
    a.axis("off")
axes_a[0].text(-0.08, 1.04, "a", transform=axes_a[0].transAxes, fontsize=10, weight="bold", ha="right", va="bottom")

axes_b = []
for j, t in enumerate(["Slice i − 1 (R)", "Slice i (G)", "Slice i + 1 (B)"]):
    a = fig.add_subplot(gs[1, j])
    a.imshow(up(win[j]), cmap="gray", vmin=0, vmax=255, interpolation="none")
    a.set_title(t, fontsize=8, pad=2)
    a.axis("off")
    axes_b.append(a)
axes_b[0].text(-0.08, 1.04, "b", transform=axes_b[0].transAxes, fontsize=10, weight="bold", ha="right", va="bottom")
a = fig.add_subplot(gs[1, 3]); a.axis("off")
a.annotate("", xy=(0.82, 0.58), xytext=(0.18, 0.58), xycoords="axes fraction",
           arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black"))
a.text(0.5, 0.30, "Stacked as the three\ncolour channels of\none 2D image", fontsize=7, ha="center", va="center",
       transform=a.transAxes)
a = fig.add_subplot(gs[1, 4])
a.imshow(np.stack([up(win[0]), up(win[1]), up(win[2])], -1), interpolation="none")
a.set_title("One window (model input)", fontsize=8, pad=2)
a.axis("off")

out = os.path.join(HERE, "latex", "figures", "fig2_mri_example.pdf")
fig.savefig(out, format="pdf", facecolor="white", dpi=300)
print("wrote", out)
