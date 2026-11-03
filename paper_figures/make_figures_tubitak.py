"""Journal versions of the chart figures for Turk J Elec Eng & Comp Sci (run: py -3 paper/make_figures_tubitak.py).

Reuses the drawing code of make_figures.py unchanged and only applies the journal's figure rules on save:
Times New Roman inside figures, every line weight between 0.5 and 1 pt, vector PDF with editable text
(TrueType fonts embedded), files in paper/latex/figures/ under the paper's figure numbers.
The MRI example (Figure 2) is drawn by make_mri_figure_tubitak.py from the raw panels."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.collections import Collection
import make_figures as mf

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "latex", "figures")
os.makedirs(OUTDIR, exist_ok=True)
NAMES = {"fig1_pipeline.png": "fig1_pipeline.pdf", "fig2_gold_auc.png": "fig3_targets_models.pdf",
         "fig4_per_finding.png": "fig4_per_finding.pdf", "fig3_public_private.png": "fig5_public_private.pdf"}
LW_MIN, LW_MAX = 0.5, 1.0

plt.rcParams.update({"font.family": "Times New Roman", "pdf.fonttype": 42, "mathtext.fontset": "stix",
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.linewidth": 0.6})


def clamp(w):
    return w if w == 0 else min(max(w, LW_MIN), LW_MAX)


def journal_lines(fig):
    """Clamp every drawn line (plot lines, marker edges, patches, arrows, scatter edges, grid, spines) to 0.5-1 pt."""
    for art in fig.findobj():
        if isinstance(art, Line2D):
            art.set_linewidth(clamp(art.get_linewidth()))
            art.set_markeredgewidth(clamp(art.get_markeredgewidth()))
        elif isinstance(art, (Patch, FancyArrowPatch)):
            if art.get_linewidth():
                art.set_linewidth(clamp(art.get_linewidth()))
        elif isinstance(art, Collection):
            art.set_linewidths([clamp(w) for w in art.get_linewidths()])


_orig_savefig = Figure.savefig


def savefig_journal(self, fname, *args, **kwargs):
    name = NAMES[os.path.basename(str(fname))]
    for t in self.findobj(matplotlib.text.Text):
        t.set_fontfamily("Times New Roman")
    journal_lines(self)
    kwargs.pop("dpi", None)
    _orig_savefig(self, os.path.join(OUTDIR, name), *args, format="pdf", **kwargs)
    print("wrote", name)


Figure.savefig = savefig_journal

if __name__ == "__main__":
    mf.fig1(); mf.fig2(); mf.fig3(); mf.fig4()
