# %% [markdown]
# # 14 — Which combination of our own reader families is best? (out-of-fold, 58 gold studies, CPU only)
# **Settings:** Accelerator = None · Internet = OFF
# **Inputs:** competition data · the training notebooks of every family to compare (04g = M6, 04h = M7, 04j = M9,
# 04k = M11, ...): each wrote `oof_<tag>_folds*.csv` (out-of-fold probabilities, training window layout).
#
# Every gold study is predicted by the fold model that did not train on it. For each family and for rank averages of
# families (with weights), the macro AUC on the gold studies and per finding. A paired bootstrap (same resampled
# studies) gives how often a combination beats the best single family. Notebook 10 showed that dense windows at test
# time add 0.000-0.006 on top of these numbers.

# %%
import os, re, itertools, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

T0 = time.time()
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

def find_inputs(match):
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series", "cache")]
        hits += [os.path.join(root, f) for f in files if match(os.path.join(root, f))]
    return sorted(hits)

DATA = os.path.dirname(find_inputs(lambda p: p.endswith("/train_series.csv"))[0])
train = pd.read_csv(f"{DATA}/train.csv")
GT = train[train[LABELS].notna().all(axis=1)].set_index("StudyInstanceUID")[LABELS].astype(int)

OOF = {}
for p in find_inputs(lambda p: re.search(r"/oof_(m\d+_reader)_folds[^/]*\.csv$", p) is not None):
    tag = re.search(r"/oof_(m\d+_reader)_folds", p).group(1)
    d = pd.read_csv(p, index_col=0)
    OOF.setdefault(tag, []).append(d)
for tag in list(OOF):
    d = pd.concat(OOF[tag])
    d = d[~d.index.duplicated()]
    OOF[tag] = d.loc[d.index.intersection(GT.index), LABELS]
    print(f"{tag}: {len(OOF[tag])} gold studies out-of-fold")
TAGS = sorted(OOF, key=lambda t: int(re.search(r"m(\d+)", t).group(1)))

def aucs(p, idx):
    y = GT.loc[idx]
    return np.array([roc_auc_score(y[L], p.loc[idx, L]) if 0 < y[L].sum() < len(y) else np.nan for L in LABELS])

rank = lambda d: d.rank(pct=True)
full = [t for t in TAGS if len(OOF[t]) >= 55]                      # families with (almost) all 58 gold predicted
IDX = sorted(set.intersection(*[set(OOF[t].index) for t in full])) if full else []
print(f"families with all folds: {full} | common gold studies {len(IDX)}")

# %%
rows = {}
for t in TAGS:
    idx = sorted(OOF[t].index)
    a = aucs(OOF[t], idx)
    rows[f"{t} alone (n={len(idx)})"] = [np.nanmean(a)] + list(a)
COMBOS = {}
for r in range(2, len(full) + 1):
    for combo in itertools.combinations(full, r):
        for w in itertools.product((1, 2), repeat=r):
            if r > 1 and len(set(w)) == 1 and w[0] == 2:
                continue
            e = sum(wi * rank(OOF[t].loc[IDX]) for t, wi in zip(combo, w)) / sum(w)
            name = " + ".join(f"{wi}x{t.split('_')[0]}" for t, wi in zip(combo, w))
            COMBOS[name] = e
            a = aucs(e, IDX)
            rows[name] = [np.nanmean(a)] + list(a)
tab = pd.DataFrame(rows, index=["macro"] + LABELS).T.sort_values("macro", ascending=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
print(tab.round(3).to_string())
tab.round(4).to_csv("/kaggle/working/ensemble_check.csv")

# %%
# paired bootstrap of the best combinations against the best single family on the same studies
rng = np.random.default_rng(0)
y = GT.loc[IDX].values
best_single = max(full, key=lambda t: np.nanmean(aucs(OOF[t], IDX))) if full else None
def boot(pa, pb, n=300):
    a, b = pa.loc[IDX, LABELS].values, pb.loc[IDX, LABELS].values
    out = []
    for _ in range(n):
        i = rng.integers(0, len(IDX), len(IDX))
        ok = [j for j in range(12) if 0 < y[i, j].sum() < len(i)]
        out.append(np.mean([roc_auc_score(y[i, j], b[i, j]) - roc_auc_score(y[i, j], a[i, j]) for j in ok]))
    out = np.array(out)
    return out.mean(), (out > 0).mean()
s = ["=== SUMMARY 14 ensemble check ===", f"common gold studies {len(IDX)} | best single family {best_single}"]
for name in list(tab.index[:12]):
    s.append(f"{name}: macro {tab.loc[name, 'macro']:.4f} | " + " ".join(f"{L[:6]} {tab.loc[name, L]:.3f}" for L in LABELS))
if best_single:
    for name in [n for n in tab.index if n in COMBOS][:6]:
        m, frac = boot(OOF[best_single], COMBOS[name])
        s.append(f"bootstrap {name} vs {best_single}: {m:+.4f}, better in {frac:.0%}")
# second pass: every family, also partial ones (e.g. only folds 0,1 trained), on the gold studies all of them predicted
part = [t for t in TAGS if t not in full]
for p in part:                                       # each partial family with the full ones, on its own gold studies
    fam = [t for t in full if t in ("m7_reader", "m9_reader")] + [p]
    IDX2 = sorted(set.intersection(*[set(OOF[t].index) for t in fam]))
    s.append(f"--- {p} (partial) with {fam[:-1]} on their common {len(IDX2)} gold studies ---")
    res = {}
    for r in range(1, len(fam) + 1):
        for combo in itertools.combinations(fam, r):
            for w in itertools.product((1, 2), repeat=r):
                if r == 1 and w[0] == 2:
                    continue
                e = sum(wi * rank(OOF[t].loc[IDX2]) for t, wi in zip(combo, w)) / sum(w)
                res[" + ".join(f"{wi}x{t.split('_')[0]}" for t, wi in zip(combo, w))] = np.nanmean(aucs(e, IDX2))
    for name, v in sorted(res.items(), key=lambda kv: -kv[1])[:12]:
        s.append(f"{name}: macro {v:.4f}")
s += [f"total {(time.time() - T0) / 60:.1f} min", "=== END ==="]
print("\n".join(s))
