# %% [markdown]
# # 09 — Label lab: build round-2 training targets (report labels + out-of-fold image predictions)
# **Settings:** Accelerator = None (CPU only, uses no GPU quota) · Internet = ON (Claude labels from GitHub)
# **Inputs:** competition data · our labelers (02, 02b) · our training notebooks 04e A-D, 04f A-C, 04g A-C (out-of-fold
# predictions of M4, M5, M6) · public label tables (consensus, JEV, nartaa, Dread Development, rayanbabur)
#
# Why: every report-label table (ours and public) sits at 0.89-0.92 macro AUC on the 58 expert-labelled studies, and our
# image models' out-of-fold (OOF) predictions are at 0.91 — as good as the labels they were trained on, with different
# errors (image vs report). Teams near the top of the leaderboard say the same and train a second round on a mix of
# both ("pseudo-labels"). This notebook measures each source and each mix on the 58 gold studies and writes
# `targets_v2.csv`, the soft target of the round-2 models (M7+). Gold labels are only used to *measure*; no weight is
# fitted on them.

# %%
import os, re, json, itertools
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

W_TEACHER = 0.5                    # weight of the OOF image predictions in the round-2 target (fixed before looking)

CLAUDE_URL = "https://raw.githubusercontent.com/NTejas-1/RSNA-Knee-Abnormality-Detection/main/labels_ds/labels_v1_opus45.csv"
COMBINE, MARGIN = "auto", 0.02
WORK = "/kaggle/working"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

def find_inputs(match):
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        hits += [os.path.join(root, f) for f in files if match(os.path.join(root, f))]
    return sorted(hits)

DATA = os.path.dirname(find_inputs(lambda p: p.endswith("/train_series.csv"))[0])
train = pd.read_csv(f"{DATA}/train.csv")
UIDS = train.StudyInstanceUID.tolist()
GT = train[train[LABELS].notna().all(axis=1)].set_index("StudyInstanceUID")[LABELS].astype(int)
print(len(UIDS), "studies |", len(GT), "gold | positives per finding", GT.sum().to_dict())

norm = lambda s: "".join(ch for ch in str(s).lower() if ch.isalnum())
ALIAS = {"ACL": ["acl"], "MCL": ["mcl"], "Medial Meniscus": ["medialmeniscus", "medmeniscus", "mm"],
         "Lateral Meniscus": ["lateralmeniscus", "latmeniscus", "lm"], "Medial OA": ["medialoa", "medoa"],
         "Lateral OA": ["lateraloa", "latoa"], "PF OA": ["pfoa", "patellofemoraloa", "pfjoa"], "Effusion": ["effusion"],
         "Synovitis": ["synovitis"], "Baker's": ["bakers", "bakerscyst", "baker", "bakercyst"],
         "Contusion": ["contusion", "bonecontusion"], "Fracture": ["fracture"]}
STRIP = ["s", "p", "prob", "score", "soft", "label", "pred", "y"]

def to_table(df, name):
    """Any label file → DataFrame indexed by study with the 12 finding columns (float), or None if columns do not map."""
    uid = next((c for c in df.columns if norm(c) in ("studyinstanceuid", "studyuid", "uid", "studyid")), None)
    if uid is None:
        print(f"[{name}] no study column in {list(df.columns)[:20]}")
        return None
    cols = {}
    for L, al in ALIAS.items():
        cand = []
        for c in df.columns:
            n = norm(c)
            variants = {n} | {n[len(s):] for s in STRIP if n.startswith(s)} | {n[:-len(s)] for s in STRIP if n.endswith(s)}
            if variants & set(al):
                cand.append(c)
        exact = [c for c in cand if norm(c) in al]
        pick = exact or cand
        if len(pick) != 1:
            print(f"[{name}] finding {L!r}: {len(pick)} matching columns {pick} in {list(df.columns)[:30]}")
            return None
        cols[L] = pick[0]
    t = df.drop_duplicates(uid).set_index(uid)[[cols[L] for L in LABELS]].apply(pd.to_numeric, errors="coerce").astype(float)
    t.columns = LABELS
    t.index = t.index.astype(str)
    t.index.name = "StudyInstanceUID"
    print(f"[{name}] rows {len(t)} | in train {t.index.isin(UIDS).sum()} | gold rows {t.index.isin(GT.index).sum()} | "
          f"range {np.nanmin(t.values):.2f}..{np.nanmax(t.values):.2f} | missing cells {int(t.isna().sum().sum())} | columns {list(cols.values())[:4]}…")
    return t.reindex(UIDS)

def read_any(path):
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

def by_name(fname, name):
    hits = find_inputs(lambda p: os.path.basename(p) == fname)
    if not hits:
        print(f"[{name}] {fname} not attached")
        return None
    return to_table(read_any(hits[0]), name)

# %% [markdown]
# ## Label sources

# %%
def load_ours(fname, name):                               # same loader as 04e
    for path in find_inputs(lambda p: os.path.basename(p) == fname):
        df = pd.read_csv(path)
        if len(df) < 4000:
            continue
        s = df.set_index("StudyInstanceUID")[[f"s_{L}" for L in LABELS]]
        s.columns = LABELS
        print(f"[{name}] {path} | rows {len(s)}")
        return s
    print(f"[{name}] not attached")
    return None

def auc_tab(s):
    """per-finding AUC on the gold studies (cells without a value → the column mean)"""
    g = s.reindex(GT.index)
    if g.notna().any(axis=1).sum() < 40:
        return {L: np.nan for L in LABELS}
    return {L: roc_auc_score(GT[L], g[L].fillna(s[L].mean())) for L in LABELS}

OWN = {n: s for n, s in [("gemma", load_ours("report_labels_v3.csv", "gemma")), ("qwen", load_ours("report_labels.csv", "qwen"))] if s is not None}
SRC = {}
if OWN:                                                   # the "auto" rule of 04e: per finding the better labeler, or both
    src_auc = pd.DataFrame({n: auc_tab(s) for n, s in OWN.items()}).loc[LABELS]
    names = list(OWN)
    ours = pd.DataFrame(index=UIDS, columns=LABELS, dtype=float)
    for L in LABELS:
        cols = pd.concat([OWN[n][L].rename(n) for n in names], axis=1).reindex(UIDS)
        pick = names
        if COMBINE == "auto" and len(names) == 2:
            d = src_auc.loc[L, names[0]] - src_auc.loc[L, names[1]]
            pick = [names[0]] if d >= MARGIN else [names[1]] if d <= -MARGIN else names
        ours[L] = cols[pick].mean(axis=1).fillna(cols.mean(axis=1))
    SRC["ours"] = ours
for key, fname in [("consensus", "consensus_labels.csv"), ("jev", "FINAL2_labels_scores.csv"),
                   ("nartaa_blend", "labels_v1_blend.parquet"), ("dread", "labels_llm_soft.csv")]:
    # not used: rayanbabur/train_targets_soft.csv — its rows for the 58 gold studies ARE the gold labels (AUC 1.000 in v1 of
    # this notebook), so it cannot be measured here
    t = by_name(fname, key)
    if t is not None:
        SRC[key] = t
try:
    cl = pd.read_csv(CLAUDE_URL, dtype={"STUDY_UID": str})
    t = to_table(cl.rename(columns={"STUDY_UID": "StudyInstanceUID"}), "claude")
    if t is not None:
        SRC["claude"] = t
except Exception as e:
    print("[claude] not loaded:", repr(e))
print("label sources:", list(SRC))

# %% [markdown]
# ## Teacher: out-of-fold predictions of our round-1 models

# %%
OOF = {}
for p in find_inputs(lambda p: re.search(r"/oof_m\d+_reader_folds[^/]*\.csv$", p) is not None):
    tag = re.search(r"/oof_(m\d+_reader)_folds", p).group(1)
    OOF.setdefault(tag, []).append(pd.read_csv(p, index_col=0))
for tag in sorted(OOF):
    o = pd.concat(OOF[tag])
    o = o[~o.index.duplicated()]
    o.index = o.index.astype(str)
    OOF[tag] = o.reindex(UIDS)[LABELS]
    print(f"[teacher] {tag}: {int(OOF[tag].notna().all(axis=1).sum())} of {len(UIDS)} studies | range "
          f"{np.nanmin(OOF[tag].values):.3f}..{np.nanmax(OOF[tag].values):.3f}")
COMPLETE = [t for t in sorted(OOF) if OOF[t].notna().all(axis=1).mean() > 0.99]
assert COMPLETE, "no complete OOF table attached"
# one teacher per target file: v2 = round-1 families (target of M7) · v3 = the zoomed families incl. round 2 (target of M8+)
TEACHER_SETS = {"v2": ["m4_reader", "m5_reader", "m6_reader"], "v3": ["m5_reader", "m6_reader", "m7_reader"],
                "v4": ["m7_reader", "m9_reader", "m9_reader"]}     # v4 = M7 + 2 x M9, the best OOF ensemble (notebook 14: 0.9315)
TEACHER_SETS = {k: v for k, v in TEACHER_SETS.items() if all(t in COMPLETE for t in v)}
TEACH = {k: sum(OOF[t] for t in v) / len(v) for k, v in TEACHER_SETS.items()}
print("complete OOF tables:", COMPLETE, "| teacher sets:", TEACHER_SETS)

# %% [markdown]
# ## Mixes, measured on the 58 gold studies

# %%
def mean_of(names):
    num = sum(SRC[n].fillna(0) for n in names)
    den = sum(SRC[n].notna().astype(float) for n in names)
    m = num / den.where(den > 0)
    return m.fillna(m.mean()).clip(0, 1)

rk = lambda d: d.rank(pct=True)
rows = {}
for n, s in SRC.items():
    rows[f"label · {n}"] = auc_tab(s)
ON_GOLD = [n for n in SRC if not np.isnan(list(rows[f"label · {n}"].values())[0])]
L_OLD = mean_of([n for n in ("ours", "consensus", "claude") if n in SRC])            # the round-1 target (M5, M6)
L_ALL = mean_of(ON_GOLD)                                                              # every source that has gold rows
rows["labels · round-1 target (ours+consensus+claude)"] = auc_tab(L_OLD)
rows["labels · mean of all measurable"] = auc_tab(L_ALL)
for t in COMPLETE:
    rows[f"oof · {t}"] = auc_tab(OOF[t])
if len(COMPLETE) > 1:
    rows["oof · mean of all families"] = auc_tab(sum(OOF[t] for t in COMPLETE) / len(COMPLETE))
for k, T in TEACH.items():
    rows[f"teacher {k} · mean of {'+'.join(t[:2] for t in TEACHER_SETS[k])}"] = auc_tab(T)
    for a in (0.25, 0.5, 0.75):
        rows[f"mix {k} · round-1 labels · teacher {a}"] = auc_tab((1 - a) * L_OLD + a * T)
    rows[f"mix {k} · rank · teacher 0.5"] = auc_tab(0.5 * rk(L_OLD) + 0.5 * rk(T))
if "m9_reader" in COMPLETE:                         # single-family teacher, for comparison only
    rows["mix · round-1 labels · m9 only 0.5"] = auc_tab(0.5 * L_OLD + 0.5 * OOF["m9_reader"])
if "m7_reader" in COMPLETE:                         # single-family teacher, for comparison only
    rows["mix · round-1 labels · m7 only 0.5"] = auc_tab(0.5 * L_OLD + 0.5 * OOF["m7_reader"])
tab = pd.DataFrame(rows).T[LABELS]
tab.insert(0, "macro", tab.mean(axis=1))
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
print(tab.round(3).to_string())
tab.round(4).to_csv(f"{WORK}/label_lab_gold_auc.csv")

# agreement between the families on all studies (mean Spearman over the findings) — how different their errors are
fam_corr = pd.DataFrame({a: {b: float(np.mean([OOF[a][L].corr(OOF[b][L], method="spearman") for L in LABELS])) for b in COMPLETE} for a in COMPLETE})
print("mean Spearman between OOF families:\n", fam_corr.round(3).to_string())
src_corr = pd.DataFrame({a: {b: float(np.mean([SRC[a][L].corr(SRC[b][L], method='spearman') for L in LABELS])) for b in SRC} for a in SRC})
print("mean Spearman between label sources:\n", src_corr.round(2).to_string())

# %% [markdown]
# ## Write the targets (labels = the round-1 target, so a round-2+ model differs from M5 only by the teacher half)

# %%
s = ["=== SUMMARY 09 label lab ===", f"label sources: {list(SRC)} | W_TEACHER {W_TEACHER}",
     "macro AUC on 58 gold: " + " | ".join(f"{k} {v:.4f}" for k, v in tab['macro'].items())]
L_OLD.to_csv(f"{WORK}/labels_round1.csv")
for k, T in TEACH.items():
    target = ((1 - W_TEACHER) * L_OLD + W_TEACHER * T).clip(0, 1)
    target.index.name = "StudyInstanceUID"
    target.to_csv(f"{WORK}/targets_{k}.csv")
    T.to_csv(f"{WORK}/teacher_{k}.csv")
    a = auc_tab(target)
    s.append(f"targets_{k}.csv (teacher {TEACHER_SETS[k]}): macro {np.mean(list(a.values())):.4f} | " + ", ".join(f"{L} {v:.2f}" for L, v in a.items()))
for t in COMPLETE:
    s.append(f"oof {t} per finding: " + ", ".join(f"{L} {v:.3f}" for L, v in rows[f'oof · {t}'].items()))
if len(COMPLETE) > 1:
    s.append("oof mean of all families per finding: " + ", ".join(f"{L} {v:.3f}" for L, v in rows['oof · mean of all families'].items()))
s.append("=== END ===")
print("\n".join(s))
json.dump(dict(sources=list(SRC), teacher_sets=TEACHER_SETS, W_TEACHER=W_TEACHER), open(f"{WORK}/targets.json", "w"))
