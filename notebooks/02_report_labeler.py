# %% [markdown]
# # 02 — Report Labeler (v3 = ensemble of prompt v1 + prompt v2)
# **Settings:** Accelerator = GPU T4 x2 · Internet = ON (downloads the model from Hugging Face)
#
# An open-weights LLM running *inside* Kaggle (no competition data leaves the platform) reads each
# radiology report with two prompts and we average their soft scores:
#
# | prompt | answer per finding | soft score from token logits |
# |---|---|---|
# | v1 | 0 / 1 (absent / present) | P(1) |
# | v2 | 0 / 1 / 2 (absent / equivocal-minimal / clearly present) + grading guide | 0.5·P(1) + P(2) |
# | **ensemble** | | mean of v1 and v2 → training target for the image model |
#
# `MODE = "validate"` → the 58 ground-truth studies only (about 8 min)
# `MODE = "full"`     → all 4,407 train studies, ground-truth studies first (about 5.5 h)
#
# The last cell prints a block starting with `=== SUMMARY` — copy that to Claude.

# %%
import os, glob, re, time, subprocess, sys
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"  # must be set before torch is imported
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

MODE = "full"                # "validate" or "full"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
RUN_PROMPTS = ["v1"]         # full run: v1 only (~2.3 h); v1+v2 costs ~5.5 h for no AUC gain on the 58 GT
MAX_BATCH = 24
TOKEN_BUDGET = 12_000        # max (batch size x longest prompt) per batch
MAX_NEW = 100
N_BOOT = 2000
WORK = "/kaggle/working"

DATA = os.path.dirname((glob.glob("/kaggle/input/*/train_series.csv") +
                        glob.glob("/kaggle/input/*/*/train_series.csv"))[0])
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

train = pd.read_csv(f"{DATA}/train.csv")
train["Report"] = train["Report"].fillna("")
gt_mask = train[LABELS].notna().all(axis=1)
print("MODE:", MODE, "| prompts:", RUN_PROMPTS,
      "| GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("ground-truth studies:", gt_mask.sum(), "| total:", len(train))

# %% [markdown]
# ## Prompts (v1 and v2 are kept verbatim from their original runs)

# %%
PROMPTS = {}

PROMPTS["v1"] = dict(
    levels="01", weights=[0.0, 1.0], fmt="<0 or 1>",
    system=("You are an expert musculoskeletal radiologist. You read knee MRI reports written in any "
            "language (English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch, French, ...) "
            "and extract findings accurately."),
    guide="""- ACL: anterior cruciate ligament injury of any grade (sprain, partial or complete tear, torn reconstruction graft). Intact ligament or intact graft = 0.
- MCL: medial collateral ligament injury of any grade (sprain, partial or complete tear).
- Medial Meniscus: tear of the medial meniscus (any tear type, including degenerative, complex, root, radial, bucket-handle). Intrasubstance degeneration/signal WITHOUT tear = 0.
- Lateral Meniscus: tear of the lateral meniscus (same rules).
- Medial OA: osteoarthritis of the medial tibiofemoral compartment (cartilage thinning/loss or chondral defects of the medial femoral condyle or medial tibial plateau, osteophytes, subchondral sclerosis/cysts/edema from degeneration).
- Lateral OA: the same for the lateral tibiofemoral compartment.
- PF OA: osteoarthritis or chondral damage of the patellofemoral compartment (patellar or trochlear cartilage loss, chondromalacia patellae, patellofemoral osteophytes).
- Effusion: joint effusion (small, moderate or large = 1; 'physiologic/minimal fluid' or 'no effusion' = 0).
- Synovitis: synovitis, synovial thickening or synovial inflammation (including Hoffa fat pad synovitis).
- Baker's: Baker's (popliteal) cyst of any size.
- Contusion: bone contusion / bone bruise / traumatic bone marrow edema.
- Fracture: any fracture (including impaction, avulsion, osteochondral, stress or insufficiency fracture).""",
    user="""Read the knee MRI report below. For each finding, answer 1 if the report says it is present, and 0 if it is absent or not mentioned. If the report describes it as possible or suspected, answer 1 only if it is more likely present than not.

Findings:
{guide}

Report:
\"\"\"
{report}
\"\"\"

Answer with exactly these 12 lines and nothing else:
{fmt}""")

PROMPTS["v2"] = dict(
    levels="012", weights=[0.0, 0.5, 1.0], fmt="<0, 1 or 2>",
    system=("You are an expert musculoskeletal radiologist. You read knee MRI reports written in any "
            "language (English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch, French, ...) "
            "and grade findings accurately and consistently."),
    guide="""- ACL (anterior cruciate ligament): 2 = partial or complete tear, torn or insufficient reconstruction graft, high-grade sprain. 1 = low-grade sprain, mucoid degeneration, thickened/edematous ligament, or a tear that is only possible/suspected. 0 = intact ligament or intact graft.
- MCL (medial collateral ligament): 2 = partial or complete tear, grade 2-3 sprain. 1 = grade 1 sprain, edema around the ligament only, chronic thickening from old injury. 0 = intact.
- Medial Meniscus: 2 = tear of any type (horizontal, vertical, radial, oblique, complex, flap, root, bucket-handle, degenerative tear), or re-tear after surgery. 1 = tear possible/suspected, or maceration/extrusion without a clearly described tear. 0 = normal, intrasubstance degeneration (grade 1-2 signal) WITHOUT a tear, or post-operative meniscus without a new tear.
- Lateral Meniscus: same grading for the lateral meniscus.
- Medial OA (medial tibiofemoral compartment: medial femoral condyle and medial tibial plateau): 2 = cartilage thinning, loss, fissures or defects, osteophytes, or degenerative subchondral changes (sclerosis, cysts, edema) in this compartment; 'medial compartment arthrosis/gonarthrosis'. 1 = only minimal/early cartilage change (e.g., grade 1 chondromalacia, subtle surface irregularity). 0 = normal cartilage in this compartment.
- Lateral OA (lateral femoral condyle and lateral tibial plateau): same grading for the lateral compartment.
- PF OA (patellofemoral compartment: patella and trochlea): 2 = patellar or trochlear cartilage thinning, fissuring, fibrillation or defects, chondromalacia patellae grade 2 or higher, patellofemoral osteophytes, 'patellofemoral arthrosis'. 1 = chondromalacia grade 1 or mild signal change of the patellar cartilage only. 0 = normal patellofemoral cartilage.
- Effusion: 2 = moderate or large joint effusion, or 'joint effusion' stated without a size. 1 = small effusion. 0 = no effusion, or only a physiologic/minimal/trace amount of joint fluid.
- Synovitis: 2 = synovitis, synovial thickening, proliferation or hypertrophy, effusion-synovitis. 1 = Hoffa fat pad edema/inflammation, thickened plica, intra-articular debris or bodies suggesting synovial reaction, or synovitis only possible. 0 = not mentioned.
- Baker's: 2 = Baker's / popliteal / gastrocnemius-semimembranosus bursal cyst of any size, including ruptured. 1 = small amount of fluid in that bursa not called a cyst. 0 = none.
- Contusion: 2 = bone contusion, bone bruise, or bone marrow edema attributed to trauma. 1 = bone marrow edema of unclear cause. 0 = none, or subchondral edema clearly attributed to degeneration/osteoarthritis.
- Fracture: 2 = any fracture (impaction, avulsion including Segond, osteochondral, tibial plateau, subchondral insufficiency, stress fracture, visible fracture line). 1 = possible fracture or trabecular microfractures only. 0 = none.""",
    user="""Read the knee MRI report below and grade each finding:
2 = clearly present
1 = equivocal or minimal (possible, suspected, cannot be excluded, trace, mild, low-grade)
0 = absent, normal, or not mentioned

Grading guide:
{guide}

Report:
\"\"\"
{report}
\"\"\"

Answer with exactly these 12 lines and nothing else:
{fmt}""")

for pv, P in PROMPTS.items():
    P["answer_format"] = "\n".join(f"{L}: {P['fmt']}" for L in LABELS)
    with open(f"{WORK}/prompt_{pv}.txt", "w") as f:
        f.write(P["system"] + "\n\n" + P["user"].format(guide=P["guide"], report="<REPORT>", fmt=P["answer_format"]))
print(open(f"{WORK}/prompt_v2.txt").read()[:1500], "...")

# %% [markdown]
# ## Load the model (fp16, split over both T4s)

# %%
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="auto",
                                             max_memory={0: "10GiB", 1: "10GiB"}).eval()
print(f"loaded in {time.time() - t0:.0f}s")

for P in PROMPTS.values():
    P["level_ids"] = []
    for d in P["levels"]:
        ids = tok.encode(d, add_special_tokens=False)
        assert len(ids) == 1, (d, ids)
        P["level_ids"].append(ids[0])

def build(report, P):
    msgs = [{"role": "system", "content": P["system"]},
            {"role": "user", "content": P["user"].format(guide=P["guide"], report=report.strip(), fmt=P["answer_format"])}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

# Tolerant key matching: "Baker's cyst", "**ACL**", "- PF OA", curly apostrophes, ...
_norm = lambda s: re.sub(r"[^a-z]", "", s.lower())
NORM2LABEL = {_norm(L): L for L in LABELS}
def canon(key):
    k = _norm(key)
    if k in NORM2LABEL:
        return NORM2LABEL[k]
    hits = [L for nk, L in NORM2LABEL.items() if len(k) >= 3 and (k.startswith(nk) or nk.startswith(k))]
    return hits[0] if len(hits) == 1 else None

def key_before(prefix):
    """If the text ends right after 'Some Key:' on the current line, return the canonical label."""
    line = prefix.split("\n")[-1]
    if ":" not in line:
        return None
    key, rest = line.rsplit(":", 1)
    return canon(key) if rest.strip(" *") == "" else None

@torch.no_grad()
def grade_batch(reports, P):
    """Per report: raw text, {label: level}, {label: soft score}."""
    enc = tok([build(r, P) for r in reports], return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, output_scores=True,
                         return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
    seqs = out.sequences[:, enc.input_ids.shape[1]:].tolist()
    ids = P["level_ids"]
    probs = torch.softmax(torch.stack([s[:, ids] for s in out.scores], 1).float(), -1).cpu().numpy()
    w = np.array(P["weights"])
    results = []
    for b, seq in enumerate(seqs):
        level, score, prefix = {}, {}, ""
        for t, tid in enumerate(seq):
            if tid in ids:
                name = key_before(prefix)
                if name and name not in level:
                    level[name] = ids.index(tid)
                    score[name] = float(probs[b, t] @ w)
            prefix += tok.decode([tid])
        results.append((tok.decode(seq, skip_special_tokens=True), level, score))
    return results

# %% [markdown]
# ## Run every prompt

# %%
def batches(d, col):  # token-budget batching over a frame sorted longest-first
    d = d.sort_values(col, ascending=False)
    start = 0
    while start < len(d):
        longest, end = d[col].iloc[start], start + 1
        while end < len(d) and end - start < MAX_BATCH and (end - start + 1) * longest <= TOKEN_BUDGET:
            end += 1
        yield d.iloc[start:end]
        start = end

subset = (train[gt_mask] if MODE == "validate" else pd.concat([train[gt_mask], train[~gt_mask]])).copy()
pred = subset[["StudyInstanceUID"]].copy().set_index("StudyInstanceUID")
OUT = f"{WORK}/report_labels.csv"
timing = {}
for pv in RUN_PROMPTS:
    P = PROMPTS[pv]
    col = f"n_tok_{pv}"
    subset[col] = [len(tok(build(r, P)).input_ids) for r in subset["Report"]]
    parts = [subset[subset.StudyInstanceUID.isin(train.StudyInstanceUID[gt_mask])]]
    if MODE == "full":
        parts.append(subset[~subset.StudyInstanceUID.isin(train.StudyInstanceUID[gt_mask])])
    rows, t0 = {}, time.time()
    for i, chunk in enumerate(b for p in parts for b in batches(p, col)):
        for uid, (raw, level, score) in zip(chunk["StudyInstanceUID"], grade_batch(chunk["Report"].tolist(), P)):
            r = {f"ok_{pv}": len(level) == len(LABELS), f"raw_{pv}": raw}
            for L in LABELS:  # unparsed answer -> NaN, so the ensemble falls back on the other prompt
                r[f"lv_{pv}_{L}"] = level.get(L, np.nan)
                r[f"s_{pv}_{L}"] = score.get(L, np.nan)
            rows[uid] = r
        if i % 20 == 0 or len(rows) == len(subset):
            rate = len(rows) / (time.time() - t0)
            print(f"[{pv}] {len(rows)}/{len(subset)}  {rate:.2f} reports/s  eta {(len(subset) - len(rows)) / rate / 60:.1f} min", flush=True)
            if i % 100 == 0:
                pred.join(pd.DataFrame.from_dict(rows, orient="index"), how="left").to_csv(OUT)
    pred = pred.join(pd.DataFrame.from_dict(rows, orient="index"))
    timing[pv] = (time.time() - t0) / 60
    fails = pred[~pred[f"ok_{pv}"]]
    print(f"[{pv}] done in {timing[pv]:.1f} min | parse failures: {len(fails)}")
    for uid, raw in fails[f"raw_{pv}"].head(3).items():
        print("  PARSE FAIL", uid[-12:], "|", raw.replace("\n", " / ")[:300])

for L in LABELS:
    pred[f"s_ens_{L}"] = pred[[f"s_{pv}_{L}" for pv in RUN_PROMPTS]].mean(axis=1)
pred.to_csv(OUT)

# %% [markdown]
# ## Validation on the 58 ground-truth studies
# Hard-label thresholds are chosen per finding on these 58 studies (v1: fixed at 1; v2: level 1 or 2;
# ensemble: 0.25 / 0.5 / 0.75). AUCs do not depend on thresholds. 95% CIs by bootstrapping studies.

# %%
m = train[gt_mask].set_index("StudyInstanceUID")[LABELS].astype(int).join(pred, how="inner")
VARIANTS = RUN_PROMPTS + ["ens"]
THR_GRID = {"v1": [1], "v2": [1, 2], "ens": [0.25, 0.5, 0.75]}

def hard(v, L, thr):
    return (m[f"lv_{v}_{L}"] >= thr) if v != "ens" else (m[f"s_ens_{L}"] >= thr)

THRESH, rows = {v: {} for v in VARIANTS}, []
for v in VARIANTS:
    for L in LABELS:
        y = m[L]
        best = max(THR_GRID[v], key=lambda t: (f1_score(y, hard(v, L, t).astype(int), zero_division=0),
                                                accuracy_score(y, hard(v, L, t).astype(int))))
        THRESH[v][L] = best
        h = hard(v, L, best).astype(int)
        rows.append(dict(variant=v, label=L, thr=best, n_pos=int(y.sum()), pred_pos=int(h.sum()),
                         acc=accuracy_score(y, h), f1=f1_score(y, h, zero_division=0),
                         auc=roc_auc_score(y, m[f"s_{v}_{L}"].fillna(0))))
res = pd.DataFrame(rows)
res.to_csv(f"{WORK}/labeler_eval.csv", index=False)
auc_tab = res.pivot(index="label", columns="variant", values="auc").loc[LABELS, VARIANTS]
print(auc_tab.round(3).to_string())

# bootstrap macro AUC (resample studies; skip labels that become single-class)
rng = np.random.default_rng(0)
Y = m[LABELS].values
S = {v: m[[f"s_{v}_{L}" for L in LABELS]].fillna(0).values for v in VARIANTS}
def macro_auc(idx, v):
    return np.nanmean([roc_auc_score(Y[idx, j], S[v][idx, j]) if Y[idx, j].min() != Y[idx, j].max() else np.nan
                       for j in range(len(LABELS))])
boot = {v: [] for v in VARIANTS}
for _ in range(N_BOOT):
    idx = rng.integers(0, len(m), len(m))
    for v in VARIANTS:
        boot[v].append(macro_auc(idx, v))
boot = {v: np.array(b) for v, b in boot.items()}
ci = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))
macro = {v: macro_auc(np.arange(len(m)), v) for v in VARIANTS}

# final labels for training: ensemble soft score + ensemble hard label
for L in LABELS:
    pred[f"s_{L}"] = pred[f"s_ens_{L}"]
    pred[f"h_{L}"] = (pred[f"s_ens_{L}"] >= THRESH["ens"][L]).astype(int)
pred.to_csv(OUT)

m_ens = m.join(pred[[f"h_{L}" for L in LABELS]])
err_counts = {}
for uid, r in m_ens.iterrows():
    for L in LABELS:
        if r[L] != r[f"h_{L}"]:
            e = ("FN " if r[L] == 1 else "FP ") + L
            err_counts[e] = err_counts.get(e, 0) + 1

# %% [markdown]
# ## Language breakdown

# %%
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "langdetect"])
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0
def lang(t):
    try: return detect(t[:1000])
    except Exception: return "unk"
lang_of = dict(zip(train["StudyInstanceUID"], train["Report"].map(lang)))
pred["lang"] = pred.index.map(lang_of)
m_ens["lang"] = m_ens.index.map(lang_of)
m_ens["acc_all"] = np.mean([m_ens[L] == m_ens[f"h_{L}"] for L in LABELS], axis=0)
by_lang_gt = m_ens.groupby("lang")["acc_all"].agg(["count", "mean"])
print("ground-truth accuracy by language (ensemble):\n", by_lang_gt.round(3).to_string())
if MODE == "full":
    prev = pred.groupby("lang")[[f"h_{L}" for L in LABELS]].mean().round(2)
    prev.columns = LABELS
    prev.insert(0, "n", pred.groupby("lang").size())
    prev.to_csv(f"{WORK}/prevalence_by_language.csv")
    print("\nLLM label prevalence by language (all train):\n", prev.to_string())
pred.to_csv(OUT)

# %% [markdown]
# ## Summary — copy this block to Claude

# %%
s = [f"=== SUMMARY labeler v3-ensemble | {MODEL} | MODE={MODE} ===",
     f"studies labeled: {len(pred)} | minutes per prompt: " + ", ".join(f"{pv} {t:.1f}" for pv, t in timing.items()),
     "parse failures: " + ", ".join(f"{pv} {int((~pred[f'ok_{pv}']).sum())}" for pv in RUN_PROMPTS),
     "MACRO on 58 GT       acc    f1     auc   [95% CI]"]
for v in VARIANTS:
    r = res[res.variant == v]
    lo, hi = ci(boot[v])
    s.append(f"  {v:<5}              {r.acc.mean():.3f}  {r.f1.mean():.3f}  {macro[v]:.3f} [{lo:.3f}, {hi:.3f}]")
for v in VARIANTS:
    if v != "v1":
        lo, hi = ci(boot[v] - boot["v1"])
        s.append(f"  delta {v} - v1 macro AUC: {macro[v] - macro['v1']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
s.append("per-label AUC            " + "  ".join(f"{v:>5}" for v in VARIANTS) + "   ens thr")
for L in LABELS:
    s.append(f"  {L:<22}" + "  ".join(f"{auc_tab.loc[L, v]:.3f}" for v in VARIANTS) + f"   {THRESH['ens'][L]}")
s.append("top errors (ens): " + ", ".join(f"{k} {v}" for k, v in sorted(err_counts.items(), key=lambda kv: -kv[1])[:8]))
s.append("GT acc by language (ens): " + ", ".join(f"{k} {v['mean']:.2f} (n={int(v['count'])})" for k, v in by_lang_gt.iterrows()))
if MODE == "full":
    s.append("LLM prevalence (all train, ens): " + ", ".join(f"{L} {pred[f'h_{L}'].mean():.2f}" for L in LABELS))
s.append("=== END ===")
s = "\n".join(s)
print(s)
with open(f"{WORK}/labeler_summary_{MODE}.txt", "w") as f:
    f.write(s)
