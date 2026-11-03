# %% [markdown]
# # 02b — Report Labeler v3: Gemma 4 26B-A4B + official annotation criteria
# **Settings:** Accelerator = GPU T4 x2 · Internet = ON
#
# - Model: Google **Gemma 4 26B-A4B-it**, QAT Q4_0 GGUF, hosted on Kaggle (`google/gemma-4/gguf`),
#   served with **llama.cpp** across both T4s (built from source for the T4 on the first run; the binary is saved
#   to `/kaggle/working/llama_bin` so later notebooks can attach it instead of rebuilding).
# - Prompt **v3** follows the host's published annotation criteria (discussion 733343): borderline → negative,
#   ACL/MCL high-grade only, meniscus signal reaching the surface, OA = ≥1 cm of >50% cartilage loss,
#   effusion / Baker's moderate-large only, fracture = acute cortical break.
# - Each finding graded 0 / 1 / 2 → soft score `0.5·P(1) + P(2)` from token log-probabilities.
#
# `MODE = "validate"` → 58 ground-truth studies · `MODE = "full"` → all 4,407 (ground truth first)
# The last cell prints a block starting with `=== SUMMARY` — copy that to Claude.

# %%
import os, glob, re, time, json, math, shutil, subprocess, sys, threading
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

MODE = "full"                 # "validate" or "full"
PROMPT_VERSION = "v3"
MODEL_HANDLE = "google/gemma-4/gguf/gemma-4-26b-a4b-it-qat-q4_0-gguf"
PARALLEL = 16                 # concurrent sequences in the llama.cpp server (24 gave no speed-up)
MAX_FAIL_FRAC = 0.01          # stop with an error (after saving) if more reports than this fail
CTX_PER_SLOT = 3072           # tokens per sequence (prompt ~1.3k + report + answer)
MAX_NEW = 200
N_BOOT = 2000
WORK = "/kaggle/working"
URL = "http://127.0.0.1:8080/v1/chat/completions"

DATA = os.path.dirname((glob.glob("/kaggle/input/*/train_series.csv") +
                        glob.glob("/kaggle/input/*/*/train_series.csv"))[0])
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
train = pd.read_csv(f"{DATA}/train.csv")
train["Report"] = train["Report"].fillna("")
gt_mask = train[LABELS].notna().all(axis=1)
T_START = time.time()
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True,
                     capture_output=True, text=True).stdout)
print("MODE:", MODE, "| ground-truth studies:", gt_mask.sum(), "| total:", len(train))

# %% [markdown]
# ## Prompt v3 (official annotation criteria)

# %%
SYSTEM = ("You are a subspecialty musculoskeletal radiologist. You read knee MRI reports written in any "
          "language (English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch, French, ...) and "
          "convert them into labels that follow a strict expert annotation standard.")

GUIDE = """General rules: the labels follow expert MRI annotation criteria that favour specificity. Findings that are borderline, "on the fence", only possible, or below the stated threshold are NOT positive. Grade only the knee that was examined.

- ACL: 2 = complete tear/rupture, non-visualised ACL, high-grade partial tear (more than half of the fibres), or torn reconstruction graft. 1 = partial tear of unstated grade, or tear only suspected. 0 = intact ligament or intact graft, mucoid degeneration, low-grade sprain, signal change or thickening without fibre discontinuity.
- MCL: 2 = acute complete tear or high-grade partial tear with disrupted fibres (grade 3, or grade 2 described as high-grade). 1 = partial tear / grade 2 sprain of unstated severity. 0 = intact, grade 1 / low-grade sprain, edema around the ligament only, chronic or old thickening (e.g. Pellegrini-Stieda).
- Medial Meniscus: 2 = tear (signal reaching the articular surface, "grade 3 signal", horizontal/vertical/radial/oblique/complex/flap/root/bucket-handle tear), or a morphologic abnormality: truncated, diminutive, macerated or displaced fragment. 1 = tear only possible/suspected. 0 = normal, intrasubstance/degenerative signal that does not reach the surface (grade 1-2), extrusion alone.
- Lateral Meniscus: same rules for the lateral meniscus.
- Medial OA (medial femoral condyle + medial tibial plateau): 2 = high-grade cartilage loss (more than half of the thickness: full-thickness or deep partial-thickness defects, grade 3-4 chondromalacia/chondropathy, denuded bone, "bone-on-bone", moderate-to-severe or advanced osteoarthritis/gonarthrosis) of a moderate or large area (about 1 cm or more). 1 = high-grade loss that is small/focal (< 1 cm), or moderate/grade-2 cartilage thinning, or osteophytes/"degenerative changes" without cartilage detail. 0 = normal cartilage, superficial fibrillation, grade 1 chondromalacia, mild changes.
- Lateral OA (lateral femoral condyle + lateral tibial plateau): same rules.
- PF OA (patella + trochlea): same rules (e.g. chondromalacia patellae grade 3-4 over a moderate area = 2; grade 1-2 = 0 or 1).
- Effusion: 2 = moderate or large (marked, significant, distending the joint). 1 = small-to-moderate, or "joint effusion" with no size stated. 0 = none, physiologic, trace or small.
- Synovitis: 2 = synovitis, synovial thickening/hypertrophy/proliferation, effusion-synovitis. 1 = Hoffa fat pad edema/"Hoffa-itis", thickened plica, intra-articular debris, or synovitis only possible. 0 = not mentioned.
- Baker's: 2 = moderate or large popliteal (Baker's / gastrocnemius-semimembranosus) cyst, or ruptured/leaking cyst. 1 = Baker's cyst of unstated size. 0 = none, small or tiny cyst, fluid in the bursa only.
- Contusion: 2 = bone contusion/bone bruise or traumatic bone-marrow edema (e.g. pivot-shift, "kissing" contusions). 1 = bone-marrow edema of unclear cause. 0 = none, or subchondral edema attributed to degeneration/osteoarthritis.
- Fracture: 2 = acute fracture with a cortical break or fracture line (tibial plateau, Segond, avulsion, osteochondral, impaction fracture). 1 = possible fracture, trabecular/microtrabecular fracture, or subchondral insufficiency/stress fracture. 0 = none, or old/healed fracture."""

ANSWER_FORMAT = "\n".join(f"{L}: <0, 1 or 2>" for L in LABELS)

USER = """Read the knee MRI report below and grade each finding with the rules:
2 = present and meets the criterion
1 = borderline, equivocal, or unclear whether the criterion is met
0 = absent, below the criterion, or not mentioned

{guide}

Report:
\"\"\"
{report}
\"\"\"

Answer with exactly these 12 lines and nothing else:
{fmt}"""

LEVELS = "012"
W = np.array([0.0, 0.5, 1.0])
prompt_text = SYSTEM + "\n\n" + USER.format(guide=GUIDE, report="<REPORT>", fmt=ANSWER_FORMAT)
with open(f"{WORK}/prompt_{PROMPT_VERSION}.txt", "w") as f:
    f.write(prompt_text)
print(prompt_text[:1200], "...")

# %% [markdown]
# ## llama.cpp server (build once for the T4, then reuse)

# %%
def sh(cmd):
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"$ {cmd[:120]}  [{time.time() - t0:.0f}s, rc={r.returncode}]")
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError(cmd)
    return r.stdout

def find_inputs(match):
    """Walk /kaggle/input for files whose path satisfies `match`, skipping the competition's DICOM folders."""
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        hits += [os.path.join(root, f) for f in files if match(os.path.join(root, f))]
    return hits

# A previous run's output (attached as an input) may already contain the compiled server.
SERVER = next(iter(find_inputs(lambda p: p.endswith("llama_bin/llama-server"))), None)
if SERVER:   # inputs are read-only → copy (and carry it into this run's output so the next run finds it too)
    shutil.copytree(os.path.dirname(SERVER), f"{WORK}/llama_bin", dirs_exist_ok=True)
    SERVER = f"{WORK}/llama_bin/llama-server"
print("prebuilt server:", SERVER)
t_build = time.time()
if SERVER is None:
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
    os.environ["CUDACXX"] = "/usr/local/cuda/bin/nvcc"
    if shutil.which("cmake") is None:
        sh(f"{sys.executable} -m pip install -q cmake")
    # Kaggle has the driver library only as libcuda.so.1 (no dev symlink), so CMake's CUDA::cuda_driver
    # target is not found unless we point at it (toolkit stub preferred, real driver library as fallback).
    libcuda = next(iter(glob.glob("/usr/local/cuda/lib64/stubs/libcuda.so") +
                        glob.glob("/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so") +
                        glob.glob("/usr/lib/x86_64-linux-gnu/libcuda.so*") +
                        glob.glob("/usr/local/nvidia/lib64/libcuda.so*")), None)
    print("libcuda:", libcuda)
    sh("rm -rf /tmp/llama.cpp && git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp")
    sh("cd /tmp/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_SHARED_LIBS=OFF "
       "-DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release "
       f"-DCUDA_cuda_driver_LIBRARY={libcuda}")
    sh("cd /tmp/llama.cpp && cmake --build build -j 4 --target llama-server")
    os.makedirs(f"{WORK}/llama_bin", exist_ok=True)
    shutil.copy("/tmp/llama.cpp/build/bin/llama-server", f"{WORK}/llama_bin/llama-server")
    commit = sh("cd /tmp/llama.cpp && git rev-parse HEAD").strip()
    open(f"{WORK}/llama_bin/COMMIT", "w").write(commit)
    SERVER = f"{WORK}/llama_bin/llama-server"
    print("built llama.cpp", commit)
t_build = (time.time() - t_build) / 60
os.chmod(SERVER, 0o755)

# The model must be attached in the editor (Add Input → Models → Gemma 4 → GGUF → gemma-4-26b-a4b-it-qat-q4_0-gguf);
# committed runs cannot attach new models on the fly.
t_dl = time.time()
ggufs = find_inputs(lambda p: p.endswith(".gguf") and "26b" in p.lower())
if not ggufs:
    import kagglehub
    mdir = kagglehub.model_download(MODEL_HANDLE)
    ggufs = glob.glob(f"{mdir}/**/*.gguf", recursive=True)
ggufs = sorted(set(ggufs), key=os.path.getsize)
print([(os.path.basename(g), round(os.path.getsize(g) / 1e9, 2)) for g in ggufs])
GGUF = ggufs[-1]                                     # largest file = language model (a small mmproj may exist)
t_dl = (time.time() - t_dl) / 60

# v4 lesson: llama-server's host-RAM prompt cache grew by ~800 MB per entry until the process was killed
# (every later request got "connection refused"). Disable it, and restart the server automatically if it dies.
BASE_ARGS = ["-m", GGUF, "-ngl", "999", "--split-mode", "layer", "-c", str(PARALLEL * CTX_PER_SLOT),
             "-np", str(PARALLEL), "--host", "127.0.0.1", "--port", "8080", "--jinja", "--reasoning-budget", "0"]
EXTRA_ARGS = ["--cache-ram", "0"]
proc, RESTARTS = None, 0
SERVER_LOCK = threading.Lock()

def healthy():
    try:
        return requests.get("http://127.0.0.1:8080/health", timeout=3).status_code == 200
    except Exception:
        return False

def start_server():
    global proc, EXTRA_ARGS
    if proc is not None and proc.poll() is None:
        proc.kill()
        proc.wait()
    log = open(f"{WORK}/llama_server.log", "a")
    proc = subprocess.Popen([SERVER] + BASE_ARGS + EXTRA_ARGS, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while not healthy():
        if proc.poll() is not None:
            tail = open(f"{WORK}/llama_server.log").read()[-3000:]
            if EXTRA_ARGS and ("cache-ram" in tail or "invalid argument" in tail or "unknown" in tail):
                print("this llama.cpp build has no --cache-ram; starting without it")
                EXTRA_ARGS = []
                return start_server()
            print(tail)
            raise RuntimeError("llama-server exited")
        if time.time() - t0 > 1200:
            raise TimeoutError("server did not become healthy")
        time.sleep(3)
    return (time.time() - t0) / 60

def ensure_server():
    """Called when a request fails to connect: restart the server once (thread-safe)."""
    global RESTARTS
    with SERVER_LOCK:
        if not healthy():
            RESTARTS += 1
            print(f"!! llama-server down — restart #{RESTARTS}", flush=True)
            print(open(f"{WORK}/llama_server.log").read()[-1500:], flush=True)
            start_server()

t_load = start_server()
print(f"build {t_build:.1f} min | model download {t_dl:.1f} min | load {t_load:.1f} min")
print(subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv", shell=True, capture_output=True, text=True).stdout)

# %% [markdown]
# ## Grading + parsing

# %%
_norm = lambda s: re.sub(r"[^a-z]", "", s.lower())
NORM2LABEL = {_norm(L): L for L in LABELS}
def canon(key):
    k = _norm(key)
    if k in NORM2LABEL:
        return NORM2LABEL[k]
    hits = [L for nk, L in NORM2LABEL.items() if len(k) >= 3 and (k.startswith(nk) or nk.startswith(k))]
    return hits[0] if len(hits) == 1 else None

def key_before(prefix):
    line = prefix.split("\n")[-1]
    if ":" not in line:
        return None
    key, rest = line.rsplit(":", 1)
    return canon(key) if rest.strip(" *") == "" else None

def ask(report):
    body = {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": USER.format(guide=GUIDE, report=report.strip(), fmt=ANSWER_FORMAT)}],
            "temperature": 0, "max_tokens": MAX_NEW, "logprobs": True, "top_logprobs": 5,
            "chat_template_kwargs": {"enable_thinking": False}}
    err = None
    for attempt in range(4):
        try:
            r = requests.post(URL, json=body, timeout=900)
            r.raise_for_status()
            return r.json()
        except requests.ConnectionError as e:      # server died → restart it, then retry this report
            err = e
            ensure_server()
        except Exception as e:
            err = e
            time.sleep(3)
    return {"error": repr(err)}

def parse(resp):
    """-> content, {label: level}, {label: soft score}"""
    if "error" in resp:
        return "ERROR " + resp["error"], {}, {}
    ch = resp["choices"][0]
    content = ch["message"].get("content") or ""
    toks = (ch.get("logprobs") or {}).get("content") or []
    level, score, prefix = {}, {}, ""
    for t in toks:
        tok = t.get("token", "")
        s = tok.strip()
        if s in LEVELS and s:
            name = key_before(prefix)
            if name and name not in level:
                p = np.zeros(3)
                for alt in t.get("top_logprobs", []):
                    a = alt.get("token", "").strip()
                    if a in LEVELS and a:
                        p[int(a)] += math.exp(alt["logprob"])
                if p.sum() == 0:
                    p[int(s)] = 1.0
                level[name] = int(s)
                score[name] = float((p / p.sum()) @ W)
        prefix += tok
    for L in LABELS:   # fallback when log-probs are missing for a line
        if L not in level:
            m = re.search(re.escape(L) + r"\**\s*:\s*\**\s*([012])", content)
            if m:
                level[L] = int(m.group(1))
                score[L] = float(W[int(m.group(1))])
    return content, level, score

test = parse(ask(train[gt_mask].Report.iloc[0]))
print(test[0])
print(test[1])
print({k: round(v, 3) for k, v in test[2].items()})

# %% [markdown]
# ## Run

# %%
subset = train[gt_mask] if MODE == "validate" else pd.concat([train[gt_mask], train[~gt_mask]])
OUT = f"{WORK}/report_labels_{PROMPT_VERSION}.csv"

# Resume: reuse rows that a previous run (attached as input) already labeled successfully.
rows = []
prev_files = find_inputs(lambda p: p.endswith(f"/report_labels_{PROMPT_VERSION}.csv"))
if prev_files and MODE == "full":
    prev = pd.read_csv(prev_files[0])
    prev = prev[prev["ok"].astype(str).str.lower() == "true"]
    prev = prev[prev.StudyInstanceUID.isin(subset.StudyInstanceUID)]
    rows = prev[["StudyInstanceUID", "ok", "raw"] + [f"lv_{L}" for L in LABELS] + [f"s_{L}" for L in LABELS]].to_dict("records")
    subset = subset[~subset.StudyInstanceUID.isin(prev.StudyInstanceUID)]
    print(f"resume: reusing {len(rows)} labeled studies from {prev_files[0]} | {len(subset)} left to label")
n_reused = len(rows)
t0 = time.time()
with ThreadPoolExecutor(PARALLEL) as pool:
    for i, (uid, resp) in enumerate(zip(subset.StudyInstanceUID, pool.map(ask, subset.Report.tolist()))):
        content, level, score = parse(resp)
        row = {"StudyInstanceUID": uid, "ok": len(level) == len(LABELS), "raw": content}
        for L in LABELS:
            row[f"lv_{L}"] = level.get(L, np.nan)
            row[f"s_{L}"] = score.get(L, np.nan)
        rows.append(row)
        if (i + 1) % 100 == 0 or i + 1 == len(subset):
            rate = (i + 1) / (time.time() - t0)
            n_bad = sum(1 for r in rows[n_reused:] if not r["ok"])
            print(f"{i + 1}/{len(subset)}  {rate:.2f} reports/s  eta {(len(subset) - i - 1) / rate / 60:.1f} min"
                  f"  | failures so far {n_bad} | server restarts {RESTARTS}", flush=True)
            if (i + 1) % 500 == 0:
                pd.DataFrame(rows).to_csv(OUT, index=False)
t_label = (time.time() - t0) / 60
pred = pd.DataFrame(rows).set_index("StudyInstanceUID")
pred["ok"] = pred["ok"].astype(str).str.lower() == "true"
pred.to_csv(OUT)
fails = pred[~pred.ok]
print(f"labeling {t_label:.1f} min | parse failures: {len(fails)} | server restarts: {RESTARTS}")
for uid, raw in fails.raw.head(3).items():
    print("  PARSE FAIL", uid[-12:], "|", str(raw).replace("\n", " / ")[:300])
if len(fails) > MAX_FAIL_FRAC * len(pred):
    raise RuntimeError(f"{len(fails)} of {len(pred)} reports failed — labels saved to {OUT}; re-run to resume")

# %% [markdown]
# ## Validation on the 58 ground-truth studies

# %%
m = train[gt_mask].set_index("StudyInstanceUID")[LABELS].astype(int).join(pred, how="inner")
res, THRESH = [], {}
for L in LABELS:
    y, lv, s = m[L], m[f"lv_{L}"], m[f"s_{L}"].fillna(0)
    f1s = {t: f1_score(y, (lv >= t).astype(int), zero_division=0) for t in (1, 2)}
    accs = {t: accuracy_score(y, (lv >= t).astype(int)) for t in (1, 2)}
    thr = max((1, 2), key=lambda t: (f1s[t], accs[t]))
    THRESH[L] = thr
    res.append(dict(label=L, thr=thr, n_pos=int(y.sum()), pred_pos=int((lv >= thr).sum()),
                    acc=accs[thr], f1=f1s[thr], f1_ge2=f1s[2], acc_ge2=accs[2], auc=roc_auc_score(y, s),
                    lv0=int((lv == 0).sum()), lv1=int((lv == 1).sum()), lv2=int((lv == 2).sum())))
res = pd.DataFrame(res)
res.to_csv(f"{WORK}/labeler_eval_{PROMPT_VERSION}.csv", index=False)
print(res.round(3).to_string(index=False))

rng = np.random.default_rng(0)
Y = m[LABELS].values
S = m[[f"s_{L}" for L in LABELS]].fillna(0).values
def macro_auc(idx):
    return np.nanmean([roc_auc_score(Y[idx, j], S[idx, j]) if Y[idx, j].min() != Y[idx, j].max() else np.nan
                       for j in range(len(LABELS))])
boot = np.array([macro_auc(rng.integers(0, len(m), len(m))) for _ in range(N_BOOT)])
macro = macro_auc(np.arange(len(m)))
lo, hi = np.percentile(boot, [2.5, 97.5])

for L in LABELS:
    pred[f"h_{L}"] = (pred[f"lv_{L}"] >= THRESH[L]).astype(int)
pred.to_csv(OUT)
err_counts = {}
for uid, r in m.join(pred[[f"h_{L}" for L in LABELS]]).iterrows():
    for L in LABELS:
        if r[L] != r[f"h_{L}"]:
            e = ("FN " if r[L] == 1 else "FP ") + L
            err_counts[e] = err_counts.get(e, 0) + 1

# %% [markdown]
# ## Summary — copy this block to Claude

# %%
proc.terminate()
s = [f"=== SUMMARY labeler {PROMPT_VERSION} | Gemma 4 26B-A4B QAT Q4_0 (llama.cpp) | MODE={MODE} ===",
     f"studies: {len(pred)} (reused {n_reused}, new {len(subset)}) | parse failures: {len(fails)} | server restarts {RESTARTS}",
     f"build {t_build:.1f} min, download {t_dl:.1f}, "
     f"load {t_load:.1f}, labeling {t_label:.1f} min ({len(subset) / max(t_label, 1e-9) / 60:.2f} reports/s) | "
     f"total {(time.time() - T_START) / 60:.1f} min",
     f"MACRO on 58 GT: acc={res.acc.mean():.3f} f1={res.f1.mean():.3f} auc={macro:.3f} [95% CI {lo:.3f}, {hi:.3f}] "
     f"| fixed thr>=2: acc={res.acc_ge2.mean():.3f} f1={res.f1_ge2.mean():.3f}",
     "  (reference, Qwen2.5-7B: v1 auc .874 [.835,.906] f1 .715 | ensemble v1+v2 auc .876 f1 .766)",
     "per label  thr  acc   f1    auc   pos/pred  levels(0/1/2)"]
for _, r in res.iterrows():
    s.append(f"  {r.label:<17}{r.thr}  {r.acc:.3f} {r.f1:.3f} {r.auc:.3f}  {r.n_pos}/{r.pred_pos}     {r.lv0}/{r.lv1}/{r.lv2}")
s.append("top errors: " + ", ".join(f"{k} {v}" for k, v in sorted(err_counts.items(), key=lambda kv: -kv[1])[:8]))
if MODE == "full":
    s.append("prevalence (all train, hard): " + ", ".join(f"{L} {pred[f'h_{L}'].mean():.2f}" for L in LABELS))
s.append("=== END ===")
s = "\n".join(s)
print(s)
open(f"{WORK}/labeler_summary_{PROMPT_VERSION}_{MODE}.txt", "w").write(s)
