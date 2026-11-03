# %% [markdown]
# # 04k — Train M11 = the M7 recipe (ConvNeXt-T, zoomed 03d cache) on targets_v3 (report labels + OOF predictions of M5, M6, M7)
# **Settings:** Accelerator = GPU T4 x2 · Internet = ON (downloads ImageNet weights)
# **Inputs to attach:** competition data · notebook "RSNA Knee 03c Preprocess M2" (cache + knee_prep.py) · notebook "RSNA Knee 09 Label Lab" (targets_v3.csv) ·
# notebook "RSNA Knee 02b Labeler Gemma4" (report_labels_v3.csv) · notebook "RSNA Knee 02 Report Labeler" (report_labels.csv) ·
# dataset "karttikjangid05/rsna-knee-public-llm-label-sets-compared" (consensus_labels.csv, CC0)
#
# Why: the public 0.950 notebooks are RAPTOR (CoAtNet, 0.949 alone) + a 3-fold 2.5D ConvNeXt reader (0.929 alone,
# goodpjw2008, Apache-2.0). The 0.951-0.954 notebooks replaced/added a *better, different* second model. M4 is our
# version of that reader architecture (ConvNeXt-T per 3-slice window → slot + slice-position embeddings → 2-layer
# transformer over all windows → one attention pooling per finding), trained on
#   target = mean(our Gemma/Qwen labels, public 6-source consensus)  (both ≈0.89-0.90 macro AUC on the 58 gold),
# 5 site-grouped folds (same folds as M2), EMA, fp16 on T4.
#
# `FOLDS` = which folds this copy trains (A: [0, 1], B: [2, 3], C: [4], D: [-1] = one model on ALL studies, no
# held-out fold, other seed); at most 2 folds per notebook. With `PARALLEL = True` the 2 folds train at the same time, one per GPU
# (forked processes that share the cache in RAM). The model code is written to `knee_reader.py`, which 05c imports. The last cell prints `=== SUMMARY`.

# %%
FOLDS = [4, -1]
FAMILY = "m11"                      # m4: 03c cache (130 mm, 240 px) + ConvNeXt-T · m5: 03d cache (100 mm, 224 px) + ConvNeXt-T ·
                                   # m6: 03d cache + EfficientNetV2-S (another network family: more diverse errors in the blend)
PREP, BACKBONE, TAG = {"m4": ("03c", "convnext_tiny.fb_in22k_ft_in1k", "m4_reader"),
                       "m5": ("03d", "convnext_tiny.fb_in22k_ft_in1k", "m5_reader"),
                       "m6": ("03d", "tf_efficientnetv2_s.in21k_ft_in1k", "m6_reader"),
                       "m7": ("03d", "convnext_tiny.fb_in22k_ft_in1k", "m7_reader"),
                       "m8": ("03e", "convnext_tiny.fb_in22k_ft_in1k", "m8_reader"),        # high resolution: 320 px cache, 224 px training crops
                       "m9": ("03d", "tf_efficientnetv2_s.in21k_ft_in1k", "m9_reader"),     # M6 on the round-3 target
                       "m10": ("03d", "vit_small_patch14_dinov2.lvd142m", "m10_reader"),
                       "m11": ("03d", "convnext_tiny.fb_in22k_ft_in1k", "m11_reader")}[FAMILY]      # M7 recipe on the round-3 target
# round 2+: the target is a file of notebook 09 (report labels + out-of-fold predictions of the earlier families)
TARGET_FILE = {"m7": "targets_v2.csv", "m8": "targets_v3.csv", "m9": "targets_v3.csv", "m10": "targets_v3.csv", "m11": "targets_v3.csv"}.get(FAMILY)
CROP_TRAIN = 224 if PREP == "03e" else None    # 03e: train on random 224 px crops (GPU cost of M7), validate and test on the full 320 px
META_PREP = "03d" if PREP == "03e" else PREP   # 03e is split over two notebooks: site folds from the complete 03d metadata (same series)
ROUND2 = TARGET_FILE is not None
PARALLEL = True                    # train the folds of this copy at the same time, one per GPU (halves the GPU quota used)
import os
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"   # torch.cuda.is_available() must not start CUDA (forked folds)
import os, sys, gc, glob, time, math, json, copy, random, faulthandler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import psutil
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

N_FOLDS = 5
EPOCHS = 12
BATCH = 8                          # studies per step (split over the 2 GPUs)
T_PER_SLOT = [8, 8, 8, 4, 4]       # 3-slice windows per slot (sag_fs, cor_fs, ax_fs, sag_nfs, cor_t1) → 32 tokens
DIM, DEPTH = 384, 2
LR_BB, LR = (5e-5 if "vit" in BACKBONE else 1e-4), 5e-4   # backbone / head (goodpjw2008 reader recipe; ViT: half)
WD, EMA_DECAY, DROP_PATH = 0.02, 0.998, 0.1
GT_WEIGHT = 3.0
W_CONSENSUS = 0.5 if PREP == "03c" else 1 / 3   # weight of the public consensus in the soft target
W_CLAUDE = 0.0 if PREP == "03c" else 1 / 3      # weight of the public Claude Opus 4.5 report labels (NTejas-1, MIT); rest = ours
CLAUDE_URL = "https://raw.githubusercontent.com/NTejas-1/RSNA-Knee-Abnormality-Detection/main/labels_ds/labels_v1_opus45.csv"
COMBINE, MARGIN = "auto", 0.02     # how our two labelers are combined (same rule as M1/M2)
SEED = 42
WORK = "/kaggle/working"
T0 = time.time()
torch.backends.cudnn.benchmark = True

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(SEED)

def find_inputs(match):
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        hits += [os.path.join(root, f) for f in files if match(os.path.join(root, f))]
    return sorted(hits)

PREP_DIR = os.path.dirname(find_inputs(lambda p: p.endswith("/knee_prep.py") and PREP in p)[0])
sys.path.insert(0, PREP_DIR)
import knee_prep as kp               # layout constants (slots, slices per slot, S) of the cache
ENC_KWARGS = {"img_size": kp.S} if "vit" in BACKBONE else {}   # ViTs are built for the cache resolution (DINOv2 default: 518)

DATA = os.path.dirname(find_inputs(lambda p: p.endswith("/train_series.csv"))[0])
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
train = pd.read_csv(f"{DATA}/train.csv")
gt_mask = train[LABELS].notna().all(axis=1)
GT = train[gt_mask].set_index("StudyInstanceUID")[LABELS].astype(int)
print("GPUs:", torch.cuda.device_count(), "| folds", FOLDS, "| parallel", PARALLEL,
      "| prep", PREP_DIR, "| slots", [s[0] for s in kp.SLOTS], kp.DS, "S", kp.S)

# %% [markdown]
# ## Model code → `knee_reader.py` (05c imports this file from this notebook's output)

# %%
%%writefile /kaggle/working/knee_reader.py
"""M4 reader model + window sampling, shared by 04e (training) and 05c (submission). Import it, never copy it.
Architecture after goodpjw2008's public 2.5D ConvNeXt-MIL reader (Apache-2.0), adapted to the 03c cache
(5 slots, 130 mm crop, 240 px) and to fp16 on T4 GPUs."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)

def windows(t_per_slot, ds, off, rng=None):
    """→ global slice indices (N, 3) of centre±1 windows, slot of each window (N,), relative position (N,) in 0..1.
    rng=None → evenly spaced centres (validation / test); else one random centre per equal-width bin (training)."""
    idx, slot, pos = [], [], []
    for k, (t, d) in enumerate(zip(t_per_slot, ds)):
        a, b = 1.0, d - 2.0
        if rng is None:
            c = np.linspace(a, b, t)
        else:
            edges = np.linspace(a, b, t + 1)
            c = edges[:-1] + rng.uniform(0, 1, t) * (edges[1:] - edges[:-1])
        c = np.clip(np.round(c).astype(int), 1, d - 2)
        idx.append(off[k] + np.stack([c - 1, c, c + 1], 1))
        slot += [k] * t
        pos.append(c / max(d - 1, 1))
    return np.concatenate(idx), np.array(slot), np.concatenate(pos).astype(np.float32)

def normalize(x):
    """x: (B, N, 3, S, S) float in [0, 1]"""
    return (x - MEAN.to(x.device)) / STD.to(x.device)

def gpu_augment(x, slot, n_slot):
    """Intensity per study; one small affine per (study, slot) shared by that slot's windows; never a left-right flip
    (the cache is lateral-first, so medial/lateral must stay where they are)."""
    B, N = x.shape[:2]
    r = lambda *s: torch.rand(*s, device=x.device)
    sh = (B, 1, 1, 1, 1)
    x = x.clamp(1e-4, 1) ** torch.exp((r(*sh) - 0.5) * 0.5)
    x = x * (1 + (r(*sh) - 0.5) * 0.2)
    ang = (r(B, n_slot) - 0.5) * math.radians(24)
    scale = 1 + (r(B, n_slot) - 0.5) * 0.24
    tx, ty = (r(B, n_slot) - 0.5) * 0.12, (r(B, n_slot) - 0.5) * 0.12
    theta = torch.stack([torch.stack([torch.cos(ang) / scale, -torch.sin(ang) / scale, tx], -1),
                         torch.stack([torch.sin(ang) / scale, torch.cos(ang) / scale, ty], -1)], -2)
    theta = theta[:, torch.as_tensor(slot, device=x.device)].flatten(0, 1)
    flat = x.flatten(0, 1)
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    return F.grid_sample(flat, grid, align_corners=False, padding_mode="zeros").view_as(x).clamp(0, 1)

class ReaderNet(nn.Module):
    """2D backbone on every present 3-slice window → projection + slot embedding + slice-position embedding →
    transformer over all windows of the study → one attention pooling per finding → 12 logits."""
    def __init__(self, backbone, slot_of_tok, n_slot, pretrained=True, dim=384, depth=2, drop=0.2, drop_path=0.1, n_out=12,
                 ckpt=False, enc_kwargs=None):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path, **(enc_kwargs or {}))
        if ckpt:                                                # gradient checkpointing: less activation memory
            self.enc.set_grad_checkpointing(True)
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(self.enc.num_features, dim), nn.LayerNorm(dim))
        self.register_buffer("slot", torch.as_tensor(slot_of_tok, dtype=torch.long), persistent=False)
        self.slot_emb = nn.Embedding(n_slot, dim)
        self.pos_emb = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, 8, dim * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False) if depth else None
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n_out))
        self.cls_w = nn.Parameter(torch.randn(n_out, dim) * 0.02)
        self.cls_b = nn.Parameter(torch.zeros(n_out))

    def forward(self, x, mask, pos):
        """x: (B, N, 3, S, S) normalised · mask: (B, N) bool (False = slot missing) · pos: (B, N) in 0..1"""
        B, N = x.shape[:2]
        mask = mask.clone()
        mask[~mask.any(1), 0] = True                            # a study with no image at all: avoid NaN
        # The backbone sees ALL windows (missing slots are black images): a fixed batch shape. Running it only on the
        # present windows made the shape change every step → cuDNN re-tuned every step (19 s/step instead of ~0.7 s).
        with torch.autocast("cuda", dtype=torch.float16):      # explicit: autocast is thread-local (DataParallel)
            f = self.enc(x.flatten(0, 1)).float()
        feat = f.view(B, N, -1) * mask[..., None]
        h = self.proj(feat) + self.slot_emb(self.slot)[None] + self.pos_emb(pos.unsqueeze(-1).float())
        if self.tf is not None:
            h = self.tf(h, src_key_padding_mask=~mask)
        a = self.att(h).masked_fill(~mask[..., None], -1e4).softmax(1)          # (B, N, 12)
        pooled = torch.einsum("bnk,bnd->bkd", a, h)
        return (pooled * self.cls_w).sum(-1) + self.cls_b

    def param_groups(self, lr_backbone, lr_head):
        enc = list(self.enc.parameters())
        ids = {id(p) for p in enc}
        return [dict(params=enc, lr=lr_backbone), dict(params=[p for p in self.parameters() if id(p) not in ids], lr=lr_head)]

# %%
sys.path.insert(0, WORK)
import knee_reader as kr
TRIP_VAL, SLOT_OF_TOK, POS_VAL = kr.windows(T_PER_SLOT, kp.DS, kp.OFF)
N_TOK = len(SLOT_OF_TOK)

# Benchmark REAL training steps (AdamW + GradScaler + EMA + augment) for a few setups and train with the fastest.
# v1/v2 lesson: a bare fwd+bwd took 0.59 s, the real step 17-19 s/step (memory pressure on the 15 GB T4s suspected).
def bench(use_dp, batch, ckpt, n=5):
    m = kr.ReaderNet(BACKBONE, SLOT_OF_TOK, kp.N_SLOT, depth=DEPTH, dim=DIM, ckpt=ckpt, enc_kwargs=ENC_KWARGS).cuda()
    e = copy.deepcopy(m).eval()
    net = nn.DataParallel(m) if use_dp else m
    opt = torch.optim.AdamW(m.param_groups(LR_BB, LR), weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda")
    x8 = torch.randint(0, 255, (batch, N_TOK, 3, kp.S, kp.S), dtype=torch.uint8)
    mk = torch.ones(batch, N_TOK, dtype=torch.bool).cuda()
    ps = torch.rand(batch, N_TOK).cuda()
    y = torch.rand(batch, 12).cuda()
    slot_t = torch.as_tensor(SLOT_OF_TOK)
    for d in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(d)
    times = []
    for i in range(n):
        torch.cuda.synchronize(); t = time.time()
        x = kr.normalize(kr.gpu_augment(x8.cuda().float().div_(255), slot_t, kp.N_SLOT))
        loss = F.binary_cross_entropy_with_logits(net(x, mk, ps).float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(m.parameters(), 3.0)
        scaler.step(opt); scaler.update()
        with torch.no_grad():
            for a, b in zip(e.state_dict().values(), m.state_dict().values()):
                if a.dtype.is_floating_point:
                    a.mul_(0.998).add_(b, alpha=0.002)
        torch.cuda.synchronize(); times.append(time.time() - t)
    mem = [round(torch.cuda.max_memory_allocated(d) / 1e9, 1) for d in range(torch.cuda.device_count())]
    del m, e, net, opt, x8
    gc.collect(); torch.cuda.empty_cache()
    return float(np.median(times[1:])) / batch, mem

ngpu = torch.cuda.device_count()                      # NVML count: does not start CUDA in this process
DRY_RUN = ngpu == 0          # CPU session: check the data path (labels, folds, cache, crops, loaders) and stop before training
PARALLEL = PARALLEL and len(FOLDS) > 1 and ngpu >= len(FOLDS)
N_WORKERS = 2 if PARALLEL else 4                      # data-loader workers per fold (the machine has 4 CPU cores)
ENC_SD = None                                         # ImageNet weights of the backbone, loaded once for the forked folds
if PARALLEL:
    # The folds run in forked processes, and a forked process cannot use CUDA once its parent has started it → no
    # benchmark here. The setup is the one the benchmark chose in every M4 run: single GPU, batch 4, no gradient
    # checkpointing (0.17-0.21 s per study at 240 px; DataParallel was 10× slower).
    USE_DP, BATCH, CKPT, SEC_PER_STUDY = False, 4, False, 0.2
    ENC_SD = timm.create_model(BACKBONE, pretrained=True, num_classes=0, **ENC_KWARGS).state_dict()   # no network use in the forked folds
    print(f"parallel folds {FOLDS}: one per GPU ({ngpu} GPUs) | batch {BATCH} | {N_WORKERS} workers per fold", flush=True)
elif DRY_RUN:
    USE_DP, BATCH, CKPT, SEC_PER_STUDY = False, 4, False, 0.2
    print("DRY RUN on CPU: no benchmark, no training", flush=True)
else:
    SETUPS = [(ngpu > 1, 8, False), (ngpu > 1, 8, True), (False, 4, False), (False, 4, True)]
    results = {}
    for s in SETUPS:
        try:
            results[s] = bench(*s)
        except torch.cuda.OutOfMemoryError:
            results[s] = (float("inf"), "OOM")
            gc.collect(); torch.cuda.empty_cache()
        print(f"bench dp={s[0]} batch={s[1]} ckpt={s[2]}: {results[s][0]:.3f} s per study | peak GB {results[s][1]}", flush=True)
    USE_DP, BATCH, CKPT = min(results, key=lambda s: results[s][0])
    SEC_PER_STUDY = results[(USE_DP, BATCH, CKPT)][0]
    print(f"chosen: dp={USE_DP} batch={BATCH} ckpt={CKPT} → ~{SEC_PER_STUDY * 3525 / 60:.1f} min per epoch", flush=True)
    if SEC_PER_STUDY * 3525 > 30 * 60:
        raise RuntimeError("every setup is too slow (> 30 min/epoch) — see the bench lines above")

# %% [markdown]
# ## Labels: our two labelers (auto rule as M1/M2) averaged with the public consensus

# %%
def load_source(fname, name):
    for path in find_inputs(lambda p: os.path.basename(p) == fname):
        df = pd.read_csv(path)
        if len(df) < 4000:
            print(f"[labels] {name}: skip {path} ({len(df)} rows)")
            continue
        s = df.set_index("StudyInstanceUID")[[f"s_{L}" for L in LABELS]]
        s.columns = LABELS
        print(f"[labels] {name}: {path} | rows {len(s)} | missing {int(s.isna().sum().sum())}")
        return s
    print(f"[labels] {name}: not attached")
    return None

UIDS = train.StudyInstanceUID.tolist()
SOURCES = {n: s for n, s in [("gemma", load_source("report_labels_v3.csv", "gemma")),
                             ("qwen", load_source("report_labels.csv", "qwen"))] if s is not None}
if not SOURCES:
    raise RuntimeError("No full labeling run attached (report_labels_v3.csv / report_labels.csv with ~4,407 rows).")
auc_on_gt = lambda s: {L: roc_auc_score(GT[L], s.reindex(GT.index)[L].fillna(s[L].mean())) for L in LABELS}
src_auc = pd.DataFrame({n: auc_on_gt(s) for n, s in SOURCES.items()}).loc[LABELS]
ours, choice = pd.DataFrame(index=UIDS, columns=LABELS, dtype=float), {}
names = list(SOURCES)
for L in LABELS:
    cols = pd.concat([SOURCES[n][L].rename(n) for n in names], axis=1).reindex(UIDS)
    if COMBINE == "auto" and len(names) == 2:
        d = src_auc.loc[L, names[0]] - src_auc.loc[L, names[1]]
        pick = [names[0]] if d >= MARGIN else [names[1]] if d <= -MARGIN else names
    else:
        pick = names
    ours[L] = cols[pick].mean(axis=1).fillna(cols.mean(axis=1))
    choice[L] = "+".join(pick)

cons_path = find_inputs(lambda p: os.path.basename(p) == "consensus_labels.csv")
if cons_path:
    cons = pd.read_csv(cons_path[0]).set_index("StudyInstanceUID").reindex(UIDS)[LABELS].astype(float)
    print(f"[labels] consensus: {cons_path[0]} | missing {int(cons.isna().sum().sum())}")
else:
    print("[labels] consensus_labels.csv NOT attached")
    cons = None
claude = None
if W_CLAUDE > 0:
    try:
        cl = pd.read_csv(CLAUDE_URL, dtype={"STUDY_UID": str}).set_index("STUDY_UID")
        claude = cl.reindex(UIDS)[LABELS].astype(float)
        claude.index.name = "StudyInstanceUID"
        print(f"[labels] claude: {len(cl)} rows | missing {int(claude.isna().sum().sum())}")
        if claude.isna().mean().mean() > 0.05:
            raise RuntimeError("too many studies without Claude labels")
    except Exception as e:
        raise RuntimeError(f"Claude labels wanted (W_CLAUDE > 0) but not loaded — is Internet ON? {e!r}")
# weighted mean of the sources that have a value (weights renormalised per study and finding)
parts = [(w, s) for w, s in [(1 - W_CONSENSUS - W_CLAUDE, ours), (W_CONSENSUS, cons), (W_CLAUDE, claude)]
         if s is not None and w > 0]
target = sum(w * s.fillna(0) for w, s in parts) / sum(w * s.notna() for w, s in parts)
target = target.fillna(target.mean()).clip(0.0, 1.0)
label_auc = {"ours": np.mean(list(auc_on_gt(ours).values())), "target": np.mean(list(auc_on_gt(target).values()))}
if ROUND2:
    v2 = find_inputs(lambda p: os.path.basename(p) == TARGET_FILE)
    if not v2:
        raise RuntimeError(f"round-2 family but {TARGET_FILE} (output of notebook 09 Label Lab) is not attached")
    t2 = pd.read_csv(v2[0], dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").reindex(UIDS)[LABELS].astype(float)
    if t2.isna().any().any():
        raise RuntimeError(f"{TARGET_FILE} does not cover every study: {int(t2.isna().any(axis=1).sum())} missing")
    print(f"[labels] round-2 target: {v2[0]} | mean |diff| to the round-1 target {float((t2 - target).abs().mean().mean()):.3f}")
    target = t2.clip(0.0, 1.0)
    label_auc["target_v2"] = np.mean(list(auc_on_gt(target).values()))
if cons is not None:
    label_auc["consensus"] = np.mean(list(auc_on_gt(cons).values()))
if claude is not None:
    label_auc["claude"] = np.mean(list(auc_on_gt(claude).values()))
print("labels used:", choice, "| macro AUC on 58 GT:", {k: round(v, 4) for k, v in label_auc.items()})

Y = target.values.astype(np.float32)
Y_hard = (Y >= 0.5).astype(int)
gt_idx = np.array([UIDS.index(u) for u in GT.index])
Y[gt_idx] = GT.values
Y_hard[gt_idx] = GT.values
W = np.ones(len(UIDS), np.float32)
W[gt_idx] = GT_WEIGHT

# %% [markdown]
# ## Folds grouped by scanner / site (identical rule to 04c → same folds as M2)

# %%
series_meta = pd.read_csv(find_inputs(lambda p: p.endswith("/series_meta.csv") and META_PREP in p)[0])
sag = (series_meta[(series_meta.slot == "sag_fs") & (series_meta.error.fillna("") == "")]
       .drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID"))
site = (sag.maker.fillna("?").str.upper().str.split().str[0].fillna("?") + "|" + sag.field.round(1).astype(str) + "|"
        + sag.rows.astype(str) + "x" + sag.cols.astype(str) + "|" + sag.px_row.round(2).astype(str))
site = site.reindex(UIDS).fillna("unknown")
FOLD = np.full(len(UIDS), -1)
for k, (_, va) in enumerate(GroupKFold(N_FOLDS).split(UIDS, groups=site.values)):
    FOLD[va] = k
pd.DataFrame({"StudyInstanceUID": UIDS, "fold": FOLD, "site": site.values}).to_csv(f"{WORK}/folds.csv", index=False)
print(f"{site.nunique()} site keys | fold sizes {np.bincount(FOLD).tolist()} | GT per fold {np.bincount(FOLD[gt_idx], minlength=N_FOLDS).tolist()}")

# %% [markdown]
# ## Volume cache → RAM

# %%
t0 = time.time()
shards = find_inputs(lambda p: "/cache/shard_" in p and p.endswith(".npz") and PREP in p)
need = len(UIDS) * kp.N_SLICES * kp.S * kp.S
avail = psutil.virtual_memory().available
print(f"{len(shards)} shards | need {need / 1e9:.1f} GB | RAM available {avail / 1e9:.1f} GB")
uid2i = {u: i for i, u in enumerate(UIDS)}
if avail >= need + 4e9:
    VOL = np.zeros((len(UIDS), kp.N_SLICES, kp.S, kp.S), np.uint8)
else:
    # a cache that leaves too little of the ~31 GB host RAM for training → keep it in a file on the
    # scratch disk; the page cache keeps most of it in memory and gives it back when the process needs RAM
    import shutil
    tmp_dir = next(d for d in ("/kaggle/tmp", "/tmp") if os.path.isdir(d) or d == "/tmp")
    free = shutil.disk_usage(tmp_dir).free
    print(f"cache → memmap in {tmp_dir} (disk free {free / 1e9:.0f} GB)")
    if free < need + 5e9:
        raise RuntimeError("Not enough RAM and not enough scratch disk for the cache")
    VOL = np.lib.format.open_memmap(f"{tmp_dir}/vol_{PREP}.npy", mode="w+", dtype=np.uint8,
                                    shape=(len(UIDS), kp.N_SLICES, kp.S, kp.S))
PRESENT = np.zeros((len(UIDS), kp.N_SLOT), bool)
for n, sh in enumerate(shards):
    with np.load(sh) as z:
        for key in z.files:
            uid, what = key.split("/")
            if uid in uid2i:
                if what == "vol":
                    VOL[uid2i[uid]] = z[key]
                else:
                    PRESENT[uid2i[uid]] = z[key]
    if n % 5 == 0:
        print(f"  shard {n + 1}/{len(shards)}  {(time.time() - t0) / 60:.1f} min", flush=True)
print(f"cache ready in {(time.time() - t0) / 60:.1f} min | slots present {PRESENT.mean(0).round(3).tolist()} | "
      f"RAM left {psutil.virtual_memory().available / 1e9:.1f} GB")
assert PRESENT.any(1).mean() > 0.99, f"cache incomplete: {PRESENT.any(1).mean():.3f} of the studies loaded (all parts attached?)"
if isinstance(VOL, np.memmap):
    VOL.flush()

def crop_slots(x):
    """(N, 3, S, S) → (N, 3, CROP_TRAIN, CROP_TRAIN): one random, centre-biased crop per series slot (its windows stay aligned)"""
    out = np.empty(x.shape[:2] + (CROP_TRAIN, CROP_TRAIN), np.uint8)
    m = kp.S - CROP_TRAIN
    for k in range(kp.N_SLOT):
        sel = SLOT_OF_TOK == k
        oy, ox = np.clip(np.round(m / 2 + np.random.normal(0, m / 4, 2)), 0, m).astype(int)
        out[sel] = x[sel][:, :, oy:oy + CROP_TRAIN, ox:ox + CROP_TRAIN]
    return out

class KneeDS(torch.utils.data.Dataset):
    def __init__(self, idx, train):
        self.idx, self.train = idx, train
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, k):
        i = self.idx[k]
        if self.train:
            trip, _, pos = kr.windows(T_PER_SLOT, kp.DS, kp.OFF, np.random)
        else:
            trip, pos = TRIP_VAL, POS_VAL
        x = VOL[i][trip]                                          # (N, 3, S, S) uint8
        if self.train and CROP_TRAIN:
            x = crop_slots(x)
        mask = PRESENT[i][SLOT_OF_TOK]
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(mask), torch.from_numpy(pos),
                torch.from_numpy(Y[i]), torch.tensor(W[i]))

if DRY_RUN:                  # fetch a few batches the way training and validation do, then stop
    _dl = lambda idx, t: torch.utils.data.DataLoader(KneeDS(idx, t), batch_size=BATCH, shuffle=t, num_workers=N_WORKERS)
    for _name, _idx, _t in (("train", np.where(FOLD != 0)[0], True), ("valid", np.where(FOLD == 0)[0], False)):
        _t0 = time.time()
        for _b, (_x, _m, _p, _y, _w) in enumerate(_dl(_idx, _t)):
            print(f"DRY RUN {_name} batch {_b}: x {tuple(_x.shape)} {_x.dtype} mean {_x.float().mean():.1f} | mask {_m.float().mean():.2f} | "
                  f"y mean {_y.mean():.3f} | {time.time() - _t0:.1f} s", flush=True)
            if _b == 5:
                break
    print(f"=== DRY RUN OK | {FAMILY} {PREP} S {kp.S} crop {CROP_TRAIN} | total {(time.time() - T0) / 60:.1f} min ===", flush=True)
    raise SystemExit(0)

def to_input(x):
    return x.cuda(non_blocking=True).float().div_(255)

def macro_auc(y, p):
    aucs = [roc_auc_score(y[:, j], p[:, j]) if 0 < y[:, j].sum() < len(y) else np.nan for j in range(y.shape[1])]
    return np.nanmean(aucs), aucs

# %% [markdown]
# ## Train (EMA weights are evaluated and saved)

# %%
@torch.no_grad()
def predict(net, loader):
    net.eval()
    out = []
    for x, m, pos, _, _ in loader:
        out.append(torch.sigmoid(net(kr.normalize(to_input(x)), m.cuda(), pos.cuda())).float().cpu())
    return torch.cat(out).numpy()

@torch.no_grad()
def ema_update(ema, model, decay):
    for e, m in zip(ema.state_dict().values(), model.state_dict().values()):
        if e.dtype.is_floating_point:
            e.mul_(decay).add_(m.detach(), alpha=1 - decay)
        else:
            e.copy_(m)

def run_fold(k):
    full = k < 0                      # FOLDS = [-1]: train on ALL studies (no held-out fold), another seed
    seed_all(SEED + (1000 - k if full else k))
    tr, va = (np.arange(len(UIDS)), gt_idx) if full else (np.where(FOLD != k)[0], np.where(FOLD == k)[0])
    name = f"{TAG}_fold_full{-k}" if full else f"{TAG}_fold{k}"   # ours_blend.py loads every {TAG}_fold*.pt
    # 04c lesson: no pinned memory, no persistent workers, ≤ 2 folds per notebook (host RAM leak otherwise)
    dl = lambda idx, t: torch.utils.data.DataLoader(KneeDS(idx, t), batch_size=BATCH, shuffle=t, drop_last=t,
                                                    num_workers=N_WORKERS, pin_memory=False, persistent_workers=False)
    tl, vl = dl(tr, True), dl(va, False)
    model = kr.ReaderNet(BACKBONE, SLOT_OF_TOK, kp.N_SLOT, pretrained=ENC_SD is None, dim=DIM, depth=DEPTH,
                         drop_path=DROP_PATH, ckpt=CKPT, enc_kwargs=ENC_KWARGS)
    if ENC_SD is not None:
        model.enc.load_state_dict(ENC_SD)
    model = model.cuda()
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    multi = USE_DP
    net = nn.DataParallel(model) if multi else model
    ema_net = nn.DataParallel(ema) if multi else ema
    opt = torch.optim.AdamW(model.param_groups(LR_BB, LR), weight_decay=WD)
    total, warm = EPOCHS * len(tl), len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, warm)) *
                                              0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    scaler = torch.amp.GradScaler("cuda")
    slot_t = torch.as_tensor(SLOT_OF_TOK)
    best, best_pred, log, gstep = -1, None, [], 0
    for ep in range(EPOCHS):
        net.train()
        t0, losses, data_wait, t_last = time.time(), [], 0.0, time.time()
        for step, (x, m, pos, y, w) in enumerate(tl):
            data_wait += time.time() - t_last
            if step % 50 == 0:
                print(f"  fold {k} ep {ep} step {step}/{len(tl)}  loss {np.mean(losses[-50:]) if losses else float('nan'):.4f}  "
                      f"{time.time() - t0:.0f}s (waiting for data {data_wait:.0f}s)", flush=True)
            x = kr.normalize(kr.gpu_augment(to_input(x), slot_t, kp.N_SLOT))
            y, w, m, pos = y.cuda(), w.cuda(), m.cuda(), pos.cuda()
            logits = net(x, m, pos)
            loss = (F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none").mean(1) * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            gstep += 1
            ema_update(ema, model, min(EMA_DECAY, (1 + gstep) / (10 + gstep)))
            losses.append(loss.item())
            t_last = time.time()
            if ep == 0 and step == 30:
                faulthandler.cancel_dump_traceback_later()
                sps = (time.time() - t0) / 31
                print(f"  step time {sps:.2f} s (bench {SEC_PER_STUDY * BATCH:.2f} s) | waiting for data {data_wait:.0f}s of {time.time() - t0:.0f}s", flush=True)
                if sps > max(4.0, 3 * SEC_PER_STUDY * BATCH):
                    raise RuntimeError(f"too slow: {sps:.1f} s/step vs bench {SEC_PER_STUDY * BATCH:.2f} — data wait {data_wait:.0f}s")
        p = predict(ema_net, vl)
        proxy, _ = macro_auc(Y_hard[va], p)
        in_gt = np.isin(va, gt_idx)
        gt_auc = macro_auc(Y_hard[va][in_gt], p[in_gt])[0] if in_gt.sum() >= 5 else np.nan
        log.append(dict(fold=k, epoch=ep, loss=np.mean(losses), proxy_auc=proxy, gt_auc=gt_auc, minutes=(time.time() - t0) / 60))
        print(f"fold {k} ep {ep}  loss {np.mean(losses):.4f}  EMA proxy AUC {proxy:.4f}  GT AUC (n={in_gt.sum()}) {gt_auc:.4f}  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if full or proxy > best:      # full-data model: no honest validation → keep the last epoch (EMA)
            best, best_pred = proxy, p
            torch.save(ema.state_dict(), f"{WORK}/models/{name}.pt")
    del model, ema, net, ema_net, opt, tl, vl
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  RAM available after fold {k}: {psutil.virtual_memory().available / 1e9:.1f} GB", flush=True)
    return va, best_pred, log

os.makedirs(f"{WORK}/models", exist_ok=True)
OOF = np.full((len(UIDS), len(LABELS)), np.nan, np.float32)
LOG = []
oof_path = f"{WORK}/oof_{TAG}_folds{''.join(map(str, FOLDS))}.csv"
if PARALLEL:
    # One forked process per fold, each on its own GPU. The children share the cache (VOL) with this process
    # (copy-on-write, never written) and write their output to log_fold<k>.txt, which is echoed here every 30 s.
    import multiprocessing as mp

    def child(k, gpu):
        sys.stdout = sys.stderr = open(f"{WORK}/log_fold{k}.txt", "w", buffering=1)
        faulthandler.dump_traceback_later(240, file=sys.stderr)   # a fold that hangs before step 30 says where
        torch.set_num_threads(1)      # the parent's OpenMP thread pool does not survive a fork (v2 hung in the model init)
        torch.cuda.set_device(gpu)
        print(f"fold {k} on GPU {gpu}: {torch.cuda.get_device_name(gpu)}", flush=True)
        va, p, log = run_fold(k)
        np.savez(f"{WORK}/part_fold{k}.npz", va=va, p=p)
        pd.DataFrame(log).to_csv(f"{WORK}/part_log_fold{k}.csv", index=False)

    procs = {k: mp.get_context("fork").Process(target=child, args=(k, i)) for i, k in enumerate(FOLDS)}
    for pr in procs.values():
        pr.start()
    seen = {k: 0 for k in FOLDS}

    def pump():
        for k in FOLDS:
            path = f"{WORK}/log_fold{k}.txt"
            if os.path.exists(path):
                lines = open(path, errors="replace").read().split("\n")[:-1]      # complete lines only
                for line in lines[seen[k]:]:
                    print(f"[fold {k}] {line}", flush=True)
                seen[k] = len(lines)

    while any(pr.is_alive() for pr in procs.values()):
        time.sleep(30)
        pump()
        bad = [k for k, pr in procs.items() if pr.exitcode not in (None, 0)]
        if bad:                       # one fold died: stop the other one too, do not burn GPU quota
            for pr in procs.values():
                pr.terminate()
            raise RuntimeError(f"fold(s) {bad} failed — see the [fold k] lines above")
    pump()
    bad = [k for k, pr in procs.items() if pr.exitcode != 0]
    if bad:
        raise RuntimeError(f"fold(s) {bad} failed — see the [fold k] lines above")
    for k in FOLDS:
        with np.load(f"{WORK}/part_fold{k}.npz") as z:
            if k >= 0:                # a full-data model has no out-of-fold predictions
                OOF[z["va"]] = z["p"]
        LOG += pd.read_csv(f"{WORK}/part_log_fold{k}.csv").to_dict("records")
        os.remove(f"{WORK}/part_fold{k}.npz"); os.remove(f"{WORK}/part_log_fold{k}.csv")
    if (~np.isnan(OOF[:, 0])).any():
        pd.DataFrame(OOF, index=UIDS, columns=LABELS).dropna().to_csv(oof_path)
    pd.DataFrame(LOG).to_csv(f"{WORK}/train_log.csv", index=False)
else:
    for k in FOLDS:
        va, p, log = run_fold(k)
        if k >= 0:                    # a full-data model has no out-of-fold predictions
            OOF[va] = p
        LOG += log
        if (~np.isnan(OOF[:, 0])).any():
            pd.DataFrame(OOF, index=UIDS, columns=LABELS).dropna().to_csv(oof_path)
        pd.DataFrame(LOG).to_csv(f"{WORK}/train_log.csv", index=False)

# %% [markdown]
# ## Evaluation on the ground-truth studies in these folds (the full 5-fold number is computed in 05c)

# %%
done = ~np.isnan(OOF[:, 0])
g = gt_idx[done[gt_idx]]
if done.any():
    yg, pg = GT.loc[[UIDS[i] for i in g]].values, OOF[g]
    model_auc, model_aucs = macro_auc(yg, pg)
    proxy_all, proxy_aucs = macro_auc(Y_hard[done], OOF[done])
else:                                 # full-data run: nothing is out-of-fold (the epoch log's AUCs are on training data)
    model_auc, proxy_all, proxy_aucs = np.nan, np.nan, [np.nan] * len(LABELS)
s = [f"=== SUMMARY train {TAG} | {BACKBONE} dim {DIM} depth {DEPTH} | folds {FOLDS} | epochs {EPOCHS} T/slot {T_PER_SLOT} batch {BATCH} ===",
     f"labels: macro AUC on 58 GT " + ", ".join(f"{k} {v:.4f}" for k, v in label_auc.items()) + f" | W_CONSENSUS {W_CONSENSUS:.2f} W_CLAUDE {W_CLAUDE:.2f} | prep {PREP}",
     f"OOF proxy AUC vs soft labels (n={done.sum()}): {proxy_all:.4f}",
     f"OOF GT AUC (n={len(g)}): {model_auc:.4f}",
     "per-label proxy AUC: " + ", ".join(f"{L} {a:.2f}" for L, a in zip(LABELS, proxy_aucs)),
     "epochs: " + " | ".join(f"f{r['fold']}e{r['epoch']} {r['proxy_auc']:.3f}/{r['gt_auc']:.3f}" for r in LOG),
     f"minutes per epoch ~{np.mean([r['minutes'] for r in LOG]):.1f} | total {(time.time() - T0) / 60:.1f} min",
     "=== END ==="]
s = "\n".join(s)
print(s)
open(f"{WORK}/train_summary_{TAG}_folds{''.join(map(str, FOLDS))}.txt", "w").write(s)
json.dump(dict(TAG=TAG, ARCH="reader", BACKBONE=BACKBONE, DIM=DIM, DEPTH=DEPTH, T_PER_SLOT=T_PER_SLOT, N_TOK=N_TOK,
               EPOCHS=EPOCHS, BATCH=BATCH, LR=LR, LR_BB=LR_BB, EMA=EMA_DECAY, W_CONSENSUS=W_CONSENSUS, W_CLAUDE=W_CLAUDE, PREP=PREP,
               COMBINE=COMBINE, MARGIN=MARGIN, GT_WEIGHT=GT_WEIGHT, choice=choice, folds=FOLDS, ROUND2=ROUND2, TARGET_FILE=TARGET_FILE, ENC_KWARGS=ENC_KWARGS, CROP_TRAIN=CROP_TRAIN,
               DS=kp.DS, S=kp.S, CROP_MM=kp.CROP_MM),
          open(f"{WORK}/config_{TAG}.json", "w"), indent=1)
