# %% [markdown]
# # 04c — Train M2 (ResNet-34) or M3 (DINOv2 ViT-S + slice transformer) on the 5-slot 130 mm cache, EMA
# **Settings:** Accelerator = GPU T4 x2 · Internet = ON (downloads ImageNet weights)
# **Inputs to attach:** competition data · notebook "RSNA Knee 03c Preprocess M2" (cache + knee_prep.py) ·
# notebook "RSNA Knee 02b Labeler Gemma4" (report_labels_v3.csv) · notebook "RSNA Knee 02 Report Labeler" (report_labels.csv)
#
# Same recipe as M1 (ResNet-34, per-finding attention pooling, LLM soft labels, GT ×3, site-grouped folds) with the
# richer M2 input: per study 24 slice-triplets = 6 from each fat-sat slot (sag/cor/ax) + 3 from each non-fat-sat
# slot (sag PD/T2/T1, cor T1). Each triplet gets its own learned position embedding. Weights are averaged with an
# **EMA** (exponential moving average), which is evaluated and saved.
#
# `FOLDS` = which folds this notebook trains. Two copies of this notebook run at the same time:
# "04c Train M2 A" (folds 0, 1, 2) and "04c Train M2 B" (folds 3, 4). The last cell prints `=== SUMMARY`.

# %%
MODEL = "m2"          # "m2" = ResNet-34 (ImageNet) · "m3" = DINOv2 ViT-S/14 + 2-layer slice transformer
FOLDS = [3, 4]
import os, sys, gc, glob, time, math, json, copy, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

N_FOLDS = 5
EPOCHS = 8
BATCH = 8                          # studies per step (split over the 2 GPUs)
T_PER_SLOT = [6, 6, 6, 3, 3]       # triplets per slot (sag_fs, cor_fs, ax_fs, sag_nfs, cor_t1)
PRESETS = {  # LR = head/aggregator, LR_BB = backbone
    "m2": dict(BACKBONE="resnet34.a1_in1k", IMG=None, N_LAYERS=0, LR=3e-4, LR_BB=3e-4, DROP_PATH=0.0),
    "m3": dict(BACKBONE="vit_small_patch14_dinov2.lvd142m", IMG=224, N_LAYERS=2, LR=3e-4, LR_BB=3e-5, DROP_PATH=0.1)}
globals().update(PRESETS[MODEL])
TAG = f"{MODEL}_{BACKBONE.split('.')[0]}"
WD, WARMUP = 1e-2, 0.5
EMA_DECAY = 0.998
GT_WEIGHT = 3.0
COMBINE, MARGIN = "auto", 0.02
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

PREP_DIR = os.path.dirname(find_inputs(lambda p: p.endswith("/knee_prep.py") and "03c" in p)[0])
sys.path.insert(0, PREP_DIR)
import knee_prep as kp               # layout constants (slots, slices per slot, S) of the cache

DATA = os.path.dirname(find_inputs(lambda p: p.endswith("/train_series.csv"))[0])
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
train = pd.read_csv(f"{DATA}/train.csv")
gt_mask = train[LABELS].notna().all(axis=1)
GT = train[gt_mask].set_index("StudyInstanceUID")[LABELS].astype(int)
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "| folds", FOLDS,
      "| prep", PREP_DIR, "| slots", [s[0] for s in kp.SLOTS], kp.DS, "S", kp.S)

# %% [markdown]
# ## Model code → `knee_model.py` (05c imports this file from this notebook's output)

# %%
%%writefile /kaggle/working/knee_model.py
"""M2 model + triplet sampling, shared by 04c (training) and 05c (submission). Import it, never copy it."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)

def triplets(t_per_slot, ds, off, rng=None):
    """Global slice indices (N, 3) of the centre±1 triplets, and the slot of each triplet (N,).
    rng=None → deterministic evenly spaced centres (validation / test); else jittered by up to half a gap."""
    idx, slot = [], []
    for k, (t, d) in enumerate(zip(t_per_slot, ds)):
        base = np.linspace(1, d - 2, t)
        if rng is not None:
            base = base + rng.uniform(-0.5, 0.5, t) * (d - 3) / max(t - 1, 1)
        c = np.clip(np.round(base).astype(int), 1, d - 2)
        idx.append(off[k] + np.stack([c - 1, c, c + 1], 1))
        slot += [k] * t
    return np.concatenate(idx), np.array(slot)

def normalize(x):
    """x: (B, N, 3, S, S) float in [0, 1]"""
    return (x - MEAN.to(x.device)) / STD.to(x.device)

def gpu_augment(x, slot, n_slot):
    """Intensity per study; one small affine per (study, slot) shared by that slot's triplets; never a left-right flip."""
    B, N = x.shape[:2]
    r = lambda *s: torch.rand(*s, device=x.device)
    sh = (B, 1, 1, 1, 1)
    x = x.clamp(1e-4, 1) ** torch.exp((r(*sh) - 0.5) * 0.6)
    x = x * (1 + (r(*sh) - 0.5) * 0.4) + (r(*sh) - 0.5) * 0.2
    ang = (r(B, n_slot) - 0.5) * math.radians(20)
    scale = 1 + (r(B, n_slot) - 0.5) * 0.2
    tx, ty = (r(B, n_slot) - 0.5) * 0.12, (r(B, n_slot) - 0.5) * 0.12
    theta = torch.stack([torch.stack([torch.cos(ang) / scale, -torch.sin(ang) / scale, tx], -1),
                         torch.stack([torch.sin(ang) / scale, torch.cos(ang) / scale, ty], -1)], -2)  # (B, n_slot, 2, 3)
    theta = theta[:, torch.as_tensor(slot, device=x.device)].flatten(0, 1)                             # (B*N, 2, 3)
    flat = x.flatten(0, 1)
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    return F.grid_sample(flat, grid, align_corners=False, padding_mode="zeros").view_as(x).clamp(0, 1)

class KneeNet2(nn.Module):
    """Shared 2D backbone on every triplet → + position embedding → (optional) transformer over all triplet tokens
    (Medical Slice Transformer style, arXiv:2411.15802) → one attention pooling per finding (attention-MIL).
    img: resize the input to this size first (ViT/DINOv2 needs a multiple of its patch size)."""
    def __init__(self, backbone, n_tok, pretrained=True, n_out=12, n_layers=0, img=None, drop_path=0.0):
        super().__init__()
        kw = dict(img_size=img, drop_path_rate=drop_path) if "vit" in backbone else {}
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kw)
        self.img = img
        C = self.enc.num_features
        self.tok_emb = nn.Parameter(torch.zeros(1, n_tok, C))
        self.agg = (nn.TransformerEncoder(nn.TransformerEncoderLayer(C, 6 if C % 6 == 0 else 8, 2 * C, 0.1,
                                                                     batch_first=True, norm_first=True), n_layers)
                    if n_layers else None)
        self.att = nn.Sequential(nn.Linear(C, 256), nn.Tanh(), nn.Linear(256, n_out))
        self.cls_w = nn.Parameter(torch.randn(n_out, C) * 0.01)
        self.cls_b = nn.Parameter(torch.zeros(n_out))
        self.drop = nn.Dropout(0.2)
    def forward(self, x, mask):
        """x: (B, N, 3, S, S) normalised · mask: (B, N) bool (False = slot missing for this study)"""
        B, N = x.shape[:2]
        mask = mask.clone()
        mask[~mask.any(1), 0] = True                            # a study with no image at all: avoid NaN
        x = x.flatten(0, 1)
        if self.img and x.shape[-1] != self.img:
            x = F.interpolate(x, size=(self.img, self.img), mode="bilinear", antialias=True, align_corners=False)
        with torch.autocast("cuda", dtype=torch.float16):      # explicit: autocast is thread-local (DataParallel)
            f = self.enc(x).float()
        f = f.view(B, N, -1) + self.tok_emb
        if self.agg is not None:
            f = self.agg(f, src_key_padding_mask=~mask)
        a = self.att(f).float().masked_fill(~mask[..., None], -1e4).softmax(1)    # (B, N, 12)
        pooled = torch.einsum("bnk,bnc->bkc", a, f)
        return (self.drop(pooled) * self.cls_w).sum(-1) + self.cls_b

    def param_groups(self, lr_backbone, lr_head):
        enc = list(self.enc.parameters())
        ids = {id(p) for p in enc}
        return [dict(params=enc, lr=lr_backbone), dict(params=[p for p in self.parameters() if id(p) not in ids], lr=lr_head)]

# %%
sys.path.insert(0, WORK)
import knee_model as km

# %% [markdown]
# ## Labels: combine report labelers per finding (same rule as M1)

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

SOURCES = {n: s for n, s in [("gemma", load_source("report_labels_v3.csv", "gemma")),
                             ("qwen", load_source("report_labels.csv", "qwen"))] if s is not None}
if not SOURCES:
    raise RuntimeError("No full labeling run attached (report_labels_v3.csv / report_labels.csv with ~4,407 rows).")
UIDS = train.StudyInstanceUID.tolist()
src_auc = pd.DataFrame({n: {L: roc_auc_score(GT[L], s.reindex(GT.index)[L].fillna(0)) for L in LABELS}
                        for n, s in SOURCES.items()}).loc[LABELS]
target, choice = pd.DataFrame(index=UIDS, columns=LABELS, dtype=float), {}
names = list(SOURCES)
for L in LABELS:
    cols = pd.concat([SOURCES[n][L].rename(n) for n in names], axis=1).reindex(UIDS)
    if COMBINE in SOURCES:
        pick = [COMBINE]
    elif COMBINE == "auto" and len(names) == 2:
        d = src_auc.loc[L, names[0]] - src_auc.loc[L, names[1]]
        pick = [names[0]] if d >= MARGIN else [names[1]] if d <= -MARGIN else names
    else:
        pick = names
    target[L] = cols[pick].mean(axis=1).fillna(cols.mean(axis=1)).fillna(0.0)
    choice[L] = "+".join(pick)
comb_auc = pd.Series({L: roc_auc_score(GT[L], target.loc[GT.index, L]) for L in LABELS})
print("labels used:", choice, "| combined label AUC on 58 GT:", round(comb_auc.mean(), 3))

Y = target.values.astype(np.float32)
Y_hard = (Y >= 0.5).astype(int)
gt_idx = np.array([UIDS.index(u) for u in GT.index])
Y[gt_idx] = GT.values
Y_hard[gt_idx] = GT.values
W = np.ones(len(UIDS), np.float32)
W[gt_idx] = GT_WEIGHT

# %% [markdown]
# ## Folds grouped by scanner / site (key from the sagittal fat-sat series; same rule as M1)

# %%
series_meta = pd.read_csv(find_inputs(lambda p: p.endswith("/series_meta.csv") and "03c" in p)[0])
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
import psutil
t0 = time.time()
shards = find_inputs(lambda p: "/cache/shard_" in p and p.endswith(".npz") and "03c" in p)
need = len(UIDS) * kp.N_SLICES * kp.S * kp.S
avail = psutil.virtual_memory().available
print(f"{len(shards)} shards | need {need / 1e9:.1f} GB | RAM available {avail / 1e9:.1f} GB")
if avail < need + 3e9:
    raise RuntimeError("Not enough RAM for the M2 cache — lower S or slices in knee_prep.py")
uid2i = {u: i for i, u in enumerate(UIDS)}
VOL = np.zeros((len(UIDS), kp.N_SLICES, kp.S, kp.S), np.uint8)
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

TRIP_VAL, SLOT_OF_TOK = km.triplets(T_PER_SLOT, kp.DS, kp.OFF)
N_TOK = len(SLOT_OF_TOK)

class KneeDS(torch.utils.data.Dataset):
    def __init__(self, idx, train):
        self.idx, self.train = idx, train
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, k):
        i = self.idx[k]
        trip = km.triplets(T_PER_SLOT, kp.DS, kp.OFF, np.random)[0] if self.train else TRIP_VAL
        x = VOL[i][trip]                                          # (N, 3, S, S) uint8
        mask = PRESENT[i][SLOT_OF_TOK]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(mask), torch.from_numpy(Y[i]), torch.tensor(W[i])

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
    for x, m, _, _ in loader:
        out.append(torch.sigmoid(net(km.normalize(to_input(x)), m.cuda())).float().cpu())
    return torch.cat(out).numpy()

@torch.no_grad()
def ema_update(ema, model, decay):
    for e, m in zip(ema.state_dict().values(), model.state_dict().values()):
        if e.dtype.is_floating_point:
            e.mul_(decay).add_(m.detach(), alpha=1 - decay)
        else:
            e.copy_(m)

def run_fold(k):
    seed_all(SEED + k)
    tr, va = np.where(FOLD != k)[0], np.where(FOLD == k)[0]
    # v1 lesson: persistent workers from earlier folds were never freed → "Kernel died" (RAM) in fold 2
    dl = lambda idx, t: torch.utils.data.DataLoader(KneeDS(idx, t), batch_size=BATCH, shuffle=t, drop_last=t,
                                                    num_workers=4, pin_memory=True, persistent_workers=False)
    tl, vl = dl(tr, True), dl(va, False)
    model = km.KneeNet2(BACKBONE, N_TOK, n_layers=N_LAYERS, img=IMG, drop_path=DROP_PATH).cuda()
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    multi = torch.cuda.device_count() > 1
    net = nn.DataParallel(model) if multi else model
    ema_net = nn.DataParallel(ema) if multi else ema
    opt = torch.optim.AdamW(model.param_groups(LR_BB, LR), weight_decay=WD)
    total, warm = EPOCHS * len(tl), int(WARMUP * len(tl))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, warm)) *
                                              0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    scaler = torch.amp.GradScaler("cuda")
    slot_t = torch.as_tensor(SLOT_OF_TOK)
    best, best_pred, log, gstep = -1, None, [], 0
    for ep in range(EPOCHS):
        net.train()
        t0, losses, data_wait, t_last = time.time(), [], 0.0, time.time()
        for step, (x, m, y, w) in enumerate(tl):
            data_wait += time.time() - t_last
            if step % 50 == 0:
                print(f"  fold {k} ep {ep} step {step}/{len(tl)}  loss {np.mean(losses[-50:]) if losses else float('nan'):.4f}  "
                      f"{time.time() - t0:.0f}s (waiting for data {data_wait:.0f}s)", flush=True)
            x = km.normalize(km.gpu_augment(to_input(x), slot_t, kp.N_SLOT))
            y, w, m = y.cuda(), w.cuda(), m.cuda()
            logits = net(x, m)
            loss = (F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none").mean(1) * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            gstep += 1
            ema_update(ema, model, min(EMA_DECAY, (1 + gstep) / (10 + gstep)))
            losses.append(loss.item())
            t_last = time.time()
        p = predict(ema_net, vl)
        proxy, _ = macro_auc(Y_hard[va], p)
        in_gt = np.isin(va, gt_idx)
        gt_auc = macro_auc(Y_hard[va][in_gt], p[in_gt])[0] if in_gt.sum() >= 5 else np.nan
        log.append(dict(fold=k, epoch=ep, loss=np.mean(losses), proxy_auc=proxy, gt_auc=gt_auc, minutes=(time.time() - t0) / 60))
        print(f"fold {k} ep {ep}  loss {np.mean(losses):.4f}  EMA proxy AUC {proxy:.4f}  GT AUC (n={in_gt.sum()}) {gt_auc:.4f}  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if proxy > best:
            best, best_pred = proxy, p
            torch.save(ema.state_dict(), f"{WORK}/models/{TAG}_fold{k}.pt")
    del model, ema, net, ema_net, opt, tl, vl
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  RAM available after fold {k}: {psutil.virtual_memory().available / 1e9:.1f} GB", flush=True)
    return va, best_pred, log

os.makedirs(f"{WORK}/models", exist_ok=True)
OOF = np.full((len(UIDS), len(LABELS)), np.nan, np.float32)
LOG = []
for k in FOLDS:
    va, p, log = run_fold(k)
    OOF[va] = p
    LOG += log
    pd.DataFrame(OOF, index=UIDS, columns=LABELS).dropna().to_csv(f"{WORK}/oof_{TAG}_folds{''.join(map(str, FOLDS))}.csv")
    pd.DataFrame(LOG).to_csv(f"{WORK}/train_log.csv", index=False)

# %% [markdown]
# ## Evaluation on the ground-truth studies in these folds (the full 5-fold number is computed in 05c)

# %%
done = ~np.isnan(OOF[:, 0])
g = gt_idx[done[gt_idx]]
yg, pg = GT.loc[[UIDS[i] for i in g]].values, OOF[g]
model_auc, model_aucs = macro_auc(yg, pg)
proxy_all, proxy_aucs = macro_auc(Y_hard[done], OOF[done])
s = [f"=== SUMMARY train {MODEL} | {BACKBONE} | folds {FOLDS} | epochs {EPOCHS} T/slot {T_PER_SLOT} batch {BATCH} EMA {EMA_DECAY} ===",
     f"slots {[s_[0] for s_ in kp.SLOTS]} slices {kp.DS} S {kp.S} crop {kp.CROP_MM:.0f} mm | present {PRESENT.mean(0).round(3).tolist()}",
     f"OOF proxy AUC vs LLM labels (n={done.sum()}): {proxy_all:.4f}",
     f"OOF GT AUC (n={len(g)}): {model_auc:.4f}",
     "per-label proxy AUC: " + ", ".join(f"{L} {a:.2f}" for L, a in zip(LABELS, proxy_aucs)),
     "epochs: " + " | ".join(f"f{r['fold']}e{r['epoch']} {r['proxy_auc']:.3f}/{r['gt_auc']:.3f}" for r in LOG),
     f"minutes per epoch ~{np.mean([r['minutes'] for r in LOG]):.1f} | total {(time.time() - T0) / 60:.1f} min",
     "=== END ==="]
s = "\n".join(s)
print(s)
open(f"{WORK}/train_summary_{TAG}_folds{''.join(map(str, FOLDS))}.txt", "w").write(s)
json.dump(dict(TAG=TAG, MODEL=MODEL, N_LAYERS=N_LAYERS, IMG=IMG, DROP_PATH=DROP_PATH, LR_BB=LR_BB, BACKBONE=BACKBONE, T_PER_SLOT=T_PER_SLOT, N_TOK=N_TOK, EPOCHS=EPOCHS, BATCH=BATCH, LR=LR, EMA=EMA_DECAY,
               COMBINE=COMBINE, MARGIN=MARGIN, GT_WEIGHT=GT_WEIGHT, choice=choice, folds=FOLDS,
               DS=kp.DS, S=kp.S, CROP_MM=kp.CROP_MM),
          open(f"{WORK}/config_{TAG}.json", "w"), indent=1)
