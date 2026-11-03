# %% [markdown]
# # 04 — Train: 2.5D multi-plane model on LLM report labels
# **Settings:** Accelerator = GPU T4 x2 · Internet = ON (downloads ImageNet weights)
# **Inputs to attach:** competition data · notebook "RSNA Knee 03 Preprocess" (volume cache) ·
# notebook "RSNA Knee 02b Labeler Gemma4" (report_labels_v3.csv) · notebook "RSNA Knee 02 Report Labeler" (report_labels.csv, Qwen)
#
# - Input per study: 3 planes (sag/cor/ax) × `T` slice-triplets (3 adjacent slices = 3 channels), 224 px, lateral-first.
# - Shared ImageNet backbone → per-slice features + plane & slice-position embeddings →
#   **one attention pooling per finding** over all 3·T slice images → 12 logits.
# - Targets: LLM soft labels, combined per finding (`COMBINE="auto"`: a labeler is used alone only when it beats the
#   other by ≥ `MARGIN` AUC on the 58 ground-truth studies, otherwise the two are averaged); the 58 ground-truth studies
#   use their true labels with a higher loss weight.
# - Folds: GroupKFold by scanner/site key (vendor + field strength + matrix + pixel spacing).
# - Main metric: out-of-fold AUC on the 58 ground-truth studies. Proxy metric: AUC vs LLM labels on the validation fold.
#
# `MODE = "fast"` → fold 0 only (pipeline check, ~30 min) · `MODE = "cv"` → all folds.
# The last cell prints a block starting with `=== SUMMARY` — copy that to Claude.

# %%
import os, glob, time, math, json, shutil, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

MODE = "fast"                      # "fast" (fold 0) or "cv" (all folds)
N_FOLDS = 5
EPOCHS = 8
BATCH = 8                          # studies per step (split over the 2 GPUs)
T = 8                              # slice triplets per plane
BACKBONE = "resnet34.a1_in1k"      # dense convs use T4 tensor cores well (EfficientNet's depthwise convs did not)
BACKBONE_CANDIDATES = ["resnet34.a1_in1k", "resnet18.a1_in1k", "efficientnet_b0.ra_in1k"]
MAX_EPOCH_MIN = 10.0
LR, WD, WARMUP = 3e-4, 1e-2, 0.5   # warmup in epochs
GT_WEIGHT = 3.0
COMBINE, MARGIN = "auto", 0.02     # "auto" | "mean" | "gemma" | "qwen"
N_BOOT = 2000
SEED = 42
WORK = "/kaggle/working"
T0 = time.time()

torch.backends.cudnn.benchmark = True   # autotune conv kernels (channels_last stays off: it crashed on T4 in v2)

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(SEED)

def find_inputs(match):
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        hits += [os.path.join(root, f) for f in files if match(os.path.join(root, f))]
    return sorted(hits)

DATA = os.path.dirname(find_inputs(lambda p: p.endswith("/train_series.csv"))[0])
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
train = pd.read_csv(f"{DATA}/train.csv")
gt_mask = train[LABELS].notna().all(axis=1)
GT = train[gt_mask].set_index("StudyInstanceUID")[LABELS].astype(int)
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "| MODE:", MODE)

# %% [markdown]
# ## Labels: combine report labelers per finding

# %%
def load_source(fname, name):
    for path in find_inputs(lambda p: os.path.basename(p) == fname):
        df = pd.read_csv(path)
        if len(df) < 4000:              # a validation-only run (58 rows) is not a usable source
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
    raise RuntimeError(
        "No full labeling run found (need a report_labels*.csv with all ~4,407 studies). Either the labeler notebooks "
        "are still running, or this session mounted their older validation outputs. Wait until '02b Labeler Gemma4' "
        "and/or '02 Report Labeler' show 'Successful', then run this notebook with Save Version (a committed run "
        "mounts their latest outputs; an already-open interactive session keeps the old ones).")
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
label_table = src_auc.assign(used=pd.Series(choice), combined=comb_auc).round(3)
print(label_table.to_string())
print("macro:", label_table[[c for c in label_table.columns if c != "used"]].mean().round(3).to_dict())

Y = target.values.astype(np.float32)
Y_hard = (Y >= 0.5).astype(int)                        # LLM hard labels, for the proxy metric
gt_idx = np.array([UIDS.index(u) for u in GT.index])
Y[gt_idx] = GT.values                                   # ground truth overrides LLM labels
Y_hard[gt_idx] = GT.values
W = np.ones(len(UIDS), np.float32)
W[gt_idx] = GT_WEIGHT
target.to_csv(f"{WORK}/train_targets.csv")

# %% [markdown]
# ## Folds grouped by scanner / site

# %%
series_meta = pd.read_csv(find_inputs(lambda p: p.endswith("/series_meta.csv"))[0])
sag = (series_meta[(series_meta.plane == "sag") & (series_meta.error.fillna("") == "")]
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
# ## Volume cache → one uint8 memmap on local disk

# %%
t0 = time.time()
shards = find_inputs(lambda p: "/cache/shard_" in p and p.endswith(".npz"))
with np.load(shards[0]) as z:
    D, S = z[z.files[0]].shape[0], z[z.files[0]].shape[1]
tmp = max([p for p in ("/kaggle/temp", "/tmp") if os.path.isdir(p) or not os.makedirs(p, exist_ok=True)],
          key=lambda p: shutil.disk_usage(p).free)
need = len(UIDS) * 3 * D * S * S
print(f"{len(shards)} shards | D={D} S={S} | need {need / 1e9:.1f} GB | {tmp} free {shutil.disk_usage(tmp).free / 1e9:.1f} GB")
PLANE = {"sag": 0, "cor": 1, "ax": 2}
uid2i = {u: i for i, u in enumerate(UIDS)}
import psutil
avail = psutil.virtual_memory().available
IN_RAM = avail > need + 5e9          # v1 lesson: random reads from the disk cache were far too slow
print(f"RAM available {avail / 1e9:.1f} GB → volumes {'in RAM' if IN_RAM else 'memory-mapped on disk'}")
if IN_RAM:
    VOL = np.empty((len(UIDS), 3, D, S, S), np.uint8)
else:
    VOL_PATH = f"{tmp}/vols.npy"
    VOL = np.lib.format.open_memmap(VOL_PATH, mode="w+", dtype=np.uint8, shape=(len(UIDS), 3, D, S, S))
PRESENT = np.zeros((len(UIDS), 3), bool)
for sh in shards:
    with np.load(sh) as z:
        for key in z.files:
            uid, plane = key.split("/")
            if uid in uid2i:
                VOL[uid2i[uid], PLANE[plane]] = z[key]
                PRESENT[uid2i[uid], PLANE[plane]] = True
if not IN_RAM:
    VOL.flush()
    del VOL
    VOL = np.load(VOL_PATH, mmap_mode="r")
print(f"cache ready in {(time.time() - t0) / 60:.1f} min | planes present: {PRESENT.mean(0).round(3).tolist()} | "
      f"RAM left {psutil.virtual_memory().available / 1e9:.1f} GB")

# %% [markdown]
# ## Dataset, GPU augmentation, model

# %%
class KneeDS(torch.utils.data.Dataset):
    def __init__(self, idx, train):
        self.idx, self.train = idx, train
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, k):
        i = self.idx[k]
        base = np.linspace(1, D - 2, T)
        if self.train:
            base = base + np.random.uniform(-1.5, 1.5, T)
        c = np.clip(np.round(base).astype(int), 1, D - 2)
        x = VOL[i][:, np.stack([c - 1, c, c + 1], 1)]            # (3 planes, T, 3, S, S) uint8
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(PRESENT[i]),
                torch.from_numpy(Y[i]), torch.tensor(W[i]))

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3, 1, 1)

def gpu_augment(x):
    """x: (B, 3, T, 3, S, S) float in [0, 1]. Intensity per study, one affine per study-plane (never a left-right flip)."""
    B, P = x.shape[:2]
    r = lambda *s: torch.rand(*s, device=x.device)
    shape = (B, 1, 1, 1, 1, 1)
    x = x.clamp(1e-4, 1) ** torch.exp((r(*shape) - 0.5) * 0.6)
    x = x * (1 + (r(*shape) - 0.5) * 0.4) + (r(*shape) - 0.5) * 0.2
    ang = (r(B * P) - 0.5) * math.radians(20)
    scale = 1 + (r(B * P) - 0.5) * 0.2
    tx, ty = (r(B * P) - 0.5) * 0.12, (r(B * P) - 0.5) * 0.12
    theta = torch.stack([torch.stack([torch.cos(ang) / scale, -torch.sin(ang) / scale, tx], 1),
                         torch.stack([torch.sin(ang) / scale, torch.cos(ang) / scale, ty], 1)], 1)
    flat = x.flatten(0, 1).flatten(1, 2)                          # (B*P, T*3, S, S)
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    flat = F.grid_sample(flat, grid, align_corners=False, padding_mode="zeros")
    return flat.view_as(x).clamp(0, 1)

class KneeNet(nn.Module):
    def __init__(self, backbone, pretrained=True, n_out=12):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        C = self.enc.num_features
        self.plane_emb = nn.Parameter(torch.zeros(1, 3, 1, C))
        self.slice_emb = nn.Parameter(torch.zeros(1, 1, T, C))
        self.att = nn.Sequential(nn.Linear(C, 256), nn.Tanh(), nn.Linear(256, n_out))
        self.cls_w = nn.Parameter(torch.randn(n_out, C) * 0.01)
        self.cls_b = nn.Parameter(torch.zeros(n_out))
        self.drop = nn.Dropout(0.2)
    def forward(self, x, present):
        B, P, Tn = x.shape[:3]
        with torch.autocast("cuda", dtype=torch.float16):         # explicit: autocast is thread-local (DataParallel)
            f = self.enc(x.flatten(0, 2)).float()                 # (B*P*T, C)
        f = f.view(B, P, Tn, -1) + self.plane_emb + self.slice_emb[:, :, :Tn]
        f = f.flatten(1, 2)                                       # (B, P*T, C)
        a = self.att(f)                                           # (B, N, 12) one attention map per finding
        mask = present[:, :, None].expand(B, P, Tn).flatten(1)
        a = a.float().masked_fill(~mask[..., None], -1e4).softmax(1)
        pooled = torch.einsum("bnk,bnc->bkc", a, f.float())       # (B, 12, C)
        return (self.drop(pooled) * self.cls_w).sum(-1) + self.cls_b

def to_input(x):
    x = x.cuda(non_blocking=True).float().div_(255)
    return x

def normalize(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)

def macro_auc(y, p):
    aucs = [roc_auc_score(y[:, j], p[:, j]) if 0 < y[:, j].sum() < len(y) else np.nan for j in range(y.shape[1])]
    return np.nanmean(aucs), aucs

# %% [markdown]
# ## Speed benchmark (data loading vs model step) — tells us which side is the bottleneck

# %%
def bench_data(n_data=30):
    dl = torch.utils.data.DataLoader(KneeDS(np.random.permutation(len(UIDS))[:BATCH * n_data], True),
                                     batch_size=BATCH, num_workers=4)
    t0 = time.time()
    for _ in dl:
        pass
    return (time.time() - t0) / n_data

def benchmark(backbone, data_s, n_model=10):
    model = KneeNet(backbone).cuda()
    net = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randint(0, 255, (BATCH, 3, T, 3, S, S), dtype=torch.uint8)
    m = torch.ones(BATCH, 3, dtype=torch.bool).cuda()
    y = torch.rand(BATCH, len(LABELS)).cuda()
    for i in range(n_model + 3):
        if i == 3:
            torch.cuda.synchronize()
            t0 = time.time()
        xx = normalize(gpu_augment(to_input(x)))
        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.binary_cross_entropy_with_logits(net(xx, m).float(), y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    model_s = (time.time() - t0) / n_model
    steps = int(len(UIDS) * (N_FOLDS - 1) / N_FOLDS / BATCH)
    est = steps * max(data_s, model_s) / 60
    print(f"BENCH {backbone:<26} data {data_s:.2f} s/batch | model {model_s:.2f} s/step | bottleneck: "
          f"{'DATA' if data_s > model_s else 'MODEL'} | est. {est:.1f} min/epoch ({steps} steps)", flush=True)
    del model, net, opt
    torch.cuda.empty_cache()
    return data_s, model_s, est

# v3 lesson: EfficientNet-B0 ran at ~49 img/s on 2×T4 (29 min/epoch); compare backbones and use the configured
# one unless it is too slow, in which case fall back to the fastest candidate.
DATA_S = bench_data()
BENCHES = {b: benchmark(b, DATA_S) for b in BACKBONE_CANDIDATES}
if BENCHES[BACKBONE][2] > MAX_EPOCH_MIN:
    fastest = min(BENCHES, key=lambda b: BENCHES[b][2])
    print(f"{BACKBONE} is too slow ({BENCHES[BACKBONE][2]:.1f} min/epoch) → using {fastest}")
    BACKBONE = fastest
BENCH = BENCHES[BACKBONE]

# %% [markdown]
# ## Train

# %%
@torch.no_grad()
def predict(net, loader):
    net.eval()
    out = []
    for x, m, _, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(net(normalize(to_input(x)), m.cuda())).float().cpu())
    return torch.cat(out).numpy()

def run_fold(k):
    seed_all(SEED + k)
    tr, va = np.where(FOLD != k)[0], np.where(FOLD == k)[0]
    dl = lambda idx, t: torch.utils.data.DataLoader(KneeDS(idx, t), batch_size=BATCH, shuffle=t, drop_last=t,
                                                    num_workers=4, pin_memory=True, persistent_workers=True)
    tl, vl = dl(tr, True), dl(va, False)
    model = KneeNet(BACKBONE).cuda()
    net = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total, warm = EPOCHS * len(tl), int(WARMUP * len(tl))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, warm)) *
                                              0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    scaler = torch.amp.GradScaler("cuda")
    best, best_pred, log = -1, None, []
    for ep in range(EPOCHS):
        net.train()
        t0, losses, data_wait, t_last = time.time(), [], 0.0, time.time()
        for step, (x, m, y, w) in enumerate(tl):
            data_wait += time.time() - t_last
            if step % 50 == 0:
                print(f"  fold {k} ep {ep} step {step}/{len(tl)}  loss {np.mean(losses[-50:]) if losses else float('nan'):.4f}  "
                      f"{time.time() - t0:.0f}s (waiting for data {data_wait:.0f}s)", flush=True)
            x = normalize(gpu_augment(to_input(x)))
            y, w, m = y.cuda(), w.cuda(), m.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = net(x, m)
            loss = (F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none").mean(1) * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            t_last = time.time()
        p = predict(net, vl)
        proxy, _ = macro_auc(Y_hard[va], p)
        in_gt = np.isin(va, gt_idx)
        gt_auc = macro_auc(Y_hard[va][in_gt], p[in_gt])[0] if in_gt.sum() >= 5 else np.nan
        log.append(dict(fold=k, epoch=ep, loss=np.mean(losses), proxy_auc=proxy, gt_auc=gt_auc, minutes=(time.time() - t0) / 60))
        print(f"fold {k} ep {ep}  loss {np.mean(losses):.4f}  proxy AUC {proxy:.4f}  GT AUC (n={in_gt.sum()}) {gt_auc:.4f}  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if proxy > best:
            best, best_pred = proxy, p
            torch.save(model.state_dict(), f"{WORK}/models/{BACKBONE.split('.')[0]}_fold{k}.pt")
    return va, best_pred, log

os.makedirs(f"{WORK}/models", exist_ok=True)
folds_to_run = [0] if MODE == "fast" else list(range(N_FOLDS))
OOF = np.full((len(UIDS), len(LABELS)), np.nan, np.float32)
LOG = []
for k in folds_to_run:
    va, p, log = run_fold(k)
    OOF[va] = p
    LOG += log
    torch.cuda.empty_cache()
oof_df = pd.DataFrame(OOF, index=UIDS, columns=LABELS)
oof_df.to_csv(f"{WORK}/oof_{BACKBONE.split('.')[0]}.csv")
pd.DataFrame(LOG).to_csv(f"{WORK}/train_log.csv", index=False)

# %% [markdown]
# ## Evaluation on the ground-truth studies (out-of-fold)

# %%
done = ~np.isnan(OOF[:, 0])
g = gt_idx[done[gt_idx]]
yg, pg, lg = GT.loc[[UIDS[i] for i in g]].values, OOF[g], target.loc[[UIDS[i] for i in g]].values
model_auc, model_aucs = macro_auc(yg, pg)
label_auc, _ = macro_auc(yg, lg)
rng = np.random.default_rng(0)
boot = []
for _ in range(N_BOOT):
    b = rng.integers(0, len(g), len(g))
    boot.append(macro_auc(yg[b], pg[b])[0])
lo, hi = np.nanpercentile(boot, [2.5, 97.5])
proxy_all, proxy_aucs = macro_auc(Y_hard[done], OOF[done])

s = [f"=== SUMMARY train | {BACKBONE} | MODE={MODE} | folds {folds_to_run} | epochs {EPOCHS} T={T} batch {BATCH} ===",
     f"labels: " + ", ".join(f"{L}={choice[L]}" for L in LABELS),
     "label AUC on 58 GT: " + " | ".join(f"{c} {label_table[c].mean():.3f}" for c in label_table.columns if c != "used"),
     f"sites {site.nunique()} | fold sizes {np.bincount(FOLD).tolist()} | GT per fold {np.bincount(FOLD[gt_idx], minlength=N_FOLDS).tolist()}",
     f"OOF proxy AUC vs LLM labels (n={done.sum()}): {proxy_all:.4f}",
     f"OOF GT AUC (n={len(g)}): model {model_auc:.4f} [95% CI {lo:.3f}, {hi:.3f}] | same studies, report labels {label_auc:.4f}",
     "per-label GT AUC (model): " + ", ".join(f"{L} {a:.2f}" for L, a in zip(LABELS, model_aucs)),
     "per-label proxy AUC: " + ", ".join(f"{L} {a:.2f}" for L, a in zip(LABELS, proxy_aucs)),
     "epochs: " + " | ".join(f"f{r['fold']}e{r['epoch']} {r['proxy_auc']:.3f}/{r['gt_auc']:.3f}" for r in LOG),
     f"minutes per epoch ~{np.mean([r['minutes'] for r in LOG]):.1f} | total {(time.time() - T0) / 60:.1f} min | "
     f"bench data {BENCH[0]:.2f}s/batch, model {BENCH[1]:.2f}s/step | volumes in RAM: {IN_RAM}",
     "bench min/epoch: " + ", ".join(f"{b.split('.')[0]} {v[2]:.1f}" for b, v in BENCHES.items()),
     "=== END ==="]
s = "\n".join(s)
print(s)
open(f"{WORK}/train_summary_{MODE}.txt", "w").write(s)
json.dump(dict(BACKBONE=BACKBONE, T=T, D=D, S=S, EPOCHS=EPOCHS, BATCH=BATCH, LR=LR, COMBINE=COMBINE, MARGIN=MARGIN,
               GT_WEIGHT=GT_WEIGHT, choice=choice, folds=folds_to_run), open(f"{WORK}/config.json", "w"), indent=1)
