#!/usr/bin/env python
"""Phase 3b (v2): train 5 bigger Laya models on the combined A+I v2 data (GPU1).

Arch v2: in_dim=36, d_model=256, nhead=16 (cross-attn; causal blocks use fixed
8 heads per laya_model.CausalBlock), num_layers=6, ffn=1024, max_len=512,
dropout=0.1 (~3.5x v1 capacity). Focal BCE gamma=2, alpha=0.75 (positive tilt).
AdamW lr=3e-4 wd=1e-2, cosine LR, bf16 autocast, torch.compile, OOM -> halve batch.
Max 12 epochs; early stop patience=3 (min 6) on val PR-AUC.
Per epoch logs train PR/ROC (fixed 200k-row train subsample) + val PR/ROC ->
logs/train_v2.jsonl (train-vs-val gap = overfit evidence).

Outputs: checkpoints_v2/v2_split_{k}_best.pt, data_v2/oof_split_{k}.npz
  {probs f32, row_id i32, y i8}
Run: CUDA_VISIBLE_DEVICES=1 /root/.conda/envs/cde2_env/bin/python 03b_train_v2.py
"""
import os
import sys
import json

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

PIPE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPE)
from laya_model import LayaFraudDecisionNet, save_ckpt  # noqa: E402

CFG = {"in_dim": 36, "d_model": 256, "nhead": 16, "num_layers": 6,
       "ffn": 1024, "max_len": 512, "dropout": 0.1}
EPOCHS, PATIENCE, MIN_EPOCHS = 12, 3, 6
BS0, LR, WD, GAMMA, ALPHA = 1024, 3e-4, 1e-2, 2.0, 0.75
WORKERS = 8
CKPT_DIR = os.path.join(PIPE, "checkpoints_v2")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(os.path.join(PIPE, "logs"), exist_ok=True)
LOG = os.path.join(PIPE, "logs", "train_v2.jsonl")

meta = json.load(open(os.path.join(PIPE, "data_v2", "meta_v2.json")))
STORED = meta["features_stored"]
dev = pd.read_parquet(os.path.join(PIPE, "data_v2", "dev.parquet"))
Xb = dev[STORED].to_numpy(np.float32)
Xb = np.insert(Xb, 2, dev["amt_raw"].to_numpy(np.float32), axis=1)
Xb = np.nan_to_num(Xb, nan=0.0, posinf=0.0, neginf=0.0)
y = dev["is_fraud"].to_numpy(np.int64)
accs = dev["acct_start"].to_numpy(np.int64)
fold = dev["fold"].to_numpy(np.int64)
row_ids = dev["row_id"].to_numpy(np.int64)
assert Xb.shape[1] == CFG["in_dim"], Xb.shape
del dev
device = "cuda"
L = CFG["max_len"]


def focal_a(logits, yy, gamma=GAMMA, alpha=ALPHA):
    bce = F.binary_cross_entropy_with_logits(logits, yy, reduction="none")
    p = torch.sigmoid(logits)
    w = yy * alpha * (1 - p) ** gamma + (1 - yy) * (1 - alpha) * p ** gamma
    return (w * bce).mean()


def make_collate_len(X, acct_start, yy, L):
    j = np.arange(L)[None, :]

    def collate(ids):
        ids = np.asarray(ids, dtype=np.int64)
        lo = np.maximum(acct_start[ids], ids - L)
        gi = ids[:, None] - L + j
        valid = (gi >= lo[:, None]) & (j < (ids - lo)[:, None])
        gi = np.where(valid, gi, 0)
        hist = X[gi.reshape(-1)].reshape(len(ids), L, X.shape[1])
        tgt = X[ids]
        return (torch.from_numpy(np.ascontiguousarray(hist)),
                torch.from_numpy(np.ascontiguousarray(tgt)),
                torch.from_numpy(valid),
                torch.from_numpy(np.ascontiguousarray(yy[ids].astype(np.float32))))

    return collate


@torch.no_grad()
def metrics(model, idx, Xarr):
    model.eval()
    out = []
    for s in range(0, len(idx), 4096):
        ids = idx[s:s + 4096]
        lo = np.maximum(accs[ids], ids - L)
        j = np.arange(L)[None, :]
        gi = ids[:, None] - L + j
        valid = (gi >= lo[:, None]) & (j < (ids - lo)[:, None])
        gi = np.where(valid, gi, 0)
        hist = torch.from_numpy(np.ascontiguousarray(Xarr[gi.reshape(-1)].reshape(len(ids), L, -1))).to(device)
        tgt = torch.from_numpy(Xarr[ids]).to(device)
        mk = torch.from_numpy(valid).to(device)
        out.append(torch.sigmoid(model(hist, mk, tgt)).cpu().numpy())
    p = np.concatenate(out)
    yy = y[idx]
    has_fraud = bool(yy.max())
    return (average_precision_score(yy, p) if has_fraud else 0.0,
            roc_auc_score(yy, p) if has_fraud else 0.0, p)


for k in range(5):
    tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
    scaler = StandardScaler().fit(Xb[tr])
    Xs = ((Xb - scaler.mean_) / scaler.scale_).astype(np.float32)
    rng = np.random.default_rng(0)
    tr_sub = np.sort(rng.choice(tr, size=min(200_000, len(tr)), replace=False))
    model = LayaFraudDecisionNet(**CFG).to(device)
    try:
        model = torch.compile(model)
        compiled = True
    except Exception as e:  # noqa: BLE001
        compiled = False
        print(f"[compile disabled: {e}]", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    bs, coll = BS0, make_collate_len(Xs, accs, y, L)

    def build_dl():
        return DataLoader(tr, batch_size=bs, shuffle=True, collate_fn=coll,
                          num_workers=WORKERS, pin_memory=True)

    dl = build_dl()
    best_pr, best_ep, bad = -1.0, 0, 0
    print(f"=== split {k}: train {len(tr)} (fraud {int(y[tr].sum())}) / "
          f"val {len(va)} (fraud {int(y[va].sum())}) params {n_params} compiled={compiled}", flush=True)
    for ep in range(1, EPOCHS + 1):
        model.train()
        while True:
            try:
                for xh, xt, mk, yl in dl:
                    xh = xh.to(device, non_blocking=True)
                    xt = xt.to(device, non_blocking=True)
                    mk = mk.to(device, non_blocking=True)
                    yl = yl.to(device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = focal_a(model(xh, mk, xt), yl)
                    loss.backward()
                    opt.step()
                break
            except torch.cuda.OutOfMemoryError:
                bs = max(64, bs // 2)
                torch.cuda.empty_cache()
                dl = build_dl()
                print(f"split {k} ep {ep}: OOM -> batch {bs}", flush=True)
        sched.step()
        tp, trc, _ = metrics(model, tr_sub, Xs)
        vp, vrc, _ = metrics(model, va, Xs)
        rec = {"split": k, "epoch": ep, "train_pr": tp, "train_roc": trc,
               "val_pr": vp, "val_roc": vrc, "batch": bs}
        with open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"split {k} ep {ep}: train PR {tp:.4f} ROC {trc:.4f} | "
              f"val PR {vp:.4f} ROC {vrc:.4f}", flush=True)
        if vp > best_pr + 1e-5:
            best_pr, best_ep, bad = vp, ep, 0
            save_ckpt(os.path.join(CKPT_DIR, f"v2_split_{k}_best.pt"), model,
                      scaler.mean_, scaler.scale_, CFG, k, best_pr, best_ep, n_params)
        else:
            bad += 1
        if bad >= PATIENCE and ep >= MIN_EPOCHS:
            print(f"split {k}: early stop ep {ep} (best {best_ep}, PR {best_pr:.4f})", flush=True)
            break
    ck = torch.load(os.path.join(CKPT_DIR, f"v2_split_{k}_best.pt"), map_location="cpu")
    base = LayaFraudDecisionNet(**CFG)
    base.load_state_dict(ck["model_state"])
    base.to(device).eval()
    _, _, pva = metrics(base, va, Xs)
    np.savez(os.path.join(PIPE, "data_v2", f"oof_split_{k}.npz"),
             probs=pva.astype(np.float32), row_id=row_ids[va].astype(np.int32),
             y=y[va].astype(np.int8))
    del model, base, Xs, dl
    torch.cuda.empty_cache()
    print(f"split {k} done: best val PR {best_pr:.4f} @ ep {best_ep}", flush=True)
print("[v2: all 5 splits done]", flush=True)
