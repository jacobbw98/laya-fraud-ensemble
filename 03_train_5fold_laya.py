#!/usr/bin/env python
"""Phase 3: Train 5 Laya sequential fraud-decision models (one per dev fold).

Per split k (seed 42+k):
  - StandardScaler fit on the TRAIN split only; amount_z derived from train stats.
  - Focal loss (gamma=2.0), AdamW(lr=3e-4, wd=1e-2), cosine LR, bf16 autocast,
    torch.compile, DataLoader(num_workers=8, pin_memory=True).
  - Eval every epoch on the split's validation fold (PR-AUC, ROC-AUC);
    save best checkpoint checkpoints/laya_split_{k}_best.pt and OOF val
    predictions oof_split_{k}.npz. GPU cache cleared between splits.

Single GPU: CUDA_VISIBLE_DEVICES=0, cuda:0 only.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # overridable: CUDA_VISIBLE_DEVICES=1 python 03_...
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # hard pin: single GPU, never multi-GPU

import argparse
import gc
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from laya_model import (  # noqa: E402
    D_MODEL, DROPOUT, FFN, MAX_LEN, N_HEAD, N_LAYER,
    LayaFraudDecisionNet, focal_loss, make_collate, save_ckpt,
)

torch.backends.cuda.matmul.allow_tf32 = True

PIPE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PIPE, "data")
CKPT_DIR = os.path.join(PIPE, "checkpoints")
LOG_DIR = os.path.join(PIPE, "logs")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

EPOCHS = 6
LR = 3e-4
WD = 1e-2
CLIP = 1.0
DEVICE = "cuda:0"


class IdDataset(Dataset):
    def __init__(self, idx):
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return int(self.idx[i])


def load_dev():
    df = pd.read_parquet(os.path.join(DATA_DIR, "dev.parquet"))
    meta = json.load(open(os.path.join(DATA_DIR, "meta.json")))
    feats = meta["features"]
    stored = [f for f in feats if f != "amount_z"]
    X = df[stored].to_numpy(dtype=np.float32)
    amt = df["amt_raw"].to_numpy(dtype=np.float32)
    X = np.insert(X, 2, amt, axis=1)  # amount_z slot filled per split by scaler
    y = df["is_fraud"].to_numpy(dtype=np.int64)
    acc = df["cc_num"].to_numpy()
    starts = np.flatnonzero(np.r_[True, acc[1:] != acc[:-1]])
    acct_start = np.repeat(starts, np.diff(np.r_[starts, len(acc)]))
    folds = df["fold"].to_numpy(dtype=np.int64)
    return X, y, acct_start, folds, feats


def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    chunks = []
    for xh, xt, mk, yb in loader:
        xh = xh.to(DEVICE, non_blocking=True)
        xt = xt.to(DEVICE, non_blocking=True)
        mk = mk.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xh, mk, xt)
        chunks.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(chunks)


def build_dl(ids, coll, bs, shuffle, seed):
    kwargs = dict(batch_size=bs, shuffle=shuffle, num_workers=8, pin_memory=True,
                  collate_fn=coll, prefetch_factor=4)
    if shuffle:
        kwargs.update(persistent_workers=True, generator=torch.Generator().manual_seed(seed))
    return DataLoader(IdDataset(ids), **kwargs)


def train_split(k, Xtr, ytr, acctr, ids_tr, Xva, yva, acctva, ids_va, row_ids_va):
    t0 = time.time()
    set_seed(42 + k)
    log_path = os.path.join(LOG_DIR, f"train_split_{k}.jsonl")

    scaler = StandardScaler()
    scaler.fit(Xtr)
    mu = scaler.mean_.astype(np.float32)
    sc = np.maximum(scaler.scale_.astype(np.float32), 1e-6)
    Xtr_s = np.clip((Xtr - mu) / sc, -20.0, 20.0).astype(np.float32)
    Xva_s = np.clip((Xva - mu) / sc, -20.0, 20.0).astype(np.float32)

    model = LayaFraudDecisionNet(in_dim=Xtr_s.shape[1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    compiled = torch.compile(model, dynamic=True)
    base = compiled._orig_mod
    opt = torch.optim.AdamW(compiled.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    coll_tr = make_collate(Xtr_s, acctr, ytr)
    coll_va = make_collate(Xva_s, acctva, yva)
    bs = 1024
    train_dl = build_dl(ids_tr, coll_tr, bs, True, 42 + k)
    val_dl = build_dl(ids_va, coll_va, 4096, False, 42 + k)

    best_pr, best_params, best_epoch = -1.0, None, -1
    with open(log_path, "w") as lf:
        for ep in range(EPOCHS):
            model.train()
            run, steps, retries = 0.0, 0, 0
            while True:
                try:
                    for xh, xt, mk, yb in train_dl:
                        xh = xh.to(DEVICE, non_blocking=True)
                        xt = xt.to(DEVICE, non_blocking=True)
                        mk = mk.to(DEVICE, non_blocking=True)
                        yb = yb.to(DEVICE, non_blocking=True)
                        opt.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = compiled(xh, mk, xt)
                        loss = focal_loss(logits.float(), yb, gamma=2.0)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(compiled.parameters(), CLIP)
                        opt.step()
                        run += loss.item()
                        steps += 1
                    break
                except torch.cuda.OutOfMemoryError:
                    retries += 1
                    if retries > 3 or bs <= 128:
                        raise
                    bs //= 2
                    del train_dl
                    gc.collect()
                    torch.cuda.empty_cache()
                    opt.zero_grad(set_to_none=True)
                    train_dl = build_dl(ids_tr, coll_tr, bs, True, 42 + k)
                    print(f"[split {k}] OOM -> batch size {bs * 2} -> {bs}", flush=True)
            sched.step()

            probs = evaluate(compiled, val_dl)
            yv = yva[ids_va]
            pr = average_precision_score(yv, probs)
            roc = roc_auc_score(yv, probs)
            rec = dict(split=k, epoch=ep + 1, loss=round(run / max(steps, 1), 5),
                       val_pr_auc=round(float(pr), 5), val_roc_auc=round(float(roc), 5),
                       batch_size=bs, sec=round(time.time() - t0, 1),
                       peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
            lf.write(json.dumps(rec) + "\n")
            print(f"[split {k} ep {ep + 1}] loss={rec['loss']} pr={pr:.5f} roc={roc:.5f} "
                  f"bs={bs} {rec['sec']}s", flush=True)
            if pr > best_pr:
                best_pr, best_epoch = pr, ep + 1
                best_params = {n: p.detach().cpu().clone() for n, p in base.named_parameters()}

    base.load_state_dict(best_params)
    probs = evaluate(compiled, val_dl)
    np.savez_compressed(os.path.join(PIPE, f"oof_split_{k}.npz"),
                        row_ids=row_ids_va, probs=probs.astype(np.float32), y=yva)

    save_ckpt(os.path.join(CKPT_DIR, f"laya_split_{k}_best.pt"), base, mu, sc,
              dict(in_dim=Xtr_s.shape[1], d_model=D_MODEL, nhead=N_HEAD, num_layers=N_LAYER,
                   ffn=FFN, max_len=MAX_LEN, dropout=DROPOUT), k, best_pr, best_epoch, n_params)
    print(f"[split {k}] done: best pr={best_pr:.5f} @ep{best_epoch}, params={n_params / 1e6:.2f}M, "
          f"{time.time() - t0:.0f}s", flush=True)

    del model, compiled, base, opt, sched, train_dl, val_dl, best_params
    gc.collect()
    torch.cuda.empty_cache()
    return float(best_pr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    free = torch.cuda.mem_get_info(0)[0]
    print(f"[init] cuda:0 = {torch.cuda.get_device_name(0)}, free={free / 1e9:.1f}GB", flush=True)
    X, y, acct_start, folds, feats = load_dev()
    print(f"[data] dev rows={len(folds)}, model features={len(feats)}, "
          f"fraud={int(y.sum())} ({y.mean():.5f})", flush=True)

    summary = {}
    for k in args.splits:
        tr, va = folds != k, folds == k
        ids_tr, ids_va = np.where(tr)[0], np.where(va)[0]
        print(f"[split {k}] train={int(tr.sum())} (fraud {int(y[tr].sum())}) "
              f"val={int(va.sum())} (fraud {int(y[va].sum())})", flush=True)
        # split-local ids for collate (Xtr/Xva are split slices); account groups never
        # cross splits, so acct starts resolve within the split
        tr_local = np.empty(len(folds), dtype=np.int64)
        va_local = np.empty(len(folds), dtype=np.int64)
        tr_local[ids_tr] = np.arange(len(ids_tr))
        va_local[ids_va] = np.arange(len(ids_va))
        summary[k] = train_split(k, X[tr], y[tr], tr_local[acct_start[tr]], np.arange(len(ids_tr)),
                                 X[va], y[va], va_local[acct_start[va]], np.arange(len(ids_va)), ids_va)

    out = os.path.join(CKPT_DIR, "val_pr_auc.json")
    json.dump({int(k): v for k, v in summary.items()}, open(out, "w"), indent=2)
    print(f"[done] val PR-AUC per split: {summary} -> {out}", flush=True)


if __name__ == "__main__":
    main()
