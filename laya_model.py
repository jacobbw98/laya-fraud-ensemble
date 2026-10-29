#!/usr/bin/env python
"""Shared Laya architecture + data utilities (used by 03/04; NOT the standalone predictor).

LayaFraudDecisionNet: causal Transformer over the account's last L=256 PAST
transactions (front-padded) + learnable NOHIST token; the target transaction
token then cross-attends to the encoded history (behavioral anomaly vector)
and an MLP head outputs the binary fraud logit.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # single GPU; overridable via env (e.g. CUDA_VISIBLE_DEVICES=1)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LEN = 256
D_MODEL = 128
N_HEAD = 8
N_LAYER = 4
FFN = 256
DROPOUT = 0.1


class CausalBlock(nn.Module):
    """Pre-LN Transformer block with causal SDPA (Flash/efficient attention)."""

    def __init__(self, d, h, ffn, p):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(p), nn.Linear(ffn, d), nn.Dropout(p))

    def forward(self, x, attn_mask=None):
        # x: (B, T, d); attn_mask: bool (B,1,1,T) or (B,T), True = must attend
        B, T, d = x.shape
        h = self.wq.out_features  # noqa (unused; kept for clarity)
        hd = d // 8
        nh = 8
        x0 = self.ln1(x)
        q = self.wq(x0).view(B, T, nh, hd).transpose(1, 2)
        k = self.wk(x0).view(B, T, nh, hd).transpose(1, 2)
        v = self.wv(x0).view(B, T, nh, hd).transpose(1, 2)
        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask = attn_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, d)
        return x + self.wo(y) + self.ffn(self.ln2(x))


class LayaFraudDecisionNet(nn.Module):
    def __init__(self, in_dim=16, d_model=D_MODEL, nhead=N_HEAD, num_layers=N_LAYER,
                 ffn=FFN, max_len=MAX_LEN, dropout=DROPOUT):
        super().__init__()
        self.max_len = max_len
        self.token_enc = nn.Sequential(nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.pos_hist = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.no_hist = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_target = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([CausalBlock(d_model, nhead, ffn, dropout) for _ in range(num_layers)])
        # Laya decision cross-attention: target -> history
        self.cross_ln_q = nn.LayerNorm(d_model)
        self.cross_ln_kv = nn.LayerNorm(d_model)
        self.cross_k = nn.Linear(d_model, d_model, bias=False)
        self.cross_v = nn.Linear(d_model, d_model, bias=False)
        self.cross_o = nn.Linear(d_model, d_model, bias=False)
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )
        self.nhead = nhead

    def forward(self, x_hist, mask, x_target):
        # x_hist: (B, L, F) front-padded; mask: (B, L) bool True=valid; x_target: (B, F)
        B = x_hist.shape[0]
        L = x_hist.shape[1]
        d = self.blocks[0].ln1.weight.shape[0]
        nh = self.nhead
        hd = d // nh

        h = self.token_enc(x_hist) + self.pos_hist
        nh_tok = self.no_hist.expand(B, 1, -1)
        h = torch.cat([nh_tok, h], dim=1)                       # (B, L+1, d)
        valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x_hist.device), mask], dim=1)
        am = valid[:, None, None, :]                            # (B,1,1,L+1) True=attend

        for blk in self.blocks:
            h = blk(h, am)

        q = (self.token_enc(x_target) + self.pos_target.squeeze(0)).unsqueeze(1)   # (B,1,d)
        qe = self.cross_ln_q(q).view(B, 1, nh, hd).transpose(1, 2)
        ke = self.cross_k(self.cross_ln_kv(h)).view(B, L + 1, nh, hd).transpose(1, 2)
        ve = self.cross_v(self.cross_ln_kv(h)).view(B, L + 1, nh, hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(qe, ke, ve, attn_mask=am)  # (B,nh,1,hd)
        out = out.transpose(1, 2).reshape(B, 1, d)
        out = self.ln_out(self.cross_o(out) + q).squeeze(1)
        feat = torch.cat([out, q.squeeze(1)], dim=1)
        return self.head(feat).squeeze(-1)


def focal_loss(logits, y, gamma=2.0):
    """Focal BCE (alpha=1): down-weights easy examples, gamma=2 default."""
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p = torch.sigmoid(logits)
    w = y * (1.0 - p) ** gamma + (1.0 - y) * p ** gamma
    return (w * bce).mean()


def make_collate(X, acct_start, y):
    """Collate sample positions -> (hist front-padded, target, mask, labels).

    X: (N, F) float32 feature matrix (rows = globally sorted account positions);
    acct_start: (N,) int, first global position of each row's account;
    y: (N,) int64 labels.
    """
    L = MAX_LEN
    j = np.arange(L)[None, :]

    def collate(ids):
        ids = np.asarray(ids, dtype=np.int64)
        lo = np.maximum(acct_start[ids], ids - L)
        idx = ids[:, None] - L + j
        valid = (idx >= lo[:, None]) & (j < (ids - lo)[:, None])
        idx = np.where(valid, idx, 0)
        hist = X[idx.reshape(-1)].reshape(len(ids), L, X.shape[1])
        tgt = X[ids]
        return (
            torch.from_numpy(np.ascontiguousarray(hist)),
            torch.from_numpy(np.ascontiguousarray(tgt)),
            torch.from_numpy(valid),
            torch.from_numpy(np.ascontiguousarray(y[ids].astype(np.float32))),
        )

    return collate


def save_ckpt(path, base_model, mu, sc, cfg, split, val_pr, best_epoch, n_params):
    import json
    from datetime import datetime, timezone

    ckpt = {
        "split": split,
        "model_state": {n: p.detach().cpu().clone() for n, p in base_model.named_parameters()},
        "config": cfg,
        "scaler_mean": mu, "scaler_scale": sc,
        "val_pr_auc": float(val_pr), "best_epoch": int(best_epoch),
        "n_params": int(n_params),
        "saved_utc": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(ckpt, path)
    return ckpt
