#!/usr/bin/env python
"""Standalone Laya 5-model ensemble fraud predictor (self-contained).

Usage:
    from laya_ensemble_predictor import predict_transaction
    result = predict_transaction(account_history_df)
    # -> {"is_fraud": int, "fraud_probability": float, "ensemble_std": float, "threshold": float}

`account_history_df`: DataFrame of the account's chronological transactions
INCLUDING the target transaction as the LAST (latest) row. Requires a timestamp
column ("unix_time" numeric epoch, or any of "trans_date_trans_time"/"timestamp"/
"time"/"date" parsed as UTC) and an amount column ("amt" or "amount").
Raw identifiers (names, card numbers, addresses) are ignored by design.

Loads the 5 split checkpoints (checkpoints/laya_split_{0..4}_best.pt) next to
this file; decision threshold from ensemble_evaluation_report.json
(tau* = argmax F1 on strict OOF; falls back to 0.5 if report missing).
Single GPU: pinned to cuda:0.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # single GPU; override via env if needed

import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LEN = 256
PIPE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(PIPE, "checkpoints")
REPORT = os.path.join(PIPE, "ensemble_evaluation_report.json")
DEVICE = "cuda:0"
WINDOWS = [(3600.0, "1h"), (86400.0, "24h"), (604800.0, "7d")]
TS_COLS = ["unix_time", "trans_date_trans_time", "timestamp", "time", "date"]
AMT_COLS = ["amt", "amount"]

# ---------------- architecture (mirrors laya_model.py) ----------------
class CausalBlock(nn.Module):
    def __init__(self, d, h, ffn, p):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(p), nn.Linear(ffn, d), nn.Dropout(p))

    def forward(self, x, attn_mask=None):
        B, T, d = x.shape
        nh, hd = 8, d // 8
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
    def __init__(self, in_dim=16, d_model=128, nhead=8, num_layers=4, ffn=256, max_len=256, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.token_enc = nn.Sequential(nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.pos_hist = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.no_hist = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_target = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([CausalBlock(d_model, nhead, ffn, dropout) for _ in range(num_layers)])
        self.cross_ln_q = nn.LayerNorm(d_model)
        self.cross_ln_kv = nn.LayerNorm(d_model)
        self.cross_k = nn.Linear(d_model, d_model, bias=False)
        self.cross_v = nn.Linear(d_model, d_model, bias=False)
        self.cross_o = nn.Linear(d_model, d_model, bias=False)
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x_hist, mask, x_target):
        B, L = x_hist.shape[0], x_hist.shape[1]
        d = self.blocks[0].ln1.weight.shape[0]
        nh, hd = self.nhead, d // self.nhead
        h = self.token_enc(x_hist) + self.pos_hist
        h = torch.cat([self.no_hist.expand(B, 1, -1), h], dim=1)
        valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x_hist.device), mask], dim=1)
        am = valid[:, None, None, :]
        for blk in self.blocks:
            h = blk(h, am)
        q = (self.token_enc(x_target) + self.pos_target.squeeze(0)).unsqueeze(1)
        qe = self.cross_ln_q(q).view(B, 1, nh, hd).transpose(1, 2)
        ke = self.cross_k(self.cross_ln_kv(h)).view(B, L + 1, nh, hd).transpose(1, 2)
        ve = self.cross_v(self.cross_ln_kv(h)).view(B, L + 1, nh, hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(qe, ke, ve, attn_mask=am)
        out = out.transpose(1, 2).reshape(B, 1, d)
        out = self.ln_out(self.cross_o(out) + q).squeeze(1)
        return self.head(torch.cat([out, q.squeeze(1)], dim=1)).squeeze(-1)


# ---------------- feature engineering (identical to 02_preprocess_and_split.py) ----------------
def _compute_features(ts, amt):
    """ts: (n,) epoch seconds float64 (chronological); amt: (n,) float64.
    Returns (n, 16) float32 with raw amount at index 2 (scaled per-model)."""
    n = len(ts)
    f = {}
    d = np.zeros(n)
    d[1:] = ts[1:] - ts[:-1]
    f["time_delta_sec_log"] = np.log1p(d)
    f["log_amount"] = np.log1p(np.maximum(amt, 0.0))
    pref1 = np.concatenate([[0.0], np.cumsum(amt)])
    pref2 = np.concatenate([[0.0], np.cumsum(amt * amt)])
    cnt = np.arange(n, dtype=np.float64)
    mu = pref1[:n] / np.maximum(cnt, 1)
    mu[0] = 0.0
    var = np.maximum(pref2[:n] / np.maximum(cnt, 1) - mu * mu, 0.0)
    f["exp_mean_amt"] = mu
    f["exp_std_amt"] = np.sqrt(var)
    f["amt_to_mean_ratio"] = np.clip(np.where(mu > 1e-6, amt / np.maximum(mu, 1e-6), 0.0), 0.0, 1e4)
    i = np.arange(n)
    for wsec, tag in WINDOWS:
        j = np.searchsorted(ts, ts - wsec, side="left")
        f[f"vel_cnt_{tag}"] = np.log1p((i - j).astype(np.float64))
        f[f"vel_sum_{tag}"] = np.log1p(pref1[i] - pref1[j])
    hod = (ts % 86400.0) / 86400.0 * 2 * np.pi
    dow = ((ts + 4 * 86400.0) % 604800.0) / 604800.0 * 2 * np.pi
    f["hod_sin"], f["hod_cos"] = np.sin(hod), np.cos(hod)
    f["dow_sin"], f["dow_cos"] = np.sin(dow), np.cos(dow)
    cols = ["time_delta_sec_log", "log_amount", "exp_mean_amt", "exp_std_amt", "amt_to_mean_ratio",
            "vel_cnt_1h", "vel_sum_1h", "vel_cnt_24h", "vel_sum_24h", "vel_cnt_7d", "vel_sum_7d",
            "hod_sin", "hod_cos", "dow_sin", "dow_cos"]
    X = np.stack([f[c].astype(np.float32) for c in cols], axis=1)
    return np.insert(X, 2, amt.astype(np.float32), axis=1)


def _pick(df, names, what):
    for c in names:
        if c in df.columns:
            return df[c]
    raise ValueError(f"no {what} column found; expected one of {names}; got {list(df.columns)}")


_state = {"models": None, "weights": None, "threshold": 0.5}


def _load():
    if _state["models"] is None:
        torch.manual_seed(0)
        models, weights = [], []
        for k in range(5):
            ck = torch.load(os.path.join(CKPT_DIR, f"laya_split_{k}_best.pt"), map_location="cpu", weights_only=False)
            m = LayaFraudDecisionNet(**ck["config"])
            m.load_state_dict(ck["model_state"])
            m.eval().to(DEVICE)
            models.append((m, ck["scaler_mean"].astype(np.float32), np.maximum(ck["scaler_scale"].astype(np.float32), 1e-6)))
            weights.append(float(ck["val_pr_auc"]))
        _state["models"] = models
        _state["weights"] = np.array(weights)
        if os.path.exists(REPORT):
            rep = json.load(open(REPORT))
            _state["threshold"] = float(rep["thresholds"]["tau_star_max_f1_oof"])
    return _state["models"], _state["weights"], _state["threshold"]


def predict_transaction(account_history_df):
    """Judge the LAST (latest) transaction in `account_history_df` as fraud or not.

    Returns {"is_fraud": int, "fraud_probability": float, "ensemble_std": float, "threshold": float}.
    """
    df = account_history_df.copy()
    amt = _pick(df, AMT_COLS, "amount").to_numpy(dtype=np.float64)
    ts = _pick(df, TS_COLS, "timestamp")
    if ts.dtype.kind in "iuf":
        t = ts.to_numpy(dtype=np.float64)
    else:
        t = pd.to_datetime(ts, utc=True, format="mixed").astype("int64").to_numpy() // 10**9
    order = np.argsort(t, kind="mergesort")
    t, amt = t[order], amt[order]

    X = _compute_features(t, amt)
    tgt = X[-1]
    hlen = min(len(t) - 1, MAX_LEN)
    hist = X[len(t) - 1 - hlen:-1]
    pad = MAX_LEN - hlen
    if pad:
        hist = np.vstack([np.zeros((pad, X.shape[1]), dtype=np.float32), hist])
    mask = np.zeros((1, MAX_LEN), dtype=bool)
    mask[0, pad:] = True

    models, weights, threshold = _load()
    xh = torch.from_numpy(hist).unsqueeze(0).to(DEVICE)
    xt = torch.from_numpy(tgt).unsqueeze(0).to(DEVICE)
    mk = torch.from_numpy(mask).to(DEVICE)

    probs, zs = [], []
    for m, mu, sc in models:
        def s_(v):
            return np.clip((v - mu) / sc, -20.0, 20.0).astype(np.float32)
        xh_s = torch.from_numpy(s_(hist)).unsqueeze(0).to(DEVICE)
        xt_s = torch.from_numpy(s_(tgt)).unsqueeze(0).to(DEVICE)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z = m(xh_s, mk, xt_s)
        z = z.float()
        zs.append(z.item())
        probs.append(float(torch.sigmoid(z).item()))
    p_mean = float(np.mean(probs))
    wsum = weights.sum()
    z_w = sum(w * z for w, z in zip(weights, zs)) / wsum
    p_weighted = float(1.0 / (1.0 + np.exp(-z_w)))
    ens = 0.5 * (p_mean + p_weighted)  # blend of the two aggregation strategies
    std_ens = float(np.std(probs, ddof=0))
    return {
        "is_fraud": int(ens >= threshold),
        "fraud_probability": ens,
        "ensemble_std": std_ens,
        "threshold": threshold,
        "detail": {"per_model_probs": probs, "mean_prob": p_mean, "weighted_logit_prob": p_weighted},
    }


if __name__ == "__main__":
    # demo: predict on the first 3 test-set accounts using their last transaction
    test = pd.read_parquet(os.path.join(PIPE, "data", "test.parquet"))
    for acc in test["cc_num"].unique()[:3]:
        g = test[test["cc_num"] == acc].tail(MAX_LEN + 1)
        hist = g[["ts", "amt_raw"]]
        r = predict_transaction(hist.rename(columns={"ts": "unix_time", "amt_raw": "amt"}))
        print(acc, "->", {k: v for k, v in r.items() if k != "detail"})
