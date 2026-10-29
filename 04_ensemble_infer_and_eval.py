#!/usr/bin/env python
"""Phase 4: 5-model Laya ensemble — inference, calibration, evaluation, artifacts.

Loads checkpoints/laya_split_{0..4}_best.pt and evaluates on:
  1. OOF: each model's strict OOF = its own validation fold (oof_split_{k}.npz),
     plus a quasi-OOF 5-model mean over all dev rows (documented in report).
  2. Holdout: locked test.parquet (15% accounts, never trained on).

Ensembles:
  - mean probability: P = (1/5) sum_k sigmoid(z_k)
  - PR-AUC weighted logit: P = sigmoid( sum_k w_k z_k / sum_k w_k ), w_k = val PR-AUC
  - epistemic uncertainty: Std(P_0..P_4) across the 5 models

Thresholds (from OOF ensemble probs): tau* = argmax F1; tau_r1 = max recall @ FPR<=1%.
Outputs: ensemble_evaluation_report.json (+ oof_ensemble.npz).
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # single-GPU pin; override to run on GPU1
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from laya_model import LayaFraudDecisionNet  # noqa: E402

PIPE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PIPE, "data")
CKPT_DIR = os.path.join(PIPE, "checkpoints")
DEVICE = "cuda:0"
N_SPLITS = 5


def load_models():
    models, cfgs, weights = [], [], []
    for k in range(N_SPLITS):
        ck = torch.load(os.path.join(CKPT_DIR, f"laya_split_{k}_best.pt"), map_location="cpu", weights_only=False)
        m = LayaFraudDecisionNet(**ck["config"])
        m.load_state_dict(ck["model_state"])
        m.eval().to(DEVICE)
        models.append(m)
        cfgs.append(ck)
        weights.append(float(ck["val_pr_auc"]))
        print(f"[load] split {k}: val_pr_auc={ck['val_pr_auc']:.5f} best_epoch={ck['best_epoch']} "
              f"params={ck['n_params'] / 1e6:.2f}M", flush=True)
    return models, cfgs, np.array(weights)


def load_test_features(cfg0):
    df = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
    meta = json.load(open(os.path.join(DATA_DIR, "meta.json")))
    stored = [f for f in meta["features"] if f != "amount_z"]
    X = df[stored].to_numpy(dtype=np.float32)
    amt = df["amt_raw"].to_numpy(dtype=np.float32)
    X = np.insert(X, 2, amt, axis=1)
    y = df["is_fraud"].to_numpy(dtype=np.int64)
    acc = df["cc_num"].to_numpy()
    starts = np.flatnonzero(np.r_[True, acc[1:] != acc[:-1]])
    acct_start = np.repeat(starts, np.diff(np.r_[starts, len(acc)]))
    row_ids = df["row_id"].to_numpy(dtype=np.int64)
    return X, y, acct_start, row_ids


def scale(X, cfg):
    mu, sc = cfg["scaler_mean"].astype(np.float32), cfg["scaler_scale"].astype(np.float32)
    return np.clip((X - mu) / sc, -20.0, 20.0).astype(np.float32)


@torch.no_grad()
def infer(model, Xs, acct_start, bs=4096, workers=8):
    from laya_model import make_collate
    from torch.utils.data import DataLoader, Dataset

    class IDS(Dataset):
        def __init__(s, i):
            s.i = i

        def __len__(s):
            return len(s.i)

        def __getitem__(s, i):
            return int(s.i[i])

    ids = np.arange(len(Xs))
    coll = make_collate(Xs, acct_start, np.zeros(len(Xs), dtype=np.int64))
    dl = DataLoader(IDS(ids), batch_size=bs, shuffle=False, num_workers=workers,
                    pin_memory=True, collate_fn=coll)
    out = []
    for xh, xt, mk, _ in dl:
        xh = xh.to(DEVICE, non_blocking=True)
        xt = xt.to(DEVICE, non_blocking=True)
        mk = mk.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xh, mk, xt)
        out.append(logits.float().cpu().numpy())
    return np.concatenate(out)


def thresholds_from_probs(probs, y):
    """tau* = argmax F1; tau_r1 = max recall with FPR <= 1%."""
    f1_best, tau_best = -1.0, 0.5
    for tau in np.linspace(0.005, 0.995, 199):
        pred = (probs >= tau).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > f1_best:
            f1_best, tau_best = f1, float(tau)
    # recall @ FPR <= 1%
    order = np.argsort(-probs)
    ys = y[order]
    n_pos = max(int(y.sum()), 1)
    tp = np.cumsum(ys)
    fp = np.arange(1, len(ys) + 1) - tp
    fpr = fp / max(len(ys) - n_pos, 1)
    ok = fpr <= 0.01
    if ok.any():
        i = int(np.where(ok)[0].max())
        recall_at = tp[i] / n_pos
        tau_r1 = float(probs[order[i]])
        fpr_at = float(fpr[i])
    else:
        recall_at, tau_r1, fpr_at = 0.0, 1.0, 0.0
    return tau_best, f1_best, tau_r1, recall_at, fpr_at


def classify_metrics(y, probs, tau):
    pred = (probs >= tau).astype(int)
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }


def main():
    t0 = time.time()
    models, cfgs, w = load_models()
    print(f"[weights] pr_auc weights = {np.round(w, 5).tolist()}", flush=True)

    X, y, acct_start, row_ids = load_test_features(cfgs[0])
    print(f"[test] rows={len(y)} fraud={int(y.sum())} ({y.mean():.5f})", flush=True)

    logits = [infer(m, scale(X, c), acct_start) for m, c in zip(models, cfgs)]
    probs = [np.exp(l) / (1 + np.exp(l)) for l in logits]  # stable sigmoid
    P = np.stack(probs, axis=1)                             # (N,5)
    mean_p = P.mean(axis=1)
    wsum = w.sum()
    z = np.stack(logits, axis=1)  # (N,5) logits
    z_w = (z * w).sum(axis=1) / wsum
    wt_logit = np.exp(z_w) / (1 + np.exp(z_w))  # PR-AUC weighted logit ensemble (stable)
    std_p = P.std(axis=1, ddof=0)

    def report_metrics(y_arr, p):
        return {
            "roc_auc": float(roc_auc_score(y_arr, p)),
            "pr_auc": float(average_precision_score(y_arr, p)),
            "f1": float(f1_score(y_arr, (p >= 0.5).astype(int), zero_division=0)),
            "precision": float(precision_score(y_arr, (p >= 0.5).astype(int), zero_division=0)),
            "recall": float(recall_score(y_arr, (p >= 0.5).astype(int), zero_division=0)),
            "mean_prob": float(p.mean()),
            "std_prob": float(p.std()),
        }

    # ---- OOF ----
    oof_rows, oof_y, oof_p = {}, {}, {}
    for k in range(N_SPLITS):
        d = np.load(os.path.join(PIPE, f"oof_split_{k}.npz"))
        oof_rows[k], oof_y[k], oof_p[k] = d["row_ids"], d["y"], d["probs"]
    # row_ids stored by 03 are positions within dev; the 5 folds tile dev exactly
    dev_row_ids = np.sort(np.concatenate([oof_rows[k] for k in range(N_SPLITS)]))
    assert (np.diff(dev_row_ids) == 1).all() and dev_row_ids[0] == 0, "OOF folds do not tile dev"
    oof_n = len(dev_row_ids)
    oof_all = np.zeros(oof_n, dtype=np.float32)   # strict-OOF: each dev row scored by the model that held it out
    y_dev = np.zeros(oof_n, dtype=np.int64)
    for k in range(N_SPLITS):
        oof_all[oof_rows[k]] = oof_p[k]
        y_dev[oof_rows[k]] = oof_y[k]
    oof_metrics = {"per_model": {}, "ensemble_strict_oof": report_metrics(y_dev, oof_all)}
    for k in range(N_SPLITS):
        oof_metrics["per_model"][f"split_{k}"] = report_metrics(oof_y[k], oof_p[k])
    tau_oof, f1_oof, tau_r1, rec_r1, fpr_r1 = thresholds_from_probs(oof_all, y_dev)

    # ensemble classification metrics at OOF-derived thresholds
    oof_metrics["ensemble_strict_oof"]["at_tau_star"] = classify_metrics(y_dev, oof_all, tau_oof)
    oof_metrics["ensemble_strict_oof"]["at_tau_recall1pctFPR"] = classify_metrics(y_dev, oof_all, tau_r1)
    for k in range(N_SPLITS):
        oof_metrics["per_model"][f"split_{k}"]["at_tau_star"] = classify_metrics(oof_y[k], oof_p[k], tau_oof)

    # ---- Holdout test metrics ----
    test_metrics = {}
    for k in range(N_SPLITS):
        m = report_metrics(y, probs[k])
        m["at_tau_star"] = classify_metrics(y, probs[k], tau_oof)
        m["at_tau_recall1pctFPR"] = classify_metrics(y, probs[k], tau_r1)
        test_metrics[f"split_{k}"] = m
    for name, p in [("ensemble_mean_prob", mean_p), ("ensemble_pr_auc_weighted_logit", wt_logit)]:
        m = report_metrics(y, p)
        m["at_tau_star"] = classify_metrics(y, p, tau_oof)
        m["at_tau_recall1pctFPR"] = classify_metrics(y, p, tau_r1)
        test_metrics[name] = m
    test_metrics["uncertainty"] = {
        "mean_ensemble_std_all": float(std_p.mean()),
        "mean_ensemble_std_borderline_|p-0.5|<0.15": float(
            std_p[np.abs(mean_p - 0.5) < 0.15].mean() if (np.abs(mean_p - 0.5) < 0.15).any() else 0.0
        ),
        "p95_ensemble_std": float(np.percentile(std_p, 95)),
    }

    best_single = max((k for k in range(N_SPLITS)), key=lambda k: test_metrics[f"split_{k}"]["pr_auc"])
    gain = {
        "best_single_split": f"split_{best_single}",
        "best_single_pr_auc": test_metrics[f"split_{best_single}"]["pr_auc"],
        "ensemble_mean_pr_auc": test_metrics["ensemble_mean_prob"]["pr_auc"],
        "ensemble_weighted_pr_auc": test_metrics["ensemble_pr_auc_weighted_logit"]["pr_auc"],
        "pr_auc_gain_mean_vs_best": test_metrics["ensemble_mean_prob"]["pr_auc"] - test_metrics[f"split_{best_single}"]["pr_auc"],
        "roc_gain_mean_vs_best": test_metrics["ensemble_mean_prob"]["roc_auc"] - test_metrics[f"split_{best_single}"]["roc_auc"],
    }

    report = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "n_splits": N_SPLITS,
        "model": "LayaFraudDecisionNet (causal transformer backbone + target cross-attention decision head)",
        "architecture": dict(cfgs[0]["config"]),
        "per_split_val_pr_auc": {f"split_{k}": float(w[k]) for k in range(N_SPLITS)},
        "thresholds": {
            "tau_star_max_f1_oof": tau_oof, "oof_f1_at_tau_star": f1_oof,
            "tau_recall_at_1pct_fpr": tau_r1, "oof_recall_at_tau_r1": rec_r1, "oof_fpr_at_tau_r1": fpr_r1,
            "note": "thresholds derived from strict-OOF ensemble probabilities (each dev row scored by the model that held it out)",
        },
        "oof_metrics": oof_metrics,
        "test_metrics_holdout": test_metrics,
        "ensemble_gain": gain,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out = os.path.join(PIPE, "ensemble_evaluation_report.json")
    json.dump(report, open(out, "w"), indent=2)

    # per-row test probabilities (for predictor cross-check + downstream use)
    np.savez_compressed(os.path.join(PIPE, "test_probs.npz"),
                        row_ids=row_ids, y=y.astype(np.int64),
                        per_model_probs=P.astype(np.float32),
                        mean_prob=mean_p.astype(np.float32),
                        weighted_logit_prob=wt_logit.astype(np.float32),
                        ensemble_std=std_p.astype(np.float32))

    # OOF ensemble artifact
    np.savez_compressed(os.path.join(PIPE, "oof_ensemble.npz"),
                        row_ids=dev_row_ids, y=y_dev, probs=oof_all)

    print(json.dumps({k: report[k] for k in ("thresholds", "ensemble_gain")}, indent=2))
    print(f"[test] split0 roc={test_metrics['split_0']['roc_auc']:.4f} pr={test_metrics['split_0']['pr_auc']:.4f}")
    print(f"[test] mean-ens roc={test_metrics['ensemble_mean_prob']['roc_auc']:.4f} "
          f"pr={test_metrics['ensemble_mean_prob']['pr_auc']:.4f}")
    print(f"[test] wt-ens   roc={test_metrics['ensemble_pr_auc_weighted_logit']['roc_auc']:.4f} "
          f"pr={test_metrics['ensemble_pr_auc_weighted_logit']['pr_auc']:.4f}")
    print(f"[done] {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
