#!/usr/bin/env python
"""Phase 4b (v2): ensemble the 5 v2 models; strict-OOF thresholds; holdout eval;
overfit audit. GPU inference (default CUDA_VISIBLE_DEVICES=1).

Outputs: ensemble_evaluation_report_v2.json, test_probs_v2.npz
  {row_id, p0..p4, p_mean, p_weighted, y, dataset}
"""
import os
import sys
import json

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve

PIPE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPE)
from laya_model import LayaFraudDecisionNet  # noqa: E402

meta = json.load(open(os.path.join(PIPE, "data_v2", "meta_v2.json")))
STORED = meta["features_stored"]
test = pd.read_parquet(os.path.join(PIPE, "data_v2", "test.parquet"))
Xt0 = test[STORED].to_numpy(np.float32)
Xt0 = np.insert(Xt0, 2, test["amt_raw"].to_numpy(np.float32), axis=1)
Xt0 = np.nan_to_num(Xt0, nan=0.0, posinf=0.0, neginf=0.0)
yt = test["is_fraud"].to_numpy(np.int64)
accs = test["acct_start"].to_numpy(np.int64)
row_ids = test["row_id"].to_numpy(np.int64)
ds = test["dataset"].to_numpy(np.int64)
L = 512
device = "cuda"


@torch.no_grad()
def infer(model, X):
    model.eval()
    out = []
    for s in range(0, len(X), 4096):
        ids = np.arange(s, s + 4096)
        lo = np.maximum(accs[ids], ids - L)
        j = np.arange(L)[None, :]
        gi = ids[:, None] - L + j
        valid = (gi >= lo[:, None]) & (j < (ids - lo)[:, None])
        gi = np.where(valid, gi, 0)
        hist = torch.from_numpy(np.ascontiguousarray(X[gi.reshape(-1)].reshape(len(ids), L, -1))).to(device)
        tgt = torch.from_numpy(X[ids]).to(device)
        mk = torch.from_numpy(valid).to(device)
        out.append(torch.sigmoid(model(hist, mk, tgt)).cpu().numpy())
    return np.concatenate(out)


P = np.zeros((5, len(yt)))
prs = {}
for k in range(5):
    ck = torch.load(os.path.join(PIPE, "checkpoints_v2", f"v2_split_{k}_best.pt"), map_location="cpu")
    m = LayaFraudDecisionNet(**ck["config"])
    m.load_state_dict(ck["model_state"])
    m.to(device)
    Xs = ((Xt0 - ck["scaler_mean"]) / ck["scaler_scale"]).astype(np.float32)
    P[k] = infer(m, Xs)
    prs[k] = float(ck["val_pr_auc"])
    del m, Xs
    torch.cuda.empty_cache()
    print(f"split {k}: holdout single-model PR-AUC {average_precision_score(yt, P[k]):.4f} "
          f"(val PR {prs[k]:.4f})", flush=True)

p_mean = P.mean(0)
w = np.array([prs[k] for k in range(5)])
w = w / w.sum()
lg = np.log(np.clip(P, 1e-7, 1 - 1e-7) / np.clip(1 - P, 1e-7, 1 - 1e-7))
p_w = 1.0 / (1.0 + np.exp(-(lg.T @ w)))


def metrics_at(p, yy, t):
    pred = (p >= t).astype(np.int64)
    tp = int(((pred == 1) & (yy == 1)).sum())
    fp = int(((pred == 1) & (yy == 0)).sum())
    fn = int(((pred == 0) & (yy == 1)).sum())
    tn = int(((pred == 0) & (yy == 0)).sum())
    return {"f1": round(2 * tp / max(2 * tp + fp + fn, 1), 6),
            "precision": round(tp / max(tp + fp, 1), 6),
            "recall": round(tp / max(tp + fn, 1), 6),
            "fpr": round(fp / max(fp + tn, 1), 6),
            "n_pos": int(yy.sum()), "n_flagged": int(pred.sum())}


def block(p, yy, tau_star, tau_r1, tau_r5):
    d = {
        "n": int(len(yy)), "n_fraud": int(yy.sum()),
        "pr_auc": round(average_precision_score(yy, p), 6),
        "roc_auc": round(roc_auc_score(yy, p), 6),
        "at_tau_star": metrics_at(p, yy, tau_star),
        "at_tau_r1": metrics_at(p, yy, tau_r1),
        "at_tau_r5": metrics_at(p, yy, tau_r5),
    }
    return d


# ---- strict OOF: each dev row scored exactly once, by the model that held it out ----
oof_mean = np.zeros(meta["n_dev_rows"])
oof_y = np.zeros(meta["n_dev_rows"], dtype=np.int64)
for k in range(5):
    z = np.load(os.path.join(PIPE, "data_v2", f"oof_split_{k}.npz"))
    oof_mean[z["row_id"]] = z["probs"]
    oof_y[z["row_id"]] = z["y"]
assert oof_y.sum() > 0 and np.isfinite(oof_mean).all()
oof_w = oof_mean  # strict OOF: one model per row, weighted == mean

prec, rec, th_ = precision_recall_curve(oof_y, oof_mean)
f1s = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1][:-1] * 0 + rec[:-1], 1e-9)
f1s = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-9)
tau_star = float(th_[int(f1s.argmax())])


def thr_recall_at_fpr(p, yy, fpr_max):
    fpr_, tpr_, th_ = roc_curve(yy, p)
    ok = np.where(fpr_ <= fpr_max, tpr_, -1.0)
    i = int(ok.argmax())
    return float(th_[i]), float(tpr_[i])


tau_r1, oof_r1 = thr_recall_at_fpr(oof_mean, oof_y, 0.01)
tau_r5, oof_r5 = thr_recall_at_fpr(oof_mean, oof_y, 0.05)

# ---- overfit audit from training log ----
overfit = {}
try:
    rows = [json.loads(l) for l in open(os.path.join(PIPE, "logs", "train_v2.jsonl")) if l.strip()]
    for k in range(5):
        r = [x for x in rows if x["split"] == k]
        if not r:
            continue
        best = max(r, key=lambda x: x["val_pr"])
        last = r[-1]
        overfit[f"split_{k}"] = {
            "best_val_pr": round(best["val_pr"], 6), "best_epoch": best["epoch"],
            "final_train_pr": round(last["train_pr"], 6),
            "final_val_pr": round(last["val_pr"], 6),
            "train_minus_val_pr": round(last["train_pr"] - last["val_pr"], 6),
            "epochs": last["epoch"],
        }
except FileNotFoundError:
    pass

missed = {}
for name, m in (("combined", np.ones_like(yt, bool)),
                ("archive_A", ds == 0), ("ieee_I", ds == 1)):
    bad = np.where((yt[m] == 1) & (p_mean[m] < tau_r1))[0]
    missed[name] = {"n_missed": int(len(bad)), "n_fraud": int(yt[m].sum()),
                    "row_ids": [int(i) for i in bad[:500]]}

report = {
    "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    "version": "v2",
    "model": "5x LayaFraudDecisionNet (d256/h8-causal/h16-cross/6L/ffn1024/L512) on A+I combined data",
    "architecture": meta,
    "per_split_val_pr": {f"split_{k}": round(prs[k], 6) for k in range(5)},
    "thresholds_from_strict_oof": {
        "tau_star_max_f1": tau_star,
        "oof_f1_at_tau_star": round(2 * (1 / (1 / prec[int(f1s.argmax())] + 1 / rec[int(f1s.argmax())])) / 2 if False else metrics_at(oof_mean, oof_y, tau_star)["f1"], 6),
        "tau_recall_at_1pct_fpr": tau_r1, "oof_recall_at_tau_r1": oof_r1,
        "tau_recall_at_5pct_fpr": tau_r5, "oof_recall_at_tau_r5": oof_r5,
    },
    "oof_ensemble": {
        "pr_auc_mean": round(average_precision_score(oof_y, oof_mean), 6),
        "roc_auc_mean": round(roc_auc_score(oof_y, oof_mean), 6),
        "pr_auc_weighted_logit": round(average_precision_score(oof_y, oof_w), 6),
        "roc_auc_weighted_logit": round(roc_auc_score(oof_y, oof_w), 6),
    },
    "holdout": {
        "combined": block(p_mean, yt, tau_star, tau_r1, tau_r5),
        "archive_A": block(p_mean[yt_ds0 := (ds == 0)], yt[yt_ds0], tau_star, tau_r1, tau_r5),
        "ieee_I": block(p_mean[ds == 1], yt[ds == 1], tau_star, tau_r1, tau_r5),
    },
    "overfit_audit": overfit,
    "oof_vs_holdout_pr": {
        "oof": round(average_precision_score(oof_y, oof_mean), 6),
        "holdout": round(average_precision_score(yt, p_mean), 6),
    },
    "missed_fraud_at_tau_r1": missed,
    "v1_reference": {"holdout_pr": 0.9059, "holdout_roc": 0.9968,
                     "recall_at_fpr_le_1pct": 0.944, "n_holdout_fraud": 1452,
                     "note": "v1 = 5x d128/4L/L256 on archive only"},
}
with open(os.path.join(PIPE, "ensemble_evaluation_report_v2.json"), "w") as f:
    json.dump(report, f, indent=2)
np.savez(os.path.join(PIPE, "test_probs_v2.npz"),
         row_id=row_ids.astype(np.int32),
         **{f"p{k}": P[k] for k in range(5)},
         p_mean=p_mean, p_weighted=p_w, y=yt.astype(np.int8), dataset=ds.astype(np.int8))

h = report["holdout"]["combined"]
print("\n===== V2 HOLDOUT (combined A+I) =====")
print(f"rows {h['n']} fraud {h['n_fraud']}  PR-AUC {h['pr_auc']}  ROC {h['roc_auc']}")
for tname, t in (("tau_star", tau_star), ("tau_r1(FPR<=1%)", tau_r1), ("tau_r5(FPR<=5%)", tau_r5)):
    m = h[f"at_{tname.split('_')[1]}" if False else {"tau_star": "at_tau_star", "tau_r1": "at_tau_r1", "tau_r5": "at_tau_r5"}[tname]]
    print(f"  {tname}={float(t):.4f}: P {m['precision']:.4f} R {m['recall']:.4f} F1 {m['f1']:.4f} FPR {m['fpr']:.4f} flagged {m['n_flagged']}")
print(f"OOF PR {report['oof_ensemble']['pr_auc_mean']} | holdout PR {report['oof_vs_holdout_pr']['holdout']} | missed@tau_r1 {missed['combined']['n_missed']}/{missed['combined']['n_fraud']}")
for k, v in overfit.items():
    print(f"  overfit {k}: train {v['final_train_pr']} vs val {v['final_val_pr']} (gap {v['train_minus_val_pr']:+}) epochs {v['epochs']}")
print("[v2 eval done]", flush=True)
