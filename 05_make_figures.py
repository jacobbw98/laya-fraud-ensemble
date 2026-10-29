#!/usr/bin/env python3
"""05_make_figures.py — build the repo figures from saved artifacts (CPU-only, no GPU).

Reads : test_probs.npz, oof_ensemble.npz, ensemble_evaluation_report.json
Writes: figures/fig1_pr_curves_holdout.png
        figures/fig2_roc_curves_holdout.png
        figures/fig3_threshold_operating_points.png
        figures/fig4_separation_and_confusion.png
        figures/fig5_pipeline_flow.png
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import (average_precision_score, auc, confusion_matrix,
                             precision_recall_curve, roc_curve)

PIPE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(PIPE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

T = np.load(os.path.join(PIPE, "test_probs.npz"))
OOF = np.load(os.path.join(PIPE, "oof_ensemble.npz"))
REP = json.load(open(os.path.join(PIPE, "ensemble_evaluation_report.json")))
TAU_STAR = REP["thresholds"]["tau_star_max_f1_oof"]
TAU_R1 = REP["thresholds"]["tau_recall_at_1pct_fpr"]
BASE = T["y"].mean()

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})


def fig1_pr_curves():
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for k in range(5):
        p = T["per_model_probs"][:, k]
        pr, rc, _ = precision_recall_curve(T["y"], p)
        ax.plot(rc, pr, lw=1.0, alpha=0.6, color="tab:gray",
                label=f"split {k}  ({average_precision_score(T['y'], p):.4f})")
    p = T["weighted_logit_prob"]
    pr, rc, _ = precision_recall_curve(T["y"], p)
    ax.plot(rc, pr, lw=1.8, ls="--", color="tab:orange",
            label=f"weighted-logit  ({average_precision_score(T['y'], p):.4f})")
    p = T["mean_prob"]
    pr, rc, _ = precision_recall_curve(T["y"], p)
    ax.plot(rc, pr, lw=2.5, color="tab:red",
            label=f"ensemble mean  ({average_precision_score(T['y'], p):.4f})")
    ax.axhline(BASE, color="k", ls=":", lw=1, label=f"chance = base rate {BASE:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall — holdout (150 unseen accounts, base rate "
                 f"{BASE:.4f})")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_pr_curves_holdout.png"))
    plt.close(fig)


def fig2_roc_curves():
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for k in range(5):
        p = T["per_model_probs"][:, k]
        fpr, tpr, _ = roc_curve(T["y"], p)
        ax.plot(fpr, tpr, lw=1.0, alpha=0.6, color="tab:gray",
                label=f"split {k}  ({auc(fpr, tpr):.4f})")
    p = T["mean_prob"]
    fpr, tpr, _ = roc_curve(T["y"], p)
    ax.plot(fpr, tpr, lw=2.5, color="tab:red",
            label=f"ensemble mean  ({auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — holdout (150 unseen accounts)")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_roc_curves_holdout.png"))
    plt.close(fig)


def fig3_thresholds():
    y, p = T["y"], T["mean_prob"]
    thr = np.linspace(0.001, 0.9, 900)
    prec, rec = [], []
    for t in thr:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        prec.append(tp / max(int((pred == 1).sum()), 1))
        rec.append(tp / int((y == 1).sum()))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot(thr, rec, lw=2.2, color="tab:blue", label="Recall")
    ax.plot(thr, prec, lw=2.2, color="tab:orange", label="Precision")
    for t, c, dy in [(TAU_STAR, "tab:red", +0.14), (TAU_R1, "tab:green", -0.22)]:
        pt = (p >= t).astype(int)
        tp = int(((pt == 1) & (y == 1)).sum())
        fp = int(((pt == 1) & (y == 0)).sum())
        pr, rc = tp / (tp + fp), tp / int((y == 1).sum())
        f1 = 2 * pr * rc / (pr + rc)
        ax.axvline(t, color=c, ls="--", lw=1.5)
        ax.plot(t, rc, "o", color=c, ms=7, zorder=5)
        ax.annotate(f"τ={t:.4f}\nP={pr:.3f} R={rc:.3f} F1={f1:.3f}",
                    xy=(t, rc), xytext=(t + 0.09, rc + dy),
                    color=c, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=c, alpha=0.9, lw=1.0),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Decision threshold on ensemble fraud probability")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall vs threshold — holdout")
    ax.text(0.99, -0.16,
            f"τ*= {TAU_STAR:.2f} (max F1) and τ= {TAU_R1:.4f} "
            f"(recall @ FPR≤1%) chosen on strict OOF",
            transform=ax.transAxes, ha="right", fontsize=8, color="0.3")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_threshold_operating_points.png"))
    plt.close(fig)


def fig4_separation_confusion():
    y, p = T["y"], T["mean_prob"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2))
    lo, hi = 0.0, 0.6
    a1.hist(p[y == 0], bins=120, range=(lo, hi), density=True, alpha=0.75,
            color="tab:blue", label=f"normal (n={int((y == 0).sum()):,})")
    a1.hist(p[y == 1], bins=60, range=(lo, hi), density=True, alpha=0.8,
            color="tab:red", label=f"fraud (n={int((y == 1).sum()):,})")
    a1.set_yscale("log")
    a1.axvline(TAU_STAR, color="k", ls="--", lw=1.2)
    a1.text(TAU_STAR + 0.01, a1.get_ylim()[0] * 3, f"τ*={TAU_STAR:.2f}", fontsize=9)
    a1.set_xlabel("Ensemble fraud probability (holdout)")
    a1.set_ylabel("Density (log scale)")
    a1.set_title("Predicted-probability separation")
    a1.legend(fontsize=9)
    pt = (p >= TAU_STAR).astype(int)
    cm = confusion_matrix(y, pt)  # [[TN, FP], [FN, TP]]
    im = a2.imshow(cm, cmap="Blues")
    a2.set_xticks([0, 1], ["pred normal", "pred fraud"])
    a2.set_yticks([0, 1], ["actual normal", "actual fraud"])
    for i in range(2):
        for j in range(2):
            a2.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=13, color="white" if cm[i, j] > cm.max() / 2 else "k")
    a2.set_title(f"Confusion matrix @ τ*={TAU_STAR:.2f} — holdout")
    fig.colorbar(im, ax=a2, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_separation_and_confusion.png"))
    plt.close(fig)


def _box(ax, x, y, w, h, text, fc, ec, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.015",
                                fc=fc, ec=ec, lw=1.5, mutation_aspect=0.35))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight="bold" if bold else "normal", linespacing=1.5)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=18, lw=1.8, color="k"))


def fig5_flow():
    fig, ax = plt.subplots(figsize=(16.5, 9.5))
    ax.set_xlim(-0.03, 1.035)
    ax.set_ylim(-0.045, 1.03)
    ax.axis("off")
    ax.set_title("Information flow — Laya fraud ensemble pipeline",
                 fontsize=14, weight="bold", pad=18)
    _box(ax, 0.010, 0.30, 0.105, 0.42,
         "RAW DATA\nieee-fraud-detection CSV\n1.85M transactions\n999 accounts\n"
         "0.521% fraud (not in repo)", "#f3f4f6", "#9ca3af", fs=8.5)
    _box(ax, 0.140, 0.30, 0.105, 0.42,
         "01_inspect_data.py\nschema, dtypes,\nimbalance, account\ndistribution",
         "#e0f2fe", "#0284c7", fs=8.5)
    _box(ax, 0.270, 0.30, 0.135, 0.42,
         "02_preprocess_and_split.py\n16 CAUSAL features per txn\n"
         "(z-score, velocity 1h/24h/7d, time,\ntime-of-day — prior txns only)\n"
         "StratifiedGroupKFold on cc_num\n85% dev → 5 folds | 15% holdout",
         "#e0f2fe", "#0284c7", fs=8.5)
    _box(ax, 0.430, 0.20, 0.215, 0.62,
         "03_train_5fold_laya.py — 5 independent LayaFraudDecisionNet\n"
         "each model: unseen-folds train → val fold, best by val PR-AUC\n\n"
         "MODEL (per split, ~0.65M params)\n"
         "  16-d features × last 256 txns (+ target)\n"
         "  → encoder: Linear · LayerNorm · GELU + positional emb\n"
         "  → 4× causal Transformer (SDPA, is_causal=True)\n"
         "  → target txn queries history via cross-attention\n"
         "  → MLP → fraud logit z_k\n\n"
         "focal loss γ=2 · AdamW lr 3e-4 · cosine · bf16\n"
         "6 epochs · per-split StandardScaler",
         "#fef3c7", "#d97706", fs=8.5)
    _box(ax, 0.680, 0.20, 0.165, 0.62,
         "04_ensemble_infer_and_eval.py\n"
         "per txn: 5 probabilities p_0..p_4\n\n"
         "ensemble = mean(p_k)  (primary)\n"
         "         = PR-AUC-weighted logit (alt)\n\n"
         "uncertainty = std(p_0..p_4) per txn\n\n"
         "thresholds from STRICT OOF:\n"
         "τ* = 0.46  → max F1\n"
         "τ  = 0.1645 → recall @ FPR ≤ 1%",
         "#e0f2fe", "#0284c7", fs=8.5)
    _box(ax, 0.870, 0.20, 0.120, 0.62,
         "HOLDOUT RESULT\n(150 unseen accounts)\n\n"
         "PR-AUC 0.9059\nROC-AUC 0.9968\n\n"
         "F1 0.8413 @ τ*\n(P 0.940 / R 0.761)\n\n"
         "recall 94.4%\n@ FPR ≤ 1%",
         "#dcfce7", "#16a34a", fs=8.5, bold=True)
    _box(ax, 0.430, 0.02, 0.415, 0.13,
         "laya_ensemble_predictor.py — predict_transaction(account_history_df)\n"
         "loads 5 ckpts + scalers → recomputes causal features → "
         "{is_fraud, fraud_probability, ensemble_std, threshold}",
         "#ede9fe", "#7c3aed", fs=8.5)
    for x0, x1 in [(0.115, 0.140), (0.245, 0.270), (0.405, 0.430),
                   (0.645, 0.680), (0.845, 0.870)]:
        _arrow(ax, x0, 0.51, x1, 0.51)
    _arrow(ax, 0.540, 0.20, 0.540, 0.15)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.03)
    fig.savefig(os.path.join(FIG_DIR, "fig5_pipeline_flow.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig1_pr_curves()
    fig2_roc_curves()
    fig3_thresholds()
    fig4_separation_confusion()
    fig5_flow()
    print("figures written to", FIG_DIR)
