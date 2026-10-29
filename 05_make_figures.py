#!/usr/bin/env python3
"""05_make_figures.py — build the repo figures from saved artifacts (CPU-only, no GPU).

Reads : test_probs.npz, oof_ensemble.npz, ensemble_evaluation_report.json
Writes: figures/fig1_pr_curves_holdout.png
        figures/fig2_roc_curves_holdout.png
        figures/fig3_threshold_operating_points.png
        figures/fig4_separation_and_confusion.png
        figures/fig5_pipeline_flow.png   (model data-flow diagram)
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
    fig, ax = plt.subplots(figsize=(17.5, 10.5))
    ax.set_xlim(-0.03, 1.035)
    ax.set_ylim(-0.095, 1.03)
    ax.axis("off")
    ax.set_title("Information flow inside the ensemble model — input (data + dtypes) → 5 Laya models → outputs",
                 fontsize=13, weight="bold", pad=18)
    # ---------------- INPUT zone ----------------
    _box(ax, 0.010, 0.865, 0.225, 0.095,
         "INPUT — one row per target transaction (batch B, e.g. 1024)",
         "#f3f4f6", "#9ca3af", fs=8.5, bold=True)
    _box(ax, 0.010, 0.635, 0.225, 0.205,
         "ACCOUNT'S LAST 256 PAST TRANSACTIONS\n"
         "x_hist (B, 256, 16) · float32 · front-padded\n"
         "mask (B, 256) · bool (True = valid txn)\n"
         "chronological, oldest → newest", "#e0f2fe", "#0284c7", fs=8)
    _box(ax, 0.010, 0.435, 0.225, 0.135,
         "TARGET TRANSACTION (being judged)\n"
         "x_target (B, 16) · float32 · same 16 features", "#e0f2fe", "#0284c7", fs=8)
    _box(ax, 0.010, 0.055, 0.225, 0.335,
         "THE 16 FEATURES (all float32, causal —\nfrom the account's own PRIOR txns only):\n"
         "  time_delta_sec_log · log_amount · amount_z\n"
         "  exp_mean_amt · exp_std_amt · amt_to_mean_ratio\n"
         "  vel_cnt / vel_sum @ 1h, 24h, 7d   (6)\n"
         "  hod_sin/hod_cos · dow_sin/dow_cos\n"
         "source: int64 unix_time + float64 amount;\n"
         "amount_z via per-split StandardScaler;\n"
         "identifiers (cc_num, …) are NOT features", "#f9fafb", "#d1d5db", fs=7.5)
    # ---------------- MODEL zone ----------------
    _box(ax, 0.275, 0.865, 0.340, 0.095,
         "MODEL — LayaFraudDecisionNet · split k\n"
         "× 5 independent copies (same arch · ~0.65M params;\ncopy k is trained only on the other 4 folds)",
         "#fef3c7", "#d97706", fs=8.5, bold=True)
    _box(ax, 0.275, 0.715, 0.340, 0.115,
         "① TOKEN ENCODING\n"
         "Linear(16→128) → LayerNorm → GELU + learned pos emb;\n"
         "prepend learnable NOHIST token\n"
         "→ h (B, 257, 128) · float32", "#fffbeb", "#f59e0b", fs=8)
    _box(ax, 0.275, 0.560, 0.340, 0.12,
         "② CAUSAL BACKBONE × 4 (Pre-LN Transformer blocks)\n"
         "8-head scaled_dot_product_attention, is_causal + padding mask;\n"
         "FFN 128→256→128 · GELU · dropout 0.1 · residual adds\n"
         "→ h (B, 257, 128) · float32  (each position sees only the past)",
         "#fffbeb", "#f59e0b", fs=8)
    _box(ax, 0.275, 0.425, 0.340, 0.095,
         "③ TARGET TOKEN\n"
         "token_enc(x_target) + pos_target → q (B, 1, 128) · float32",
         "#fffbeb", "#f59e0b", fs=8)
    _box(ax, 0.275, 0.265, 0.340, 0.125,
         "④ LAYA DECISION CROSS-ATTENTION (target queries history)\n"
         "Q = LN(q) proj 8×16;  K,V = LN(h) proj 8×16;  masked SDPA\n"
         "→ (B, 128) + residual q + LayerNorm\n"
         "→ anomaly vector a (B, 128) · float32", "#fffbeb", "#f59e0b", fs=8)
    _box(ax, 0.275, 0.155, 0.340, 0.07,
         "⑤ CONCATENATION   feat = [a ‖ q] → (B, 256) · float32",
         "#fffbeb", "#f59e0b", fs=8)
    _box(ax, 0.275, 0.045, 0.340, 0.075,
         "⑥ MLP HEAD   Linear(256→128) → GELU → Dropout(0.1) → Linear(128→1)\n"
         "→ fraud logit z_k (B,) · float32", "#fffbeb", "#f59e0b", fs=8)
    _box(ax, 0.275, -0.055, 0.340, 0.07,
         "⑦ SIGMOID   p_k = σ(z_k) → (B,) · float32 ∈ (0, 1)",
         "#fef9c3", "#ca8a04", fs=8.5, bold=True)
    # ---------------- ENSEMBLE / OUTPUT zone ----------------
    _box(ax, 0.655, 0.865, 0.330, 0.095,
         "ENSEMBLE — 5 independent split models;\nmean of probabilities is the final fraud score",
         "#f3f4f6", "#9ca3af", fs=8.5, bold=True)
    _box(ax, 0.655, -0.055, 0.330, 0.07,
         "5 models × every target   p_0 … p_4 → (N, 5) · float32\n"
         "[N = #target txns; holdout N = 269,993]",
         "#e0f2fe", "#0284c7", fs=8, bold=True)
    _box(ax, 0.655, 0.095, 0.330, 0.20,
         "AGGREGATION (per target txn · float32)\n"
         "  mean_prob = mean(p_0…p_4)        ← primary score\n"
         "  p_w = σ( Σ_k w_k·z_k / Σ_k w_k ),   w_k = val PR-AUC of split k\n"
         "  uncertainty = std(p_0…p_4)   (epistemic)", "#e0f2fe", "#0284c7", fs=8)
    _box(ax, 0.655, 0.365, 0.330, 0.315,
         "OUTPUT — per target transaction\n"
         "  is_fraud:          int (0/1) = 1{ mean_prob ≥ 0.46 }\n"
         "  fraud_probability: float32 ∈ (0, 1)\n"
         "  ensemble_std:      float32 (inter-model disagreement)\n"
         "  threshold:         0.46 (max-F1 on strict OOF)\n"
         "  detail: per-model probs + weighted-logit variant\n"
         "predictor variant: fraud_probability = 50/50 blend of\nmean_prob and p_w",
         "#dcfce7", "#16a34a", fs=8, bold=True)
    _box(ax, 0.655, 0.755, 0.330, 0.13,
         "HOLDOUT SCORE (150 unseen accounts)\n"
         "PR-AUC 0.9059 · ROC-AUC 0.9968\n"
         "F1 0.8413 @ τ* (P 0.940 / R 0.761) · recall 94.4% @ FPR ≤ 1%",
         "#dcfce7", "#16a34a", fs=8)
    # ---------------- arrows ----------------
    _arrow(ax, 0.445, 0.865, 0.445, 0.830)
    _arrow(ax, 0.445, 0.715, 0.445, 0.680)
    _arrow(ax, 0.445, 0.560, 0.445, 0.520)
    _arrow(ax, 0.445, 0.425, 0.445, 0.390)
    _arrow(ax, 0.445, 0.265, 0.445, 0.225)
    _arrow(ax, 0.445, 0.155, 0.445, 0.120)
    _arrow(ax, 0.445, 0.045, 0.445, 0.015)
    _arrow(ax, 0.235, 0.7375, 0.275, 0.7725)   # x_hist -> token encoding
    _arrow(ax, 0.235, 0.5025, 0.275, 0.4725)   # x_target -> target token
    _arrow(ax, 0.615, -0.02, 0.655, -0.02)     # p_k -> (N,5)
    _arrow(ax, 0.820, 0.015, 0.820, 0.095)     # (N,5) -> aggregation
    _arrow(ax, 0.820, 0.295, 0.820, 0.365)     # aggregation -> output
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02)
    fig.savefig(os.path.join(FIG_DIR, "fig5_pipeline_flow.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig1_pr_curves()
    fig2_roc_curves()
    fig3_thresholds()
    fig4_separation_confusion()
    fig5_flow()
    print("figures written to", FIG_DIR)
