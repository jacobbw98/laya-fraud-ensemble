#!/usr/bin/env python
"""Phase 2: Leakage-free chronological sequence features + 5-split generation.

Primary dataset: synthetic credit-card fraud (data/raw_archive/fraudTrain.csv +
fraudTest.csv, 1.85M transactions). Groups by cc_num, sorts by [cc_num, ts],
computes strictly causal per-transaction features (past transactions only),
carves a 15% account-level holdout test set, and assigns 5 StratifiedGroupKFold
folds over the remaining 85% dev accounts (grouped by cc_num, stratified by
is_fraud, seed 42).

Outputs (under laya_ensemble_pipeline/data/):
  dev.parquet   (dev rows, all features, fold in 0..4)
  test.parquet  (locked holdout rows, fold = -1)
  fold_mapping.parquet  (cc_num -> partition + fold)
  meta.json     (feature names, dtypes, counts, constants)

No raw identifiers (cc_num, names, card numbers, addresses) ever become
model input features; cc_num is used only for grouping/split assignment.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # hard pin: single GPU

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

PIPE = os.path.dirname(os.path.abspath(__file__))   # laya_ensemble_pipeline/
BASE = os.path.dirname(PIPE)                        # Fraud_Pred/
RAW_DIR = os.path.join(BASE, "data")                # raw extraction lives here
OUT_DIR = os.path.join(PIPE, "data")
os.makedirs(OUT_DIR, exist_ok=True)

RAW_PARTS = ["fraudTrain.csv", "fraudTest.csv"]
SEED = 42
TEST_FRAC = 0.15
N_FOLDS = 5
WINDOWS = [(3600.0, "1h"), (86400.0, "24h"), (604800.0, "7d")]

# Fixed (non-fit) transforms: log1p where noted. amount_z is fit per-split in Phase 3.
FEATURES = [
    "time_delta_sec_log",   # log1p(sec since previous txn of same account; 0 for first)
    "log_amount",           # log1p(amt)
    "amount_z",             # z-score of amt — scaled per-split in Phase 3 from amt_raw
    "exp_mean_amt",         # causal expanding mean of PAST amounts (0 if none)
    "exp_std_amt",          # causal expanding std of PAST amounts (0 if none)
    "amt_to_mean_ratio",    # amt / causal expanding mean (0 if no past)
    "vel_cnt_1h", "vel_sum_1h",
    "vel_cnt_24h", "vel_sum_24h",
    "vel_cnt_7d", "vel_sum_7d",
    "hod_sin", "hod_cos",
    "dow_sin", "dow_cos",
]
FEATURES_STORED = [f for f in FEATURES if f != "amount_z"]
VELOCITY_WINDOWS = WINDOWS  # (sec, tag)

EPS = 1e-6


def load_raw():
    frames = []
    for part in RAW_PARTS:
        p = os.path.join(RAW_DIR, "raw_archive", part)
        df = pd.read_csv(p, low_memory=False)
        df["part"] = part
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df


def main():
    t0 = datetime.now(timezone.utc)
    df = load_raw()
    n = len(df)
    print(f"[load] {n} rows, {df['cc_num'].nunique()} accounts, "
          f"fraud={int(df['is_fraud'].sum())} ({df['is_fraud'].mean():.5f})")

    # ---- 1. Timestamps -> UTC epoch seconds; sort by [cc_num, ts] ----
    ts = df["unix_time"].to_numpy(dtype=np.float64)
    if np.isnan(ts).any():
        parsed = pd.to_datetime(df["trans_date_trans_time"], utc=True, format="mixed")
        ts = np.where(np.isnan(ts), parsed.astype("int64").to_numpy() // 10**9, ts)
    df["ts"] = ts.astype(np.int64)
    df = df.sort_values(["cc_num", "ts"], kind="mergesort").reset_index(drop=True)

    cc = df["cc_num"].to_numpy()
    amt = df["amt"].to_numpy(dtype=np.float64)
    t = df["ts"].to_numpy(dtype=np.float64)
    n = len(df)

    # Group boundaries (positional; data sorted by cc_num)
    starts = np.flatnonzero(np.r_[True, cc[1:] != cc[:-1]])
    n_accounts = len(starts)
    ends = np.r_[starts[1:], n]
    print(f"[sort] {n_accounts} accounts, boundaries ok")

    # ---- 2. Causal features (strictly PAST transactions only) ----
    feat = {}

    # time delta to previous txn of same account (0 for first)
    d = np.empty(n, dtype=np.float64)
    for a in range(n_accounts):
        s, e = starts[a], ends[a]
        seg = t[s:e]
        d[s] = 0.0
        d[s + 1:e] = seg[1:] - seg[:-1]
    feat["time_delta_sec_log"] = np.log1p(d)

    feat["log_amount"] = np.log1p(np.maximum(amt, 0.0))

    # expanding mean/std of PAST amounts (shifted)
    em = np.zeros(n); es = np.zeros(n)
    for a in range(n_accounts):
        s, e = starts[a], ends[a]
        m = amt[s:e]
        c1 = np.concatenate([[0.0], np.cumsum(m)])
        c2 = np.concatenate([[0.0], np.cumsum(m * m)])
        cnt = np.arange(0, e - s, dtype=np.float64)          # #past txns for row s+i
        mu = c1[: e - s] / np.maximum(cnt, 1)
        var = np.maximum(c2[: e - s] / np.maximum(cnt, 1) - mu * mu, 0.0)
        mu[0] = 0.0
        em[s:e] = mu
        es[s:e] = np.sqrt(var)
    feat["exp_mean_amt"] = em
    feat["exp_std_amt"] = es
    feat["amt_to_mean_ratio"] = np.where(em > EPS, amt / np.maximum(em, EPS), 0.0)
    feat["amt_to_mean_ratio"] = np.clip(feat["amt_to_mean_ratio"], 0.0, 1e4)

    # rolling velocity: count + sum of PAST txns in prior 1h / 24h / 7d
    for wsec, tag in VELOCITY_WINDOWS:
        cnt = np.zeros(n); tot = np.zeros(n)
        for a in range(n_accounts):
            s, e = starts[a], ends[a]
            seg_t = t[s:e]; seg_a = amt[s:e]
            pref = np.concatenate([[0.0], np.cumsum(seg_a)])
            i = np.arange(e - s)
            j = np.searchsorted(seg_t, seg_t - wsec, side="left")  # first idx with time >= t_i - w
            cnt[s:e] = i - j
            tot[s:e] = pref[i] - pref[j]
        feat[f"vel_cnt_{tag}"] = np.log1p(cnt)
        feat[f"vel_sum_{tag}"] = np.log1p(tot)

    # cyclical time (UTC)
    hod = (t % 86400.0) / 86400.0 * 2 * np.pi
    dow = ((t + 4 * 86400.0) % 604800.0) / 604800.0 * 2 * np.pi  # stable cyclical week code
    feat["hod_sin"] = np.sin(hod)
    feat["hod_cos"] = np.cos(hod)
    feat["dow_sin"] = np.sin(dow)
    feat["dow_cos"] = np.cos(dow)

    # ---- 3. Assemble feature frame ----
    X = np.stack([feat[f].astype(np.float32) for f in FEATURES_STORED], axis=1)
    out = pd.DataFrame(X, columns=FEATURES_STORED)
    out["cc_num"] = df["cc_num"]
    out["ts"] = df["ts"].astype(np.int64)
    out["amt_raw"] = amt.astype(np.float32)
    out["is_fraud"] = df["is_fraud"].to_numpy(dtype=np.int8)
    out["part"] = df["part"].to_numpy()
    out["row_id"] = np.arange(n, dtype=np.int32)
    print(f"[feat] X shape {out.shape}")

    # ---- 4. Splits: 15% account-level holdout, then StratifiedGroupKFold(5) ----
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

    unique_accounts = np.unique(cc)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SEED)
    tr_idx, te_idx = next(gss.split(np.zeros(len(unique_accounts)), groups=unique_accounts))
    test_accounts = set(unique_accounts[te_idx].tolist())
    test_arr = np.fromiter(test_accounts, dtype=cc.dtype)
    partition = np.where(np.isin(cc, test_arr), -1, 0)  # -1 test, 0 dev
    n_test_rows = int((partition == -1).sum())
    print(f"[split] holdout: {len(test_accounts)} accounts, {n_test_rows} rows "
          f"({n_test_rows / n:.3f}), fraud rate {out.loc[partition == -1, 'is_fraud'].mean():.5f}")

    dev_mask = partition == 0
    dev_cc = cc[dev_mask]
    dev_y = out["is_fraud"].to_numpy()[dev_mask]
    dev_rows = np.where(dev_mask)[0]
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.full(n, -10, dtype=np.int8)
    for k, (tr_i, va_i) in enumerate(sgkf.split(np.zeros(dev_cc.shape[0]), groups=dev_cc, y=dev_y)):
        fold[dev_rows[va_i]] = k
    fold[partition == -1] = -1  # sentinel for holdout rows
    for k in range(N_FOLDS):
        m = fold == k
        print(f"  fold {k}: {int(m.sum())} rows, {len(np.unique(cc[m]))} accounts, "
              f"fraud={int(out['is_fraud'].to_numpy()[m].sum())}")

    out["fold"] = fold
    assert (out.loc[dev_mask, "fold"] >= 0).all() and (out.loc[partition == -1, "fold"] == -1).all()

    # ---- 5. Save ----
    try:
        import pyarrow  # noqa
        use_parquet = True
    except ImportError:
        use_parquet = False
    print(f"[save] parquet={use_parquet}")

    def save(df_, name):
        if use_parquet:
            df_.to_parquet(os.path.join(OUT_DIR, name), index=False)
        else:  # pragma: no cover
            df_.to_pickle(os.path.join(OUT_DIR, name.replace(".parquet", ".pkl")))

    save(out[partition == -1], "test.parquet")
    save(out[partition == 0], "dev.parquet")
    mapping = pd.DataFrame(
        {
            "cc_num": unique_accounts,
            "partition": np.where(np.isin(unique_accounts, test_arr), "test", "dev"),
        }
    )
    mapping["fold"] = mapping["cc_num"].map(dict(zip(cc[dev_mask], fold[dev_mask]))).fillna(-1).astype(np.int8)
    save(mapping, "fold_mapping.parquet")

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "test_frac": TEST_FRAC,
        "n_folds": N_FOLDS,
        "max_len": 256,
        "n_rows_total": int(n),
        "n_accounts": int(n_accounts),
        "n_test_accounts": int(len(test_accounts)),
        "n_test_rows": int(n_test_rows),
        "n_dev_rows": int(dev_mask.sum()),
        "features": FEATURES,  # model input order (amount_z computed from amt_raw per split)
        "features_stored": FEATURES_STORED,
        "velocity_windows_sec": [w for w, _ in VELOCITY_WINDOWS],
        "row_id": "int32, stable across dev.parquet/test.parquet (0-based global row)",
        "fold": "0..4 = dev validation fold; -1 = holdout test",
        "dow_encoding": "dow = ((t + 4*86400) % 604800)/604800 * 2pi  (stable cyclical code)",
        "elapsed_sec": round((datetime.now(timezone.utc) - t0).total_seconds(), 1),
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] elapsed {meta['elapsed_sec']}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
