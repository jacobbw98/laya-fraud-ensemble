#!/usr/bin/env python
"""Phase 2c (v3): unified strictly-causal features across TWO datasets + v2 splits.

Data (both already downloaded; v1 data/ outputs are untouched):
  A (archive): data_anal/Fraud_Pred/data/raw_archive/{fraudTrain,fraudTest}.csv
               1.85M txns, 999 accounts, 2,507 frauds
  I (IEEE-CIS): data_anal/Fraud_Pred/data/raw_ieee/train_transaction.csv
                590,540 txns, ~474k accounts, 6,024 frauds
    (IEEE TransactionDT = seconds since 2017-11-30 00:00 UTC)

v2 model input = 36 dims: the 15 v1 core features (amount_z inserted at index 2
from amt_raw per-split by the scaler, same convention as v1) +
  dataset_ieee (0/1), d1_log..d14_log (IEEE D1..D14 = days since kth last txn;
  0 for A), prod_cd_1/2/3 (IEEE ProductCD one-hot; 0 for A),
  merch_cnt_24h_log / merch_sum_24h_log (same-merchant 24h velocity).

Splits: GroupShuffleSplit(15% accounts, seed 42) over combined account codes
(A: 1e9+cc_num, I: 2e9+TransactionID) -> data_v3/test.parquet (fold=-1);
StratifiedGroupKFold(5, shuffle, seed 42) on dev rows -> folds 0..4.
All feature math is fully vectorized (no per-account Python loops).

Outputs: data_v3/{dev,test}.parquet, data_v3/meta_v3.json
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU script

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

PIPE = os.path.dirname(os.path.abspath(__file__))
RAW_A = "/home/jacobbw/roell_koastore/data_anal/Fraud_Pred/data/raw_archive"
RAW_I = "/home/jacobbw/roell_koastore/data_anal/Fraud_Pred/data/raw_ieee"
OUT = os.path.join(PIPE, "data_v3")
os.makedirs(OUT, exist_ok=True)

IEEE_TS0 = 1512000000  # epoch seconds of 2017-11-30 00:00 UTC
SEED, TEST_FRAC, N_FOLDS = 42, 0.15, 5
WINDOWS = ((3600.0, "1h"), (86400.0, "24h"), (604800.0, "7d"))
EPS = 1e-6



# v3: 62 IEEE cols with univariate AUC>=0.62 (v3_feature_scan.json); 0 for archive rows
SCAN62 = [
    "C4",
    "C8",
    "C10",
    "C12",
    "V303",
    "V304",
    "D5",
    "V302",
    "C7",
    "V283",
    "V218",
    "V264",
    "C5",
    "V94",
    "V52",
    "V219",
    "C9",
    "V34",
    "V265",
    "V51",
    "V33",
    "V91",
    "V70",
    "V90",
    "V282",
    "V69",
    "V30",
    "V29",
    "V79",
    "V93",
    "V203",
    "card3",
    "V92",
    "V168",
    "V204",
    "V217",
    "V40",
    "V263",
    "V15",
    "V16",
    "V258",
    "V50",
    "C2",
    "V39",
    "V81",
    "V80",
    "V74",
    "V43",
    "V32",
    "V31",
    "V18",
    "V17",
    "V85",
    "V42",
    "V84",
    "V22",
    "V21",
    "V229",
    "D10",
    "V73",
    "V167",
    "V202",
]

# Stored columns (parquet order). Model input = this list with amount_z
# (from amt_raw) inserted at position 2.
STORED = [
    "time_delta_sec_log", "log_amount",
    "exp_mean_amt", "exp_std_amt", "amt_to_mean_ratio",
    "vel_cnt_1h", "vel_sum_1h", "vel_cnt_24h", "vel_sum_24h", "vel_cnt_7d", "vel_sum_7d",
    "hod_sin", "hod_cos", "dow_sin", "dow_cos",
    "dataset_ieee",
    *[f"d{k}_log" for k in range(1, 15)],
    "prod_cd_1", "prod_cd_2", "prod_cd_3",
    "merch_cnt_24h_log", "merch_sum_24h_log",
    *SCAN62,  # v3: scan-selected IEEE cols
]
IN_DIM = len(STORED) + 1  # 36

def _bounds(acc, n):
    starts = np.flatnonzero(np.r_[True, acc[1:] != acc[:-1]])
    counts = np.diff(np.r_[starts, n]).astype(np.int64)
    return starts, counts, np.repeat(starts, counts)


def causal_core(acc, t, amt):
    """Vectorized strictly-causal core features; input sorted by (acc, t)."""
    n = len(t)
    starts, _, row_start = _bounds(acc, n)
    out = {}
    d = np.empty(n)
    d[starts] = 0.0
    d[1:] = np.where(acc[1:] == acc[:-1], t[1:] - t[:-1], 0.0)
    out["time_delta_sec_log"] = np.log1p(d)
    out["log_amount"] = np.log1p(np.maximum(amt, 0.0))
    c1 = np.concatenate([[0.0], np.cumsum(amt)])
    c2 = np.concatenate([[0.0], np.cumsum(amt * amt)])
    b1 = c1[np.maximum(row_start - 1, 0)]
    b2 = c2[np.maximum(row_start - 1, 0)]
    idx = np.arange(n)
    cnt = (idx - row_start).astype(np.float64)
    mu = np.where(cnt > 0, (c1[idx] - b1) / np.maximum(cnt, 1.0), 0.0)
    var = np.where(cnt > 0, (c2[idx] - b2) / np.maximum(cnt, 1.0) - mu * mu, 0.0)
    out["exp_mean_amt"] = mu
    out["exp_std_amt"] = np.sqrt(np.maximum(var, 0.0))
    out["amt_to_mean_ratio"] = np.clip(np.where(mu > EPS, amt / np.maximum(mu, EPS), 0.0), 0.0, 1e4)
    t2 = t.copy()
    t2[starts] = -np.inf  # account blocks -> globally non-decreasing for searchsorted
    b0 = t[row_start]  # each row's account-start timestamp
    for w, tag in WINDOWS:
        q = t - w
        j = np.clip(np.where(b0 >= q, row_start, np.searchsorted(t2, q, side="left")), row_start, idx)
        out[f"vel_cnt_{tag}"] = np.log1p(idx - j)
        out[f"vel_sum_{tag}"] = np.log1p(c1[idx] - c1[j])
    hod = (t % 86400.0) / 86400.0 * 2 * np.pi
    dow = ((t + 4 * 86400.0) % 604800.0) / 604800.0 * 2 * np.pi
    out["hod_sin"], out["hod_cos"] = np.sin(hod), np.cos(hod)
    out["dow_sin"], out["dow_cos"] = np.sin(dow), np.cos(dow)
    return out


def merch_24h(keys, t, amt):
    """Same-(account,merchant) strictly-past 24h velocity. Returns (log1p cnt, log1p sum)
    in the original row order."""
    order = np.lexsort((t, keys))
    tk, ak, kk = t[order], amt[order], keys[order]
    starts, counts, _ = _bounds(kk, len(tk))
    row_start = np.repeat(starts, counts)
    c1 = np.concatenate([[0.0], np.cumsum(ak)])
    ii = np.arange(len(tk))
    tk2 = tk.copy()
    tk2[starts] = -np.inf  # merchant blocks -> globally non-decreasing
    b0 = tk[row_start]
    q = tk - 86400.0
    j = np.clip(np.where(b0 >= q, row_start, np.searchsorted(tk2, q, side="left")), row_start, ii)
    cnt = np.log1p(ii - j)
    summ = np.log1p(c1[ii] - c1[j])
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return cnt[inv], summ[inv]


def build_frame(df, dataset, acc_raw, t, amt, merch_key, extra):
    n = len(df)
    order = np.lexsort((t, acc_raw))
    core = causal_core(acc_raw[order], t[order], amt[order])
    inv = np.empty_like(order)
    inv[order] = np.arange(n)
    F = {k: v[inv] for k, v in core.items()}
    mc, ms = merch_24h(merch_key, t, amt)
    F["merch_cnt_24h_log"], F["merch_sum_24h_log"] = mc, ms
    F["dataset_ieee"] = np.full(n, float(dataset), dtype=np.float32)
    F.update(extra)
    out = pd.DataFrame(F)
    for c in out.columns:
        out[c] = np.nan_to_num(out[c].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    out["amt_raw"] = np.nan_to_num(amt.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    out["ts"] = t.astype(np.int64)
    out["is_fraud"] = df["fraud"].to_numpy(dtype=np.int8)
    out["dataset"] = np.full(n, dataset, dtype=np.int8)
    return out


def main():
    t0 = datetime.now(timezone.utc)

    # ---------------- A: archive ----------------
    fr = []
    for part in ("fraudTrain.csv", "fraudTest.csv"):
        d = pd.read_csv(os.path.join(RAW_A, part), low_memory=False)
        fr.append(d)
    a = pd.concat(fr, ignore_index=True)
    ts = a["unix_time"].to_numpy(dtype=np.float64)
    if np.isnan(ts).any():
        parsed = pd.to_datetime(a["trans_date_trans_time"], utc=True, format="mixed")
        ts = np.where(np.isnan(ts), parsed.astype("int64").to_numpy() // 10**9, ts)
    ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
    a["amt"] = a["amt"].fillna(0.0)
    acc = a["cc_num"].to_numpy(dtype=np.int64)
    mcode, _ = pd.factorize(a["merchant"].to_numpy())
    extra = {f"d{k}_log": np.zeros(len(a), dtype=np.float64) for k in range(1, 15)}
    extra.update({f"prod_cd_{p}": np.zeros(len(a), dtype=np.float64) for p in (1, 2, 3)})
    extra.update({c: np.zeros(len(a), dtype=np.float64) for c in SCAN62})
    a["fraud"] = a["is_fraud"]
    fa = build_frame(a, 0, 10**9 + acc, ts, a["amt"].to_numpy(np.float64),
                     mcode, extra)
    print(f"[A] {len(a)} rows, {len(np.unique(10**9 + acc))} accounts, fraud={int(a['is_fraud'].sum())}")

    # ---------------- I: IEEE-CIS ----------------
    dcols = [f"D{k}" for k in range(1, 15)]
    cols = ["TransactionID", "ProductCD", "TransactionAmt", "TransactionDT",
            "isFraud"] + list(dict.fromkeys([*dcols, *SCAN62]))
    i = pd.read_csv(os.path.join(RAW_I, "train_transaction.csv"), usecols=cols, low_memory=False)
    i["TransactionAmt"] = i["TransactionAmt"].fillna(0.0)
    txid = i["TransactionID"].to_numpy(dtype=np.int64)
    mid = np.arange(len(i))  # no merchant col in this file -> per-row keys => zero merch velocity
    ti = (IEEE_TS0 + i["TransactionDT"].to_numpy(np.int64)).astype(np.float64)
    extra = {f"d{k}_log": np.log1p(np.clip(
        np.nan_to_num(i[f"D{k}"].to_numpy(np.float64), nan=0.0), 0.0, None))
             for k in range(1, 15)}
    extra.update({c: np.nan_to_num(i[c].to_numpy(np.float64), nan=0.0) for c in SCAN62})
    pc = i["ProductCD"].to_numpy()
    extra["prod_cd_1"] = (pc == "W").astype(np.float64)
    extra["prod_cd_2"] = (pc == "C").astype(np.float64)
    extra["prod_cd_3"] = np.isin(pc, ("R", "H", "S")).astype(np.float64)
    i["fraud"] = i["isFraud"]
    fi = build_frame(i, 1, 2 * 10**9 + txid, ti, i["TransactionAmt"].to_numpy(np.float64),
                     mid, extra)
    print(f"[I] {len(i)} rows, {len(np.unique(2 * 10**9 + txid))} accounts, fraud={int(i['isFraud'].sum())}")

    # ---------------- combine, sort, split ----------------
    df = pd.concat([fa, fi], ignore_index=True)
    df = df.sort_values(["acc_raw", "ts"], kind="mergesort").reset_index(drop=True) if "acc_raw" in df else df
    # re-attach numeric account code for boundaries/splits
    code = np.concatenate([10**9 + a["cc_num"].to_numpy(np.int64),
                           2 * 10**9 + i["TransactionID"].to_numpy(np.int64)])
    # rows were concatenated in [A, I] order before sorting -> follow the sort
    pre = pd.DataFrame({"code": code}).reset_index(drop=True)
    order = np.argsort(pre["code"].to_numpy() * 0 + np.r_[fa.index, fi.index], kind="stable")  # placeholder, replaced below
    # simpler: rebuild code column in original concat order, then argsort by (code, ts)
    code_col = np.concatenate([10**9 + a["cc_num"].to_numpy(np.int64),
                               2 * 10**9 + i["TransactionID"].to_numpy(np.int64)])
    ts_col = np.concatenate([fa["ts"].to_numpy(np.int64), fi["ts"].to_numpy(np.int64)])
    perm = np.lexsort((ts_col, code_col))
    df = df.iloc[perm].reset_index(drop=True)
    df["code"] = code_col[perm].astype(np.int64)
    n = len(df)

    starts, counts, row_start = _bounds(df["code"].to_numpy(np.int64), n)
    df["acct_start"] = row_start.astype(np.int32)
    print(f"[comb] {n} rows, {len(starts)} accounts, fraud={int(df['is_fraud'].sum())}")

    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
    unique = np.unique(df["code"].to_numpy())
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SEED)
    tr_i, te_i = next(gss.split(np.zeros(len(unique)), groups=unique))
    test_codes = unique[te_i]
    is_test = np.isin(df["code"].to_numpy(), test_codes)
    y = df["is_fraud"].to_numpy()
    print(f"[split] holdout: {len(test_codes)} accounts, {int(is_test.sum())} rows, "
          f"fraud={int(y[is_test].sum())} (A={int((df['dataset'][is_test] == 0).sum())} rows, "
          f"I={int((df['dataset'][is_test] == 1).sum())} rows)")

    fold = np.full(n, -10, dtype=np.int8)
    dev_idx = np.where(~is_test)[0]
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cc_dev = df["code"].to_numpy()[dev_idx]
    for k, (_, va_i) in enumerate(sgkf.split(np.zeros(len(dev_idx)), groups=cc_dev, y=y[dev_idx])):
        fold[dev_idx[va_i]] = k
    fold[is_test] = -1
    for k in range(N_FOLDS):
        m = fold == k
        print(f"  fold {k}: {int(m.sum())} rows, {len(np.unique(df['code'][m]))} accts, "
              f"fraud={int(y[m].sum())}")

    out_cols = STORED + ["amt_raw", "is_fraud", "dataset", "ts", "row_id", "fold", "acct_start"]
    df["row_id"] = np.arange(n, dtype=np.int32)
    df["fold"] = fold
    save_dev = df.loc[~is_test, out_cols]
    save_te = df.loc[is_test, out_cols]
    save_te.to_parquet(os.path.join(OUT, "test.parquet"), index=False)
    save_dev.to_parquet(os.path.join(OUT, "dev.parquet"), index=False)

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "version": "v3",
        "seed": SEED, "test_frac": TEST_FRAC, "n_folds": N_FOLDS, "max_len": 512,
        "in_dim": IN_DIM,
        "features": STORED[:2] + ["amount_z"] + STORED[2:],
        "features_stored": STORED,
        "amount_z": "inserted at index 2 from amt_raw via per-split StandardScaler (v1 convention)",
        "acc_code": "A: 1e9+cc_num; I: 2e9+TransactionID",
        "ieee_ts_offset": IEEE_TS0,
        "n_rows_total": int(n),
        "n_accounts": int(len(starts)),
        "n_test_accounts": int(len(test_codes)),
        "n_test_rows": int(is_test.sum()),
        "n_test_fraud": int(y[is_test].sum()),
        "n_dev_rows": int((~is_test).sum()),
        "n_dev_fraud": int(y[~is_test].sum()),
        "elapsed_sec": round((datetime.now(timezone.utc) - t0).total_seconds(), 1),
    }
    with open(os.path.join(OUT, "meta_v3.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {meta['elapsed_sec']}s -> {OUT}")


if __name__ == "__main__":
    main()
