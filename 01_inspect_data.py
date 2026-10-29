#!/usr/bin/env python
"""Phase 1: Data discovery & schema mapping for Fraud_Pred datasets.

Inspects every dataset file under Fraud_Pred/ (zips + extracted CSVs):
column names, dtypes, row counts, missing values, candidate account
identifier / amount / timestamp / fraud-label columns, class imbalance,
and unique-account counts.

Writes: logs/data_inspection_report.json
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # hard pin: single GPU, never multi-GPU

import json
import glob
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Fraud_Pred/
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(OUT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Columns that would identify the account / amount / time / label.
CAND_ACCOUNT = ["cc_num", "account_id", "customer_id", "anon_name", "TransactionID"]
CAND_AMOUNT = ["amt", "amount", "TransactionAmt"]
CAND_TIME = ["trans_date_trans_time", "unix_time", "TransactionDT", "time"]
CAND_LABEL = ["is_fraud", "isFraud", "target"]


def fmt(o, depth=0, prefix=""):
    return o if depth > 2 else o


def inspect_file(path, nrows=None):
    out = {"path": path, "size_mb": round(os.path.getsize(path) / 1e6, 1)}
    if path.endswith(".csv"):
        with open(path) as f:
            header = f.readline().strip()
        cols = [c.strip() for c in header.split(",")]
        out["n_columns"] = len(cols)
        out["columns"] = cols
        sample = pd.read_csv(path, nrows=nrows or 200_000, low_memory=False)
        out["n_rows_sampled"] = len(sample)
        # Full row count (fast, no parse): count newlines
        with open(path, "rb") as f:
            total = sum(1 for _ in f)
        out["n_rows_total"] = total - 1
        # Dtypes / missing on the sample
        out["dtypes"] = {c: str(sample[c].dtype) for c in sample.columns}
        out["missing"] = {c: int(sample[c].isna().sum()) for c in sample.columns if sample[c].isna().any()}
        # Value-range probes on key-ish columns
        for c in sample.columns:
            if c in CAND_AMOUNT:
                out["amount_probe"] = {"col": c, "min": float(sample[c].min()), "max": float(sample[c].max()), "median": float(sample[c].median())}
            if c in CAND_TIME and sample[c].dtype.kind in "iuf":
                out["time_probe"] = {"col": c, "min": int(sample[c].min()), "max": int(sample[c].max())}
            if c in CAND_LABEL:
                out["label_probe"] = {"col": c, "value_counts": {int(k): int(v) for k, v in sample[c].value_counts().items()}}
            if c in CAND_ACCOUNT:
                out["account_probe"] = {"col": c, "unique_in_sample": int(sample[c].nunique())}
        return out
    out["note"] = "non-csv; listed only"
    return out


def main():
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "base_dir": BASE,
        "files": [],
        "datasets": {},
    }

    # 1. All files at top level
    for p in sorted(glob.glob(os.path.join(BASE, "*"))):
        if os.path.isfile(p):
            report["files"].append({"name": os.path.basename(p), "size_mb": round(os.path.getsize(p) / 1e6, 1)})

    # 2. Inspect every extracted CSV
    csvs = sorted(glob.glob(os.path.join(BASE, "data", "raw_*", "*.csv")))
    for p in csvs:
        name = os.path.basename(p)
        big = os.path.getsize(p) > 150e6  # IEEE transaction files: sample-based
        report["datasets"][name] = inspect_file(p, nrows=200_000 if big else None)
        print(f"[inspect] {name}: rows={report['datasets'][name].get('n_rows_total')}")

    # 3. Deep statistics on the primary (synthetic) dataset: full-pass label/account stats
    print("[stats] full-pass statistics on fraudTrain.csv + fraudTest.csv ...")
    stat_frames = []
    for part in ["raw_archive/fraudTrain.csv", "raw_archive/fraudTest.csv"]:
        path = os.path.join(BASE, "data", part)
        df = pd.read_csv(
            path,
            usecols=["cc_num", "amt", "trans_date_trans_time", "unix_time", "is_fraud"],
            low_memory=False,
        )
        df["part"] = part
        stat_frames.append(df)
    s = pd.concat(stat_frames, ignore_index=True)
    label_counts = s["is_fraud"].value_counts().to_dict()
    fraud_n = int(label_counts.get(1, 0))
    normal_n = int(label_counts.get(0, 0))
    primary = {
        "name": "synthetic_credit_card (fraudTrain.csv + fraudTest.csv)",
        "n_rows_total": int(len(s)),
        "rows_by_part": {k: int(v) for k, v in s.groupby("part").size().items()},
        "n_accounts": int(s["cc_num"].nunique()),
        "fraud_count": fraud_n,
        "normal_count": normal_n,
        "fraud_rate": fraud_n / len(s),
        "imbalance_ratio_normal_to_fraud": round(normal_n / max(fraud_n, 1), 1),
        "amt_min": float(s["amt"].min()),
        "amt_max": float(s["amt"].max()),
        "amt_median": float(s["amt"].median()),
        "amt_negatives": int((s["amt"] < 0).sum()),
        "unix_time_min": int(s["unix_time"].min()),
        "unix_time_max": int(s["unix_time"].max()),
        "unix_time_min_iso": datetime.fromtimestamp(int(s["unix_time"].min()), tz=timezone.utc).isoformat(),
        "unix_time_max_iso": datetime.fromtimestamp(int(s["unix_time"].max()), tz=timezone.utc).isoformat(),
        "unix_time_missing": int(s["unix_time"].isna().sum()),
        "trans_date_missing": int(s["trans_date_trans_time"].isna().sum()),
        "amt_missing": int(s["amt"].isna().sum()),
        "label_missing": int(s["is_fraud"].isna().sum()),
        "transactions_per_account": {
            "min": int(s.groupby("cc_num").size().min()),
            "max": int(s.groupby("cc_num").size().max()),
            "median": int(s.groupby("cc_num").size().median()),
            "mean": round(float(s.groupby("cc_num").size().mean()), 2),
        },
        "fraud_by_part": {k: int(v) for k, v in s.groupby("part")["is_fraud"].sum().items()},
        "schema": {
            "account_id": "cc_num",
            "amount": "amt",
            "timestamp": "trans_date_trans_time (str) + unix_time (epoch s)",
            "label": "is_fraud (0/1)",
            "identity_cols_never_as_features": ["first", "last", "gender", "street", "city", "state", "zip", "lat", "long", "job", "dob", "merchant", "trans_num", "merch_lat", "merch_long"],
        },
    }
    report["primary_dataset"] = primary
    print(json.dumps(primary, indent=2, default=str))

    out_path = os.path.join(LOG_DIR, "data_inspection_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[done] report -> {out_path}")


if __name__ == "__main__":
    main()
