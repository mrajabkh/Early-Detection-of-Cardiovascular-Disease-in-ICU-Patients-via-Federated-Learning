# run_gru_sweep.py
# Run GRU experiments with top-K feature limits

from __future__ import annotations
from typing import List, Optional

import pandas as pd
import numpy as np

import config
from train_eval_gru import TrainConfig, train_and_eval


def _round_3dp(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        if pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = df2[c].round(3)
    return df2


def run_sweep(
    ks: List[Optional[int]],
    rank_path: Optional[str] = None,
) -> pd.DataFrame:

    disease = config.DISEASE
    cfg = TrainConfig()

    rows = []

    for k in ks:
        k_name = "all" if k is None else str(int(k))

        print("========================================")
        print(f"GRU run | top_k={k_name}")
        print(f"rank_path={rank_path}")
        print("========================================")

        out = train_and_eval(disease=disease, cfg=cfg, top_k=k, rank_path=rank_path)

        strat = out.get("strat_test", {})

        # Debug once per run: helps catch key mismatches
        if isinstance(strat, dict) and strat:
            print("Stratified keys:", sorted(strat.keys()))
        else:
            print("Stratified keys: (empty)")

        row = {
            "top_k": k_name,
            "n_features": out["extra"]["n_features"],
            "train_auroc": out["train"]["auroc"],
            "train_auprc": out["train"]["auprc"],
            "val_auroc": out["val"]["auroc"],
            "val_auprc": out["val"]["auprc"],
            "test_auroc": out["test"]["auroc"],
            "test_auprc": out["test"]["auprc"],
            # move these after test_auprc
            "cpu_peak_mib": out["extra"]["cpu_peak_mib"],
            "runtime_sec": out["extra"]["runtime_sec"],
            # stratified test performance
            "test_auroc_0_2h": strat.get("auroc_0_2h", np.nan),
            "test_auprc_0_2h": strat.get("auprc_0_2h", np.nan),
            "test_auroc_2_6h": strat.get("auroc_2_6h", np.nan),
            "test_auprc_2_6h": strat.get("auprc_2_6h", np.nan),
            "test_auroc_6_12h": strat.get("auroc_6_12h", np.nan),
            "test_auprc_6_12h": strat.get("auprc_6_12h", np.nan),
            # optional 24h bucket (will be NaN on 12h horizon)
            "test_auroc_12_24h": strat.get("auroc_12_24h", np.nan),
            "test_auprc_12_24h": strat.get("auprc_12_24h", np.nan),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    col_order = [
        "top_k",
        "n_features",
        "train_auroc",
        "train_auprc",
        "val_auroc",
        "val_auprc",
        "test_auroc",
        "test_auprc",
        "cpu_peak_mib",
        "runtime_sec",
        "test_auroc_0_2h",
        "test_auprc_0_2h",
        "test_auroc_2_6h",
        "test_auprc_2_6h",
        "test_auroc_6_12h",
        "test_auprc_6_12h",
        "test_auroc_12_24h",
        "test_auprc_12_24h",
    ]
    df = df[col_order]

    df_out = _round_3dp(df)

    out_path = config.gru_results_path(disease)
    df_out.to_csv(out_path, index=False)

    print("========================================")
    print("Summary:")
    print(df_out.to_string(index=False))
    print("----------------------------------------")
    print(f"Saved results CSV: {out_path}")
    print("========================================")

    return df_out


if __name__ == "__main__":
    ks = [20, 40, 60, 80, 100, None]
    rank_path = str(config.stability_combined_path(config.DISEASE))
    run_sweep(ks=ks, rank_path=rank_path)
