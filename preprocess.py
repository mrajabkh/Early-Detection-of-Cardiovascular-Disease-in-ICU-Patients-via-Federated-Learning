# preprocess.py
# Shared preprocessing for rolling-window feature matrices.
# Uses patient-level split if samples.csv contains a 'split' column.
# Location: Project/Code/preprocess.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Set

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#############################
# Leak-prone columns
#############################
def _leak_cols() -> Set[str]:
    return {
        "t_event",
        "lead_time_mins",
        "t_event_missing",
        "lead_time_mins_missing",
    }


def _is_vital_feature(col: str) -> bool:
    # Define vitals as bedside physiologic signals from:
    # - vitalPeriodic (vp_)
    # - vitalAperiodic (va_)
    c = str(col)
    return c.startswith("vp_") or c.startswith("va_")


#############################
# Data containers
#############################
@dataclass
class PreprocessArtifacts:
    imputer: SimpleImputer
    scaler: Optional[StandardScaler]
    valid_feature_names: List[str]


@dataclass
class SplitData:
    X_train_imputed: np.ndarray
    X_test_imputed: np.ndarray
    X_train_scaled: Optional[np.ndarray]
    X_test_scaled: Optional[np.ndarray]
    y_train: np.ndarray
    y_test: np.ndarray
    artifacts: PreprocessArtifacts


#############################
# Loading helpers
#############################
def load_samples(samples_csv_path: str | pd.PathLike) -> pd.DataFrame:
    df = pd.read_csv(samples_csv_path)
    required = {"patientunitstayid", "t_end", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"samples.csv missing required columns: {sorted(missing)}")

    df["patientunitstayid"] = pd.to_numeric(df["patientunitstayid"], errors="coerce")
    df["t_end"] = pd.to_numeric(df["t_end"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["patientunitstayid", "t_end", "label"]).copy()

    df["patientunitstayid"] = df["patientunitstayid"].astype(np.int64)
    df["t_end"] = df["t_end"].astype(np.int64)
    df["label"] = df["label"].astype(np.int64)

    if "split" in df.columns:
        df["split"] = df["split"].astype(str)

    return df


def load_features(features_parquet_path: str | pd.PathLike) -> pd.DataFrame:
    df = pd.read_parquet(features_parquet_path)

    required = {"patientunitstayid", "t_end"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"features.parquet missing required columns: {sorted(missing)}")

    df["patientunitstayid"] = pd.to_numeric(df["patientunitstayid"], errors="coerce")
    df["t_end"] = pd.to_numeric(df["t_end"], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["patientunitstayid", "t_end"]).copy()

    df["patientunitstayid"] = df["patientunitstayid"].astype(np.int64)
    df["t_end"] = df["t_end"].astype(np.int64)
    return df


#############################
# Matrix construction
#############################
def build_xy(
    features_df: pd.DataFrame,
    samples_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    key_cols = ["patientunitstayid", "t_end"]

    merged = samples_df.merge(
        features_df,
        on=key_cols,
        how="left",
        validate="one_to_one",
    )

    if merged.isna().all(axis=1).any():
        bad = merged[merged.isna().all(axis=1)][key_cols].head(10)
        raise ValueError(
            "Some sampled windows have no matching features (all-NaN row after merge). "
            "Example missing keys:\n"
            f"{bad.to_string(index=False)}"
        )

    y = merged["label"].astype(int)

    split_col = None
    if "split" in merged.columns:
        split_col = merged["split"].astype(str)

    # Drop identifiers, label, split, and leak columns if present
    drop_cols = set(key_cols + ["label"])
    if "split" in merged.columns:
        drop_cols.add("split")

    for c in _leak_cols():
        if c in merged.columns:
            drop_cols.add(c)

    feature_cols = [c for c in merged.columns if c not in drop_cols]

    # Numeric only
    X_numeric = merged[feature_cols].select_dtypes(include=[np.number]).copy()

    # Missingness indicators ONLY for vitals (vp_, va_)
    vital_cols = [c for c in X_numeric.columns if _is_vital_feature(c)]
    if vital_cols:
        cols_with_missing = X_numeric[vital_cols].isna().any(axis=0)
        vital_missing_cols = cols_with_missing[cols_with_missing].index.tolist()

        if vital_missing_cols:
            missing_mask = X_numeric[vital_missing_cols].isna().astype(int)
            missing_mask.columns = [f"{c}_missing" for c in missing_mask.columns]
            X = pd.concat([X_numeric, missing_mask], axis=1)
        else:
            X = X_numeric
    else:
        X = X_numeric

    X.index = pd.RangeIndex(start=0, stop=len(X), step=1)
    y.index = X.index
    if split_col is not None:
        split_col.index = X.index

    return X, y, split_col


#############################
# Split + preprocess
#############################
def split_and_preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    split_col: Optional[pd.Series],
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    impute_strategy: str = "median",
    scale_numeric: bool = True,
) -> SplitData:
    if split_col is not None:
        train_mask = (split_col == "train").to_numpy()
        test_mask = (split_col == "test").to_numpy()

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            raise ValueError("split column exists but has no train/test rows. Expected values: train/test.")

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        y_train = y.loc[train_mask]
        y_test = y.loc[test_mask]
    else:
        strat = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=strat,
        )

    valid_mask = ~X_train.isna().all(axis=0)
    valid_feature_names = X_train.columns[valid_mask].tolist()

    X_train_valid = X_train.loc[:, valid_feature_names]
    X_test_valid = X_test.loc[:, valid_feature_names]

    imputer = SimpleImputer(strategy=impute_strategy)

    X_train_imputed = imputer.fit_transform(X_train_valid)
    X_test_imputed = imputer.transform(X_test_valid)

    scaler = None
    X_train_scaled = None
    X_test_scaled = None

    if scale_numeric:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

    artifacts = PreprocessArtifacts(
        imputer=imputer,
        scaler=scaler,
        valid_feature_names=valid_feature_names,
    )

    return SplitData(
        X_train_imputed=X_train_imputed,
        X_test_imputed=X_test_imputed,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train.to_numpy(),
        y_test=y_test.to_numpy(),
        artifacts=artifacts,
    )


#############################
# Convenience wrapper (used by scripts)
#############################
def load_build_split(
    features_parquet_path: str | pd.PathLike,
    samples_csv_path: str | pd.PathLike,
    test_size: float,
    random_state: int,
    stratify: bool,
    impute_strategy: str,
    scale_numeric: bool,
) -> Tuple[pd.DataFrame, pd.Series, SplitData]:
    samples_df = load_samples(samples_csv_path)
    features_df = load_features(features_parquet_path)

    X, y, split_col = build_xy(features_df=features_df, samples_df=samples_df)

    split_data = split_and_preprocess(
        X=X,
        y=y,
        split_col=split_col,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
        impute_strategy=impute_strategy,
        scale_numeric=scale_numeric,
    )
    return X, y, split_data


#############################
# Feature index map (for top-K slicing)
#############################
def make_feature_index_map(valid_feature_names: List[str]) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(valid_feature_names)}
