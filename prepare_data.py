# prepare_data.py
# Build rolling-window samples.csv for a chosen disease definition.
# Adds patient-level train/val/test split to avoid patient overlap across sets.
# Also stores t_event (onset time) and lead_time_mins for true lead-time stratification.
# Location: Project/Code/prepare_data.py

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import config


#############################
# Helpers
#############################
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_diagnoses(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "diagnosis.csv.gz"
    df = pd.read_csv(path, compression="infer", low_memory=False)
    required = {"diagnosisstring", "patientunitstayid", "diagnosisoffset"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"diagnosis.csv.gz missing columns: {sorted(missing)}")
    return df


def read_patients(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "patient.csv.gz"
    df = pd.read_csv(path, compression="infer", low_memory=False, usecols=["patientunitstayid"])
    return df


def split_diagnosisstring(diagnoses: pd.DataFrame) -> pd.DataFrame:
    parts = diagnoses["diagnosisstring"].astype(str).str.split("|", n=2, expand=True)
    diagnoses = diagnoses.copy()
    diagnoses["majorcategory"] = parts[0].str.strip()
    diagnoses["subcategory"] = parts[1].str.strip() if parts.shape[1] > 1 else ""
    diagnoses["diagnosisname"] = parts[2].str.strip() if parts.shape[1] > 2 else ""
    return diagnoses


def filter_disease_rows(diagnoses: pd.DataFrame, disease: config.DiseaseSpec) -> pd.DataFrame:
    df = split_diagnosisstring(diagnoses)

    major = disease.major.strip().lower()
    subcat = disease.subcategory.strip().lower() if disease.subcategory else None

    if subcat is None:
        out = df[df["majorcategory"].str.lower() == major]
    else:
        out = df[
            (df["majorcategory"].str.lower() == major) &
            (df["subcategory"].str.lower() == subcat)
        ]
    return out


def compute_onset_times(disease_rows: pd.DataFrame) -> pd.Series:
    onset = disease_rows.groupby("patientunitstayid")["diagnosisoffset"].min()
    onset = pd.to_numeric(onset, errors="coerce").dropna()
    onset.index = onset.index.astype(int)
    onset = onset.astype(int)
    return onset


def compute_max_offsets_across_tables(data_dir: Path, chunksize: int = 1_000_000) -> pd.Series:
    sources = [
        ("vitalPeriodic.csv.gz", "observationoffset", ["patientunitstayid", "observationoffset"]),
        ("vitalAperiodic.csv.gz", "observationoffset", ["patientunitstayid", "observationoffset"]),
        ("lab.csv.gz", "labresultoffset", ["patientunitstayid", "labresultoffset"]),
        ("intakeOutput.csv.gz", "intakeoutputoffset", ["patientunitstayid", "intakeoutputoffset"]),
        ("respiratoryCare.csv.gz", "respcarestatusoffset", ["patientunitstayid", "respcarestatusoffset"]),
        ("respiratoryCharting.csv.gz", "respchartoffset", ["patientunitstayid", "respchartoffset"]),
        ("infusionDrug.csv.gz", "infusionoffset", ["patientunitstayid", "infusionoffset"]),
        ("medication.csv.gz", "drugstartoffset", ["patientunitstayid", "drugstartoffset"]),
        ("treatment.csv.gz", "treatmentoffset", ["patientunitstayid", "treatmentoffset"]),
        ("nurseAssessment.csv.gz", "nurseassessoffset", ["patientunitstayid", "nurseassessoffset"]),
        ("nurseCare.csv.gz", "nursecareoffset", ["patientunitstayid", "nursecareoffset"]),
        ("nurseCharting.csv.gz", "nursingchartoffset", ["patientunitstayid", "nursingchartoffset"]),
    ]

    max_dict: Dict[int, int] = defaultdict(int)

    for fname, offset_col, usecols in sources:
        path = data_dir / fname
        if not path.exists():
            print(f"WARNING: missing file for coverage, skipping: {fname}")
            continue

        print("#############################")
        print(f"Coverage scan: {fname} ({offset_col})")
        print("#############################")

        reader = pd.read_csv(
            path,
            compression="infer",
            usecols=usecols,
            chunksize=chunksize,
            low_memory=False,
        )

        for chunk in reader:
            if "patientunitstayid" not in chunk.columns or offset_col not in chunk.columns:
                continue

            chunk["patientunitstayid"] = pd.to_numeric(chunk["patientunitstayid"], errors="coerce")
            chunk[offset_col] = pd.to_numeric(chunk[offset_col], errors="coerce")
            chunk = chunk.dropna(subset=["patientunitstayid", offset_col])
            if chunk.empty:
                continue

            chunk["patientunitstayid"] = chunk["patientunitstayid"].astype(int)
            grp = chunk.groupby("patientunitstayid")[offset_col].max()

            for pid, max_off in grp.items():
                if pd.isna(max_off):
                    continue
                max_off_int = int(max_off)
                if max_off_int > max_dict[pid]:
                    max_dict[pid] = max_off_int

    return pd.Series(max_dict, name="max_offset_any_source")


def generate_patient_windows(
    patient_id: int,
    max_offset: int,
    onset_time: Optional[int],
    history_mins: int,
    horizon_mins: int,
    stride_mins: int,
    require_full_horizon: bool,
    stop_after_event: bool,
) -> List[Tuple[int, int, int, float, float]]:
    """
    Returns rows:
      (patientunitstayid, t_end, label, t_event, lead_time_mins)

    t_event is onset_time for this patient (float, NaN if None)
    lead_time_mins = onset_time - t_end for event patients, else NaN
    """
    last_t_end = max_offset
    if require_full_horizon:
        last_t_end = max_offset - horizon_mins

    if last_t_end < history_mins:
        return []

    t_ends = list(range(history_mins, last_t_end + 1, stride_mins))

    t_event = float(onset_time) if onset_time is not None else float("nan")

    rows: List[Tuple[int, int, int, float, float]] = []
    for t_end in t_ends:
        if onset_time is not None and stop_after_event and t_end >= onset_time:
            break

        label = 0
        if onset_time is not None:
            if (t_end < onset_time) and (onset_time <= t_end + horizon_mins):
                label = 1

        lead_time = float(onset_time - t_end) if onset_time is not None else float("nan")
        rows.append((patient_id, t_end, label, t_event, lead_time))

    return rows


def add_patient_level_splits(samples_df: pd.DataFrame) -> pd.DataFrame:
    """
    Patient-level split: all windows from a patient go to the same split.

    Produces train/val/test:
      - First split test using config.TEST_SIZE
      - Then split remaining into train/val using config.VAL_SIZE (or default 0.15)

    Stratify by whether the patient ever has a positive window.
    """
    per_patient = (
        samples_df.groupby("patientunitstayid")["label"]
        .max()
        .astype(int)
        .rename("patient_has_positive")
        .reset_index()
    )

    pids = per_patient["patientunitstayid"].to_numpy()
    strat = per_patient["patient_has_positive"].to_numpy()

    test_size = float(getattr(config, "TEST_SIZE", 0.2))
    val_size = float(getattr(config, "VAL_SIZE", 0.15))
    rs = int(getattr(config, "SPLIT_RANDOM_STATE", 42))

    # Split off test
    trainval_pids, test_pids = train_test_split(
        pids,
        test_size=test_size,
        random_state=rs,
        stratify=strat,
    )

    # Split train/val from remaining
    trainval_strat = per_patient.set_index("patientunitstayid").loc[trainval_pids, "patient_has_positive"].to_numpy()

    # val_size is fraction of total; convert to fraction of trainval
    trainval_frac = 1.0 - test_size
    if trainval_frac <= 0.0:
        raise ValueError("TEST_SIZE too large.")

    val_frac_of_trainval = val_size / trainval_frac
    val_frac_of_trainval = float(np.clip(val_frac_of_trainval, 0.01, 0.8))

    train_pids, val_pids = train_test_split(
        trainval_pids,
        test_size=val_frac_of_trainval,
        random_state=rs,
        stratify=trainval_strat,
    )

    train_set = set(int(x) for x in train_pids.tolist())
    val_set = set(int(x) for x in val_pids.tolist())

    out = samples_df.copy()
    out["split"] = "test"
    out.loc[out["patientunitstayid"].isin(train_set), "split"] = "train"
    out.loc[out["patientunitstayid"].isin(val_set), "split"] = "val"

    return out


#############################
# Main
#############################
def main() -> None:
    set_seeds(config.SEED)

    data_dir = config.EICU_DATA_DIR
    out_samples = config.samples_path(config.DISEASE)
    out_meta = config.run_dir(config.DISEASE) / f"meta__{config.disease_tag(config.DISEASE)}.json"

    print("#############################")
    print("Preparing rolling-window samples")
    print("#############################")
    print(f"Data dir: {data_dir}")
    print(f"Output samples: {out_samples}")

    print("#############################")
    print("Loading diagnosis and patient tables")
    print("#############################")
    diagnoses = read_diagnoses(data_dir)
    patients = read_patients(data_dir)
    all_patients = patients["patientunitstayid"].astype(int).unique().tolist()
    print(f"Total ICU stays in patient.csv.gz: {len(all_patients)}")

    print("#############################")
    print("Computing max coverage per patient (across multiple tables)")
    print("#############################")
    max_offsets = compute_max_offsets_across_tables(data_dir)
    print(f"Patients with any coverage in selected tables: {len(max_offsets)}")

    print("#############################")
    print("Filtering diagnoses for target disease")
    print("#############################")
    disease_rows = filter_disease_rows(diagnoses, config.DISEASE)
    onset_times = compute_onset_times(disease_rows)
    print(f"Patients with at least one matching diagnosis: {len(onset_times)}")

    print("#############################")
    print("Generating rolling windows")
    print("#############################")
    history_mins = config.HISTORY_MINS
    horizon_mins = config.HORIZON_MINS
    stride_mins = config.STRIDE_MINS

    require_full_horizon = config.REQUIRE_FULL_HORIZON
    stop_after_event = True

    all_rows: List[Tuple[int, int, int, float, float]] = []

    eligible_patients = [pid for pid in all_patients if pid in max_offsets.index]
    print(f"Patients present in coverage index: {len(eligible_patients)}")

    for pid in eligible_patients:
        max_off = int(max_offsets.loc[pid])
        onset = int(onset_times.loc[pid]) if pid in onset_times.index else None

        rows = generate_patient_windows(
            patient_id=pid,
            max_offset=max_off,
            onset_time=onset,
            history_mins=history_mins,
            horizon_mins=horizon_mins,
            stride_mins=stride_mins,
            require_full_horizon=require_full_horizon,
            stop_after_event=stop_after_event,
        )
        if rows:
            all_rows.extend(rows)

    samples_df = pd.DataFrame(
        all_rows,
        columns=["patientunitstayid", "t_end", "label", "t_event", "lead_time_mins"],
    )
    if samples_df.empty:
        raise RuntimeError("No samples were generated. Check disease filter and coverage rules.")

    n_total = len(samples_df)
    n_pos = int(samples_df["label"].sum())
    n_neg = int((samples_df["label"] == 0).sum())

    print(f"Generated samples total: {n_total}")
    print(f"Positives: {n_pos}")
    print(f"Negatives: {n_neg}")

    print("#############################")
    print("Balancing negatives to match positives (1:1)")
    print("#############################")
    pos_df = samples_df[samples_df["label"] == 1].copy()
    neg_df = samples_df[samples_df["label"] == 0].copy()

    if n_pos == 0:
        raise RuntimeError(
            "No positive samples generated. This can happen if the disease onset "
            "never falls into any future horizon window. Check the disease definition."
        )

    cap = config.NEGATIVE_WINDOWS_PER_PATIENT_CAP
    if cap is not None:
        neg_df = (
            neg_df.groupby("patientunitstayid", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), cap), random_state=config.SEED))
            .reset_index(drop=True)
        )

    target_neg = int(round(config.NEGATIVE_POSITIVE_RATIO * len(pos_df)))
    target_neg = min(target_neg, len(neg_df))

    neg_df = neg_df.sample(n=target_neg, random_state=config.SEED).reset_index(drop=True)

    balanced_df = (
        pd.concat([pos_df, neg_df], axis=0)
        .sample(frac=1.0, random_state=config.SEED)
        .reset_index(drop=True)
    )

    print(f"Balanced samples total: {len(balanced_df)}")
    print(f"Balanced positives: {int(balanced_df['label'].sum())}")
    print(f"Balanced negatives: {int((balanced_df['label'] == 0).sum())}")

    print("#############################")
    print("Assigning patient-level train/val/test split")
    print("#############################")
    balanced_df = add_patient_level_splits(balanced_df)

    train_n = int((balanced_df["split"] == "train").sum())
    val_n = int((balanced_df["split"] == "val").sum())
    test_n = int((balanced_df["split"] == "test").sum())
    print(f"Train windows: {train_n}")
    print(f"Val windows:   {val_n}")
    print(f"Test windows:  {test_n}")

    print("#############################")
    print("Saving samples and meta")
    print("#############################")
    out_samples.parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(out_samples, index=False)

    meta = {
        "disease": {
            "major": config.DISEASE.major,
            "subcategory": config.DISEASE.subcategory,
        },
        "rolling_window": {
            "history_hours": config.HISTORY_HRS,
            "horizon_hours": config.HORIZON_HRS,
            "stride_minutes": config.STRIDE_MINS,
            "require_full_horizon": config.REQUIRE_FULL_HORIZON,
            "stop_after_event": stop_after_event,
        },
        "split": {
            "type": "patient_level_train_val_test",
            "test_size": float(getattr(config, "TEST_SIZE", 0.2)),
            "val_size": float(getattr(config, "VAL_SIZE", 0.15)),
            "random_state": int(getattr(config, "SPLIT_RANDOM_STATE", 42)),
        },
        "seed": config.SEED,
        "counts": {
            "generated_total": int(n_total),
            "generated_pos": int(n_pos),
            "generated_neg": int(n_neg),
            "balanced_total": int(len(balanced_df)),
            "balanced_pos": int(balanced_df["label"].sum()),
            "balanced_neg": int((balanced_df["label"] == 0).sum()),
            "train_windows": train_n,
            "val_windows": val_n,
            "test_windows": test_n,
            "train_patients": int(balanced_df.loc[balanced_df["split"] == "train", "patientunitstayid"].nunique()),
            "val_patients": int(balanced_df.loc[balanced_df["split"] == "val", "patientunitstayid"].nunique()),
            "test_patients": int(balanced_df.loc[balanced_df["split"] == "test", "patientunitstayid"].nunique()),
        },
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Done.")
    print(f"Saved samples: {out_samples}")
    print(f"Saved meta: {out_meta}")


if __name__ == "__main__":
    main()
