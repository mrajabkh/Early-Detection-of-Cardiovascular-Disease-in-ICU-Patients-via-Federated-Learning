# config.py
# Central config for the rolling-window ICU prediction pipeline.
# Location: Project/Code/config.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any


#############################
# Project layout (auto-detected)
#############################
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent

EICU_DATA_DIR = PROJECT_ROOT / "eICU(v2.0)"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"


#############################
# Task definition (disease label)
#############################
@dataclass(frozen=True)
class DiseaseSpec:
    major: str
    subcategory: Optional[str] = None


DISEASE = DiseaseSpec(
    major="cardiovascular",
    subcategory="cardiac arrest",
)


#############################
# Rolling window settings
#############################
HISTORY_HRS = 24
HORIZON_HRS = 24
STRIDE_MINS = 60

HISTORY_MINS = HISTORY_HRS * 60
HORIZON_MINS = HORIZON_HRS * 60


#############################
# Sample generation / balancing
#############################
SEED = 42
REQUIRE_FULL_HORIZON = True
NEGATIVE_POSITIVE_RATIO = 1.0
NEGATIVE_WINDOWS_PER_PATIENT_CAP: Optional[int] = None


#############################
# Train/test split + preprocessing
#############################
TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42
STRATIFY_SPLIT = True

IMPUTE_STRATEGY = "median"
SCALE_NUMERIC = True

SHAP_BACKGROUND_N = 2000


#############################
# Feature selection (Top-K)
#############################
TOPK_LIST: List[int] = [20, 40, 60, 80, 100]


#############################
# Respiratory and nursing charting feature extraction
#############################
RESPCHART_TOP_LABELS = 40  # unknown max
NURSECHART_TOP_LABELS = 48  # 48 max


#############################
# Stability selection settings
#############################
# If True, run_topk_models.py will prefer stability_combined ranking when present.
USE_STABILITY_RANKING = True

# Number of bootstrap runs for stability selection
STAB_N_BOOTSTRAPS = 15

# Bootstrap sample size as a fraction of training set
STAB_BOOTSTRAP_FRAC = 0.7

# Stability frequency is computed relative to this reference Top-K
# Example: a feature appears in top-100 in 12/15 runs => freq 0.80
STAB_TOPK_REF = 100

# Combine rule: keep features if (freq_lgbm >= THR) OR (freq_rf >= THR)
STAB_FREQ_THRESHOLD = 0.6

# SHAP parameters (LGBM selector)
STAB_SHAP_BACKGROUND_N = 1000  # per bootstrap

# RF selector importance mode:
# - "mdi" = impurity-based importance (fast, default)
# - "permutation" = permutation importance (slow unless restricted)
RF_IMPORTANCE_MODE = "mdi"

# Keep permutation mode bounded
# (only permute top-M features by RF MDI to avoid hours of runtime)
PERM_N_REPEATS = 1
PERM_SCORING = "average_precision"
PERM_MAX_SAMPLES = 2000
PERM_MAX_FEATURES = 200


#############################
# Model toggles (default only 3 on)
#############################
MODEL_ENABLED = {
    "knn": True,
    "random_forest": True,
    "lightgbm": True,
    "log_reg": False,
    "svm": False,
    "decision_tree": False,
    "naive_bayes": False,
    "xgboost": False,
    "catboost": False,
}


#############################
# Tuned model params (locked in)
#############################
LGBM_TUNED_PARAMS: Dict[str, Any] = {
    "subsample": 1.0,
    "num_leaves": 63,
    "n_estimators": 200,
    "min_child_samples": 20,
    "max_depth": 20,
    "learning_rate": 0.1,
    "colsample_bytree": 0.7,
    "objective": "binary",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

RF_TUNED_PARAMS: Dict[str, Any] = {
    "n_estimators": 500,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 0.5,
    "max_depth": None,
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced",
}


#############################
# Output naming helpers
#############################
def slugify(s: str) -> str:
    out = s.strip().lower()
    out = out.replace("/", "_").replace("\\", "_")
    out = out.replace("|", "_")
    out = out.replace(" ", "_")
    while "__" in out:
        out = out.replace("__", "_")
    return out


def disease_tag(disease: DiseaseSpec = DISEASE) -> str:
    major = slugify(disease.major)
    sub = slugify(disease.subcategory) if disease.subcategory else "all"
    return f"{major}__{sub}"


def run_name(disease: DiseaseSpec = DISEASE) -> str:
    return (
        f"{disease_tag(disease)}"
        f"__hist{HISTORY_HRS}h"
        f"__hor{HORIZON_HRS}h"
        f"__stride{STRIDE_MINS}m"
    )


def run_dir(disease: DiseaseSpec = DISEASE) -> Path:
    d = OUTPUTS_DIR / run_name(disease)
    d.mkdir(parents=True, exist_ok=True)
    return d


#############################
# Output filenames (include disease name)
#############################
def samples_filename(disease: DiseaseSpec = DISEASE) -> str:
    return f"samples__{disease_tag(disease)}.csv"


def features_filename(disease: DiseaseSpec = DISEASE) -> str:
    return f"features__{disease_tag(disease)}.parquet"


def shap_filename(disease: DiseaseSpec = DISEASE) -> str:
    return f"shap_importance__{disease_tag(disease)}.csv"


def results_filename(disease: DiseaseSpec = DISEASE) -> str:
    return f"ML_results__{disease_tag(disease)}.csv"


def stability_lgbm_filename(disease: DiseaseSpec = DISEASE) -> str:
    return f"stability_lgbm_shap__{disease_tag(disease)}.csv"


def stability_rf_filename(disease: DiseaseSpec = DISEASE) -> str:
    return f"stability_rf_perm__{disease_tag(disease)}.csv"


def stability_combined_filename(disease: DiseaseSpec = DISEASE) -> str:
    return f"stability_combined__{disease_tag(disease)}.csv"


# NEW: GRU output filenames
def gru_results_filename(disease: DiseaseSpec = DISEASE) -> str:
    return f"gru_results__{disease_tag(disease)}.csv"


def gru_checkpoint_filename(disease: DiseaseSpec = DISEASE) -> str:
    return f"gru_checkpoint__{disease_tag(disease)}.pt"


def samples_path(disease: DiseaseSpec = DISEASE) -> Path:
    return run_dir(disease) / samples_filename(disease)


def features_path(disease: DiseaseSpec = DISEASE) -> Path:
    return run_dir(disease) / features_filename(disease)


def shap_path(disease: DiseaseSpec = DISEASE) -> Path:
    return run_dir(disease) / shap_filename(disease)


def results_path(disease: DiseaseSpec = DISEASE) -> Path:
    return run_dir(disease) / results_filename(disease)


def stability_lgbm_path(disease: DiseaseSpec = DISEASE) -> Path:
    return run_dir(disease) / stability_lgbm_filename(disease)


def stability_rf_path(disease: DiseaseSpec = DISEASE) -> Path:
    return run_dir(disease) / stability_rf_filename(disease)


def stability_combined_path(disease: DiseaseSpec = DISEASE) -> Path:
    return run_dir(disease) / stability_combined_filename(disease)


# NEW: GRU output paths
def gru_results_path(disease: DiseaseSpec = DISEASE) -> Path:
    return run_dir(disease) / gru_results_filename(disease)


def gru_checkpoint_path(disease: DiseaseSpec = DISEASE) -> Path:
    return run_dir(disease) / gru_checkpoint_filename(disease)
