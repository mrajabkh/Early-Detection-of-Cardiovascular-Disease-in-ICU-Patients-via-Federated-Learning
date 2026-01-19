# train_eval_gru.py
# Train + evaluate a supervised GRU early detection model on window-sequence data.
# Adds true lead-time stratified performance using lead_time_mins from samples.csv.
# Also saves a PR-curve plot (precision-recall) for lead-time buckets on the TEST split.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import config
from sequence_dataset_gru import PatientSequenceDataset, pad_collate
from gru_risk import GRURisk

try:
    from memory_profiler import memory_usage
except Exception:
    memory_usage = None


# -------------------------
# Metrics
# -------------------------
def roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(y_score)) + 1

    sorted_scores = y_score[order]
    sorted_pos = pos[order]

    i = 0
    sum_ranks_pos = 0.0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (ranks[order[i]] + ranks[order[j]])
        sum_ranks_pos += int(sorted_pos[i : j + 1].sum()) * avg_rank
        i = j + 1

    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def avg_precision_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp = 0
    fp = 0
    ap = 0.0
    prev_rec = 0.0

    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        rec = tp / n_pos
        prec = tp / max(tp + fp, 1)
        if rec > prev_rec:
            ap += prec * (rec - prev_rec)
            prev_rec = rec

    return ap


def _precision_recall_curve_manual(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (recall, precision) arrays.
    """
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return np.array([0.0, 1.0], dtype=np.float64), np.array([1.0, 0.0], dtype=np.float64)

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    recall = np.concatenate([np.array([0.0]), recall, np.array([1.0])])
    precision = np.concatenate([np.array([1.0]), precision, np.array([precision[-1] if len(precision) else 0.0])])

    return recall, precision


def flatten_valid(logits, y, mask):
    probs = torch.sigmoid(logits)
    valid = mask > 0.5
    return (
        y[valid].detach().cpu().numpy().astype(np.int64),
        probs[valid].detach().cpu().numpy().astype(np.float64),
    )


def masked_bce(logits, targets, mask, pos_weight=None):
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    loss = loss_fn(logits, targets.float())
    loss = loss * mask.float()
    return loss.sum() / mask.sum().clamp_min(1.0)


# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    max_len: int = 128
    batch_size: int = 32
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 20
    patience: int = 3
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y, all_s = [], []
    total_loss = 0.0

    for x, y, mask, lengths in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        logits = model(x, lengths)
        loss = masked_bce(logits, y, mask)
        total_loss += float(loss.item())

        yt, ys = flatten_valid(logits, y, mask)
        if len(yt):
            all_y.append(yt)
            all_s.append(ys)

    if not all_y:
        return {"loss": float("nan"), "auroc": float("nan"), "auprc": float("nan")}

    y_true = np.concatenate(all_y)
    y_score = np.concatenate(all_s)

    return {
        "loss": total_loss / max(len(loader), 1),
        "auroc": roc_auc_score_manual(y_true, y_score),
        "auprc": avg_precision_score_manual(y_true, y_score),
    }


def _get_leadtime_buckets() -> Dict[str, Tuple[int, int]]:
    """
    Choose lead-time buckets based on the configured horizon.
    - For 12h horizon: 3 buckets (0-2, 2-6, 6-12)
    - For 24h (or larger) horizon: 4 buckets (0-2, 2-6, 6-12, 12-24)
    This prevents errors if you switch config back and forth.
    """
    hor_hrs = int(getattr(config, "HORIZON_HRS", 12))

    if hor_hrs >= 24:
        return {
            "0_2h": (0, 120),
            "2_6h": (120, 360),
            "6_12h": (360, 720),
            "12_24h": (720, 1440),
        }

    # default: 12h-style buckets
    return {
        "0_2h": (0, 120),
        "2_6h": (120, 360),
        "6_12h": (360, 720),
    }


@torch.no_grad()
def evaluate_stratified_lead_time(
    model,
    dataset: PatientSequenceDataset,
    device: str,
    disease: config.DiseaseSpec,
    top_k: int | None,
) -> Tuple[Dict[str, float], str | None]:
    """
    True lead-time stratification using lead_time_mins from samples.csv.

    For a given bucket, keep:
      - all negatives
      - only positives whose lead_time is in the bucket

    Also saves a PR curve plot for the buckets into:
      Outputs/<run_name>/PR_Plots/
    """
    model.eval()

    buckets = _get_leadtime_buckets()

    metrics_out: Dict[str, float] = {}
    plot_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for bname, (lo, hi) in buckets.items():
        y_all: List[int] = []
        s_all: List[float] = []

        for i in range(len(dataset)):
            idx = dataset._groups[i]
            sub = dataset.df.iloc[idx]

            x = sub[dataset.feature_cols].to_numpy(dtype=np.float32)
            y = sub["label"].to_numpy(dtype=np.int64)

            if "lead_time_mins" in sub.columns:
                lt = pd.to_numeric(sub["lead_time_mins"], errors="coerce").to_numpy(dtype=np.float32)
            else:
                lt = np.full(len(y), np.nan, dtype=np.float32)

            if len(y) > dataset.max_len:
                x = x[-dataset.max_len :]
                y = y[-dataset.max_len :]
                lt = lt[-dataset.max_len :]

            if len(y) == 0:
                continue

            xt = torch.from_numpy(x[None, :, :]).to(device)
            lengths = torch.tensor([len(y)], dtype=torch.long, device=device)

            logits = model(xt, lengths)
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

            neg_mask = (y == 0)
            pos_mask = (y == 1) & np.isfinite(lt) & (lt >= lo) & (lt < hi)

            keep = neg_mask | pos_mask
            if not np.any(keep):
                continue

            y_all.extend(y[keep].tolist())
            s_all.extend(probs[keep].tolist())

        if len(y_all) == 0:
            metrics_out[f"auroc_{bname}"] = float("nan")
            metrics_out[f"auprc_{bname}"] = float("nan")
            plot_data[bname] = (np.array([], dtype=np.int64), np.array([], dtype=np.float64))
            continue

        y_true = np.array(y_all, dtype=np.int64)
        y_score = np.array(s_all, dtype=np.float64)

        metrics_out[f"auroc_{bname}"] = roc_auc_score_manual(y_true, y_score)
        metrics_out[f"auprc_{bname}"] = avg_precision_score_manual(y_true, y_score)
        plot_data[bname] = (y_true, y_score)

    plot_path: str | None = None
    try:
        out_dir = config.run_dir(disease)

        PR_Plots_dir = out_dir / "PR_Plots"
        PR_Plots_dir.mkdir(parents=True, exist_ok=True)

        topk_tag = "all" if top_k is None else str(int(top_k))
        plot_path = str(PR_Plots_dir / f"gru_pr_by_leadtime__topk{topk_tag}.png")

        plt.figure()
        for bname in buckets.keys():
            y_true, y_score = plot_data.get(bname, (np.array([], dtype=np.int64), np.array([], dtype=np.float64)))
            if y_true.size == 0:
                continue
            recall, precision = _precision_recall_curve_manual(y_true, y_score)
            plt.plot(recall, precision, label=bname)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR curves by lead-time bucket (top_k={topk_tag})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
    except Exception:
        plot_path = None

    return metrics_out, plot_path


# -------------------------
# Train + Eval
# -------------------------
def train_and_eval(
    disease: config.DiseaseSpec,
    cfg: TrainConfig,
    top_k: int | None = None,
    rank_path: str | None = None,
) -> Dict[str, Dict]:

    t0 = time.perf_counter()

    def _run():
        train_ds = PatientSequenceDataset(
            split="train",
            disease=disease,
            max_len=cfg.max_len,
            seed=cfg.seed,
            normalize=True,
            top_k=top_k,
            rank_path=rank_path,
        )
        val_ds = PatientSequenceDataset(
            split="val",
            disease=disease,
            max_len=cfg.max_len,
            seed=cfg.seed,
            normalize=True,
            top_k=top_k,
            rank_path=rank_path,
        )
        test_ds = PatientSequenceDataset(
            split="test",
            disease=disease,
            max_len=cfg.max_len,
            seed=cfg.seed,
            normalize=True,
            top_k=top_k,
            rank_path=rank_path,
        )

        train_loader = DataLoader(train_ds, cfg.batch_size, True, collate_fn=pad_collate)
        val_loader = DataLoader(val_ds, cfg.batch_size, False, collate_fn=pad_collate)
        test_loader = DataLoader(test_ds, cfg.batch_size, False, collate_fn=pad_collate)

        model = GRURisk(
            input_dim=len(train_ds.feature_cols),
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(cfg.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        best_val = -1.0
        best_state = None
        bad_epochs = 0

        for epoch in range(cfg.epochs):
            model.train()
            for x, y, mask, lengths in train_loader:
                x = x.to(cfg.device)
                y = y.to(cfg.device)
                mask = mask.to(cfg.device)

                optimizer.zero_grad()
                loss = masked_bce(model(x, lengths), y, mask)
                loss.backward()
                optimizer.step()

            val = evaluate(model, val_loader, cfg.device)
            if val["auroc"] > best_val:
                best_val = val["auroc"]
                best_state = model.state_dict()
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    break

        model.load_state_dict(best_state)

        train_metrics = evaluate(model, train_loader, cfg.device)
        val_metrics = evaluate(model, val_loader, cfg.device)
        test_metrics = evaluate(model, test_loader, cfg.device)

        strat_test, pr_plot_path = evaluate_stratified_lead_time(
            model=model,
            dataset=test_ds,
            device=cfg.device,
            disease=disease,
            top_k=top_k,
        )

        return {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "strat_test": strat_test,
            "n_features": len(train_ds.feature_cols),
            "pr_plot_path": pr_plot_path,
        }

    if memory_usage is not None:
        mem, out = memory_usage((_run, (), {}), retval=True, interval=0.1)
        cpu_peak = float(max(mem))
    else:
        out = _run()
        cpu_peak = float("nan")

    runtime = time.perf_counter() - t0

    out["extra"] = {
        "runtime_sec": runtime,
        "cpu_peak_mib": cpu_peak,
        "n_features": out["n_features"],
        "pr_plot_path": out.get("pr_plot_path", None),
    }

    return out
