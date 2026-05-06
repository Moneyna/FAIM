#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a lightweight tabular LR/SR consistency fusion model.

Expected data layout:
  LR: {data_base}/part2/Bicubic/RealDeg/{relative_image_name}.png
  SR: {data_base}/part2/{method}/RealDeg/{relative_image_name}.png

This script supports two stages:
  1) export: call the existing sr_consistency_score_overall5.py pipeline and
     aggregate mask-level outputs into image-level tabular features.
  2) train: read features.csv + GT CSV, split by image, and train a gated
     branch-fusion model using FGResQ-style scene-aware pairwise ranking loss.

Run:
  python train_consistency_fusion4.py --config config4.yaml --mode all
  python train_consistency_fusion4.py --config config4.yaml --mode export
  python train_consistency_fusion4.py --config config4.yaml --mode train
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import pickle
import random
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise ImportError("pandas is required. Install with: pip install pandas") from exc

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required. Install a torch build matching your CUDA environment.") from exc

try:
    from PIL import Image, ImageFilter
except ImportError as exc:  # pragma: no cover
    raise ImportError("Pillow is required. Install with: pip install pillow") from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

try:
    from scipy.optimize import curve_fit, OptimizeWarning
    from scipy.stats import kendalltau, pearsonr, spearmanr
    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    SCIPY_AVAILABLE = False
    curve_fit = OptimizeWarning = None
    kendalltau = pearsonr = spearmanr = None


DEFAULT_METHODS = [
    "UPSR",
    "StableSR",
    "SeeSR",
    "CoSeR",
    "PASD",
    "DiT4SR",
    "HYPIR",
    "LucidFlux",
]

META_COLUMNS = {
    "part",
    "dataset",
    "method",
    "relative_image_name",
    "image1_path",
    "image2_path",
    "overall_path",
    "debug_dir",
    "mask_scores_csv",
    "status",
    "error",
    "split",
}

LABEL_COLUMNS = {
    "consistency_ye_score",
    "consistency_ye_score_1_5",
    "consistency_ye_rank",
    "label",
    "gt_label",
    "n_expected_annotators",
    "lr_quality_mos_1_5",
    "image_quality_mos_1_5",
    "consistency_mos_1_5",
}

BRANCH_PENALTY_COLUMNS = [
    "lowfreq_penalty",
    "texture_penalty",
    "semantic_under_penalty",
    "unmatched_penalty",
]


# -----------------------------------------------------------------------------
# Config and filesystem helpers
# -----------------------------------------------------------------------------


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def default_config() -> Dict[str, Any]:
    return {
        "data": {
            "data_base": "data/LQMetric",
            "part": "part2",
            "dataset": "RealDeg",
            "lr_method": "Bicubic",
            "image_ext": ".png",
            "methods": DEFAULT_METHODS,
            # Optional explicit train/val data specs. When provided, export will
            # traverse all specs and train/val split will be fixed by these specs.
            "splits": None,
        },
        "gt": {
            "csv_path": "mos_per_dataset_method_image_mos_only.csv",
            "dataset_column": "dataset",
            "method_column": "method",
            "image_column": "relative_image_name",
            "label_column": "consistency_ye_score",
            "label_normalize": False,
            "label_is_relative_only": True,
        },
        "paths": {
            "output_dir": "outputs/FAIM",
            "features_csv": "outputs/FAIM/features.csv",
            "resolved_config": "outputs/FAIM/config_resolved.yaml",
        },
        "export": {
            "enabled": True,
            "scoring_script": "sr_consistency_score_overall5.py",
            "repo_root": ".",
            "device": "cuda",
            "amp": True,
            "save_debug": True,
            "skip_existing_features": True,
            "overwrite_feature_csv": False,
            "continue_on_error": True,
            "max_images": None,
            "max_pairs": None,
            "min_base_masks": 6,
            "texture_spatial_radius": 2,
            "empty_cache_each_pair": True,
            "debug_subdir": "pair_debug",
            "extra_scoring_args": {},
        },
        "split": {
            "type": "leave_image_out",
            "train_ratio": 0.8,
            "val_ratio": 0.2,
            "seed": 42,
        },
        "model": {
            "type": "gated_residual",
            "hidden_dim": 64,
            "mid_dim": 32,
            "dropout": 0.1,
            "residual_alpha": 0.2,
            "score_clip": True,
        },
        "features": {
            "auto_select_numeric": True,
            "columns": None,
            "exclude_columns": ["consistency_score", "final_penalty"],
            "include_dataset_method_onehot": False,
            "min_non_nan_ratio": 0.25,
            "fill_nan_with_train_mean": True,
        },
        "loss": {
            "type": "same_image_plus_cross_image_rank_zscore",
            "tie_threshold": 1.0e-3,
            "pred_temperature": 1.0,
            "same_image_rank_weight": 1.0,
            "cross_image_rank_weight": 0.2,
            "zscore_reg_weight": 0.02,
        },
        "train": {
            "epochs": 200,
            "steps_per_epoch": 200,
            "batch_size": 64,
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
            "num_workers": 0,
            "device": "cuda",
            "seed": 42,
            "early_stop_patience": 30,
            "grad_clip_norm": 5.0,
            "val_pair_mode": "all",  # all or random
            "val_random_pairs": 4096,
            "save_every_epoch": False,
        },
        "selection": {
            "scope": "per_dataset",
            "part": "part1",
            "dataset": "Real47",
            "metric": "KRCC",
            "mode": "max",
        },
        "outputs": {
            "best_model": "outputs/FAIM/best_model.pt",
            "last_model": "outputs/FAIM/last_model.pt",
            "scaler": "outputs/FAIM/scaler.pkl",
            "feature_columns": "outputs/FAIM/feature_columns.json",
            "train_log": "outputs/FAIM/train_log.csv",
            "val_predictions": "outputs/FAIM/val_predictions.csv",
            "val_metrics": "outputs/FAIM/val_metrics.json",
            "val_metrics_csv": "outputs/FAIM/val_metrics.csv",
        },
    }


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return deep_update(default_config(), user_cfg)


def save_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def expand_path(p: str | Path, base_dir: Optional[str | Path] = None) -> Path:
    p = Path(str(p)).expanduser()
    if not p.is_absolute() and base_dir is not None:
        p = Path(base_dir).expanduser() / p
    return p


def ensure_parent(path: str | Path) -> None:
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)


def rel_no_suffix(rel_name: str) -> Path:
    return Path(rel_name).with_suffix("")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Feature export
# -----------------------------------------------------------------------------


def import_scoring_module(scoring_script: Path, repo_root: Path):
    repo_root = repo_root.resolve()
    scoring_script = scoring_script.resolve()
    if not scoring_script.exists():
        raise FileNotFoundError(f"scoring_script not found: {scoring_script}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(scoring_script.parent) not in sys.path:
        sys.path.insert(0, str(scoring_script.parent))

    spec = importlib.util.spec_from_file_location("sr_consistency_score_overall5_imported", scoring_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import scoring script: {scoring_script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    required = ["build_insid3", "process_one_pair", "apply_internal_defaults"]
    missing = [name for name in required if not hasattr(mod, name)]
    if missing:
        raise AttributeError(f"scoring script is missing required objects: {missing}")
    return mod


def read_existing_feature_keys(features_csv: Path) -> set[Tuple[str, str, str, str]]:
    """Read completed feature keys.

    New files use key=(part,dataset,method,relative_image_name). For backward
    compatibility with old features.csv that did not have a part column, also
    include a legacy key with part="".
    """
    if not features_csv.exists() or features_csv.stat().st_size == 0:
        return set()
    keys: set[Tuple[str, str, str, str]] = set()
    try:
        with open(features_csv, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                part = str(r.get("part", "") or "")
                dataset = str(r.get("dataset", "") or "")
                method = str(r.get("method", "") or "")
                rel = str(r.get("relative_image_name", "") or "")
                keys.add((part, dataset, method, rel))
                # Legacy compatibility: old exported rows have no part column.
                if part == "":
                    keys.add(("", dataset, method, rel))
    except Exception:
        return set()
    return keys

def append_feature_rows(features_csv: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    features_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = features_csv.exists() and features_csv.stat().st_size > 0

    # For robustness, use a stable union header. If appending to an existing CSV,
    # keep the current header and silently ignore brand-new fields only if needed.
    if exists:
        with open(features_csv, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            try:
                fieldnames = next(reader)
            except StopIteration:
                fieldnames = []
        extra = [k for r in rows for k in r.keys() if k not in fieldnames]
        if extra:
            # Rewrite with the expanded schema.
            old_rows = []
            with open(features_csv, "r", newline="", encoding="utf-8-sig") as f:
                old_reader = csv.DictReader(f)
                old_rows = list(old_reader)
            fieldnames = fieldnames + sorted(set(extra))
            with open(features_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(old_rows)
                writer.writerows(rows)
            return
    else:
        preferred = [
            "part", "dataset", "method", "relative_image_name", "image1_path", "image2_path",
            "consistency_score", "final_penalty",
            "lowfreq_penalty", "lowfreq_raw", "lowfreq_luma_l1", "lowfreq_grad_l1",
            "texture_penalty", "texture_penalty_layer6", "texture_penalty_layer12", "texture_penalty_layer18",
            "semantic_penalty", "semantic_under_penalty", "semantic_over_penalty",
            "unmatched_penalty", "unmatched_area_ratio",
            "num_unmatched_masks", "num_weakly_supported_masks",
            "support_ratio_mean", "support_ratio_min", "support_ratio_p10", "support_ratio_area_weighted_mean",
            "semantic_under_mean", "semantic_under_max", "semantic_under_p90",
            "tau_pair_used", "K_image1", "K_image2", "valid_masks_image1", "valid_masks_image2",
            "num_masks", "K_ratio", "valid_K_ratio", "image2_largest_mask_ratio", "image2_mask_area_entropy",
            "sr_highfreq_energy", "lr_highfreq_energy", "highfreq_energy_ratio",
            "texture_penalty_times_sr_highfreq_energy", "overall_path", "status", "error",
        ]
        keys = []
        for k in preferred:
            if any(k in r for r in rows):
                keys.append(k)
        rest = sorted({k for r in rows for k in r.keys()} - set(keys))
        fieldnames = keys + rest

    with open(features_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def weighted_average(mask_rows: List[Dict[str, Any]], key: str, default: float = np.nan) -> float:
    vals = []
    weights = []
    for r in mask_rows:
        if key not in r:
            continue
        try:
            val = float(r[key])
            w = float(r.get("weight_area_pixels", r.get("image2_area_pixels", 1.0)))
        except Exception:
            continue
        if math.isfinite(val):
            vals.append(val)
            weights.append(max(w, 0.0))
    if not vals:
        return float(default)
    weights_arr = np.asarray(weights, dtype=np.float64)
    vals_arr = np.asarray(vals, dtype=np.float64)
    if weights_arr.sum() <= 0:
        return float(vals_arr.mean())
    weights_arr = weights_arr / weights_arr.sum()
    return float((weights_arr * vals_arr).sum())


def percentile(values: Sequence[float], q: float, default: float = np.nan) -> float:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float(default)
    return float(np.percentile(arr, q))


def entropy_from_areas(areas: Sequence[float]) -> float:
    arr = np.asarray([max(float(x), 0.0) for x in areas], dtype=np.float64)
    s = arr.sum()
    if s <= 0:
        return 0.0
    p = arr / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def highfreq_energy(image_path: str | Path, resize_to: Optional[Tuple[int, int]] = None, sigma: float = 3.0) -> float:
    img = Image.open(image_path).convert("RGB")
    if resize_to is not None and img.size != resize_to:
        img = img.resize(resize_to, Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC)
    gray = img.convert("L")
    low = gray.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    a = np.asarray(gray).astype(np.float32) / 255.0
    b = np.asarray(low).astype(np.float32) / 255.0
    return float(np.mean(np.abs(a - b)))


def aggregate_pair_features(
    pair_score: Dict[str, Any],
    mask_rows: List[Dict[str, Any]],
    image1_path: Path,
    image2_path: Path,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    row = dict(pair_score)
    row["image1_path"] = str(image1_path)
    row["image2_path"] = str(image2_path)

    # Weighted image-level values from mask rows.
    for key in [
        "lowfreq_penalty", "lowfreq_raw", "lowfreq_luma_l1", "lowfreq_grad_l1",
        "texture_penalty", "semantic_penalty", "semantic_over_penalty", "semantic_under_penalty",
        "unmatched_penalty", "final_penalty", "consistency_score",
    ]:
        if mask_rows and key in mask_rows[0]:
            row[key] = weighted_average(mask_rows, key, default=row.get(key, np.nan))

    for layer in [6, 12, 18, 24]:
        key = f"texture_penalty_layer{layer}"
        if any(key in r for r in mask_rows):
            row[key] = weighted_average(mask_rows, key)

    # Support / semantic statistics.
    support = []
    semantic_under = []
    areas = []
    unmatched_count = 0
    weak_count = 0
    for r in mask_rows:
        try:
            support.append(float(r.get("support_ratio", np.nan)))
        except Exception:
            pass
        try:
            semantic_under.append(float(r.get("semantic_under_penalty", np.nan)))
        except Exception:
            pass
        try:
            areas.append(float(r.get("weight_area_pixels", r.get("image2_area_pixels", 0.0))))
        except Exception:
            areas.append(0.0)
        try:
            unmatched_count += int(float(r.get("is_unmatched", 0)))
        except Exception:
            pass
        try:
            weak_count += int(float(r.get("is_weakly_supported", 0)))
        except Exception:
            pass

    support_arr = np.asarray([x for x in support if math.isfinite(x)], dtype=np.float64)
    if support_arr.size:
        row["support_ratio_mean"] = float(support_arr.mean())
        row["support_ratio_min"] = float(support_arr.min())
        row["support_ratio_p10"] = float(np.percentile(support_arr, 10))
    else:
        row["support_ratio_mean"] = np.nan
        row["support_ratio_min"] = np.nan
        row["support_ratio_p10"] = np.nan
    row["support_ratio_area_weighted_mean"] = weighted_average(mask_rows, "support_ratio")

    sem_under_arr = np.asarray([x for x in semantic_under if math.isfinite(x)], dtype=np.float64)
    if sem_under_arr.size:
        row["semantic_under_mean"] = float(sem_under_arr.mean())
        row["semantic_under_max"] = float(sem_under_arr.max())
        row["semantic_under_p90"] = float(np.percentile(sem_under_arr, 90))
    else:
        row["semantic_under_mean"] = np.nan
        row["semantic_under_max"] = np.nan
        row["semantic_under_p90"] = np.nan

    row["num_unmatched_masks"] = unmatched_count
    row["num_weakly_supported_masks"] = weak_count

    try:
        k1 = float(row.get("K_image1", np.nan))
        k2 = float(row.get("K_image2", np.nan))
        row["K_ratio"] = k1 / max(k2, 1.0)
    except Exception:
        row["K_ratio"] = np.nan
    try:
        v1 = float(row.get("valid_masks_image1", np.nan))
        v2 = float(row.get("valid_masks_image2", np.nan))
        row["valid_K_ratio"] = v1 / max(v2, 1.0)
    except Exception:
        row["valid_K_ratio"] = np.nan

    if areas:
        total_area = max(float(np.sum(areas)), 1e-12)
        row["image2_largest_mask_ratio"] = float(np.max(areas) / total_area)
        row["image2_mask_area_entropy"] = entropy_from_areas(areas)
    else:
        row["image2_largest_mask_ratio"] = np.nan
        row["image2_mask_area_entropy"] = np.nan

    # Optional high-frequency features. LR and SR have the same size in this task,
    # but resize defensively if needed.
    try:
        sr_img = Image.open(image2_path)
        sr_size = sr_img.size
        sr_energy = highfreq_energy(image2_path, sigma=float(cfg.get("export", {}).get("highfreq_sigma", 3.0)))
        lr_energy = highfreq_energy(image1_path, resize_to=sr_size, sigma=float(cfg.get("export", {}).get("highfreq_sigma", 3.0)))
        row["sr_highfreq_energy"] = sr_energy
        row["lr_highfreq_energy"] = lr_energy
        row["highfreq_energy_ratio"] = sr_energy / max(lr_energy, 1e-8)
        row["texture_penalty_times_sr_highfreq_energy"] = float(row.get("texture_penalty", np.nan)) * sr_energy
    except Exception:
        row["sr_highfreq_energy"] = np.nan
        row["lr_highfreq_energy"] = np.nan
        row["highfreq_energy_ratio"] = np.nan
        row["texture_penalty_times_sr_highfreq_energy"] = np.nan

    return row




def get_data_specs(cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return explicit dataset specs with split/part/dataset.

    Any key under ``data.splits`` is accepted. This lets custom split modes
    include auxiliary validation/test datasets for feature loading/export even
    when train/val membership is decided later by ``split:``.
    """
    data_cfg = cfg["data"]
    splits = data_cfg.get("splits")
    specs: List[Dict[str, str]] = []
    if isinstance(splits, dict):
        for split_name, items in splits.items():
            for item in items or []:
                if not isinstance(item, dict):
                    raise ValueError(f"data.splits.{split_name} items must be dicts, got: {item!r}")
                specs.append({
                    "split": str(split_name),
                    "part": str(item.get("part", data_cfg.get("part", "part2"))),
                    "dataset": str(item.get("dataset", data_cfg.get("dataset", "RealDeg"))),
                })
    if not specs:
        specs.append({
            "split": "all",
            "part": str(data_cfg.get("part", "part2")),
            "dataset": str(data_cfg.get("dataset", "RealDeg")),
        })
    seen = set()
    uniq: List[Dict[str, str]] = []
    for spec in specs:
        key = (spec["split"], spec["part"], spec["dataset"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(spec)
    return uniq

def dataset_to_part_map(cfg: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for spec in get_data_specs(cfg):
        mapping.setdefault(str(spec["dataset"]), str(spec["part"]))
    return mapping


def dataset_to_split_map(cfg: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for spec in get_data_specs(cfg):
        mapping.setdefault(str(spec["dataset"]), str(spec["split"]))
    return mapping


def fill_missing_part_and_split_columns(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Fill missing part/split for old feature rows using dataset-level specs."""
    df = df.copy()
    part_map = dataset_to_part_map(cfg)
    split_map = dataset_to_split_map(cfg)
    if "part" not in df.columns:
        df["part"] = ""
    if "split" not in df.columns:
        df["split"] = ""
    part_series = df["part"].fillna("").astype(str)
    split_series = df["split"].fillna("").astype(str)
    dataset_series = df["dataset"].astype(str)
    missing_part = part_series.eq("") | part_series.eq("nan")
    missing_split = split_series.eq("") | split_series.eq("nan")
    df.loc[missing_part, "part"] = dataset_series[missing_part].map(part_map).fillna("")
    df.loc[missing_split, "split"] = dataset_series[missing_split].map(split_map).fillna("")
    return df
def load_gt_subset(cfg: Dict[str, Any]) -> pd.DataFrame:
    gt_cfg = cfg["gt"]
    methods = cfg["data"].get("methods", DEFAULT_METHODS)
    wanted_datasets = {spec["dataset"] for spec in get_data_specs(cfg)}
    gt_path = expand_path(gt_cfg["csv_path"])
    df = pd.read_csv(gt_path)
    df = df[df[gt_cfg["dataset_column"]].astype(str).isin(wanted_datasets)].copy()
    df = df[df[gt_cfg["method_column"]].astype(str).isin(methods)].copy()
    return df

def export_features(cfg: Dict[str, Any]) -> Path:
    data_cfg = cfg["data"]
    export_cfg = cfg["export"]
    paths_cfg = cfg["paths"]

    output_dir = expand_path(paths_cfg["output_dir"])
    features_csv = expand_path(paths_cfg["features_csv"])
    output_dir.mkdir(parents=True, exist_ok=True)
    features_csv.parent.mkdir(parents=True, exist_ok=True)

    if bool(export_cfg.get("overwrite_feature_csv", False)) and features_csv.exists():
        features_csv.unlink()

    data_base = expand_path(data_cfg["data_base"])
    lr_method = data_cfg.get("lr_method", "Bicubic")
    methods = list(data_cfg.get("methods", DEFAULT_METHODS))
    image_ext = str(data_cfg.get("image_ext", ".png"))
    data_specs = get_data_specs(cfg)

    gt = load_gt_subset(cfg)
    gt_image_col = cfg["gt"]["image_column"]
    gt_method_col = cfg["gt"]["method_column"]
    gt_dataset_col = cfg["gt"]["dataset_column"]

    max_pairs = export_cfg.get("max_pairs")
    if max_pairs is not None:
        max_pairs = int(max_pairs)

    scoring_script = expand_path(export_cfg["scoring_script"])
    repo_root = expand_path(export_cfg.get("repo_root", "."))
    scoring = import_scoring_module(scoring_script, repo_root)

    args = SimpleNamespace()
    scoring.apply_internal_defaults(args)
    args.device = export_cfg.get("device", "cuda")
    args.amp = bool(export_cfg.get("amp", True))
    args.save_debug = bool(export_cfg.get("save_debug", True))
    args.min_base_masks = int(export_cfg.get("min_base_masks", 6))
    args.texture_spatial_radius = int(export_cfg.get("texture_spatial_radius", 2))
    args.empty_cache_each_pair = bool(export_cfg.get("empty_cache_each_pair", True))
    args.dataset = ""
    args.method = ""

    # Allow advanced overrides for any internal scoring arg.
    for k, v in dict(export_cfg.get("extra_scoring_args", {}) or {}).items():
        setattr(args, k, v)

    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Building INSID3 model for feature export...")
    model = scoring.build_insid3(
        model_size=args.model_size,
        image_size=args.image_size,
        svd_components=args.svd_comps,
        tau=args.tau,
        merge_threshold=args.merge_thresh,
        mask_refiner="bilinear",
        resize_to_orig_size=False,
        device=args.device,
    )
    model.to(args.device)
    model.eval()

    existing_keys = read_existing_feature_keys(features_csv) if export_cfg.get("skip_existing_features", True) else set()
    rows_written = 0
    rows_buffer: List[Dict[str, Any]] = []
    flush_every = max(1, int(export_cfg.get("flush_every", 8)))

    all_jobs: List[Tuple[str, str, str, str, Path, Path]] = []  # split, part, dataset, method, lr, sr
    for spec in data_specs:
        split_name = str(spec["split"])
        part = str(spec["part"])
        dataset = str(spec["dataset"])
        lr_root = data_base / part / lr_method / dataset
        if not lr_root.exists():
            raise FileNotFoundError(f"LR folder not found: {lr_root}")

        gt_ds = gt[gt[gt_dataset_col].astype(str).eq(dataset)].copy()
        wanted_images = sorted(gt_ds[gt_image_col].astype(str).unique().tolist())
        if export_cfg.get("max_images") is not None:
            wanted_images = wanted_images[: int(export_cfg["max_images"])]
        gt_pairs = set(zip(gt_ds[gt_method_col].astype(str), gt_ds[gt_image_col].astype(str)))

        for method in methods:
            sr_root = data_base / part / method / dataset
            for rel_name in wanted_images:
                if (method, rel_name) not in gt_pairs:
                    continue
                if max_pairs is not None and len(all_jobs) >= max_pairs:
                    break
                rel_path = Path(rel_name)
                if rel_path.suffix == "":
                    rel_path = rel_path.with_suffix(image_ext)
                image1_path = lr_root / rel_path
                image2_path = sr_root / rel_path
                all_jobs.append((split_name, part, dataset, method, str(rel_path), image1_path, image2_path))
            if max_pairs is not None and len(all_jobs) >= max_pairs:
                break
        if max_pairs is not None and len(all_jobs) >= max_pairs:
            break

    print("Feature export specs:")
    for spec in data_specs:
        print(f"  {spec['split']}: {spec['part']}/{spec['dataset']}")
    print(f"Feature export jobs: {len(all_jobs)}")
    debug_root = output_dir / str(export_cfg.get("debug_subdir", "pair_debug"))

    for split_name, part, dataset, method, rel_name, image1_path, image2_path in tqdm(all_jobs, desc="Exporting LR/SR features"):
        key = (part, dataset, method, rel_name)
        legacy_key = ("", dataset, method, rel_name)
        if key in existing_keys or legacy_key in existing_keys:
            continue
        pair_out_dir = debug_root / split_name / dataset / method / rel_no_suffix(rel_name)
        args.dataset = dataset
        args.method = method
        try:
            if not image1_path.exists():
                raise FileNotFoundError(f"Missing LR image: {image1_path}")
            if not image2_path.exists():
                raise FileNotFoundError(f"Missing SR image: {image2_path}")
            pair_score, mask_rows = scoring.process_one_pair(
                model=model,
                image1_path=image1_path,
                image2_path=image2_path,
                rel_key=Path(rel_name),
                pair_out_dir=pair_out_dir,
                args=args,
            )
            row = aggregate_pair_features(pair_score, mask_rows, image1_path, image2_path, cfg)
            row["status"] = row.get("status", "ok") or "ok"
            row["error"] = row.get("error", "") or ""
        except Exception as exc:
            err = traceback.format_exc()
            print(f"[EXPORT ERROR] {part}/{dataset}/{method}/{rel_name}: {exc}")
            row = {
                "part": part,
                "dataset": dataset,
                "method": method,
                "relative_image_name": rel_name,
                "image1_path": str(image1_path),
                "image2_path": str(image2_path),
                "split": split_name,
                "status": "error",
                "error": err,
            }
            if not export_cfg.get("continue_on_error", True):
                raise

        # Force canonical identifiers from config/path, in case the scoring script
        # omits part/split or writes a normalized relative name.
        row["part"] = part
        row["dataset"] = dataset
        row["method"] = method
        row["relative_image_name"] = rel_name
        row["split"] = split_name
        row["image1_path"] = str(image1_path)
        row["image2_path"] = str(image2_path)

        rows_buffer.append(row)
        rows_written += 1
        existing_keys.add(key)
        if len(rows_buffer) >= flush_every:
            append_feature_rows(features_csv, rows_buffer)
            rows_buffer = []
        if args.empty_cache_each_pair and str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if rows_buffer:
        append_feature_rows(features_csv, rows_buffer)

    print(f"Feature export finished. New rows written: {rows_written}. features_csv={features_csv}")
    return features_csv


# -----------------------------------------------------------------------------
# Model and training data
# -----------------------------------------------------------------------------



def _safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(float(eps), 1.0 - float(eps))
    return torch.log(p) - torch.log1p(-p)


class GatedResidualConsistencyFusion(nn.Module):
    """Interpretable gated branch fusion plus a small residual calibration head.

    Returns:
      score_01:   sigmoid(raw_score), final [0,1] consistency score.
      weights:    learned branch weights for lowfreq/texture/semantic/unmatched.
      penalty:    gated weighted penalty before residual correction.
      raw_score:  unbounded score used by ranking and z-score regression losses.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        mid_dim: int = 32,
        dropout: float = 0.1,
        n_branches: int = 4,
        residual_alpha: float = 0.2,
    ):
        super().__init__()
        self.residual_alpha = float(residual_alpha)
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
        )
        self.weight_head = nn.Linear(mid_dim, n_branches)
        self.residual_head = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, x: torch.Tensor, branch_penalties: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, D], branch_penalties: [B, 4] in order lowfreq, texture, semantic_under, unmatched
        h = self.backbone(x)
        w = torch.softmax(self.weight_head(h), dim=-1)
        final_penalty = (w * branch_penalties).sum(dim=-1).clamp(0.0, 1.0)
        branch_score = (1.0 - final_penalty).clamp(1e-6, 1.0 - 1e-6)
        base_raw = _safe_logit(branch_score)
        residual = self.residual_alpha * self.residual_head(h).squeeze(-1)
        raw_score = base_raw + residual
        score_01 = torch.sigmoid(raw_score)
        return score_01, w, final_penalty, raw_score


class GatedConsistencyFusion(nn.Module):
    """Legacy gated fusion without residual head, kept for ablations."""
    def __init__(self, in_dim: int, hidden_dim: int = 64, mid_dim: int = 32, dropout: float = 0.1, n_branches: int = 4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
        )
        self.weight_head = nn.Linear(mid_dim, n_branches)

    def forward(self, x: torch.Tensor, branch_penalties: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        w = torch.softmax(self.weight_head(h), dim=-1)
        final_penalty = (w * branch_penalties).sum(dim=-1).clamp(0.0, 1.0)
        score_01 = (1.0 - final_penalty).clamp(1e-6, 1.0 - 1e-6)
        raw_score = _safe_logit(score_01)
        return score_01, w, final_penalty, raw_score


class ConsistencyFusionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, mid_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, x: torch.Tensor, branch_penalties: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_score = self.net(x).squeeze(-1)
        score_01 = torch.sigmoid(raw_score)
        dummy_w = torch.empty((x.shape[0], 0), device=x.device)
        final_penalty = 1.0 - score_01
        return score_01, dummy_w, final_penalty, raw_score


@dataclass
class PreparedData:
    df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    feature_columns: List[str]
    scaler: Dict[str, np.ndarray]
    label_zscore_stats: Dict[str, Tuple[float, float]]


def merge_features_with_gt(cfg: Dict[str, Any]) -> pd.DataFrame:
    gt_cfg = cfg["gt"]
    features_csv = expand_path(cfg["paths"]["features_csv"])
    if not features_csv.exists():
        raise FileNotFoundError(f"features_csv not found: {features_csv}")
    feat = pd.read_csv(features_csv)
    feat = fill_missing_part_and_split_columns(feat, cfg)
    if "status" in feat.columns:
        status = feat["status"].fillna("ok").astype(str).str.lower()
        feat = feat[status.isin(["", "ok", "nan"])].copy()

    gt = pd.read_csv(expand_path(gt_cfg["csv_path"]))
    methods = cfg["data"].get("methods", DEFAULT_METHODS)
    wanted_datasets = {spec["dataset"] for spec in get_data_specs(cfg)}
    gt = gt[gt[gt_cfg["dataset_column"]].astype(str).isin(wanted_datasets)].copy()
    gt = gt[gt[gt_cfg["method_column"]].astype(str).isin(methods)].copy()

    gt_renamed = gt.rename(columns={
        gt_cfg["dataset_column"]: "dataset",
        gt_cfg["method_column"]: "method",
        gt_cfg["image_column"]: "relative_image_name",
        gt_cfg["label_column"]: "label",
    })
    gt_renamed = fill_missing_part_and_split_columns(gt_renamed, cfg)
    key_cols = ["part", "dataset", "method", "relative_image_name"]
    for c in key_cols:
        feat[c] = feat[c].astype(str)
        gt_renamed[c] = gt_renamed[c].astype(str)

    merged = feat.merge(gt_renamed[key_cols + ["label"]], on=key_cols, how="inner")
    if merged.empty:
        # Compatibility fallback for old features without part where dataset names are unique.
        key_cols_legacy = ["dataset", "method", "relative_image_name"]
        merged = feat.merge(gt_renamed[key_cols_legacy + ["label"]], on=key_cols_legacy, how="inner")
        merged = fill_missing_part_and_split_columns(merged, cfg)
    if merged.empty:
        raise RuntimeError("No rows after merging features.csv with GT. Check dataset/method/image names.")
    return merged


def _part_dataset_mask(df: pd.DataFrame, part: str, dataset: str) -> pd.Series:
    return df["part"].astype(str).eq(str(part)) & df["dataset"].astype(str).eq(str(dataset))


def split_by_image(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_cfg = cfg["split"]
    split_type = str(split_cfg.get("type", "leave_image_out"))

    if split_type in {"half_dataset_plus_extra_train", "realdeg_half_plus_real47", "custom_realdeg_half_plus_real47"}:
        """Split one holdout dataset by image; add extra datasets wholly to train.

        Intended experiment:
          train = 50% part2/RealDeg images + all part1/Real47 images
          val   = remaining 50% part2/RealDeg images

        The split is always by relative_image_name, never by rows, so all
        methods for the same LR image stay on the same side.
        """
        df = fill_missing_part_and_split_columns(df, cfg)
        seed = int(split_cfg.get("seed", 42))
        holdout_cfg = dict(split_cfg.get("holdout", {}) or {})
        holdout_part = str(holdout_cfg.get("part", "part2"))
        holdout_dataset = str(holdout_cfg.get("dataset", "RealDeg"))
        train_ratio = float(holdout_cfg.get("train_ratio", split_cfg.get("train_ratio", 0.5)))
        train_ratio = min(max(train_ratio, 1e-6), 1.0 - 1e-6)

        holdout_mask = _part_dataset_mask(df, holdout_part, holdout_dataset)
        holdout_df = df[holdout_mask].copy()
        if holdout_df.empty:
            raise RuntimeError(
                f"Custom split holdout dataset is empty: {holdout_part}/{holdout_dataset}. "
                "Check features.csv, GT csv, and data.splits."
            )

        images = sorted(holdout_df["relative_image_name"].astype(str).unique().tolist())
        rng = np.random.default_rng(seed)
        rng.shuffle(images)
        n_train = int(round(len(images) * train_ratio))
        n_train = min(max(n_train, 1), len(images) - 1)
        train_images = set(images[:n_train])
        val_images = set(images[n_train:])

        holdout_train = holdout_df[holdout_df["relative_image_name"].astype(str).isin(train_images)].copy()
        holdout_val = holdout_df[holdout_df["relative_image_name"].astype(str).isin(val_images)].copy()

        extra_train_cfgs = split_cfg.get("extra_train", None)
        if extra_train_cfgs is None:
            extra_train_cfgs = [{"part": "part1", "dataset": "Real47"}]
        extra_frames: List[pd.DataFrame] = []
        for item in extra_train_cfgs or []:
            if not isinstance(item, dict):
                raise ValueError(f"split.extra_train entries must be dicts, got: {item!r}")
            part = str(item.get("part", cfg["data"].get("part", "part2")))
            dataset = str(item.get("dataset", cfg["data"].get("dataset", "RealDeg")))
            extra_df = df[_part_dataset_mask(df, part, dataset)].copy()
            if extra_df.empty:
                raise RuntimeError(f"Custom split extra_train dataset is empty: {part}/{dataset}")
            extra_frames.append(extra_df)

        extra_val_cfgs = split_cfg.get("extra_val", None)
        extra_val_frames: List[pd.DataFrame] = []
        for item in extra_val_cfgs or []:
            if not isinstance(item, dict):
                raise ValueError(f"split.extra_val entries must be dicts, got: {item!r}")
            part = str(item.get("part", cfg["data"].get("part", "part2")))
            dataset = str(item.get("dataset", cfg["data"].get("dataset", "RealDeg")))
            extra_df = df[_part_dataset_mask(df, part, dataset)].copy()
            if extra_df.empty:
                raise RuntimeError(f"Custom split extra_val dataset is empty: {part}/{dataset}")
            extra_val_frames.append(extra_df)

        train_df = pd.concat([holdout_train] + extra_frames, ignore_index=False).copy()
        val_df = pd.concat([holdout_val] + extra_val_frames, ignore_index=False).copy()
        train_df["split"] = "train"
        val_df["split"] = "val"

        if train_df.empty or val_df.empty:
            raise RuntimeError(
                f"Custom split produced empty train or val. train={len(train_df)}, val={len(val_df)}"
            )
        print(
            f"Custom split: holdout={holdout_part}/{holdout_dataset}, "
            f"train_images={len(train_images)}, val_images={len(val_images)}, "
            f"extra_train={[str(x.get('part')) + '/' + str(x.get('dataset')) for x in (extra_train_cfgs or [])]}, "
            f"extra_val={[str(x.get('part')) + '/' + str(x.get('dataset')) for x in (extra_val_cfgs or [])]}"
        )
        return train_df, val_df

    if split_type in {"fixed", "fixed_by_dataset", "fixed_by_dataset_part"}:
        df = fill_missing_part_and_split_columns(df, cfg)
        train_specs = {(spec["part"], spec["dataset"]) for spec in get_data_specs(cfg) if spec["split"] == "train"}
        val_specs = {(spec["part"], spec["dataset"]) for spec in get_data_specs(cfg) if spec["split"] == "val"}
        part_ds = list(zip(df["part"].astype(str), df["dataset"].astype(str)))
        train_mask = pd.Series([x in train_specs for x in part_ds], index=df.index)
        val_mask = pd.Series([x in val_specs for x in part_ds], index=df.index)
        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        train_df["split"] = "train"
        val_df["split"] = "val"
        if train_df.empty or val_df.empty:
            raise RuntimeError(
                f"Fixed split produced empty train or val. train={len(train_df)}, val={len(val_df)}. "
                "Check data.splits and features.csv."
            )
        return train_df, val_df

    seed = int(split_cfg.get("seed", 42))
    train_ratio = float(split_cfg.get("train_ratio", 0.8))
    # Keep legacy behavior: split by image name within the merged dataframe.
    images = sorted(df["relative_image_name"].astype(str).unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(images)
    n_train = int(round(len(images) * train_ratio))
    n_train = min(max(n_train, 1), len(images) - 1)
    train_images = set(images[:n_train])
    train_df = df[df["relative_image_name"].astype(str).isin(train_images)].copy()
    val_df = df[~df["relative_image_name"].astype(str).isin(train_images)].copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    return train_df, val_df

def auto_feature_columns(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[str]:
    feat_cfg = cfg["features"]
    explicit = feat_cfg.get("columns")
    if explicit:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise KeyError(f"Configured feature columns missing from dataframe: {missing}")
        return list(explicit)

    exclude = set(META_COLUMNS) | set(LABEL_COLUMNS) | {"label"}
    exclude.update(feat_cfg.get("exclude_columns", []) or [])
    min_ratio = float(feat_cfg.get("min_non_nan_ratio", 0.25))
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if c.startswith("consistency_ye") or c.endswith("_mos_1_5") or c.endswith("_rank"):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= min_ratio:
            cols.append(c)

    # Ensure branch penalties are present and first if available.
    ordered = [c for c in BRANCH_PENALTY_COLUMNS if c in cols]
    ordered += [c for c in cols if c not in ordered]
    if not ordered:
        raise RuntimeError("No numeric feature columns selected. Check features.csv content.")
    return ordered



def _dataset_key_from_row(row: pd.Series) -> str:
    return f"{row.get('part', '')}/{row.get('dataset', '')}"


def compute_label_zscore_stats(train_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Compute train-only label z-score stats per part/dataset."""
    stats: Dict[str, Tuple[float, float]] = {}
    for key, g in train_df.groupby(["part", "dataset"], sort=False):
        name = f"{key[0]}/{key[1]}"
        y = pd.to_numeric(g["label"], errors="coerce").to_numpy(dtype=np.float64)
        y = y[np.isfinite(y)]
        if y.size == 0:
            stats[name] = (0.0, 1.0)
            continue
        mean = float(y.mean())
        std = float(y.std())
        if not math.isfinite(std) or std < 1e-8:
            std = 1.0
        stats[name] = (mean, std)
    return stats


def add_label_zscore_column(df: pd.DataFrame, stats: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    df = df.copy()
    z = []
    global_mean = float(pd.to_numeric(df["label"], errors="coerce").mean())
    global_std = float(pd.to_numeric(df["label"], errors="coerce").std())
    if not math.isfinite(global_mean):
        global_mean = 0.0
    if not math.isfinite(global_std) or global_std < 1e-8:
        global_std = 1.0
    for _, row in df.iterrows():
        key = _dataset_key_from_row(row)
        mean, std = stats.get(key, (global_mean, global_std))
        y = float(row["label"])
        z.append((y - mean) / max(std, 1e-8))
    df["label_z"] = np.asarray(z, dtype=np.float32)
    return df


def prepare_data(cfg: Dict[str, Any]) -> PreparedData:
    df = merge_features_with_gt(cfg)
    train_df, val_df = split_by_image(df, cfg)
    feature_columns = auto_feature_columns(train_df, cfg)

    for c in BRANCH_PENALTY_COLUMNS:
        if c not in df.columns:
            raise KeyError(f"Required branch penalty column missing from features.csv: {c}")

    label_zscore_stats = compute_label_zscore_stats(train_df)
    train_df = add_label_zscore_column(train_df, label_zscore_stats)
    val_df = add_label_zscore_column(val_df, label_zscore_stats)

    train_X = train_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    mean = train_X.mean(axis=0, skipna=True).to_numpy(dtype=np.float32)
    std = train_X.std(axis=0, skipna=True).replace(0, 1.0).fillna(1.0).to_numpy(dtype=np.float32)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    scaler = {"mean": mean, "std": std}
    return PreparedData(
        df=df,
        train_df=train_df,
        val_df=val_df,
        feature_columns=feature_columns,
        scaler=scaler,
        label_zscore_stats=label_zscore_stats,
    )


class ScenePairSampler:
    def __init__(self, df: pd.DataFrame, rng: np.random.Generator, tie_threshold: float):
        self.df = df.reset_index(drop=True).copy()
        self.rng = rng
        self.tie_threshold = float(tie_threshold)
        self.scenes: Dict[Tuple[str, str, str], np.ndarray] = {}
        for key, group in self.df.groupby(["part", "dataset", "relative_image_name"], sort=False):
            if group["method"].nunique() >= 2:
                self.scenes[(str(key[0]), str(key[1]), str(key[2]))] = group.index.to_numpy(dtype=np.int64)
        self.scene_keys = list(self.scenes.keys())
        if not self.scene_keys:
            raise RuntimeError("No valid scenes with at least two methods.")
        # Formula 7 is equivalent to averaging pairwise losses within each scene,
        # then averaging over scenes. For stochastic training, sample scenes
        # uniformly and then sample one pair inside the scene. For exhaustive
        # validation, all_pairs() uses inverse pair-count weights.
        self.scene_weights = self._make_scene_weights()
        self.scene_sample_probs = np.ones(len(self.scene_keys), dtype=np.float64) / float(len(self.scene_keys))

    def _make_scene_weights(self) -> np.ndarray:
        weights = []
        for key in self.scene_keys:
            n = len(self.scenes[key])
            n_pairs = max(n * (n - 1) // 2, 1)
            weights.append(1.0 / float(n_pairs))
        arr = np.asarray(weights, dtype=np.float64)
        arr = arr / arr.sum()
        return arr

    def sample_pairs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx_a = np.empty(batch_size, dtype=np.int64)
        idx_b = np.empty(batch_size, dtype=np.int64)
        target = np.empty(batch_size, dtype=np.float32)
        weights = np.empty(batch_size, dtype=np.float32)
        chosen_scene_idx = self.rng.choice(len(self.scene_keys), size=batch_size, replace=True, p=self.scene_sample_probs)
        for t, sidx in enumerate(chosen_scene_idx):
            key = self.scene_keys[int(sidx)]
            inds = self.scenes[key]
            a, b = self.rng.choice(inds, size=2, replace=False)
            ya = float(self.df.loc[a, "label"])
            yb = float(self.df.loc[b, "label"])
            diff = ya - yb
            if abs(diff) < self.tie_threshold:
                g = 0.5
            elif diff > 0:
                g = 1.0
            else:
                g = 0.0
            idx_a[t] = a
            idx_b[t] = b
            target[t] = g
            weights[t] = 1.0
        return idx_a, idx_b, target, weights

    def all_pairs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx_a, idx_b, target, weights = [], [], [], []
        for sidx, key in enumerate(self.scene_keys):
            inds = self.scenes[key]
            n = len(inds)
            n_pairs = max(n * (n - 1) // 2, 1)
            w = 1.0 / float(n_pairs)
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = int(inds[i]), int(inds[j])
                    ya = float(self.df.loc[a, "label"])
                    yb = float(self.df.loc[b, "label"])
                    diff = ya - yb
                    if abs(diff) < self.tie_threshold:
                        g = 0.5
                    elif diff > 0:
                        g = 1.0
                    else:
                        g = 0.0
                    idx_a.append(a); idx_b.append(b); target.append(g); weights.append(w)
        weights_arr = np.asarray(weights, dtype=np.float32)
        if weights_arr.size:
            weights_arr = weights_arr / max(float(weights_arr.mean()), 1e-12)
        return (
            np.asarray(idx_a, dtype=np.int64),
            np.asarray(idx_b, dtype=np.int64),
            np.asarray(target, dtype=np.float32),
            weights_arr,
        )



class TripleSampler:
    """Sample A/B/C triples for the three-term training loss.

    A and B are two methods for the same LR image.
    C is one LR/SR pair from the same part/dataset but a different LR image.
    """

    def __init__(self, df: pd.DataFrame, rng: np.random.Generator, tie_threshold: float):
        self.df = df.reset_index(drop=True).copy()
        self.rng = rng
        self.tie_threshold = float(tie_threshold)

        self.scenes: Dict[Tuple[str, str, str], np.ndarray] = {}
        self.dataset_to_indices: Dict[Tuple[str, str], np.ndarray] = {}
        self.dataset_to_scene_keys: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {}

        for key, group in self.df.groupby(["part", "dataset", "relative_image_name"], sort=False):
            scene_key = (str(key[0]), str(key[1]), str(key[2]))
            if group["method"].nunique() >= 2:
                self.scenes[scene_key] = group.index.to_numpy(dtype=np.int64)

        for key, group in self.df.groupby(["part", "dataset"], sort=False):
            ds_key = (str(key[0]), str(key[1]))
            self.dataset_to_indices[ds_key] = group.index.to_numpy(dtype=np.int64)
            scene_keys = []
            for rel in group["relative_image_name"].astype(str).unique().tolist():
                sk = (ds_key[0], ds_key[1], str(rel))
                if sk in self.scenes:
                    scene_keys.append(sk)
            self.dataset_to_scene_keys[ds_key] = scene_keys

        self.scene_keys = [
            sk for sk in self.scenes.keys()
            if len(self.dataset_to_scene_keys.get((sk[0], sk[1]), [])) >= 2
        ]
        if not self.scene_keys:
            raise RuntimeError("No valid scenes for triple sampling. Need at least two images per train dataset and >=2 methods per image.")

        self.scene_sample_probs = np.ones(len(self.scene_keys), dtype=np.float64) / float(len(self.scene_keys))

    def _target(self, ia: int, ib: int) -> float:
        ya = float(self.df.loc[ia, "label"])
        yb = float(self.df.loc[ib, "label"])
        diff = ya - yb
        if abs(diff) < self.tie_threshold:
            return 0.5
        return 1.0 if diff > 0 else 0.0

    def sample_triples(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx_a = np.empty(batch_size, dtype=np.int64)
        idx_b = np.empty(batch_size, dtype=np.int64)
        idx_c = np.empty(batch_size, dtype=np.int64)
        target_ab = np.empty(batch_size, dtype=np.float32)
        target_ac = np.empty(batch_size, dtype=np.float32)
        target_bc = np.empty(batch_size, dtype=np.float32)

        chosen_scene_idx = self.rng.choice(len(self.scene_keys), size=batch_size, replace=True, p=self.scene_sample_probs)
        for t, sidx in enumerate(chosen_scene_idx):
            scene_key = self.scene_keys[int(sidx)]
            inds = self.scenes[scene_key]
            a, b = self.rng.choice(inds, size=2, replace=False)

            ds_key = (scene_key[0], scene_key[1])
            candidate_scene_keys = [sk for sk in self.dataset_to_scene_keys[ds_key] if sk[2] != scene_key[2]]
            c_scene_key = candidate_scene_keys[int(self.rng.integers(0, len(candidate_scene_keys)))]
            c_inds = self.scenes[c_scene_key]
            c = int(self.rng.choice(c_inds, size=1, replace=True)[0])

            idx_a[t] = int(a)
            idx_b[t] = int(b)
            idx_c[t] = int(c)
            target_ab[t] = self._target(int(a), int(b))
            target_ac[t] = self._target(int(a), int(c))
            target_bc[t] = self._target(int(b), int(c))

        return idx_a, idx_b, idx_c, target_ab, target_ac, target_bc


def dataframe_to_tensors(df: pd.DataFrame, feature_columns: List[str], scaler: Dict[str, np.ndarray], device: torch.device):
    X_np = df[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    mean = scaler["mean"]
    std = scaler["std"]
    # Fill NaN with train mean before standardization; since X_np is raw, use mean.
    nan_mask = ~np.isfinite(X_np)
    if nan_mask.any():
        X_np[nan_mask] = np.take(mean, np.where(nan_mask)[1])
    X_np = (X_np - mean) / std
    X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    B_np = df[BRANCH_PENALTY_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    B_np = np.nan_to_num(B_np, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
    B_np = np.clip(B_np, 0.0, 1.0)
    y_np = pd.to_numeric(df["label"], errors="coerce").to_numpy(dtype=np.float32)
    return (
        torch.from_numpy(X_np).to(device),
        torch.from_numpy(B_np).to(device),
        torch.from_numpy(y_np).to(device),
    )


def make_model(cfg: Dict[str, Any], in_dim: int) -> nn.Module:
    model_cfg = cfg["model"]
    model_type = str(model_cfg.get("type", "gated_residual")).lower()
    if model_type == "mlp":
        return ConsistencyFusionMLP(
            in_dim=in_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            mid_dim=int(model_cfg.get("mid_dim", 32)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    if model_type in {"gated", "gated_legacy"}:
        return GatedConsistencyFusion(
            in_dim=in_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            mid_dim=int(model_cfg.get("mid_dim", 32)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            n_branches=4,
        )
    return GatedResidualConsistencyFusion(
        in_dim=in_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        mid_dim=int(model_cfg.get("mid_dim", 32)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        n_branches=4,
        residual_alpha=float(model_cfg.get("residual_alpha", 0.2)),
    )


def pairwise_ranking_loss(
    score_a: torch.Tensor,
    score_b: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    pred_temperature: float = 1.0,
) -> torch.Tensor:
    """Pairwise ranking BCE on logits.

    score_a/score_b should be raw, unbounded scores. Using BCEWithLogits is
    numerically more stable than sigmoid + BCE.
    """
    temp = max(float(pred_temperature), 1e-8)
    logits = (score_a - score_b) / temp
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1e-8)


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    if SCIPY_AVAILABLE:
        return float(pearsonr(x, y)[0])
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_average(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_a = a[order]
    n = len(a)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return float("nan")
    if SCIPY_AVAILABLE:
        return float(spearmanr(x, y).correlation)
    return safe_pearson(rankdata_average(x), rankdata_average(y))


def safe_kendall(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return float("nan")
    if SCIPY_AVAILABLE:
        return float(kendalltau(x, y).correlation)
    # Slow fallback for small validation sets.
    concordant = discordant = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            sx = np.sign(x[i] - x[j])
            sy = np.sign(y[i] - y[j])
            if sx == 0 or sy == 0:
                continue
            if sx == sy:
                concordant += 1
            else:
                discordant += 1
    den = concordant + discordant
    return float((concordant - discordant) / den) if den > 0 else float("nan")


def logistic5(x, beta1, beta2, beta3, beta4, beta5):
    # Common IQA 5-parameter logistic mapping.
    #return beta2 + (beta1 - beta2) / (1.0 + np.exp(-(x - beta3) / (np.abs(beta4) + 1e-8))) + beta5 * x
    z = -(x - beta3) / (np.abs(beta4) + 1e-8)
    z = np.clip(z, -60.0, 60.0)
    return beta2 + (beta1 - beta2) / (1.0 + np.exp(z)) + beta5 * x



def plcc_after_logistic(pred: np.ndarray, gt: np.ndarray) -> float:
    """PLCC after 5-parameter logistic mapping.

    The fitting is used only for evaluation. To avoid noisy console warnings and
    numerical instability, fit on standardized prediction values, clip the
    logistic exponent, suppress OptimizeWarning, and fall back to raw PLCC when
    fitting fails.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(gt)
    pred, gt = pred[mask], gt[mask]
    if pred.size < 5:
        return safe_pearson(pred, gt)
    if np.std(pred) < 1e-12 or np.std(gt) < 1e-12:
        return safe_pearson(pred, gt)
    if not SCIPY_AVAILABLE:
        return safe_pearson(pred, gt)

    x = (pred - float(pred.mean())) / max(float(pred.std()), 1e-12)
    try:
        beta0 = [float(gt.max()), float(gt.min()), 0.0, 1.0, 0.0]
        with warnings.catch_warnings():
            if OptimizeWarning is not None:
                warnings.simplefilter("ignore", OptimizeWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            popt, _ = curve_fit(logistic5, x, gt, p0=beta0, maxfev=20000)
        mapped = logistic5(x, *popt)
        return safe_pearson(mapped, gt)
    except Exception:
        return safe_pearson(pred, gt)

def pairwise_accuracy(df: pd.DataFrame, tie_threshold: float) -> Dict[str, float]:
    correct = 0
    total = 0
    tie_correct = 0
    tie_total = 0
    for _, g in df.groupby(["part", "dataset", "relative_image_name"], sort=False):
        if len(g) < 2:
            continue
        rows = g.reset_index(drop=True)
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                gt_diff = float(rows.loc[i, "label"]) - float(rows.loc[j, "label"])
                pr_diff = float(rows.loc[i, "pred_score"]) - float(rows.loc[j, "pred_score"])
                if abs(gt_diff) < tie_threshold:
                    tie_total += 1
                    if abs(pr_diff) < tie_threshold:
                        tie_correct += 1
                    continue
                total += 1
                if np.sign(gt_diff) == np.sign(pr_diff):
                    correct += 1
    return {
        "pair_acc_excl_tie": float(correct / total) if total else float("nan"),
        "pair_total_excl_tie": int(total),
        "tie_acc": float(tie_correct / tie_total) if tie_total else float("nan"),
        "tie_total": int(tie_total),
    }


def metrics_for_df(df: pd.DataFrame, tie_threshold: float) -> Dict[str, Any]:
    pred = df["pred_score"].to_numpy(dtype=np.float64)
    gt = df["label"].to_numpy(dtype=np.float64)
    out: Dict[str, Any] = {
        "num_samples": int(len(df)),
        "num_scenes": int(df[["part", "dataset", "relative_image_name"]].drop_duplicates().shape[0]),
        "SRCC": safe_spearman(pred, gt),
        "KRCC": safe_kendall(pred, gt),
        "PLCC_logistic": plcc_after_logistic(pred, gt),
        "PLCC_raw": safe_pearson(pred, gt),
    }
    out.update(pairwise_accuracy(df, tie_threshold=tie_threshold))
    return out


def evaluate_model(
    model: nn.Module,
    df: pd.DataFrame,
    feature_columns: List[str],
    scaler: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        X, B, _ = dataframe_to_tensors(df.reset_index(drop=True), feature_columns, scaler, device)
        scores, weights, penalties, raw_scores = model(X, B)
        pred = scores.detach().cpu().numpy().astype(np.float64)
        raw_pred = raw_scores.detach().cpu().numpy().astype(np.float64)
        pen = penalties.detach().cpu().numpy().astype(np.float64)
        w_np = weights.detach().cpu().numpy().astype(np.float64) if weights.numel() else np.empty((len(df), 0))

    out_df = df.reset_index(drop=True).copy()
    out_df["pred_score"] = pred
    out_df["pred_raw_score"] = raw_pred
    out_df["pred_final_penalty"] = pen
    if w_np.shape[1] == 4:
        out_df["w_low"] = w_np[:, 0]
        out_df["w_texture"] = w_np[:, 1]
        out_df["w_semantic"] = w_np[:, 2]
        out_df["w_unmatched"] = w_np[:, 3]

    tie = float(cfg["loss"].get("tie_threshold", 1e-3))
    metrics: Dict[str, Any] = {"overall": metrics_for_df(out_df, tie)}

    # Per-dataset metrics. Include part when available so that datasets with the
    # same name in different parts cannot collide.
    dataset_group_cols = ["part", "dataset"] if "part" in out_df.columns else ["dataset"]
    per_dataset: Dict[str, Any] = {}
    for key, g in out_df.groupby(dataset_group_cols, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        name = "/".join(str(x) for x in key)
        per_dataset[name] = metrics_for_df(g, tie)
    metrics["per_dataset"] = per_dataset

    metrics["per_method"] = {
        str(k): metrics_for_df(g, tie) for k, g in out_df.groupby("method", sort=True)
    }

    # Useful for diagnosing which method fails on which validation dataset.
    dm_group_cols = (["part", "dataset", "method"] if "part" in out_df.columns
                     else ["dataset", "method"])
    per_dataset_method: Dict[str, Any] = {}
    for key, g in out_df.groupby(dm_group_cols, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        name = "/".join(str(x) for x in key)
        per_dataset_method[name] = metrics_for_df(g, tie)
    metrics["per_dataset_method"] = per_dataset_method

    return out_df, metrics


def val_pair_loss(
    model: nn.Module,
    val_df: pd.DataFrame,
    feature_columns: List[str],
    scaler: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    device: torch.device,
) -> float:
    model.eval()
    sampler = ScenePairSampler(val_df, np.random.default_rng(123), float(cfg["loss"].get("tie_threshold", 1e-3)))
    if cfg["train"].get("val_pair_mode", "all") == "all":
        ia, ib, target, weight = sampler.all_pairs()
    else:
        ia, ib, target, weight = sampler.sample_pairs(int(cfg["train"].get("val_random_pairs", 4096)))
    with torch.no_grad():
        X, B, _ = dataframe_to_tensors(val_df.reset_index(drop=True), feature_columns, scaler, device)
        _, _, _, raw_scores = model(X, B)
        ia_t = torch.from_numpy(ia).to(device)
        ib_t = torch.from_numpy(ib).to(device)
        target_t = torch.from_numpy(target).to(device)
        weight_t = torch.from_numpy(weight).to(device)
        loss = pairwise_ranking_loss(raw_scores[ia_t], raw_scores[ib_t], target_t, weight_t, cfg["loss"].get("pred_temperature", 1.0))
    return float(loss.detach().cpu().item())


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def flatten_metrics(metrics: Dict[str, Any]) -> pd.DataFrame:
    """Flatten nested metric dicts into a CSV-friendly table."""
    rows: List[Dict[str, Any]] = []

    def add_row(scope: str, name: str, d: Dict[str, Any]) -> None:
        row: Dict[str, Any] = {"scope": scope, "name": name}
        for k, v in d.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                row[k] = float(v) if isinstance(v, (np.floating, float)) else int(v)
            elif isinstance(v, str):
                row[k] = v
            elif v is None:
                row[k] = v
        rows.append(row)

    if "overall" in metrics:
        add_row("overall", "overall", metrics["overall"])
    for scope in ["per_dataset", "per_method", "per_dataset_method"]:
        for name, d in metrics.get(scope, {}).items():
            add_row(scope, str(name), d)
    if "selection" in metrics:
        sel = metrics["selection"]
        add_row("selection", str(sel.get("dataset", "selection")), {
            "selection_metric_name": str(sel.get("metric", "")),
            "selection_mode": str(sel.get("mode", "")),
            "selection_raw_value": sel.get("raw_value", float("nan")),
            "selection_normalized_value": sel.get("normalized_value", float("nan")),
            "selection_value": sel.get("selection_value", float("nan")),
        })
    return pd.DataFrame(rows)


def save_metrics_outputs(metrics: Dict[str, Any], json_path: str | Path, csv_path: Optional[str | Path] = None) -> None:
    save_json(metrics, json_path)
    if csv_path is not None:
        csv_path = Path(csv_path).expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        flatten_metrics(metrics).to_csv(csv_path, index=False)



def get_selection_metric(metrics: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return configured best-model selection metric.

    Default for this experiment: maximize KRCC on part1/Real47.
    """
    sel_cfg = cfg.get("selection", {}) or {}
    part = str(sel_cfg.get("part", "part1"))
    dataset = str(sel_cfg.get("dataset", "Real47"))
    metric = str(sel_cfg.get("metric", "KRCC"))
    mode = str(sel_cfg.get("mode", "max")).lower()
    ds_name = f"{part}/{dataset}"

    per_dataset = metrics.get("per_dataset", {})
    raw_value = float("nan")
    if ds_name in per_dataset:
        raw_value = float(per_dataset[ds_name].get(metric, float("nan")))
    else:
        # Fallback for configs without part in metric names.
        for name, d in per_dataset.items():
            if str(name).endswith(f"/{dataset}") or str(name) == dataset:
                raw_value = float(d.get(metric, float("nan")))
                ds_name = str(name)
                break

    if not math.isfinite(raw_value):
        overall = metrics.get("overall", {})
        raw_value = float(overall.get(metric, float("nan")))
        ds_name = "overall"

    if metric.upper() in {"SRCC", "KRCC", "PLCC", "PLCC_LOGISTIC", "PLCC_RAW"}:
        normalized_value = (raw_value + 1.0) * 0.5 if math.isfinite(raw_value) else float("nan")
    else:
        normalized_value = raw_value

    selection_value = normalized_value
    if mode == "min" and math.isfinite(selection_value):
        selection_value = -selection_value

    return {
        "dataset": ds_name,
        "metric": metric,
        "mode": mode,
        "raw_value": raw_value,
        "normalized_value": normalized_value,
        "selection_value": selection_value,
    }


def train_fusion(cfg: Dict[str, Any]) -> Dict[str, Any]:
    train_cfg = cfg["train"]
    loss_cfg = cfg["loss"]
    outputs = cfg["outputs"]
    set_seed(int(train_cfg.get("seed", 42)))

    prep = prepare_data(cfg)
    print(f"Merged samples: {len(prep.df)} | train: {len(prep.train_df)} | val: {len(prep.val_df)}")
    print("Train datasets:")
    print(prep.train_df.groupby(["part", "dataset"])["relative_image_name"].nunique().to_string())
    print("Val datasets:")
    print(prep.val_df.groupby(["part", "dataset"])["relative_image_name"].nunique().to_string())
    print(f"Feature dim: {len(prep.feature_columns)}")
    print("Feature columns:", prep.feature_columns)

    for p in [
        outputs["best_model"], outputs["last_model"], outputs["scaler"], outputs["feature_columns"],
        outputs["train_log"], outputs["val_predictions"], outputs["val_metrics"],
        outputs.get("val_metrics_csv", outputs["val_metrics"].replace(".json", ".csv")),
    ]:
        ensure_parent(expand_path(p))

    device = torch.device(train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = make_model(cfg, in_dim=len(prep.feature_columns)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    train_df_reset = prep.train_df.reset_index(drop=True)
    X_train, B_train, y_train = dataframe_to_tensors(train_df_reset, prep.feature_columns, prep.scaler, device)
    y_z_train = torch.from_numpy(
        pd.to_numeric(train_df_reset["label_z"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    ).to(device)

    sampler = TripleSampler(
        train_df_reset,
        np.random.default_rng(int(train_cfg.get("seed", 42))),
        float(loss_cfg.get("tie_threshold", 1e-3)),
    )

    with open(expand_path(outputs["scaler"]), "wb") as f:
        pickle.dump({
            "mean": prep.scaler["mean"],
            "std": prep.scaler["std"],
            "feature_columns": prep.feature_columns,
            "label_zscore_stats": prep.label_zscore_stats,
        }, f)
    save_json({
        "feature_columns": prep.feature_columns,
        "branch_penalty_columns": BRANCH_PENALTY_COLUMNS,
        "label_zscore_stats": {k: [float(v[0]), float(v[1])] for k, v in prep.label_zscore_stats.items()},
    }, expand_path(outputs["feature_columns"]))

    best_metric = -np.inf
    best_epoch = -1
    no_improve = 0
    train_log_rows: List[Dict[str, Any]] = []

    epochs = int(train_cfg.get("epochs", 200))
    steps_per_epoch = int(train_cfg.get("steps_per_epoch", 200))
    batch_size = int(train_cfg.get("batch_size", 64))
    pred_temp = float(loss_cfg.get("pred_temperature", 1.0))
    grad_clip = float(train_cfg.get("grad_clip_norm", 5.0))

    same_w = float(loss_cfg.get("same_image_rank_weight", 1.0))
    cross_w = float(loss_cfg.get("cross_image_rank_weight", 0.2))
    zreg_w = float(loss_cfg.get("zscore_reg_weight", 0.02))

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        losses_same = []
        losses_cross = []
        losses_reg = []

        for _ in range(steps_per_epoch):
            ia, ib, ic, target_ab, target_ac, target_bc = sampler.sample_triples(batch_size)
            ia_t = torch.from_numpy(ia).to(device)
            ib_t = torch.from_numpy(ib).to(device)
            ic_t = torch.from_numpy(ic).to(device)
            target_ab_t = torch.from_numpy(target_ab).to(device)
            target_ac_t = torch.from_numpy(target_ac).to(device)
            target_bc_t = torch.from_numpy(target_bc).to(device)
            weight_t = torch.ones_like(target_ab_t)

            _, _, _, raw_scores = model(X_train, B_train)

            loss_same = pairwise_ranking_loss(
                raw_scores[ia_t], raw_scores[ib_t], target_ab_t, weight_t, pred_temp
            )
            loss_cross_ac = pairwise_ranking_loss(
                raw_scores[ia_t], raw_scores[ic_t], target_ac_t, weight_t, pred_temp
            )
            loss_cross_bc = pairwise_ranking_loss(
                raw_scores[ib_t], raw_scores[ic_t], target_bc_t, weight_t, pred_temp
            )
            loss_cross = 0.5 * (loss_cross_ac + loss_cross_bc)

            reg_idx = torch.unique(torch.cat([ia_t, ib_t, ic_t], dim=0))
            loss_reg = F.smooth_l1_loss(raw_scores[reg_idx], y_z_train[reg_idx])

            loss = same_w * loss_same + cross_w * loss_cross + zreg_w * loss_reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            losses.append(float(loss.detach().cpu().item()))
            losses_same.append(float(loss_same.detach().cpu().item()))
            losses_cross.append(float(loss_cross.detach().cpu().item()))
            losses_reg.append(float(loss_reg.detach().cpu().item()))

        val_loss = val_pair_loss(model, prep.val_df, prep.feature_columns, prep.scaler, cfg, device)
        val_pred_df, val_metrics = evaluate_model(model, prep.val_df, prep.feature_columns, prep.scaler, cfg, device)
        selection_info = get_selection_metric(val_metrics, cfg)
        val_metrics["selection"] = selection_info

        val_srcc = float(val_metrics["overall"].get("SRCC", float("nan")))
        val_krcc = float(val_metrics["overall"].get("KRCC", float("nan")))
        val_plcc = float(val_metrics["overall"].get("PLCC_logistic", float("nan")))
        selection_metric = float(selection_info.get("selection_value", float("nan")))
        if not math.isfinite(selection_metric):
            selection_metric = -np.inf

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_loss_same": float(np.mean(losses_same)),
            "train_loss_cross": float(np.mean(losses_cross)),
            "train_loss_zreg": float(np.mean(losses_reg)),
            "val_pair_loss": val_loss,
            "val_SRCC": val_srcc,
            "val_KRCC": val_krcc,
            "val_PLCC_logistic": val_plcc,
            "selection_dataset": str(selection_info.get("dataset", "")),
            "selection_metric_name": str(selection_info.get("metric", "")),
            "selection_raw_value": float(selection_info.get("raw_value", float("nan"))),
            "selection_normalized_value": float(selection_info.get("normalized_value", float("nan"))),
            "selection_value": selection_metric,
        }
        train_log_rows.append(row)
        pd.DataFrame(train_log_rows).to_csv(expand_path(outputs["train_log"]), index=False)

        print(
            f"epoch {epoch:04d} | train_loss={row['train_loss']:.6f} "
            f"(same={row['train_loss_same']:.6f}, cross={row['train_loss_cross']:.6f}, zreg={row['train_loss_zreg']:.6f}) | "
            f"val_loss={val_loss:.6f} | SRCC={val_srcc:.4f} | KRCC={val_krcc:.4f} | PLCC-log={val_plcc:.4f} | "
            f"select[{row['selection_dataset']}/{row['selection_metric_name']}]={row['selection_raw_value']:.4f}"
        )
        for ds_name, ds_m in val_metrics.get("per_dataset", {}).items():
            print(
                f"  val[{ds_name}] "
                f"N={int(ds_m.get('num_samples', 0))} "
                f"SRCC={float(ds_m.get('SRCC', float('nan'))):.4f} "
                f"KRCC={float(ds_m.get('KRCC', float('nan'))):.4f} "
                f"PLCC-log={float(ds_m.get('PLCC_logistic', float('nan'))):.4f} "
                f"pair_acc={float(ds_m.get('pair_acc_excl_tie', float('nan'))):.4f}"
            )

        last_ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
            "feature_columns": prep.feature_columns,
            "branch_penalty_columns": BRANCH_PENALTY_COLUMNS,
            "scaler": prep.scaler,
            "label_zscore_stats": prep.label_zscore_stats,
            "val_metrics": val_metrics,
            "selection_info": selection_info,
        }
        torch.save(last_ckpt, expand_path(outputs["last_model"]))

        if selection_metric > best_metric:
            best_metric = selection_metric
            best_epoch = epoch
            no_improve = 0
            torch.save(last_ckpt, expand_path(outputs["best_model"]))
            val_pred_df.to_csv(expand_path(outputs["val_predictions"]), index=False)
            save_metrics_outputs(
                val_metrics,
                expand_path(outputs["val_metrics"]),
                expand_path(outputs.get("val_metrics_csv", str(outputs["val_metrics"]).replace(".json", ".csv"))),
            )
        else:
            no_improve += 1

        if bool(train_cfg.get("save_every_epoch", False)):
            ep_path = Path(expand_path(outputs["best_model"])).with_name(f"epoch_{epoch:04d}.pt")
            torch.save(last_ckpt, ep_path)

        patience = int(train_cfg.get("early_stop_patience", 30))
        if patience > 0 and no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch}. Best epoch={best_epoch}, "
                f"best selection={best_metric:.4f}"
            )
            break

    # Reload best checkpoint for final saved predictions/metrics consistency.
    best_path = expand_path(outputs["best_model"])
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    val_pred_df, val_metrics = evaluate_model(model, prep.val_df, prep.feature_columns, prep.scaler, cfg, device)
    selection_info = get_selection_metric(val_metrics, cfg)
    val_metrics["selection"] = selection_info
    val_pred_df.to_csv(expand_path(outputs["val_predictions"]), index=False)
    save_metrics_outputs(
        val_metrics,
        expand_path(outputs["val_metrics"]),
        expand_path(outputs.get("val_metrics_csv", str(outputs["val_metrics"]).replace(".json", ".csv"))),
    )
    print(f"Best model saved to: {best_path}")
    print(
        f"Best selection target: {selection_info.get('dataset')}/{selection_info.get('metric')}="
        f"{float(selection_info.get('raw_value', float('nan'))):.4f}"
    )
    return val_metrics


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--mode", choices=["export", "train", "all"], default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = expand_path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, expand_path(cfg["paths"].get("resolved_config", out_dir / "config_resolved.yaml")))

    if args.mode in {"export", "all"}:
        export_features(cfg)
    if args.mode in {"train", "all"}:
        metrics = train_fusion(cfg)
        print(json.dumps(metrics.get("overall", metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
