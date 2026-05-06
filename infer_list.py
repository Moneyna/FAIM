#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference for the trained LR/SR consistency fusion model.

The script supports:
  1) single LR/SR image pair;
  2) folder LR/SR inference with matched relative filenames.

It performs the same two-step pipeline used during training/export:
  LR/SR image pair -> sr_consistency_score_overall5.py feature extraction
  -> image-level feature aggregation -> scaler/feature_columns preprocessing
  -> trained gated-residual fusion model -> final consistency score.

Example, single pair:
  python infer_consistency_fusion.py \
    --config config4.yaml \
    --lr /path/to/part2/Bicubic/RealDeg/0001.png \
    --sr /path/to/part2/HYPIR/RealDeg/0001.png \
    --dataset RealDeg --method HYPIR --part part2 \
    --out_dir outputs/infer_single

Example, folder mode:
  python infer_consistency_fusion.py \
    --config config4.yaml \
    --lr_dir data/LQMetric/part2/Bicubic/RealDeg \
    --sr_dir data/LQMetric/part2/HYPIR/RealDeg \
    --dataset RealDeg --method HYPIR --part part2 \
    --out_dir outputs/infer_realdeg_val

By default, when dataset=RealDeg and part matches split.holdout in config4.yaml,
folder mode only evaluates the held-out val half defined by that config split.
Disable with --no_realdeg_val_filter.

Feature-CSV shortcut mode:
  python infer_consistency_fusion.py \
    --config config4.yaml \
    --features_csv outputs/.../features.csv \
    --part part2 --datasets RealDeg --methods HYPIR UPSR \
    --out_dir outputs/infer_from_features

For multiple datasets/methods without --features_csv, provide --data_base or rely on
config.data.data_base. The script will construct folders as:
  {data_base}/{part}/Bicubic/{dataset}
  {data_base}/{part}/{method}/{dataset}
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import pickle
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def expand_path(p: str | Path, base_dir: str | Path | None = None) -> Path:
    p = Path(str(p)).expanduser()
    if not p.is_absolute() and base_dir is not None:
        p = Path(base_dir).expanduser() / p
    return p


def import_module_from_path(module_name: str, path: str | Path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Python file not found: {path}")
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from: {path}")
    mod = importlib.util.module_from_spec(spec)
    # Important for dataclasses/type annotations: register the module before exec.
    # Without this, @dataclass can fail with sys.modules.get(cls.__module__) is None
    # when dynamically importing train_consistency_fusion4.py.
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def infer_part_dataset_method_from_path(sr_path_or_dir: str | Path) -> Tuple[str, str, str]:
    """Best-effort inference for layout {data_base}/{part}/{method}/{dataset}/{image}."""
    p = Path(sr_path_or_dir).expanduser()
    if p.is_file():
        dataset = p.parent.name
        method = p.parent.parent.name if p.parent.parent != p.parent else "unknown"
        part = p.parent.parent.parent.name if p.parent.parent.parent != p.parent.parent else "infer"
    else:
        dataset = p.name
        method = p.parent.name if p.parent != p else "unknown"
        part = p.parent.parent.name if p.parent.parent != p.parent else "infer"
    return part, dataset, method


def list_image_files(folder: Path, recursive: bool = False) -> Dict[str, Path]:
    pattern = "**/*" if recursive else "*"
    files = [p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if recursive:
        return {str(p.relative_to(folder)): p for p in sorted(files)}
    return {p.name: p for p in sorted(files)}


def normalize_rel_name(name: str, image_ext: str = ".png") -> str:
    p = Path(str(name))
    if p.suffix == "":
        p = p.with_suffix(image_ext)
    return str(p).replace("\\", "/")


def get_realdeg_val_images_from_config(cfg: Dict[str, Any], lr_dir: Optional[Path] = None) -> set[str]:
    """Recreate the half-dataset holdout val image list used by train_consistency_fusion4.py."""
    split_cfg = cfg.get("split", {}) or {}
    holdout_cfg = split_cfg.get("holdout", {}) or {}
    holdout_dataset = str(holdout_cfg.get("dataset", "RealDeg"))
    seed = int(split_cfg.get("seed", 42))
    train_ratio = float(holdout_cfg.get("train_ratio", split_cfg.get("train_ratio", 0.5)))
    image_ext = str(cfg.get("data", {}).get("image_ext", ".png"))

    images: List[str] = []

    # Prefer GT csv because the training split was made from merged GT/features rows.
    gt_cfg = cfg.get("gt", {}) or {}
    gt_path = gt_cfg.get("csv_path")
    if gt_path:
        gt_path = expand_path(gt_path)
        if gt_path.exists():
            try:
                gt = pd.read_csv(gt_path)
                dataset_col = gt_cfg.get("dataset_column", "dataset")
                image_col = gt_cfg.get("image_column", "relative_image_name")
                if dataset_col in gt.columns and image_col in gt.columns:
                    sub = gt[gt[dataset_col].astype(str).eq(holdout_dataset)]
                    images = sorted({normalize_rel_name(x, image_ext) for x in sub[image_col].astype(str).tolist()})
            except Exception:
                images = []

    # Fallback to LR folder if GT is unavailable.
    if not images and lr_dir is not None and lr_dir.exists():
        images = sorted(list_image_files(lr_dir, recursive=False).keys())

    if len(images) < 2:
        return set()

    rng = np.random.default_rng(seed)
    images = list(images)
    rng.shuffle(images)
    train_ratio = min(max(train_ratio, 1e-6), 1.0 - 1e-6)
    n_train = int(round(len(images) * train_ratio))
    n_train = min(max(n_train, 1), len(images) - 1)
    return set(images[n_train:])


def make_pairs_from_args(args, cfg: Dict[str, Any]) -> List[Tuple[str, Path, Path]]:
    image_ext = str(cfg.get("data", {}).get("image_ext", ".png"))

    if args.lr and args.sr:
        lr = expand_path(args.lr)
        sr = expand_path(args.sr)
        if not lr.exists():
            raise FileNotFoundError(f"LR image not found: {lr}")
        if not sr.exists():
            raise FileNotFoundError(f"SR image not found: {sr}")
        rel = args.relative_name or sr.name
        rel = normalize_rel_name(rel, image_ext)
        return [(rel, lr, sr)]

    if args.lr_dir and args.sr_dir:
        lr_dir = expand_path(args.lr_dir)
        sr_dir = expand_path(args.sr_dir)
        if not lr_dir.exists():
            raise FileNotFoundError(f"LR folder not found: {lr_dir}")
        if not sr_dir.exists():
            raise FileNotFoundError(f"SR folder not found: {sr_dir}")

        lr_map = list_image_files(lr_dir, recursive=bool(args.recursive))
        sr_map = list_image_files(sr_dir, recursive=bool(args.recursive))
        common = sorted(set(lr_map.keys()) & set(sr_map.keys()))

        if args.realdeg_val_filter:
            holdout_cfg = cfg.get("split", {}).get("holdout", {}) or {}
            holdout_part = str(holdout_cfg.get("part", "part2"))
            holdout_dataset = str(holdout_cfg.get("dataset", "RealDeg"))
            part = args.part or infer_part_dataset_method_from_path(sr_dir)[0]
            dataset = args.dataset or infer_part_dataset_method_from_path(sr_dir)[1]
            if str(part) == holdout_part and str(dataset) == holdout_dataset:
                val_images = get_realdeg_val_images_from_config(cfg, lr_dir=lr_dir)
                before = len(common)
                common = [x for x in common if normalize_rel_name(x, image_ext) in val_images]
                print(
                    f"RealDeg val filter enabled: kept {len(common)}/{before} images "
                    f"using split seed={cfg.get('split', {}).get('seed', 42)}."
                )

        if args.max_pairs is not None:
            common = common[: int(args.max_pairs)]
        if not common:
            raise RuntimeError(f"No matched LR/SR image files found: {lr_dir} vs {sr_dir}")
        return [(normalize_rel_name(rel, image_ext), lr_map[rel], sr_map[rel]) for rel in common]

    raise ValueError("Provide either --lr and --sr for single-pair mode, or --lr_dir and --sr_dir for folder mode.")



def make_feature_rows_from_csv(
    args,
    cfg: Dict[str, Any],
    part: str,
    dataset: str,
    method: str,
) -> List[Dict[str, Any]]:
    """Load already-exported image-level features and select inference rows.

    This mode skips sr_consistency_score_overall5.py entirely. It expects that
    the CSV was produced by train_consistency_fusion*.py export and already
    contains the image-level columns used by the trained fusion model.
    """
    features_csv = expand_path(args.features_csv)
    if not features_csv.exists():
        raise FileNotFoundError(f"features_csv not found: {features_csv}")

    image_ext = str(cfg.get("data", {}).get("image_ext", ".png"))
    df = pd.read_csv(features_csv)
    if df.empty:
        raise RuntimeError(f"features_csv is empty: {features_csv}")

    # Normalize required id columns. Older feature files may not have part/split.
    if "part" not in df.columns:
        df["part"] = ""
    if "dataset" not in df.columns:
        raise ValueError("features_csv must contain a dataset column")
    if "method" not in df.columns:
        raise ValueError("features_csv must contain a method column")
    if "relative_image_name" not in df.columns:
        raise ValueError("features_csv must contain a relative_image_name column")

    df["part"] = df["part"].fillna("").astype(str)
    df["dataset"] = df["dataset"].fillna("").astype(str)
    df["method"] = df["method"].fillna("").astype(str)
    df["relative_image_name"] = df["relative_image_name"].fillna("").astype(str).map(
        lambda x: normalize_rel_name(x, image_ext)
    )

    # Drop failed rows if status/error exist.
    if "status" in df.columns:
        df = df[~df["status"].fillna("").astype(str).str.lower().eq("error")].copy()
    if "error" in df.columns:
        # Keep rows with empty error; some old files use NaN.
        df = df[df["error"].fillna("").astype(str).str.len().eq(0)].copy()

    # Metadata filtering. If the file has no usable part values, do not force part.
    df = df[df["dataset"].eq(str(dataset)) & df["method"].eq(str(method))].copy()
    if df["part"].fillna("").astype(str).str.len().gt(0).any():
        df = df[df["part"].eq(str(part))].copy()

    # Optional row restriction from CLI paths or RealDeg held-out split.
    allowed_rels: Optional[set[str]] = None
    path_map: Dict[str, Tuple[Path, Path]] = {}

    if args.lr and args.sr:
        lr = expand_path(args.lr)
        sr = expand_path(args.sr)
        rel = args.relative_name or sr.name
        rel = normalize_rel_name(rel, image_ext)
        allowed_rels = {rel}
        path_map[rel] = (lr, sr)
    elif args.lr_dir and args.sr_dir:
        # Reuse the existing pair matching + RealDeg val filtering logic.
        pairs = make_pairs_from_args(args, cfg)
        allowed_rels = {rel for rel, _lr, _sr in pairs}
        path_map = {rel: (_lr, _sr) for rel, _lr, _sr in pairs}
    else:
        # If no image paths are supplied, still respect RealDeg holdout filtering
        # when the metadata matches config split.holdout.
        if args.realdeg_val_filter:
            holdout_cfg = cfg.get("split", {}).get("holdout", {}) or {}
            holdout_part = str(holdout_cfg.get("part", "part2"))
            holdout_dataset = str(holdout_cfg.get("dataset", "RealDeg"))
            if str(part) == holdout_part and str(dataset) == holdout_dataset:
                val_images = get_realdeg_val_images_from_config(cfg, lr_dir=None)
                if val_images:
                    allowed_rels = set(val_images)
                    print(
                        f"features_csv RealDeg val filter enabled: kept only held-out val images "
                        f"using split seed={cfg.get('split', {}).get('seed', 42)}."
                    )

    if allowed_rels is not None:
        before = len(df)
        df = df[df["relative_image_name"].isin(allowed_rels)].copy()
        print(f"features_csv filter: kept {len(df)}/{before} rows after relative-image restriction.")

    if args.max_pairs is not None:
        df = df.sort_values("relative_image_name").head(int(args.max_pairs)).copy()

    # If duplicate keys exist, keep the last successful export row.
    key_cols = ["part", "dataset", "method", "relative_image_name"]
    df = df.drop_duplicates(key_cols, keep="last")

    if df.empty:
        raise RuntimeError(
            "No matching feature rows found. Check --features_csv, --part, --dataset, --method, "
            "and RealDeg val filtering."
        )

    rows: List[Dict[str, Any]] = []
    for _, s in df.sort_values("relative_image_name").iterrows():
        row = s.to_dict()
        rel = normalize_rel_name(str(row.get("relative_image_name", "")), image_ext)
        row["part"] = str(part) if not str(row.get("part", "")).strip() else str(row.get("part"))
        row["dataset"] = str(dataset)
        row["method"] = str(method)
        row["relative_image_name"] = rel
        row["status"] = "ok"
        row["error"] = ""
        if rel in path_map:
            row["image1_path"] = str(path_map[rel][0])
            row["image2_path"] = str(path_map[rel][1])
        rows.append(row)
    return rows

def load_feature_columns(path: Path, checkpoint: Optional[Dict[str, Any]] = None) -> List[str]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [str(x) for x in obj]
        if isinstance(obj, dict) and "feature_columns" in obj:
            return [str(x) for x in obj["feature_columns"]]
    if checkpoint and "feature_columns" in checkpoint:
        return [str(x) for x in checkpoint["feature_columns"]]
    raise FileNotFoundError(f"Could not load feature columns from {path} or checkpoint")


def load_scaler(path: Path, checkpoint: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
    if path.exists():
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if "mean" in obj and "std" in obj:
            return {"mean": np.asarray(obj["mean"], dtype=np.float32), "std": np.asarray(obj["std"], dtype=np.float32)}
    if checkpoint and "scaler" in checkpoint:
        obj = checkpoint["scaler"]
        return {"mean": np.asarray(obj["mean"], dtype=np.float32), "std": np.asarray(obj["std"], dtype=np.float32)}
    raise FileNotFoundError(f"Could not load scaler from {path} or checkpoint")


def row_to_model_tensors(row: Dict[str, Any], feature_columns: List[str], scaler: Dict[str, np.ndarray], branch_cols: Sequence[str], device: torch.device):
    vals = []
    for c in feature_columns:
        try:
            v = float(row.get(c, np.nan))
        except Exception:
            v = np.nan
        vals.append(v)
    x = np.asarray(vals, dtype=np.float32)[None, :]
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    if mean.shape[0] != len(feature_columns) or std.shape[0] != len(feature_columns):
        raise ValueError(
            f"Scaler dim mismatch: mean={mean.shape}, std={std.shape}, feature_columns={len(feature_columns)}"
        )
    nan_mask = ~np.isfinite(x)
    if nan_mask.any():
        x[nan_mask] = np.take(mean, np.where(nan_mask)[1])
    x = (x - mean) / std
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    bvals = []
    for c in branch_cols:
        try:
            v = float(row.get(c, np.nan))
        except Exception:
            v = np.nan
        bvals.append(v)
    b = np.asarray(bvals, dtype=np.float32)[None, :]
    b = np.nan_to_num(b, nan=1.0, posinf=1.0, neginf=1.0)
    b = np.clip(b, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(x).to(device), torch.from_numpy(b).to(device)


def build_scoring_args(train_mod, scoring_mod, cfg: Dict[str, Any], device: str, save_debug: bool):
    export_cfg = cfg.get("export", {}) or {}
    args = SimpleNamespace()
    scoring_mod.apply_internal_defaults(args)
    args.device = device
    args.amp = bool(export_cfg.get("amp", True))
    args.save_debug = bool(save_debug)
    args.min_base_masks = int(export_cfg.get("min_base_masks", 6))
    args.texture_spatial_radius = int(export_cfg.get("texture_spatial_radius", 2))
    args.empty_cache_each_pair = bool(export_cfg.get("empty_cache_each_pair", True))
    args.dataset = "infer"
    args.method = "unknown"
    for k, v in dict(export_cfg.get("extra_scoring_args", {}) or {}).items():
        setattr(args, k, v)
    return args


def build_scoring_model(scoring_mod, args):
    print("Building INSID3 model for inference feature extraction...")
    model = scoring_mod.build_insid3(
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
    return model


def format_value(v: Any) -> str:
    if isinstance(v, (float, np.floating)):
        if not math.isfinite(float(v)):
            return str(v)
        return f"{float(v):.10g}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def write_txt_report(path: Path, rows: List[Dict[str, Any]], feature_columns: List[str], branch_cols: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("LR/SR Consistency Inference Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"num_images: {len(rows)}\n\n")
        for i, row in enumerate(rows, start=1):
            f.write("-" * 80 + "\n")
            f.write(f"[{i}] {row.get('relative_image_name', '')}\n")
            f.write(f"part: {row.get('part', '')}\n")
            f.write(f"dataset: {row.get('dataset', '')}\n")
            f.write(f"method: {row.get('method', '')}\n")
            f.write(f"lr_path: {row.get('image1_path', '')}\n")
            f.write(f"sr_path: {row.get('image2_path', '')}\n")
            f.write("\nFINAL RESULT\n")
            f.write(f"score: {format_value(row.get('pred_score', np.nan))}\n")
            f.write(f"raw_score: {format_value(row.get('pred_raw_score', np.nan))}\n")
            f.write(f"pred_final_penalty: {format_value(row.get('pred_final_penalty', np.nan))}\n")
            f.write("\nBRANCH WEIGHTS\n")
            for k in ["w_low", "w_texture", "w_semantic", "w_unmatched"]:
                f.write(f"{k}: {format_value(row.get(k, np.nan))}\n")
            f.write("\nBRANCH PENALTIES\n")
            for k in branch_cols:
                f.write(f"{k}: {format_value(row.get(k, np.nan))}\n")
            f.write("\nFEATURE COLUMNS USED BY MODEL\n")
            for k in feature_columns:
                f.write(f"{k}: {format_value(row.get(k, np.nan))}\n")
            f.write("\nALL RAW/AGGREGATED VALUES\n")
            for k in sorted(row.keys()):
                if k in set(feature_columns) or k in set(branch_cols):
                    continue
                f.write(f"{k}: {format_value(row.get(k))}\n")
            f.write("\n")


def save_results_files(
    out_dir: Path,
    csv_name: str,
    txt_name: str,
    rows: List[Dict[str, Any]],
    feature_columns: List[str],
    branch_cols: Sequence[str],
) -> Tuple[Path, Path]:
    """Write current inference results immediately.

    This intentionally rewrites the whole CSV/TXT after each completed image so
    users can inspect partial results while a long folder inference is still
    running, and so an interrupted run still leaves infer_results.* files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / csv_name
    txt_path = out_dir / txt_name

    preferred = [
        "part", "dataset", "method", "relative_image_name", "status", "pred_score",
        "pred_raw_score", "pred_final_penalty", "w_low", "w_texture", "w_semantic", "w_unmatched",
        "image1_path", "image2_path", "debug_dir", "error",
    ]
    all_cols: List[str] = []
    for c in preferred + list(branch_cols) + list(feature_columns):
        if c not in all_cols and any(c in r for r in rows):
            all_cols.append(c)
    for c in sorted({k for r in rows for k in r.keys()}):
        if c not in all_cols:
            all_cols.append(c)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    write_txt_report(txt_path, rows, feature_columns, branch_cols)
    return txt_path, csv_path



def make_pairs_for_spec(args, cfg: Dict[str, Any], part: str, dataset: str, method: str) -> List[Tuple[str, Path, Path]]:
    """Build LR/SR pairs for one part/dataset/method spec.

    If explicit --lr/--sr or --lr_dir/--sr_dir are provided, this keeps the
    single-pair/single-folder behavior. Otherwise, it constructs folders from
    config.data.data_base or --data_base.
    """
    # Explicit single/folder mode is only sensible for one spec; reuse existing logic.
    if args.lr or args.sr or args.lr_dir or args.sr_dir:
        return make_pairs_from_args(args, cfg)

    data_base = expand_path(args.data_base or cfg.get("data", {}).get("data_base", "."))
    lr_method = str(cfg.get("data", {}).get("lr_method", "Bicubic"))
    lr_dir = data_base / str(part) / lr_method / str(dataset)
    sr_dir = data_base / str(part) / str(method) / str(dataset)

    # Reuse folder matching and RealDeg val filtering by creating a lightweight args copy.
    spec_args = SimpleNamespace(**vars(args))
    spec_args.lr_dir = str(lr_dir)
    spec_args.sr_dir = str(sr_dir)
    spec_args.lr = None
    spec_args.sr = None
    return make_pairs_from_args(spec_args, cfg)

def main() -> None:
    parser = argparse.ArgumentParser(description="Infer trained LR/SR consistency fusion score for single images or folders.")
    parser.add_argument("--config", default="config4.yaml", help="Training config yaml. Default: config4.yaml")
    parser.add_argument("--train_script", default="train_consistency_fusion4.py", help="Training script containing model/helper definitions.")
    parser.add_argument("--scoring_script", default=None, help="Override scoring script. Default: config export.scoring_script")

    # Single-pair mode.
    parser.add_argument("--lr", default=None, help="Single LR/Bicubic image path.")
    parser.add_argument("--sr", default=None, help="Single SR/restored image path.")

    # Folder mode.
    parser.add_argument("--lr_dir", default=None, help="LR/Bicubic folder.")
    parser.add_argument("--sr_dir", default=None, help="SR/restored folder.")
    parser.add_argument("--recursive", action="store_true", help="Match folders recursively by relative path.")
    parser.add_argument("--max_pairs", type=int, default=None, help="Optional max number of pairs in folder mode.")

    # Metadata.
    parser.add_argument("--part", default=None, help="Dataset part metadata, e.g. part2. Inferred from SR path if omitted.")
    parser.add_argument("--dataset", default=None, help="Dataset metadata, e.g. RealDeg. Inferred from SR path if omitted.")
    parser.add_argument("--method", default=None, help="SR method metadata, e.g. HYPIR. Inferred from SR path if omitted.")
    parser.add_argument("--datasets", nargs="+", default=None, help="One or more datasets, e.g. RealDeg RealSR RealPhoto60. Overrides --dataset.")
    parser.add_argument("--methods", nargs="+", default=None, help="One or more SR methods, e.g. HYPIR UPSR. Overrides --method.")
    parser.add_argument("--data_base", default=None, help="Dataset root used for multi dataset/method image inference. Default: config data.data_base.")
    parser.add_argument("--relative_name", default=None, help="Relative image name for single-pair mode. Default: SR filename.")

    # Model artifacts. Defaults come from config outputs.
    parser.add_argument("--checkpoint", default=None, help="best_model.pt path. Default: config outputs.best_model")
    parser.add_argument("--scaler", default=None, help="scaler.pkl path. Default: config outputs.scaler")
    parser.add_argument("--feature_columns", default=None, help="feature_columns.json path. Default: config outputs.feature_columns")
    parser.add_argument("--features_csv", default=None, help="Optional exported image-level features.csv. If set, skip sr_consistency_score_overall5 feature extraction and infer directly from matching rows. Pass 'config' to use config paths.features_csv.")
    parser.add_argument("--use_features_csv", action="store_true", help="Use config paths.features_csv and skip first-stage feature extraction.")

    parser.add_argument("--out_dir", default="outputs/infer_consistency", help="Output directory.")
    parser.add_argument("--txt_name", default="infer_results.txt", help="TXT report filename.")
    parser.add_argument("--csv_name", default="infer_results.csv", help="CSV report filename.")
    parser.add_argument("--write_every", type=int, default=1, help="Rewrite infer_results.* every N completed pairs. Default: 1.")
    parser.add_argument("--save_debug", action="store_true", default=True, help="Save scoring debug outputs. Default: true.")
    parser.add_argument("--no_save_debug", action="store_false", dest="save_debug", help="Disable scoring debug outputs.")
    parser.add_argument("--device", default=None, help="cuda/cpu. Default: config train.device/export.device.")

    parser.add_argument(
        "--realdeg_val_filter",
        action="store_true",
        default=True,
        help="In folder mode, if part/dataset matches config split.holdout, only infer the held-out val half. Default: true.",
    )
    parser.add_argument("--no_realdeg_val_filter", action="store_false", dest="realdeg_val_filter")

    args = parser.parse_args()

    train_mod = import_module_from_path("train_consistency_fusion4_imported", args.train_script)
    cfg = train_mod.load_config(args.config)

    out_dir = expand_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = cfg.get("outputs", {}) or {}
    ckpt_path = expand_path(args.checkpoint or outputs.get("best_model", "best_model.pt"))
    scaler_path = expand_path(args.scaler or outputs.get("scaler", "scaler.pkl"))
    feature_columns_path = expand_path(args.feature_columns or outputs.get("feature_columns", "feature_columns.json"))

    device_name = args.device or cfg.get("train", {}).get("device") or cfg.get("export", {}).get("device", "cuda")
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    feature_columns = load_feature_columns(feature_columns_path, checkpoint=checkpoint)
    scaler = load_scaler(scaler_path, checkpoint=checkpoint)
    branch_cols = list(getattr(train_mod, "BRANCH_PENALTY_COLUMNS", [
        "lowfreq_penalty", "texture_penalty", "semantic_under_penalty", "unmatched_penalty"
    ]))

    model = train_mod.make_model(cfg, in_dim=len(feature_columns)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded fusion model. feature_dim={len(feature_columns)}, branch_cols={branch_cols}")

    # Infer metadata and expand list-valued dataset/method specs.
    meta_source = args.sr or args.sr_dir or args.features_csv or "."
    inferred_part, inferred_dataset, inferred_method = infer_part_dataset_method_from_path(meta_source)
    part = args.part or inferred_part
    datasets = [str(x) for x in (args.datasets if args.datasets else [args.dataset or inferred_dataset])]
    methods = [str(x) for x in (args.methods if args.methods else [args.method or inferred_method])]

    # Resolve config feature CSV shortcut.
    if args.use_features_csv and not args.features_csv:
        args.features_csv = "config"
    if args.features_csv == "config":
        args.features_csv = cfg.get("paths", {}).get("features_csv")
        if not args.features_csv:
            raise ValueError("--features_csv config requested, but config paths.features_csv is missing")

    if (len(datasets) > 1 or len(methods) > 1) and (args.lr or args.sr or args.lr_dir or args.sr_dir):
        print("[warning] Multiple datasets/methods with explicit --lr/--sr/--lr_dir/--sr_dir: "
              "the same explicit paths will be reused for each spec. Usually you should omit "
              "explicit paths and use --data_base for multi-spec image inference, or use --features_csv.")

    rows: List[Dict[str, Any]] = []
    debug_root = out_dir / "pair_debug"
    # Create empty result files immediately so the user can see where final results will appear.
    txt_path, csv_path = save_results_files(out_dir, args.csv_name, args.txt_name, rows, feature_columns, branch_cols)

    if args.features_csv:
        total_expected = 0
        for dataset in datasets:
            for method in methods:
                try:
                    feature_rows = make_feature_rows_from_csv(args, cfg, str(part), str(dataset), str(method))
                except Exception as exc:
                    print(f"[FEATURE CSV SPEC WARNING] part={part}, dataset={dataset}, method={method}: {exc}")
                    continue
                total_expected += len(feature_rows)
                print(
                    f"Inference feature rows: {len(feature_rows)} | part={part}, dataset={dataset}, method={method} | "
                    f"features_csv={expand_path(args.features_csv)}"
                )
                with torch.no_grad():
                    for src_row in tqdm(feature_rows, desc=f"Inferring from features {dataset}/{method}"):
                        rel_name = str(src_row.get("relative_image_name", ""))
                        row = dict(src_row)
                        try:
                            X, B = row_to_model_tensors(row, feature_columns, scaler, branch_cols, device)
                            score_01, weights, pred_penalty, raw_score = model(X, B)
                            row["pred_score"] = float(score_01.detach().cpu().numpy()[0])
                            row["pred_raw_score"] = float(raw_score.detach().cpu().numpy()[0])
                            row["pred_final_penalty"] = float(pred_penalty.detach().cpu().numpy()[0])
                            w = weights.detach().cpu().numpy()[0]
                            if len(w) == 4:
                                row["w_low"] = float(w[0])
                                row["w_texture"] = float(w[1])
                                row["w_semantic"] = float(w[2])
                                row["w_unmatched"] = float(w[3])
                            row["status"] = "ok"
                            row["error"] = ""
                        except Exception as exc:
                            row["status"] = "error"
                            row["error"] = traceback.format_exc()
                            row["pred_score"] = np.nan
                            row["pred_raw_score"] = np.nan
                            row["pred_final_penalty"] = np.nan
                            print(f"[INFER ERROR] {dataset}/{method}/{rel_name}: {exc}")
                        rows.append(row)
                        if int(args.write_every) > 0 and (len(rows) % int(args.write_every) == 0):
                            txt_path, csv_path = save_results_files(out_dir, args.csv_name, args.txt_name, rows, feature_columns, branch_cols)
                            print(f"[RESULT WRITTEN] {len(rows)} rows -> score={format_value(row.get('pred_score', np.nan))} | {txt_path}")
        if total_expected == 0:
            print("[warning] No feature rows matched any requested dataset/method spec.")
    else:
        scoring_script = args.scoring_script or cfg.get("export", {}).get("scoring_script", "sr_consistency_score_overall5.py")
        repo_root = expand_path(cfg.get("export", {}).get("repo_root", "."))
        scoring_mod = train_mod.import_scoring_module(expand_path(scoring_script), repo_root)
        scoring_args = build_scoring_args(train_mod, scoring_mod, cfg, str(device), bool(args.save_debug))
        scoring_model = build_scoring_model(scoring_mod, scoring_args)

        all_specs = [(str(part), str(dataset), str(method)) for dataset in datasets for method in methods]
        for spec_part, dataset, method in all_specs:
            pairs = make_pairs_for_spec(args, cfg, spec_part, dataset, method)
            print(f"Inference pairs: {len(pairs)} | part={spec_part}, dataset={dataset}, method={method}")

            with torch.no_grad():
                for rel_name, lr_path, sr_path in tqdm(pairs, desc=f"Inferring {dataset}/{method}"):
                    pair_out_dir = debug_root / str(dataset) / str(method) / Path(rel_name).with_suffix("")
                    scoring_args.dataset = str(dataset)
                    scoring_args.method = str(method)
                    try:
                        pair_score, mask_rows = scoring_mod.process_one_pair(
                            model=scoring_model,
                            image1_path=lr_path,
                            image2_path=sr_path,
                            rel_key=Path(rel_name),
                            pair_out_dir=pair_out_dir,
                            args=scoring_args,
                        )
                        row = train_mod.aggregate_pair_features(pair_score, mask_rows, lr_path, sr_path, cfg)
                        row["status"] = "ok"
                        row["error"] = ""

                        row["part"] = str(spec_part)
                        row["dataset"] = str(dataset)
                        row["method"] = str(method)
                        row["relative_image_name"] = str(rel_name)
                        row["image1_path"] = str(lr_path)
                        row["image2_path"] = str(sr_path)
                        row["debug_dir"] = str(pair_out_dir)

                        X, B = row_to_model_tensors(row, feature_columns, scaler, branch_cols, device)
                        score_01, weights, pred_penalty, raw_score = model(X, B)
                        row["pred_score"] = float(score_01.detach().cpu().numpy()[0])
                        row["pred_raw_score"] = float(raw_score.detach().cpu().numpy()[0])
                        row["pred_final_penalty"] = float(pred_penalty.detach().cpu().numpy()[0])
                        w = weights.detach().cpu().numpy()[0]
                        if len(w) == 4:
                            row["w_low"] = float(w[0])
                            row["w_texture"] = float(w[1])
                            row["w_semantic"] = float(w[2])
                            row["w_unmatched"] = float(w[3])
                    except Exception as exc:
                        row = {
                            "part": str(spec_part),
                            "dataset": str(dataset),
                            "method": str(method),
                            "relative_image_name": str(rel_name),
                            "image1_path": str(lr_path),
                            "image2_path": str(sr_path),
                            "status": "error",
                            "error": traceback.format_exc(),
                            "pred_score": np.nan,
                            "pred_raw_score": np.nan,
                            "pred_final_penalty": np.nan,
                        }
                        print(f"[INFER ERROR] {dataset}/{method}/{rel_name}: {exc}")
                    rows.append(row)
                    if int(args.write_every) > 0 and (len(rows) % int(args.write_every) == 0):
                        txt_path, csv_path = save_results_files(out_dir, args.csv_name, args.txt_name, rows, feature_columns, branch_cols)
                        print(f"[RESULT WRITTEN] {len(rows)} rows -> score={format_value(row.get('pred_score', np.nan))} | {txt_path}")

                    if bool(getattr(scoring_args, "empty_cache_each_pair", False)) and str(device).startswith("cuda"):
                        torch.cuda.empty_cache()

    # Save once more at the end to ensure the final files are complete.
    txt_path, csv_path = save_results_files(out_dir, args.csv_name, args.txt_name, rows, feature_columns, branch_cols)

    # Compact summary.
    ok_scores = [float(r["pred_score"]) for r in rows if str(r.get("status")) == "ok" and math.isfinite(float(r.get("pred_score", np.nan)))]
    print("\nInference finished.")
    print(f"Saved TXT: {txt_path}")
    print(f"Saved CSV: {csv_path}")
    if ok_scores:
        print(f"num_ok={len(ok_scores)} | mean_score={np.mean(ok_scores):.6f} | min={np.min(ok_scores):.6f} | max={np.max(ok_scores):.6f}")
    else:
        print("No successful scores were produced. Check error column in output files.")


if __name__ == "__main__":
    main()
