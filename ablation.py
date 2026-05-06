#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convenient ablation launcher for train_consistency_fusion4.py.

This wrapper does not require changing the training code for most ablations.
It generates a per-ablation config and, when needed, a per-ablation features.csv.

Examples:
  python ablation.py --config ablation.yaml --list
  python ablation.py --config ablation.yaml --ablation full --mode train
  python ablation.py --config ablation.yaml --ablation wo_texture --mode train
  python ablation.py --config ablation.yaml --ablation fixed_sr_to_lr --mode all
  python ablation.py --config ablation.yaml --ablation wo_texture --mode config
  python ablation.py --config ablation.yaml --ablation wo_texture --print_infer
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


BRANCH_PENALTY_COLUMNS = [
    "lowfreq_penalty",
    "texture_penalty",
    "semantic_under_penalty",
    "unmatched_penalty",
]

BRANCH_TO_PENALTY_COLUMN = {
    "lowfreq": "lowfreq_penalty",
    "texture": "texture_penalty",
    "semantic": "semantic_under_penalty",
    "unmatched": "unmatched_penalty",
}

# Columns containing these substrings will be dropped for the corresponding
# branch, except the core branch penalty column itself, which is kept and set 0.
BRANCH_DROP_KEYWORDS = {
    "lowfreq": [
        "lowfreq",
        "luma",
        "grad",
    ],
    "texture": [
        "texture",
        "highfreq",
        "layer6",
        "layer12",
        "layer18",
        "layer24",
    ],
    "semantic": [
        "semantic",
        "tau_pair",
        "k_image",
        "valid_masks",
        "valid_k",
        "k_ratio",
        "num_masks",
        "mask_area_entropy",
        "largest_mask",
        "semantic_ref",
        "semantic_correspondence",
    ],
    "unmatched": [
        "unmatched",
        "weakly_supported",
        "support_ratio",
    ],
}


def expand_path(p: str | Path, base_dir: Optional[str | Path] = None) -> Path:
    path = Path(str(p)).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir).expanduser() / path
    return path


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def import_module_from_path(module_name: str, path: str | Path):
    path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from: {path}")
    mod = importlib.util.module_from_spec(spec)
    # Required for @dataclass in dynamically imported train scripts.
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def set_output_paths(cfg: Dict[str, Any], out_dir: Path, features_csv: Path) -> Dict[str, Any]:
    cfg = dict(cfg)
    cfg.setdefault("paths", {})
    cfg.setdefault("outputs", {})

    out_dir_str = str(out_dir)
    cfg["paths"]["output_dir"] = out_dir_str
    cfg["paths"]["features_csv"] = str(features_csv)
    cfg["paths"]["resolved_config"] = str(out_dir / "config_resolved.yaml")

    cfg["outputs"]["best_model"] = str(out_dir / "best_model.pt")
    cfg["outputs"]["last_model"] = str(out_dir / "last_model.pt")
    cfg["outputs"]["scaler"] = str(out_dir / "scaler.pkl")
    cfg["outputs"]["feature_columns"] = str(out_dir / "feature_columns.json")
    cfg["outputs"]["train_log"] = str(out_dir / "train_log.csv")
    cfg["outputs"]["val_predictions"] = str(out_dir / "val_predictions.csv")
    cfg["outputs"]["val_metrics"] = str(out_dir / "val_metrics.json")
    cfg["outputs"]["val_metrics_csv"] = str(out_dir / "val_metrics.csv")
    return cfg


def normalize_disabled(disabled: Sequence[str]) -> List[str]:
    out = []
    aliases = {
        "unsupported": "unmatched",
        "support": "unmatched",
        "hallucination": "unmatched",
        "low-frequency": "lowfreq",
        "low_frequency": "lowfreq",
    }
    for x in disabled or []:
        key = str(x).strip().lower()
        key = aliases.get(key, key)
        if key not in BRANCH_TO_PENALTY_COLUMN:
            raise ValueError(f"Unknown disabled branch: {x}. Valid: {sorted(BRANCH_TO_PENALTY_COLUMN)}")
        if key not in out:
            out.append(key)
    return out


def ablate_feature_csv(base_csv: Path, out_csv: Path, disabled: Sequence[str]) -> Tuple[Path, Dict[str, Any]]:
    disabled = normalize_disabled(disabled)
    if not disabled:
        return base_csv, {"disabled_branches": [], "dropped_columns": [], "zeroed_columns": []}
    if not base_csv.exists():
        raise FileNotFoundError(f"Base features_csv not found: {base_csv}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(base_csv)

    zeroed: List[str] = []
    keep_always = set(BRANCH_PENALTY_COLUMNS)
    drop_cols: List[str] = []

    for branch in disabled:
        penalty_col = BRANCH_TO_PENALTY_COLUMN[branch]
        if penalty_col not in df.columns:
            raise KeyError(f"Required penalty column missing in features_csv: {penalty_col}")
        df[penalty_col] = 0.0
        zeroed.append(penalty_col)

        keywords = [k.lower() for k in BRANCH_DROP_KEYWORDS.get(branch, [])]
        for c in df.columns:
            c_low = c.lower()
            if c in keep_always:
                continue
            if any(k in c_low for k in keywords):
                if c not in drop_cols:
                    drop_cols.append(c)

    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    df.to_csv(out_csv, index=False)
    meta = {
        "base_csv": str(base_csv),
        "out_csv": str(out_csv),
        "disabled_branches": disabled,
        "zeroed_columns": zeroed,
        "dropped_columns": drop_cols,
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
    }
    with open(out_csv.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return out_csv, meta



def seed_features_for_fixed_sr_to_lr(base_csv: Path, out_csv: Path) -> Tuple[Path, Dict[str, Any]]:
    """Create a seed features.csv for fixed SR->LR matching.

    The full/adaptive features can be reused whenever adaptive matching already
    used SR/image2 as reference. Only rows where LR/image1 had fewer valid masks
    than SR/image2 need recomputation, because adaptive used LR->SR there while
    fixed SR->LR must force image2/SR as ref.

    Recompute condition:
      valid_masks_image1 < valid_masks_image2
    Robust fallback:
      semantic_ref_is_sr == 0 or semantic_correspondence_direction starts with lr_ref

    The output CSV contains only reusable OK rows. Training should run export/all
    afterward so missing rows are appended by train_consistency_fusion4.py.
    """
    if not base_csv.exists():
        raise FileNotFoundError(f"Base features_csv not found: {base_csv}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(base_csv)
    total = int(len(df))

    # Drop old error rows so export can retry them.
    if "status" in df.columns:
        status = df["status"].fillna("ok").astype(str).str.lower()
        is_error = status.eq("error")
    else:
        is_error = pd.Series(False, index=df.index)

    need_recompute = pd.Series(False, index=df.index)

    if {"valid_masks_image1", "valid_masks_image2"}.issubset(df.columns):
        v1 = pd.to_numeric(df["valid_masks_image1"], errors="coerce")
        v2 = pd.to_numeric(df["valid_masks_image2"], errors="coerce")
        need_recompute = need_recompute | (v1 < v2)

    if "semantic_ref_is_sr" in df.columns:
        ref_is_sr = pd.to_numeric(df["semantic_ref_is_sr"], errors="coerce")
        need_recompute = need_recompute | ref_is_sr.eq(0)

    if "semantic_correspondence_direction" in df.columns:
        direction = df["semantic_correspondence_direction"].fillna("").astype(str).str.lower()
        need_recompute = need_recompute | direction.str.startswith("lr_ref")

    reusable = ~(need_recompute | is_error)
    seed_df = df.loc[reusable].copy()

    # Make sure export does not skip recompute rows: they must be absent here.
    seed_df.to_csv(out_csv, index=False)

    meta = {
        "mode": "reuse_adaptive_then_export_missing_fixed_sr_to_lr",
        "base_csv": str(base_csv),
        "out_csv": str(out_csv),
        "rule": "reuse rows where adaptive already used SR/image2 ref; recompute rows where valid_masks_image1 < valid_masks_image2 or semantic_ref_is_sr == 0",
        "total_rows_base": total,
        "seed_rows_reused": int(len(seed_df)),
        "rows_marked_for_recompute": int(need_recompute.sum()),
        "error_rows_dropped_for_retry": int(is_error.sum()),
        "rows_absent_from_seed": int(total - len(seed_df)),
    }
    with open(out_csv.with_suffix(".seed_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return out_csv, meta

def build_ablation_config(ablation_cfg_path: Path, name: str) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    abl_root = load_yaml(ablation_cfg_path)
    base = abl_root.get("base", {})
    variants = abl_root.get("variants", {})
    if name not in variants:
        raise KeyError(f"Unknown ablation '{name}'. Available: {sorted(variants)}")

    train_script = expand_path(base.get("train_script", "train_consistency_fusion4.py"), ablation_cfg_path.parent)
    train_mod = import_module_from_path("train_consistency_fusion4_for_ablation", train_script)

    base_config_path = expand_path(base.get("config", "config4.yaml"), ablation_cfg_path.parent)
    cfg = train_mod.load_config(base_config_path)

    # Common overrides, then variant-specific overrides.
    cfg = deep_update(cfg, abl_root.get("common_overrides", {}) or {})
    variant = variants[name] or {}
    cfg = deep_update(cfg, variant.get("config_overrides", {}) or {})

    cfg.setdefault("ablation", {})
    cfg["ablation"].update({
        "name": name,
        "description": variant.get("description", ""),
        "disabled_branches": normalize_disabled(variant.get("disabled_branches", [])),
        "disable_residual": bool(variant.get("disable_residual", False)),
        "semantic_ref_mode": str(variant.get("semantic_ref_mode", "fewer_valid_masks")),
        "feature_source": str(variant.get("feature_source", "base")),
    })

    if cfg["ablation"].get("disable_residual", False):
        cfg.setdefault("model", {})
        cfg["model"]["type"] = "gated"

    cfg.setdefault("export", {})
    cfg["export"]["scoring_script"] = str(expand_path(base.get("scoring_script", cfg["export"].get("scoring_script", "sr_consistency_score_overall5.py")), ablation_cfg_path.parent))
    cfg["export"].setdefault("extra_scoring_args", {})
    if cfg["ablation"]["semantic_ref_mode"] != "fewer_valid_masks":
        cfg["export"]["extra_scoring_args"]["semantic_ref_mode"] = cfg["ablation"]["semantic_ref_mode"]

    output_root = expand_path(base.get("output_root", "outputs/ablations"), ablation_cfg_path.parent)
    out_dir = output_root / name

    base_features_csv = base.get("base_features_csv", None)
    if base_features_csv:
        base_features = expand_path(base_features_csv, ablation_cfg_path.parent)
    else:
        base_features = expand_path(cfg["paths"]["features_csv"], ablation_cfg_path.parent)

    feature_source = cfg["ablation"]["feature_source"]
    disabled = cfg["ablation"]["disabled_branches"]
    if feature_source == "export":
        features_csv = out_dir / "features.csv"
    elif feature_source in {"reuse_adaptive_then_export_missing", "partial_export_from_base"}:
        features_csv = out_dir / "features.csv"
        features_csv, meta = seed_features_for_fixed_sr_to_lr(base_features, features_csv)
        cfg["ablation"]["feature_seed_meta"] = meta
        cfg.setdefault("export", {})
        cfg["export"]["skip_existing_features"] = True
        cfg["export"]["overwrite_feature_csv"] = False
    elif disabled:
        features_csv = out_dir / f"features_{name}.csv"
        features_csv, meta = ablate_feature_csv(base_features, features_csv, disabled)
        cfg["ablation"]["feature_ablation_meta"] = meta
    else:
        features_csv = base_features

    cfg = set_output_paths(cfg, out_dir, features_csv)

    generated_config = out_dir / f"config_{name}.yaml"
    save_yaml(cfg, generated_config)
    return cfg, variant, generated_config


def list_variants(ablation_cfg_path: Path) -> None:
    cfg = load_yaml(ablation_cfg_path)
    variants = cfg.get("variants", {})
    print("Available ablations:")
    for name, v in variants.items():
        desc = (v or {}).get("description", "")
        source = (v or {}).get("feature_source", "base")
        disabled = (v or {}).get("disabled_branches", [])
        residual = "no-residual" if (v or {}).get("disable_residual", False) else "residual"
        print(f"  {name:18s} | source={source:17s} | disabled={disabled} | {residual} | {desc}")


def print_infer_command(abl_cfg: Dict[str, Any], generated_config: Path, ablation_cfg_path: Path) -> None:
    root = load_yaml(ablation_cfg_path).get("base", {})
    infer_script = expand_path(root.get("infer_script", "infer_consistency_fusion.py"), ablation_cfg_path.parent)
    features_csv = abl_cfg["paths"]["features_csv"]
    out_dir = Path(abl_cfg["paths"]["output_dir"]) / "infer"
    data_cfg = abl_cfg.get("data", {})
    methods = data_cfg.get("methods", [])
    # Keep command generic; user can edit datasets/methods.
    cmd = [
        "python", str(infer_script),
        "--config", str(generated_config),
        "--train_script", str(expand_path(root.get("train_script", "train_consistency_fusion4.py"), ablation_cfg_path.parent)),
        "--use_features_csv",
        "--features_csv", str(features_csv),
        "--part", str(abl_cfg.get("split", {}).get("holdout", {}).get("part", data_cfg.get("part", "part2"))),
        "--datasets", str(abl_cfg.get("split", {}).get("holdout", {}).get("dataset", data_cfg.get("dataset", "RealDeg"))),
        "--out_dir", str(out_dir),
    ]
    if methods:
        cmd += ["--methods"] + [str(m) for m in methods]
    print("Infer command template:")
    print(" \\\n  ".join(shlex.quote(x) for x in cmd))


def run_train_script(train_script: Path, generated_config: Path, mode: str, extra_args: Sequence[str], dry_run: bool) -> None:
    cmd = [sys.executable, str(train_script), "--config", str(generated_config), "--mode", mode]
    cmd.extend(extra_args)
    print("Command:")
    print(" ".join(shlex.quote(x) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LR/SR consistency fusion ablations.")
    parser.add_argument("--config", default="ablation.yaml", help="Ablation YAML path.")
    parser.add_argument("--ablation", default=None, help="Ablation name, e.g. full, wo_texture.")
    parser.add_argument("--mode", default="train", choices=["config", "export", "train", "all"], help="Action to pass to training script. 'config' only writes generated config.")
    parser.add_argument("--list", action="store_true", help="List available ablations.")
    parser.add_argument("--dry_run", action="store_true", help="Print command but do not execute.")
    parser.add_argument("--print_infer", action="store_true", help="Print matching infer command template.")
    parser.add_argument("--extra_train_args", nargs=argparse.REMAINDER, default=[], help="Extra args after -- passed to train script.")
    args = parser.parse_args()

    ablation_cfg_path = Path(args.config).expanduser().resolve()
    if args.list:
        list_variants(ablation_cfg_path)
        return
    if not args.ablation:
        raise SystemExit("Please pass --ablation NAME, or use --list.")

    cfg, variant, generated_config = build_ablation_config(ablation_cfg_path, args.ablation)
    print(f"Generated config: {generated_config}")
    print(f"Ablation: {args.ablation}")
    print(f"Output dir: {cfg['paths']['output_dir']}")
    print(f"Features CSV: {cfg['paths']['features_csv']}")
    print(f"Disabled branches: {cfg.get('ablation', {}).get('disabled_branches', [])}")
    print(f"Model type: {cfg.get('model', {}).get('type')}")
    print(f"Semantic ref mode: {cfg.get('ablation', {}).get('semantic_ref_mode')}")

    if args.print_infer:
        print_infer_command(cfg, generated_config, ablation_cfg_path)

    if args.mode == "config":
        return

    train_script = expand_path(load_yaml(ablation_cfg_path).get("base", {}).get("train_script", "train_consistency_fusion4.py"), ablation_cfg_path.parent)

    if cfg.get("ablation", {}).get("feature_source") in {"export", "reuse_adaptive_then_export_missing", "partial_export_from_base"} and args.mode == "train":
        if not Path(cfg["paths"]["features_csv"]).exists():
            print("[warning] This ablation needs an exported/seeded features_csv, but it does not exist yet.")
            print("          Use --mode export or --mode all first, otherwise training will fail.")
        elif cfg.get("ablation", {}).get("feature_source") in {"reuse_adaptive_then_export_missing", "partial_export_from_base"}:
            meta_path = Path(cfg["paths"]["features_csv"]).with_suffix(".seed_meta.json")
            print("[note] This is a partially seeded features_csv. For fixed_sr_to_lr, run --mode export or --mode all before final training so missing LR-ref rows are recomputed.")
            if meta_path.exists():
                print(f"[note] Seed meta: {meta_path}")

    run_train_script(
        train_script=train_script,
        generated_config=generated_config,
        mode=args.mode,
        extra_args=args.extra_train_args,
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
