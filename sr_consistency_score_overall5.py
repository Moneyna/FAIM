#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch LR/SR consistency scoring with INSID3-style semantic correspondence
and DINOv3 multi-layer texture feature distribution matching.

Convention:
  image1 = LR / bicubic LR / low-quality input side
  image2 = SR / restored image side

Core design:
  1) Run DINOv3 twice: semantic branch resizes images whose longest side >768
     and extracts layer 24; texture branch keeps original size and extracts layers 6/12/18.
  2) Use semantic layer 24 for self-clustering and image2 -> image1 semantic matching.
  3) Select an adaptive pair-level tau using image2 only, so that image2 has
     at least --min-base-masks valid self masks. Use the same tau for image1.
  4) For each image2 mask, find the corresponding image1 mask region.
  5) Compute:
       - low-frequency fidelity penalty from blurred luminance + blurred gradient difference
       - semantic area inconsistency penalty
       - texture inconsistency penalty using spatially constrained layer 6/12/18 NN cosine residual
       - unmatched / hallucination penalty for SR regions weakly supported by LR
  6) Save compact outputs by default:
       output_dir/all_scores.csv
       output_dir/<image_name>/overall.png
     If --save-debug is set, additionally save per-image debug files under:
       output_dir/<image_name>/fused_texture_anomaly_overlay.png
       output_dir/<image_name>/mask_scores.csv
       output_dir/<image_name>/meta.json

Run from the INSID3 repository root, because this script imports:
  from models import build_insid3
  from utils.clustering import agglomerative_clustering, compute_cluster_prototypes
  from utils.refinement import upsample_mask
"""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import os

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

from models import build_insid3
from utils.clustering import agglomerative_clustering, compute_cluster_prototypes
from utils.refinement import upsample_mask


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EPS = 1e-8

# -----------------------------------------------------------------------------
# Internal algorithm defaults. Keep CLI compact; expose only the two most useful
# scoring knobs requested by the user: min_base_masks and texture_spatial_radius.
# -----------------------------------------------------------------------------
DEFAULT_MODEL_SIZE = "large"
DEFAULT_IMAGE_SIZE = 768
DEFAULT_SEMANTIC_MAX_SIZE = 1024 #2048 #768
DEFAULT_SVD_COMPS = 500
DEFAULT_SEMANTIC_LAYER = 24
DEFAULT_TEXTURE_LAYERS = "6,12,18"
DEFAULT_TEXTURE_WEIGHTS = "0.5,0.3,0.2"
DEFAULT_TAU = 0.88
DEFAULT_ADAPTIVE_TAU_STEP = 0.01
DEFAULT_ADAPTIVE_TAU_MAX = 0.98
DEFAULT_MERGE_THRESH = 0.35
DEFAULT_BATCH_SIZE = 64
DEFAULT_NN_CHUNK_SIZE = 2048
DEFAULT_TEXTURE_NN_CHUNK_SIZE = 2048
DEFAULT_TEXTURE_STAT = "mean_p90_p95"
DEFAULT_TEXTURE_TOL = 0.05
DEFAULT_TEXTURE_CAP = 0.30
DEFAULT_MIN_PATCHES = 8
DEFAULT_MAX_REF_MASKS = None
DEFAULT_AREA_MODE = "auto"
DEFAULT_SEMANTIC_TOL = 0.01
DEFAULT_SEMANTIC_CAP = 1.0
DEFAULT_SEMANTIC_OVER_WEIGHT = 0.0 #1.0
DEFAULT_SEMANTIC_UNDER_WEIGHT = 2.0
DEFAULT_MIN_SUPPORT_RATIO = 0.15 #0.05
# Low-frequency branch defaults.
# It compares low-pass luminance and low-pass gradients between image1 and image2.
DEFAULT_LOWFREQ_SIGMA = 3.0
DEFAULT_LOWFREQ_CAP = 0.12

# Low-frequency fidelity is the main consistency signal; DINO texture/semantic are auxiliary.
DEFAULT_LAMBDA_LOWFREQ = 1.0
DEFAULT_LAMBDA_SEMANTIC = 0.2 #0.3 #1.0
DEFAULT_LAMBDA_TEXTURE = 0.5 #1.0 #0.7
DEFAULT_LAMBDA_UNMATCHED = 1.0 #2.0
DEFAULT_OVERVIEW_TILE_HEIGHT = 360
DEFAULT_HEATMAP_ALPHA = 0.60


# -----------------------------------------------------------------------------
# Basic image / folder utilities
# -----------------------------------------------------------------------------

def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def id_to_color(idx: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(12345 + int(idx))
    return tuple(int(x) for x in rng.integers(40, 255, size=3))


def list_images(folder: Path, recursive: bool = False) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted([p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in IMG_EXTS])


def find_pairs(image1_dir: Path, image2_dir: Path, recursive: bool = False):
    image1_files = list_images(image1_dir, recursive=recursive)
    image2_files = list_images(image2_dir, recursive=recursive)

    if recursive:
        image1_map = {p.relative_to(image1_dir): p for p in image1_files}
        image2_map = {p.relative_to(image2_dir): p for p in image2_files}
    else:
        image1_map = {p.name: p for p in image1_files}
        image2_map = {p.name: p for p in image2_files}

    common = sorted(set(image1_map.keys()) & set(image2_map.keys()))
    missing_in_image2 = sorted(set(image1_map.keys()) - set(image2_map.keys()))
    missing_in_image1 = sorted(set(image2_map.keys()) - set(image1_map.keys()))
    pairs = [(Path(k), image1_map[k], image2_map[k]) for k in common]
    return pairs, missing_in_image2, missing_in_image1


def safe_pair_dir_name(rel_key: Path) -> Path:
    return rel_key.with_suffix("")


def get_model_patch_size(model, fallback: int = 16) -> int:
    candidates = [
        model,
        getattr(model, "encoder", None),
        getattr(model, "_encoder", None),
        getattr(model, "backbone", None),
        getattr(model, "_backbone", None),
    ]
    for obj in candidates:
        if obj is None:
            continue
        patch_size = getattr(obj, "patch_size", None)
        if patch_size is not None:
            if isinstance(patch_size, (tuple, list)):
                patch_size = patch_size[0]
            return int(patch_size)
    return int(fallback)


class LocalDINOv3Wrapper(torch.nn.Module):
    """Small local wrapper: support tuple n + batched intermediate layers."""
    def __init__(self, enc, patch_size=16):
        super().__init__()
        self.enc = enc
        self.model = getattr(enc, "model", enc)
        ps = getattr(enc, "patch_size", None) or getattr(getattr(self.model, "config", None), "patch_size", patch_size)
        self.patch_size = int(ps[0] if isinstance(ps, (tuple, list)) else ps)

    @torch.no_grad()
    def get_intermediate_layers(self, x, n=1, reshape=False, return_class_token=False, norm=True):
        out = self.model(pixel_values=x, output_hidden_states=True, return_dict=True)
        hs = list(out.hidden_states)[1:]  # drop embedding output; hs[0] = block 1
        idxs = list(range(len(hs) - int(n), len(hs))) if isinstance(n, int) else [int(i) % len(hs) for i in n]
        B, _, H, W = x.shape
        gh, gw = H // self.patch_size, W // self.patch_size
        n_patch = gh * gw
        ret = []
        for i in idxs:
            t = hs[i]
            n_special = t.shape[1] - n_patch
            cls = t[:, 0]
            patch = t[:, n_special:n_special + n_patch]
            if reshape:
                patch = patch.reshape(B, gh, gw, -1).permute(0, 3, 1, 2).contiguous()
            ret.append((patch, cls) if return_class_token else patch)
        return tuple(ret)


def get_dino_encoder(model):
    for name in ("_encoder", "encoder", "_backbone", "backbone"):
        enc = getattr(model, name, None)
        if enc is not None and hasattr(enc, "get_intermediate_layers"):
            if hasattr(enc, "model"):
                return LocalDINOv3Wrapper(enc, patch_size=get_model_patch_size(model, 16))
            return enc
    raise AttributeError(
        "Could not find a DINO encoder with get_intermediate_layers() inside the INSID3 model."
    )


def load_image_for_dino(
    image_path: str | Path,
    device: str,
    patch_size: int = 16,
    pad_mode: str = "replicate",
    resize_max_size: int | None = None,
):
    """
    Load RGB image for DINO.

    If resize_max_size is not None, only images with max(H,W) greater than
    resize_max_size are resized so their longest side equals resize_max_size.
    Then only bottom/right padding is added to make H/W divisible by patch_size.
    """
    pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil.size
    input_w, input_h = orig_w, orig_h

    if resize_max_size is not None and max(orig_h, orig_w) > int(resize_max_size):
        scale = float(resize_max_size) / float(max(orig_h, orig_w))
        input_w = max(1, int(round(orig_w * scale)))
        input_h = max(1, int(round(orig_h * scale)))
        resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
        pil = pil.resize((input_w, input_h), resample=resample)

    arr = np.asarray(pil).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    pad_h = (patch_size - input_h % patch_size) % patch_size
    pad_w = (patch_size - input_w % patch_size) % patch_size

    if pad_h or pad_w:
        if pad_mode == "constant":
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        else:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)

    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std

    orig_size = (orig_h, orig_w)
    input_size = (input_h, input_w)
    padded_size = (int(x.shape[-2]), int(x.shape[-1]))
    return x.to(device, non_blocking=True), orig_size, input_size, padded_size


def load_image_keep_original_size(
    image_path: str | Path,
    device: str,
    patch_size: int = 16,
    pad_mode: str = "replicate",
):
    """Load RGB image without resizing; only pad bottom/right to patch-size multiples."""
    x, orig_size, _input_size, padded_size = load_image_for_dino(
        image_path=image_path,
        device=device,
        patch_size=patch_size,
        pad_mode=pad_mode,
        resize_max_size=None,
    )
    return x, orig_size, padded_size

# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------

@torch.no_grad()
def extract_dino_layers_batch(
    encoder,
    image_tensor: torch.Tensor,
    layers_1based: Iterable[int],
    use_amp: bool,
    device: str,
) -> Dict[int, torch.Tensor]:
    """
    Extract selected DINOv3 ViT block outputs for a whole mini-batch.

    Args:
      image_tensor: [B, 3, H, W]
      layers_1based: e.g. [6, 12, 18, 24]

    Returns:
      dict layer -> L2-normalized feature map [B, C, Hf, Wf].
    """
    layers = sorted(set(int(l) for l in layers_1based))
    if any(l <= 0 for l in layers):
        raise ValueError(f"layers must be 1-based positive integers, got {layers}")
    block_indices = tuple(l - 1 for l in layers)

    amp_enabled = use_amp and str(device).startswith("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
        # Different INSID3 / DINOv3 wrappers expose slightly different
        # get_intermediate_layers signatures. In particular, the local
        # HuggingFaceDINOv3Wrapper used in some INSID3 setups does not accept
        # return_class_token. Try the most specific call first, then fall back
        # while preserving batch extraction.
        call_variants = [
            dict(n=block_indices, reshape=True, return_class_token=False, norm=True),
            dict(n=block_indices, reshape=True, norm=True),
            dict(n=block_indices, reshape=True, return_class_token=False),
            dict(n=block_indices, reshape=True),
        ]
        last_type_error = None
        outs = None
        for kwargs in call_variants:
            try:
                outs = encoder.get_intermediate_layers(image_tensor, **kwargs)
                break
            except TypeError as e:
                last_type_error = e
        if outs is None:
            raise last_type_error

    if not isinstance(outs, (tuple, list)):
        outs = [outs]
    if len(outs) != len(layers):
        raise RuntimeError(
            f"Expected {len(layers)} feature maps from get_intermediate_layers, got {len(outs)}. "
            f"layers={layers}, block_indices={block_indices}"
        )

    feats = {}
    for layer, fmap in zip(layers, outs):
        # fmap: [B, C, Hf, Wf]
        if fmap.ndim != 4:
            raise RuntimeError(f"Layer {layer}: expected [B, C, H, W], got {tuple(fmap.shape)}")
        feat = fmap.float().contiguous()
        feat = F.normalize(feat, p=2, dim=1, eps=1e-6)
        feats[layer] = feat
    return feats


@torch.no_grad()
def extract_dino_layers_single(
    encoder,
    image_tensor: torch.Tensor,
    layers_1based: Iterable[int],
    use_amp: bool,
    device: str,
) -> Dict[int, torch.Tensor]:
    """Backward-compatible wrapper for a single image [1,3,H,W]."""
    feats_batched = extract_dino_layers_batch(
        encoder=encoder,
        image_tensor=image_tensor,
        layers_1based=layers_1based,
        use_amp=use_amp,
        device=device,
    )
    feats = {layer: fmap[0].contiguous() for layer, fmap in feats_batched.items()}
    return feats


@torch.no_grad()
def extract_pair_layers_from_tensors(
    model,
    img1: torch.Tensor,
    img2: torch.Tensor,
    layers: List[int],
    use_amp: bool,
    device: str,
):
    """Extract selected layers for image1/image2; batch them when shapes match."""
    encoder = get_dino_encoder(model)
    if img1.shape == img2.shape:
        imgs = torch.cat([img2, img1], dim=0)  # keep [image2, image1] order
        feats_batched = extract_dino_layers_batch(encoder, imgs, layers, use_amp, device)
        feats2 = {layer: fmap[0].contiguous() for layer, fmap in feats_batched.items()}
        feats1 = {layer: fmap[1].contiguous() for layer, fmap in feats_batched.items()}
        joint = True
    else:
        feats1 = extract_dino_layers_single(encoder, img1, layers, use_amp, device)
        feats2 = extract_dino_layers_single(encoder, img2, layers, use_amp, device)
        joint = False
    return feats1, feats2, joint


@torch.no_grad()
def extract_pair_multilayer_features(
    model,
    image1_path: str | Path,
    image2_path: str | Path,
    device: str,
    use_amp: bool,
    layers: List[int],
    semantic_layer: int,
):
    """
    Two-encoder-pass extraction:
      1) semantic pass: resize only images whose longest side > 768, extract layer 24;
      2) texture pass: keep original size, extract layers 6/12/18.
    """
    patch_size = get_model_patch_size(model, fallback=16)

    # Semantic branch: downsize only large images for layer-24 masks.
    sem_img1, orig1, sem_input1, sem_padded1 = load_image_for_dino(
        image1_path, device=device, patch_size=patch_size, resize_max_size=DEFAULT_SEMANTIC_MAX_SIZE
    )
    sem_img2, orig2, sem_input2, sem_padded2 = load_image_for_dino(
        image2_path, device=device, patch_size=patch_size, resize_max_size=DEFAULT_SEMANTIC_MAX_SIZE
    )

    # Texture branch: original resolution for layers 6/12/18.
    tex_img1, orig1_tex, tex_input1, tex_padded1 = load_image_for_dino(
        image1_path, device=device, patch_size=patch_size, resize_max_size=None
    )
    tex_img2, orig2_tex, tex_input2, tex_padded2 = load_image_for_dino(
        image2_path, device=device, patch_size=patch_size, resize_max_size=None
    )
    if orig1 != orig1_tex or orig2 != orig2_tex:
        raise RuntimeError("Original-size bookkeeping mismatch between semantic and texture branches.")

    sem_feats1, sem_feats2, sem_joint = extract_pair_layers_from_tensors(
        model, sem_img1, sem_img2, [semantic_layer], use_amp, device
    )
    tex_feats1, tex_feats2, tex_joint = extract_pair_layers_from_tensors(
        model, tex_img1, tex_img2, layers, use_amp, device
    )

    feat1_sem = sem_feats1[semantic_layer].contiguous()
    feat2_sem = sem_feats2[semantic_layer].contiguous()

    # Debias semantic layer only. If grids match, debias jointly to preserve the
    # paired INSID3 behavior; otherwise debias independently.
    if feat1_sem.shape[-2:] == feat2_sem.shape[-2:]:
        fmaps_norm = torch.stack([feat2_sem, feat1_sem], dim=0).unsqueeze(0)  # [1, 2, C, H, W]
        fmaps_deb = model._debias_features(fmaps_norm)
        feat2_deb = fmaps_deb[0, 0].contiguous()
        feat1_deb = fmaps_deb[0, 1].contiguous()
    else:
        print(
            "[warning] semantic feature grids differ: "
            f"image1={feat1_sem.shape[-2:]}, image2={feat2_sem.shape[-2:]}. "
            "Debiasing semantic features independently."
        )
        feat1_deb = model._debias_features(feat1_sem.unsqueeze(0).unsqueeze(0))[0, 0].contiguous()
        feat2_deb = model._debias_features(feat2_sem.unsqueeze(0).unsqueeze(0))[0, 0].contiguous()

    return {
        "img1": sem_img1,
        "img2": sem_img2,
        "orig1": orig1,
        "orig2": orig2,
        # Backward-compatible keys used by semantic masks/overlays.
        "padded1": sem_padded1,
        "padded2": sem_padded2,
        "semantic_input1": sem_input1,
        "semantic_input2": sem_input2,
        "semantic_padded1": sem_padded1,
        "semantic_padded2": sem_padded2,
        "texture_input1": tex_input1,
        "texture_input2": tex_input2,
        "texture_padded1": tex_padded1,
        "texture_padded2": tex_padded2,
        "patch_size": patch_size,
        "joint_batched_forward_semantic": bool(sem_joint),
        "joint_batched_forward_texture": bool(tex_joint),
        # feats1/feats2 are texture features only.
        "feats1": tex_feats1,
        "feats2": tex_feats2,
        "feat1_sem": feat1_sem,
        "feat2_sem": feat2_sem,
        "feat1_deb": feat1_deb,
        "feat2_deb": feat2_deb,
    }

# -----------------------------------------------------------------------------
# Mask upsampling and visualizations
# -----------------------------------------------------------------------------

def upsample_bool_mask(
    mask_hw: torch.Tensor,
    orig_h: int,
    orig_w: int,
    padded_h: int | None = None,
    padded_w: int | None = None,
) -> torch.Tensor:
    out_h = int(padded_h) if padded_h is not None else int(orig_h)
    out_w = int(padded_w) if padded_w is not None else int(orig_w)
    up = upsample_mask(mask_hw.bool(), out_h, out_w)
    return up[:orig_h, :orig_w]


def upsample_float_map(
    fmap_hw: torch.Tensor,
    orig_h: int,
    orig_w: int,
    padded_h: int | None = None,
    padded_w: int | None = None,
    mode: str = "bilinear",
) -> torch.Tensor:
    out_h = int(padded_h) if padded_h is not None else int(orig_h)
    out_w = int(padded_w) if padded_w is not None else int(orig_w)
    x = fmap_hw[None, None].float()
    if mode == "nearest":
        up = F.interpolate(x, size=(out_h, out_w), mode="nearest")[0, 0]
    else:
        up = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)[0, 0]
    return up[:orig_h, :orig_w]


def upsample_labels_nearest(
    labels_hw: torch.Tensor,
    orig_h: int,
    orig_w: int,
    padded_h: int | None = None,
    padded_w: int | None = None,
) -> torch.Tensor:
    out_h = int(padded_h) if padded_h is not None else int(orig_h)
    out_w = int(padded_w) if padded_w is not None else int(orig_w)
    labels = labels_hw[None, None].float()
    up = F.interpolate(labels, size=(out_h, out_w), mode="nearest")[0, 0].long()
    return up[:orig_h, :orig_w]



def resize_bool_mask_to_shape(mask_hw: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """Nearest-neighbor resize of a bool mask from semantic grid to texture grid."""
    x = mask_hw[None, None].float()
    y = F.interpolate(x, size=(int(out_h), int(out_w)), mode="nearest")[0, 0]
    return y.bool().to(mask_hw.device)

def make_label_overlay_image(
    image_path: str | Path,
    labels_hw: torch.Tensor,
    orig_size: Tuple[int, int],
    padded_size: Tuple[int, int],
    alpha: float = 0.45,
) -> Image.Image:
    orig_h, orig_w = orig_size
    padded_h, padded_w = padded_size
    img = Image.open(image_path).convert("RGB").resize((orig_w, orig_h))
    img_np = np.asarray(img).astype(np.float32)

    labels_up = upsample_labels_nearest(labels_hw, orig_h, orig_w, padded_h, padded_w).cpu().numpy()
    k_max = int(labels_up.max()) + 1
    palette = np.zeros((k_max, 3), dtype=np.uint8)
    for k in range(k_max):
        palette[k] = np.array(id_to_color(k), dtype=np.uint8)
    color_np = palette[labels_up].astype(np.float32)
    overlay = ((1.0 - alpha) * img_np + alpha * color_np).clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def make_multi_mask_overlay_image(
    image_path: str | Path,
    masks_and_ids: List[Tuple[torch.Tensor, int]],
    alpha: float = 0.50,
) -> Image.Image:
    """Overlay masks on the image resized to the mask resolution.

    This is needed because semantic masks may be produced at the semantic branch
    resolution, e.g. 768, while the source image can be larger, e.g. 1024.
    """
    img = Image.open(image_path).convert("RGB")
    if masks_and_ids:
        first_mask = masks_and_ids[0][0].detach().cpu().numpy().astype(bool)
        target_h, target_w = first_mask.shape
        img = img.resize((target_w, target_h))
    out = np.asarray(img).astype(np.float32)
    for mask, idx in masks_and_ids:
        m = mask.detach().cpu().numpy().astype(bool)
        if m.shape != out.shape[:2]:
            raise ValueError(f"Mask shape {m.shape} does not match image shape {out.shape[:2]}")
        color = np.array(id_to_color(idx), dtype=np.float32)
        out[m] = (1.0 - alpha) * out[m] + alpha * color
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def make_texture_overlay_image(
    image2_path: str | Path,
    heatmap_feat_hw: torch.Tensor,
    orig2: Tuple[int, int],
    padded2: Tuple[int, int],
    alpha: float = 0.60,
) -> Image.Image:
    orig_h, orig_w = orig2
    padded_h, padded_w = padded2
    heat = upsample_float_map(heatmap_feat_hw, orig_h, orig_w, padded_h, padded_w, mode="bilinear")
    heat_np = heat.detach().cpu().numpy().clip(0.0, 1.0).astype(np.float32)

    img = Image.open(image2_path).convert("RGB").resize((orig_w, orig_h))
    img_np = np.asarray(img).astype(np.float32)

    # Red heat overlay. No matplotlib dependency.
    red = np.array([255, 0, 0], dtype=np.float32)
    weight = alpha * heat_np[..., None]
    out = img_np * (1.0 - weight) + red * weight
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def save_texture_overlay(
    image2_path: str | Path,
    heatmap_feat_hw: torch.Tensor,
    orig2: Tuple[int, int],
    padded2: Tuple[int, int],
    out_path: Path,
    alpha: float = 0.60,
) -> Image.Image:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil = make_texture_overlay_image(image2_path, heatmap_feat_hw, orig2, padded2, alpha)
    pil.save(out_path)
    return pil


def add_label_bar(tile: Image.Image, label: str) -> Image.Image:
    label_h = 26
    out = Image.new("RGB", (tile.width, tile.height + label_h), (255, 255, 255))
    out.paste(tile, (0, label_h))
    draw = ImageDraw.Draw(out)
    draw.text((6, 6), label, fill=(0, 0, 0))
    return out


def make_overview_image(
    image1_self: Image.Image,
    image2_self: Image.Image,
    corr_img1: Image.Image,
    texture_overlay: Image.Image,
    tile_height: int = 360,
) -> Image.Image:
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    tiles = []
    for im, label in [
        (image1_self, "image1 self masks"),
        (image2_self, "image2 self masks"),
        (corr_img1, "image2 -> image1 semantic match"),
        (texture_overlay, "image2 texture anomaly"),
    ]:
        w, h = im.size
        new_w = max(1, int(round(w * tile_height / h)))
        im_r = im.resize((new_w, tile_height), resample=resample)
        tiles.append(add_label_bar(im_r, label))

    total_w = sum(t.width for t in tiles)
    max_h = max(t.height for t in tiles)
    canvas = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 0))
        x += tile.width
    return canvas


def make_overview(
    image1_self: Image.Image,
    image2_self: Image.Image,
    corr_img1: Image.Image,
    texture_overlay: Image.Image,
    out_path: Path,
    tile_height: int = 360,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    make_overview_image(
        image1_self=image1_self,
        image2_self=image2_self,
        corr_img1=corr_img1,
        texture_overlay=texture_overlay,
        tile_height=tile_height,
    ).save(out_path)


def make_overall_image(items: List[Tuple[str, float, Image.Image]], out_path: Path, row_width: int = 1400) -> None:
    """Create one overall PNG containing compact per-image overview rows."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not items:
        Image.new("RGB", (row_width, 80), (255, 255, 255)).save(out_path)
        return
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    rows = []
    for name, score, im in items:
        w, h = im.size
        new_h = max(1, int(round(h * row_width / max(w, 1))))
        im_r = im.resize((row_width, new_h), resample=resample)
        rows.append(add_label_bar(im_r, f"{name} | consistency={score:.4f}"))
    total_h = sum(r.height for r in rows)
    canvas = Image.new("RGB", (row_width, total_h), (255, 255, 255))
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    canvas.save(out_path)


# -----------------------------------------------------------------------------
# Clustering and semantic correspondence
# -----------------------------------------------------------------------------

@torch.no_grad()
def self_cluster_features(feat_chw: torch.Tensor, tau: float):
    C, h, w = feat_chw.shape
    feat_flat = feat_chw.reshape(C, -1).permute(1, 0).contiguous()
    labels = agglomerative_clustering(feat_flat, tau).reshape(h, w)
    labels = labels.to(feat_chw.device)
    K = int(labels.max().item()) + 1
    return labels, K


@torch.no_grad()
def adaptive_self_cluster_features(
    feat_chw: torch.Tensor,
    tau_start: float,
    min_base_masks: int,
    min_patches: int,
    tau_step: float,
    tau_max: float,
):
    """Select tau using this image only, requiring enough valid masks if possible."""
    tau_start = float(tau_start)
    tau_step = float(tau_step)
    tau_max = float(tau_max)
    min_base_masks = int(min_base_masks)
    min_patches = int(min_patches)

    if min_base_masks <= 0:
        labels, K = self_cluster_features(feat_chw, tau_start)
        areas = torch.bincount(labels.reshape(-1), minlength=K)
        valid_count = int((areas >= min_patches).sum().item())
        return labels, K, tau_start, valid_count

    if tau_step <= 0:
        raise ValueError("--adaptive-tau-step must be > 0")

    tau = max(0.0, min(tau_start, tau_max))
    tau_max = max(tau, min(tau_max, 0.999))

    best = None
    while tau <= tau_max + 1e-12:
        tau_cur = min(tau, tau_max)
        labels, K = self_cluster_features(feat_chw, tau_cur)
        areas = torch.bincount(labels.reshape(-1), minlength=K)
        valid_count = int((areas >= min_patches).sum().item())
        if best is None or valid_count > best[3]:
            best = (labels, K, tau_cur, valid_count)
        if valid_count >= min_base_masks:
            return labels, K, tau_cur, valid_count
        tau += tau_step

    labels, K, tau_used, valid_count = best
    print(
        "[warning] Could not reach requested image2 base-mask count. "
        f"valid_masks={valid_count}, requested={min_base_masks}, "
        f"best_tau={tau_used:.4f}, tau_max={tau_max:.4f}."
    )
    return labels, K, tau_used, valid_count


@torch.no_grad()
def nearest_ref_cluster_for_each_target_patch(
    feat_ref_deb_flat: torch.Tensor,
    feat_tgt_deb_flat: torch.Tensor,
    ref_labels_flat: torch.Tensor,
    chunk_size: int = 2048,
):
    n_tgt = feat_tgt_deb_flat.shape[0]
    best_ref_labels = []
    for start in range(0, n_tgt, chunk_size):
        end = min(start + chunk_size, n_tgt)
        sim = feat_ref_deb_flat @ feat_tgt_deb_flat[start:end].T
        best_ref_idx = sim.argmax(dim=0)
        best_ref_labels.append(ref_labels_flat[best_ref_idx])
    return torch.cat(best_ref_labels, dim=0)


@torch.no_grad()
def batched_correspondence_from_ref_masks_to_target(
    feat_target: torch.Tensor,
    feat_ref: torch.Tensor,
    feat_target_deb: torch.Tensor,
    feat_ref_deb: torch.Tensor,
    labels_target: torch.Tensor,
    labels_ref: torch.Tensor,
    K_target: int,
    K_ref: int,
    ref_mask_ids: torch.Tensor,
    merge_threshold: float,
    batch_size: int,
    nn_chunk_size: int,
    desc: str = "Matching ref masks -> target",
):
    """INSID3-style semantic linking: ref self-mask -> target region.

    The original implementation was hard-coded as image2/SR -> image1/LR.
    This generic version keeps the same matching-and-aggregation mechanism but
    lets the caller choose which image is the coarse reference. The intended
    usage for LR/SR consistency is to use the side with fewer valid self-masks
    as ref, so one coarse semantic region can aggregate multiple smaller masks
    on the target side.
    """
    device = feat_target.device
    C, h_t, w_t = feat_target.shape
    N_t = h_t * w_t

    labels_target = labels_target.to(device)
    labels_ref = labels_ref.to(device)
    labels_target_flat = labels_target.reshape(-1).long()
    labels_ref_flat = labels_ref.reshape(-1).long()

    feat_target_flat = feat_target.reshape(C, -1).permute(1, 0).contiguous()
    feat_ref_flat = feat_ref.reshape(C, -1).permute(1, 0).contiguous()
    feat_target_deb_flat = feat_target_deb.reshape(C, -1).permute(1, 0).contiguous()
    feat_ref_deb_flat = feat_ref_deb.reshape(C, -1).permute(1, 0).contiguous()

    proto_target_orig = compute_cluster_prototypes(feat_target_flat, labels_target_flat, K_target)
    proto_target_deb = compute_cluster_prototypes(feat_target_deb_flat, labels_target_flat, K_target)
    proto_ref_deb = compute_cluster_prototypes(feat_ref_deb_flat, labels_ref_flat, K_ref)

    counts_target = torch.bincount(labels_target_flat, minlength=K_target).float().to(device).clamp_min(1.0)
    intra_target = proto_target_orig @ proto_target_orig.T

    best_ref_label_for_target_patch = nearest_ref_cluster_for_each_target_patch(
        feat_ref_deb_flat=feat_ref_deb_flat,
        feat_tgt_deb_flat=feat_target_deb_flat,
        ref_labels_flat=labels_ref_flat,
        chunk_size=nn_chunk_size,
    )

    all_results = []
    ref_mask_ids = ref_mask_ids.to(device)

    for start in tqdm(range(0, len(ref_mask_ids), batch_size), desc=desc):
        ids = ref_mask_ids[start:start + batch_size]
        B = ids.numel()
        ref_proto = proto_ref_deb[ids]

        fg_sim_flat = ref_proto @ feat_target_deb_flat.T
        forward_mask = fg_sim_flat > 0
        no_forward = forward_mask.sum(dim=1) == 0
        if no_forward.any():
            q = torch.quantile(fg_sim_flat[no_forward], 0.9, dim=1)
            forward_mask[no_forward] = fg_sim_flat[no_forward] > q[:, None]

        backward_mask = best_ref_label_for_target_patch[None, :].eq(ids[:, None])
        candidate = forward_mask & backward_mask

        labels_target_expand = labels_target_flat[None, :].expand(B, N_t)
        overlap_counts = torch.zeros(B, K_target, device=device)
        overlap_counts.scatter_add_(1, labels_target_expand, candidate.float())
        area_weights = overlap_counts / counts_target[None, :]

        matched = overlap_counts > 0
        found = matched.any(dim=1)

        seed_cross = ref_proto @ proto_target_deb.T
        seed_cross = seed_cross.masked_fill(~matched, torch.finfo(seed_cross.dtype).min)
        seed_ids = seed_cross.argmax(dim=1)

        cross_sum = torch.zeros(B, K_target, device=device)
        cross_sum.scatter_add_(1, labels_target_expand, fg_sim_flat)
        cross_sim = cross_sum / counts_target[None, :]

        intra_sim = intra_target[seed_ids]
        area_weights[torch.arange(B, device=device), seed_ids] = 1.0
        combined = cross_sim * intra_sim * area_weights
        patch_scores = combined.gather(1, labels_target_expand)
        pred = (patch_scores > merge_threshold).reshape(B, h_t, w_t)
        pred[~found] = False

        for local_i in range(B):
            all_results.append({
                "ref_cluster_id": int(ids[local_i].item()),
                "found": bool(found[local_i].item()),
                "seed_target_cluster_id": int(seed_ids[local_i].item()) if found[local_i] else -1,
                "pred_mask_target_feat": pred[local_i].detach().clone(),
                "candidate_area_patches": int(candidate[local_i].sum().item()),
            })
    return all_results


@torch.no_grad()
def batched_correspondence_from_image2_masks_to_image1(
    feat1: torch.Tensor,
    feat2: torch.Tensor,
    feat1_deb: torch.Tensor,
    feat2_deb: torch.Tensor,
    labels1: torch.Tensor,
    labels2: torch.Tensor,
    K1: int,
    K2: int,
    image2_mask_ids: torch.Tensor,
    merge_threshold: float,
    batch_size: int,
    nn_chunk_size: int,
):
    """Backward-compatible wrapper: image2/SR ref -> image1/LR target."""
    results = batched_correspondence_from_ref_masks_to_target(
        feat_target=feat1,
        feat_ref=feat2,
        feat_target_deb=feat1_deb,
        feat_ref_deb=feat2_deb,
        labels_target=labels1,
        labels_ref=labels2,
        K_target=K1,
        K_ref=K2,
        ref_mask_ids=image2_mask_ids,
        merge_threshold=merge_threshold,
        batch_size=batch_size,
        nn_chunk_size=nn_chunk_size,
        desc="Matching image2 masks -> image1",
    )
    for r in results:
        r["image2_cluster_id"] = r["ref_cluster_id"]
        r["seed_image1_cluster_id"] = r["seed_target_cluster_id"]
        r["pred_mask_feat"] = r["pred_mask_target_feat"]
    return results


# -----------------------------------------------------------------------------
# Penalties
# -----------------------------------------------------------------------------

def compute_semantic_penalties(
    area1: float,
    area2: float,
    total1: float,
    total2: float,
    area_mode: str,
    semantic_tol: float,
    semantic_cap: float,
):
    """
    Returns support ratio and normalized semantic penalties in [0,1].

    auto mode uses pixel area if image1/image2 have the same total pixel count,
    otherwise normalized area fractions.
    """
    if area_mode == "auto":
        use_pixels = abs(float(total1) - float(total2)) < 0.5
    elif area_mode == "pixels":
        use_pixels = True
    elif area_mode == "normalized":
        use_pixels = False
    else:
        raise ValueError(f"Unknown area_mode={area_mode}")

    if use_pixels:
        a1 = float(area1)
        a2 = max(float(area2), EPS)
    else:
        a1 = float(area1) / max(float(total1), EPS)
        a2 = max(float(area2) / max(float(total2), EPS), EPS)

    support_ratio = a1 / max(a2, EPS)
    abs_ratio = abs(a1 - a2) / max(a2, EPS)
    over_raw = max(0.0, (a1 - a2) / max(a2, EPS) - semantic_tol)
    under_raw = max(0.0, (a2 - a1) / max(a2, EPS) - semantic_tol)
    sem_raw = max(0.0, abs_ratio - semantic_tol)
    cap = max(float(semantic_cap), EPS)
    return {
        "semantic_area_ratio": support_ratio,
        "semantic_abs_area_diff_ratio": abs_ratio,
        "semantic_over_penalty": min(1.0, over_raw / cap),
        "semantic_under_penalty": min(1.0, under_raw / cap),
        "semantic_penalty": min(1.0, sem_raw / cap),
        "area_compare_mode_used": "pixels" if use_pixels else "normalized",
    }


@torch.no_grad()
def texture_nn_residuals_for_mask(
    feat1_chw: torch.Tensor,
    feat2_chw: torch.Tensor,
    mask1_hw: torch.Tensor,
    mask2_hw: torch.Tensor,
    chunk_size: int,
    spatial_radius: int = 2,
):
    """
    Spatially constrained texture matching.

    For every SR/image2 patch feature y inside mask2, find the nearest
    LR/image1 feature x inside mask1 AND within a local coordinate window.
    Residual = (1 - max cosine(y, x)) / 2 in [0,1].

    If no LR candidate exists in the spatial window, residual is set to 1.
    Use spatial_radius=-1 to fall back to global mask-level NN.
    """
    device = feat2_chw.device
    mask1 = mask1_hw.bool().to(device)
    mask2 = mask2_hw.bool().to(device)

    n2 = int(mask2.sum().item())
    if n2 == 0:
        return torch.empty(0, device=device)
    if int(mask1.sum().item()) == 0:
        return torch.ones(n2, device=device)

    x = feat1_chw[:, mask1].T.contiguous().float()
    y = feat2_chw[:, mask2].T.contiguous().float()
    x = F.normalize(x, p=2, dim=1, eps=1e-6)
    y = F.normalize(y, p=2, dim=1, eps=1e-6)

    if spatial_radius is None or int(spatial_radius) < 0:
        residuals = []
        for start in range(0, y.shape[0], chunk_size):
            end = min(start + chunk_size, y.shape[0])
            sim = y[start:end] @ x.T
            max_sim = sim.max(dim=1).values.clamp(-1.0, 1.0)
            residuals.append(((1.0 - max_sim) * 0.5).clamp(0.0, 1.0))
        return torch.cat(residuals, dim=0)

    r = float(spatial_radius)
    H1, W1 = mask1.shape
    H2, W2 = mask2.shape
    coords1 = mask1.nonzero(as_tuple=False).float().to(device)  # [N1, 2]
    coords2 = mask2.nonzero(as_tuple=False).float().to(device)  # [N2, 2]

    # Map SR/image2 grid coordinates to LR/image1 grid coordinates.
    scale_h = (float(H1 - 1) / max(float(H2 - 1), 1.0)) if H2 > 1 else 1.0
    scale_w = (float(W1 - 1) / max(float(W2 - 1), 1.0)) if W2 > 1 else 1.0
    coords2_mapped = coords2.clone()
    coords2_mapped[:, 0] *= scale_h
    coords2_mapped[:, 1] *= scale_w

    residuals = []
    neg_inf = torch.finfo(y.dtype).min

    for start in range(0, y.shape[0], chunk_size):
        end = min(start + chunk_size, y.shape[0])
        sim = y[start:end] @ x.T

        q = coords2_mapped[start:end]
        valid_h = (coords1[:, 0][None, :] - q[:, 0:1]).abs() <= r
        valid_w = (coords1[:, 1][None, :] - q[:, 1:2]).abs() <= r
        valid = valid_h & valid_w

        has_candidate = valid.any(dim=1)
        sim = sim.masked_fill(~valid, neg_inf)

        max_sim = torch.full((end - start,), -1.0, device=device, dtype=sim.dtype)
        if has_candidate.any():
            max_sim[has_candidate] = sim[has_candidate].max(dim=1).values.clamp(-1.0, 1.0)

        residuals.append(((1.0 - max_sim) * 0.5).clamp(0.0, 1.0))

    return torch.cat(residuals, dim=0)


def residual_stats(residuals: torch.Tensor) -> Dict[str, float]:
    if residuals.numel() == 0:
        return {"mean": 1.0, "p50": 1.0, "p90": 1.0, "p95": 1.0}
    r = residuals.detach().float().cpu()
    return {
        "mean": float(r.mean().item()),
        "p50": float(torch.quantile(r, 0.50).item()),
        "p90": float(torch.quantile(r, 0.90).item()),
        "p95": float(torch.quantile(r, 0.95).item()),
    }


def residual_score_from_stats(stats: Dict[str, float], mode: str = "mean_p90_p95") -> float:
    if mode == "mean":
        return float(stats["mean"])
    if mode == "p90":
        return float(stats["p90"])
    if mode == "p95":
        return float(stats["p95"])
    if mode == "mean_p90_p95":
        return float(0.5 * stats["mean"] + 0.3 * stats["p90"] + 0.2 * stats["p95"])
    raise ValueError(f"Unknown texture_stat={mode}")


def normalize_texture_value(raw: float, texture_tol: float, texture_cap: float) -> float:
    return float(np.clip((float(raw) - float(texture_tol)) / max(float(texture_cap), EPS), 0.0, 1.0))


def normalize_texture_tensor(raw: torch.Tensor, texture_tol: float, texture_cap: float) -> torch.Tensor:
    return ((raw.float() - float(texture_tol)) / max(float(texture_cap), EPS)).clamp(0.0, 1.0)


def _gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sigma = float(max(sigma, EPS))
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / k.sum().clamp_min(EPS)
    return k


def _gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur a [1,1,H,W] tensor with separable Gaussian filtering."""
    k = _gaussian_kernel1d(sigma, x.device, x.dtype)
    pad = k.numel() // 2
    kh = k.view(1, 1, 1, -1)
    kv = k.view(1, 1, -1, 1)
    x = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, kh)
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    x = F.conv2d(x, kv)
    return x


def _pil_to_luma_tensor(pil: Image.Image, device: str) -> torch.Tensor:
    arr = np.asarray(pil.convert("RGB")).astype(np.float32) / 255.0
    rgb = torch.from_numpy(arr).to(device=device).permute(2, 0, 1)
    y = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return y[None, None].contiguous()


def compute_lowfreq_penalty(
    image1_path: str | Path,
    image2_path: str | Path,
    device: str,
    sigma: float,
    cap: float,
    luma_weight: float = 0.7,
    grad_weight: float = 0.3,
) -> Dict[str, float]:
    """
    Low-frequency fidelity penalty between image1 and image2.

    image1 is bicubic-resized to image2 size, then both images are converted to
    luminance, Gaussian blurred, and compared by blurred L1 plus blurred-gradient L1.

    Returns raw terms and normalized lowfreq_penalty in [0,1].
    """
    img2 = Image.open(image2_path).convert("RGB")
    img1 = Image.open(image1_path).convert("RGB")
    if img1.size != img2.size:
        resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
        img1 = img1.resize(img2.size, resample=resample)

    y1 = _pil_to_luma_tensor(img1, device=device)
    y2 = _pil_to_luma_tensor(img2, device=device)
    b1 = _gaussian_blur_2d(y1, sigma=sigma)
    b2 = _gaussian_blur_2d(y2, sigma=sigma)

    luma_l1 = (b1 - b2).abs().mean()

    # Simple finite-difference gradients on blurred luminance.
    dx1 = b1[..., :, 1:] - b1[..., :, :-1]
    dx2 = b2[..., :, 1:] - b2[..., :, :-1]
    dy1 = b1[..., 1:, :] - b1[..., :-1, :]
    dy2 = b2[..., 1:, :] - b2[..., :-1, :]
    grad_l1 = 0.5 * ((dx1 - dx2).abs().mean() + (dy1 - dy2).abs().mean())

    raw = float(luma_weight * float(luma_l1.item()) + grad_weight * float(grad_l1.item()))
    penalty = float(np.clip(raw / max(float(cap), EPS), 0.0, 1.0))
    return {
        "lowfreq_raw": raw,
        "lowfreq_luma_l1": float(luma_l1.item()),
        "lowfreq_grad_l1": float(grad_l1.item()),
        "lowfreq_penalty": penalty,
    }


# -----------------------------------------------------------------------------
# Per-pair processing
# -----------------------------------------------------------------------------

def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_ordered_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    """Always write a CSV with header, even when rows is empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def process_one_pair(model, image1_path: Path, image2_path: Path, rel_key: Path, pair_out_dir: Path, args):
    pair_out_dir.mkdir(parents=True, exist_ok=True)

    texture_layers = parse_int_list(args.texture_layers)
    texture_weights = parse_float_list(args.texture_weights)
    if len(texture_layers) != len(texture_weights):
        raise ValueError("Internal texture layer/weight defaults are inconsistent.")
    weight_sum = sum(texture_weights)
    if weight_sum <= 0:
        raise ValueError("Internal texture weights must sum to > 0.")
    texture_weights = [w / weight_sum for w in texture_weights]

    data = extract_pair_multilayer_features(
        model=model,
        image1_path=image1_path,
        image2_path=image2_path,
        device=args.device,
        use_amp=args.amp,
        layers=texture_layers,
        semantic_layer=args.semantic_layer,
    )

    lowfreq = compute_lowfreq_penalty(
        image1_path=image1_path,
        image2_path=image2_path,
        device=args.device,
        sigma=args.lowfreq_sigma,
        cap=args.lowfreq_cap,
    )

    feat1 = data["feat1_sem"]
    feat2 = data["feat2_sem"]
    feat1_deb = data["feat1_deb"]
    feat2_deb = data["feat2_deb"]

    orig1_h, orig1_w = data["orig1"]
    orig2_h, orig2_w = data["orig2"]

    # Semantic masks live at semantic-branch resolution, which can be smaller
    # than the original image when the long side is > 768. Use semantic sizes
    # for semantic-mask area computation and semantic overlays.
    sem1_h, sem1_w = data["semantic_input1"]
    sem2_h, sem2_w = data["semantic_input2"]
    sem_padded1_h, sem_padded1_w = data["semantic_padded1"]
    sem_padded2_h, sem_padded2_w = data["semantic_padded2"]
    total1_pixels = float(sem1_h * sem1_w)
    total2_pixels = float(sem2_h * sem2_w)

    print(f"Self-segmenting image2 with adaptive tau: {rel_key}")
    labels2, K2, tau_pair_used, valid_K2 = adaptive_self_cluster_features(
        feat2,
        tau_start=args.tau,
        min_base_masks=args.min_base_masks,
        min_patches=args.min_patches,
        tau_step=args.adaptive_tau_step,
        tau_max=args.adaptive_tau_max,
    )

    print(f"Self-segmenting image1 with image2-selected tau={tau_pair_used:.4f}: {rel_key}")
    labels1, K1 = self_cluster_features(feat1, tau_pair_used)

    labels1 = labels1.to(feat1.device)
    labels2 = labels2.to(feat2.device)
    areas1 = torch.bincount(labels1.reshape(-1), minlength=K1)
    areas2 = torch.bincount(labels2.reshape(-1), minlength=K2)
    valid_K1 = int((areas1 >= args.min_patches).sum().item())

    print(
        f"{rel_key} | tau_pair={tau_pair_used:.4f} | "
        f"K1={K1} valid1={valid_K1} | K2={K2} valid2={valid_K2}"
    )

    keep_ids1 = torch.where(areas1 >= args.min_patches)[0]
    keep_ids1 = keep_ids1[torch.argsort(areas1[keep_ids1], descending=True)]
    keep_ids2 = torch.where(areas2 >= args.min_patches)[0]
    keep_ids2 = keep_ids2[torch.argsort(areas2[keep_ids2], descending=True)]

    # Coarse-ref correspondence mode. The correspondence stage is ref-mask driven:
    # one ref mask aggregates target-side patches/masks. Therefore the more stable
    # default is to use the side with fewer valid masks as ref. This avoids assuming
    # that SR/image2 is always coarser than LR/image1.
    semantic_ref_mode = getattr(args, "semantic_ref_mode", "fewer_valid_masks")
    if semantic_ref_mode in {"image2", "sr", "sr_ref"}:
        use_image2_as_ref = True
    elif semantic_ref_mode in {"image1", "lr", "lr_ref"}:
        use_image2_as_ref = False
    elif semantic_ref_mode in {"fewer_valid_masks", "coarse_ref", "auto"}:
        use_image2_as_ref = valid_K2 <= valid_K1
    else:
        raise ValueError(f"Unknown semantic_ref_mode={semantic_ref_mode}")

    if use_image2_as_ref:
        correspondence_direction = "sr_ref_to_lr_target"
        ref_side = "image2"
        target_side = "image1"
        ref_ids = keep_ids2
        if args.max_ref_masks is not None:
            ref_ids = ref_ids[:args.max_ref_masks]
        results = batched_correspondence_from_ref_masks_to_target(
            feat_target=feat1,
            feat_ref=feat2,
            feat_target_deb=feat1_deb,
            feat_ref_deb=feat2_deb,
            labels_target=labels1,
            labels_ref=labels2,
            K_target=K1,
            K_ref=K2,
            ref_mask_ids=ref_ids,
            merge_threshold=args.merge_thresh,
            batch_size=args.batch_size,
            nn_chunk_size=args.nn_chunk_size,
            desc="Matching SR/image2 ref masks -> LR/image1",
        )
    else:
        correspondence_direction = "lr_ref_to_sr_target"
        ref_side = "image1"
        target_side = "image2"
        ref_ids = keep_ids1
        if args.max_ref_masks is not None:
            ref_ids = ref_ids[:args.max_ref_masks]
        results = batched_correspondence_from_ref_masks_to_target(
            feat_target=feat2,
            feat_ref=feat1,
            feat_target_deb=feat2_deb,
            feat_ref_deb=feat1_deb,
            labels_target=labels2,
            labels_ref=labels1,
            K_target=K2,
            K_ref=K1,
            ref_mask_ids=ref_ids,
            merge_threshold=args.merge_thresh,
            batch_size=args.batch_size,
            nn_chunk_size=args.nn_chunk_size,
            desc="Matching LR/image1 ref masks -> SR/image2",
        )

    print(
        f"{rel_key} | semantic_ref_mode={semantic_ref_mode} | "
        f"direction={correspondence_direction} | ref_masks={len(ref_ids)}"
    )

    image1_self_overlay = make_label_overlay_image(
        image1_path, labels1, data["semantic_input1"], data["semantic_padded1"]
    )
    image2_self_overlay = make_label_overlay_image(
        image2_path, labels2, data["semantic_input2"], data["semantic_padded2"]
    )

    tex_h1, tex_w1 = data["feats1"][texture_layers[0]].shape[-2:]
    tex_h2, tex_w2 = data["feats2"][texture_layers[0]].shape[-2:]
    fused_texture_map = torch.zeros((tex_h2, tex_w2), device=feat2.device, dtype=torch.float32)

    mask_rows = []
    correspondence_pixel_masks = []

    for rank, item in enumerate(results):
        src_id = int(item.get("ref_cluster_id", item.get("image2_cluster_id")))
        found = bool(item["found"])
        pred_target_feat = item.get("pred_mask_target_feat", item.get("pred_mask_feat"))

        if correspondence_direction == "sr_ref_to_lr_target":
            mask2_feat = labels2 == src_id  # SR/image2 ref mask
            mask1_feat = pred_target_feat.to(feat1.device).bool()  # matched LR/image1 region
            ref_area_side = "image2"
        else:
            mask1_feat = labels1 == src_id  # LR/image1 ref mask
            mask2_feat = pred_target_feat.to(feat2.device).bool()  # matched SR/image2 region
            ref_area_side = "image1"

        mask2_tex = resize_bool_mask_to_shape(mask2_feat, tex_h2, tex_w2)
        mask1_tex = resize_bool_mask_to_shape(mask1_feat, tex_h1, tex_w1)

        ref_mask_img2 = upsample_bool_mask(mask2_feat, sem2_h, sem2_w, sem_padded2_h, sem_padded2_w)
        pred_mask_img1 = upsample_bool_mask(mask1_feat, sem1_h, sem1_w, sem_padded1_h, sem_padded1_w)
        correspondence_pixel_masks.append((pred_mask_img1, src_id))

        area2_pixels = float(ref_mask_img2.sum().item())   # SR/image2 area for this correspondence
        area1_pixels = float(pred_mask_img1.sum().item())  # LR/image1 area for this correspondence
        ref_area_pixels = area2_pixels if ref_area_side == "image2" else area1_pixels
        target_area_pixels = area1_pixels if ref_area_side == "image2" else area2_pixels

        sem = compute_semantic_penalties(
            area1=area1_pixels,
            area2=area2_pixels,
            total1=total1_pixels,
            total2=total2_pixels,
            area_mode=args.area_mode,
            semantic_tol=args.semantic_tol,
            semantic_cap=args.semantic_cap,
        )
        semantic_symmetric = float(sem["semantic_penalty"])
        sem["semantic_penalty_symmetric"] = semantic_symmetric
        sem["semantic_under_penalty_raw"] = float(sem["semantic_under_penalty"])
        sem["semantic_over_penalty_raw"] = float(sem["semantic_over_penalty"])

        # Direction-aware semantic support.
        # - SR ref -> LR target: penalize SR area not supported by LR => semantic_under.
        # - LR ref -> SR target: penalize LR area not recovered/supported by SR => semantic_over.
        if correspondence_direction == "sr_ref_to_lr_target":
            support_ratio = area1_pixels / max(area2_pixels, EPS)
            semantic_directional = float(sem["semantic_under_penalty_raw"])
        else:
            support_ratio = area2_pixels / max(area1_pixels, EPS)
            semantic_directional = float(sem["semantic_over_penalty_raw"])

        sem["semantic_directional_penalty"] = float(np.clip(semantic_directional, 0.0, 1.0))
        # Keep the legacy branch column useful for the training script.
        sem["semantic_penalty"] = sem["semantic_directional_penalty"]
        sem["semantic_under_penalty"] = sem["semantic_directional_penalty"]

        is_unmatched = (not found) or target_area_pixels <= 0.0
        is_weakly_supported = support_ratio < args.min_support_ratio
        if is_unmatched:
            unmatched_penalty = 1.0
        elif is_weakly_supported:
            unmatched_penalty = max(0.0, 1.0 - support_ratio / max(args.min_support_ratio, EPS))
        else:
            unmatched_penalty = 0.0
        unmatched_penalty = float(np.clip(unmatched_penalty, 0.0, 1.0))

        texture_layer_penalty = {}
        fused_residual_vec = None

        for layer, w_layer in zip(texture_layers, texture_weights):
            if is_unmatched:
                residuals = torch.ones(int(mask2_tex.sum().item()), device=feat2.device)
            else:
                residuals = texture_nn_residuals_for_mask(
                    feat1_chw=data["feats1"][layer],
                    feat2_chw=data["feats2"][layer],
                    mask1_hw=mask1_tex,
                    mask2_hw=mask2_tex,
                    chunk_size=args.texture_nn_chunk_size,
                    spatial_radius=args.texture_spatial_radius,
                )
            st = residual_stats(residuals)
            layer_raw_score = residual_score_from_stats(st, args.texture_stat)
            texture_layer_penalty[layer] = normalize_texture_value(layer_raw_score, args.texture_tol, args.texture_cap)

            if residuals.numel() == int(mask2_tex.sum().item()):
                if fused_residual_vec is None:
                    fused_residual_vec = w_layer * residuals.float()
                else:
                    fused_residual_vec = fused_residual_vec + w_layer * residuals.float()

        if fused_residual_vec is None or fused_residual_vec.numel() == 0:
            texture_raw_fused = 1.0
            texture_penalty_fused = 1.0
            fused_texture_map[mask2_tex] = 1.0
        else:
            fused_stats = residual_stats(fused_residual_vec)
            texture_raw_fused = residual_score_from_stats(fused_stats, args.texture_stat)
            texture_penalty_fused = normalize_texture_value(texture_raw_fused, args.texture_tol, args.texture_cap)
            fused_texture_map[mask2_tex] = normalize_texture_tensor(
                fused_residual_vec, args.texture_tol, args.texture_cap
            ).to(fused_texture_map.device)

        numerator = (
            args.lambda_lowfreq * lowfreq["lowfreq_penalty"]
            + args.lambda_semantic * sem["semantic_penalty"]
            + args.lambda_texture * texture_penalty_fused
        )
        denominator = args.lambda_lowfreq + args.lambda_semantic + args.lambda_texture
        if unmatched_penalty > EPS:
            numerator += args.lambda_unmatched * unmatched_penalty
            denominator += args.lambda_unmatched
        final_penalty = float(np.clip(numerator / max(denominator, EPS), 0.0, 1.0))
        consistency_score = 1.0 - final_penalty

        row = {
            "dataset": getattr(args, "dataset", ""),
            "method": getattr(args, "method", ""),
            "relative_image_name": str(rel_key),
            "rank": int(rank),
            "ref_cluster_id": src_id,
            "image2_cluster_id": src_id if correspondence_direction == "sr_ref_to_lr_target" else -1,
            "found": int(found),
            "semantic_correspondence_direction": correspondence_direction,
            "semantic_ref_side": ref_side,
            "semantic_target_side": target_side,
            "semantic_ref_is_sr": int(ref_side == "image2"),
            "image2_area_pixels": int(area2_pixels),
            "image1_matched_area_pixels": int(area1_pixels),
            "semantic_ref_area_pixels": int(ref_area_pixels),
            "semantic_target_area_pixels": int(target_area_pixels),
            "support_ratio": support_ratio,
            "is_unmatched": int(is_unmatched),
            "is_weakly_supported": int(is_weakly_supported),
            "semantic_over_penalty": sem["semantic_over_penalty"],
            "semantic_under_penalty": sem["semantic_under_penalty"],
            "semantic_over_penalty_raw": sem["semantic_over_penalty_raw"],
            "semantic_under_penalty_raw": sem["semantic_under_penalty_raw"],
            "semantic_directional_penalty": sem["semantic_directional_penalty"],
            "semantic_penalty_symmetric": sem["semantic_penalty_symmetric"],
            "semantic_penalty": sem["semantic_penalty"],
            "lowfreq_penalty": lowfreq["lowfreq_penalty"],
            "lowfreq_raw": lowfreq["lowfreq_raw"],
            "lowfreq_luma_l1": lowfreq["lowfreq_luma_l1"],
            "lowfreq_grad_l1": lowfreq["lowfreq_grad_l1"],
            "texture_penalty": texture_penalty_fused,
            "unmatched_penalty": unmatched_penalty,
            "final_penalty": final_penalty,
            "consistency_score": consistency_score,
            "weight_area_pixels": int(max(ref_area_pixels, EPS)),
        }
        for layer in texture_layers:
            row[f"texture_penalty_layer{layer}"] = texture_layer_penalty[layer]
        mask_rows.append(row)

    texture_overlay = make_texture_overlay_image(
        image2_path=image2_path,
        heatmap_feat_hw=fused_texture_map,
        orig2=data["orig2"],
        padded2=data["texture_padded2"],
        alpha=args.heatmap_alpha,
    )
    corr_overlay_img1 = make_multi_mask_overlay_image(image1_path, correspondence_pixel_masks, alpha=0.50)
    overview_img = make_overview_image(
        image1_self=image1_self_overlay,
        image2_self=image2_self_overlay,
        corr_img1=corr_overlay_img1,
        texture_overlay=texture_overlay,
        tile_height=args.overview_tile_height,
    )
    overall_path = pair_out_dir / "overall.png"
    overview_img.save(overall_path)

    if args.save_debug:
        mask_scores_csv = pair_out_dir / "mask_scores.csv"
        write_csv(mask_scores_csv, mask_rows)
        texture_overlay.save(pair_out_dir / "fused_texture_anomaly_overlay.png")
        meta = {
            "dataset": getattr(args, "dataset", ""),
            "method": getattr(args, "method", ""),
            "relative_image_name": str(rel_key),
            "image1_path": str(image1_path),
            "image2_path": str(image2_path),
            "orig1_hw": list(data["orig1"]),
            "orig2_hw": list(data["orig2"]),
            "semantic_input1_hw": list(data["semantic_input1"]),
            "semantic_input2_hw": list(data["semantic_input2"]),
            "semantic_padded1_hw": list(data["semantic_padded1"]),
            "semantic_padded2_hw": list(data["semantic_padded2"]),
            "texture_input1_hw": list(data["texture_input1"]),
            "texture_input2_hw": list(data["texture_input2"]),
            "texture_padded1_hw": list(data["texture_padded1"]),
            "texture_padded2_hw": list(data["texture_padded2"]),
            "patch_size": data["patch_size"],
            "semantic_layer": args.semantic_layer,
            "texture_layers": texture_layers,
            "texture_weights": texture_weights,
            "texture_spatial_radius": args.texture_spatial_radius,
            "texture_stat": args.texture_stat,
            "texture_tol": args.texture_tol,
            "texture_cap": args.texture_cap,
            "lowfreq_sigma": args.lowfreq_sigma,
            "lowfreq_cap": args.lowfreq_cap,
            "lambda_lowfreq": args.lambda_lowfreq,
            "lowfreq_penalty": lowfreq["lowfreq_penalty"],
            "lowfreq_raw": lowfreq["lowfreq_raw"],
            "semantic_over_weight": args.semantic_over_weight,
            "semantic_under_weight": args.semantic_under_weight,
            "tau_pair_used": tau_pair_used,
            "K_image1": K1,
            "K_image2": K2,
            "valid_base_masks_image1": valid_K1,
            "valid_base_masks_image2": valid_K2,
            "semantic_ref_mode": semantic_ref_mode,
            "semantic_correspondence_direction": correspondence_direction,
            "semantic_ref_side": ref_side,
            "semantic_target_side": target_side,
            "num_ref_masks_processed": len(mask_rows),
            "num_image2_masks_processed": len(mask_rows) if correspondence_direction == "sr_ref_to_lr_target" else 0,
        }
        (pair_out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    else:
        mask_scores_csv = ""

    if mask_rows:
        weights = np.array([max(float(r["weight_area_pixels"]), 0.0) for r in mask_rows], dtype=np.float64)
        if weights.sum() <= 0:
            weights = np.ones(len(mask_rows), dtype=np.float64)
        weights = weights / weights.sum()

        def wavg(name: str) -> float:
            vals = np.array([float(r[name]) for r in mask_rows], dtype=np.float64)
            return float((weights * vals).sum())

        lowfreq_penalty_img = wavg("lowfreq_penalty")
        semantic_penalty_img = wavg("semantic_penalty")
        semantic_over_img = wavg("semantic_over_penalty")
        semantic_under_img = wavg("semantic_under_penalty")
        texture_penalty_img = wavg("texture_penalty")
        unmatched_penalty_img = wavg("unmatched_penalty")
        final_penalty_img = wavg("final_penalty")
        consistency_score_img = 1.0 - final_penalty_img
        unmatched_area_ratio = float(
            sum(float(r["weight_area_pixels"]) for r in mask_rows if int(r["is_unmatched"]) or int(r["is_weakly_supported"]))
            / max(sum(float(r["weight_area_pixels"]) for r in mask_rows), EPS)
        )
    else:
        lowfreq_penalty_img = 1.0
        semantic_penalty_img = 1.0
        semantic_over_img = 0.0
        semantic_under_img = 1.0
        texture_penalty_img = 1.0
        unmatched_penalty_img = 1.0
        final_penalty_img = 1.0
        consistency_score_img = 0.0
        unmatched_area_ratio = 1.0

    pair_score = {
        "dataset": getattr(args, "dataset", ""),
        "method": getattr(args, "method", ""),
        "relative_image_name": str(rel_key),
        "consistency_score": consistency_score_img,
        "final_penalty": final_penalty_img,
        "lowfreq_penalty": lowfreq_penalty_img,
        "lowfreq_raw": lowfreq["lowfreq_raw"],
        "semantic_penalty": semantic_penalty_img,
        "semantic_over_penalty": semantic_over_img,
        "semantic_under_penalty": semantic_under_img,
        "texture_penalty": texture_penalty_img,
        "unmatched_penalty": unmatched_penalty_img,
        "unmatched_area_ratio": unmatched_area_ratio,
        "semantic_ref_mode": semantic_ref_mode,
        "semantic_correspondence_direction": correspondence_direction,
        "semantic_ref_side": ref_side,
        "semantic_target_side": target_side,
        "semantic_ref_is_sr": int(ref_side == "image2"),
        "tau_pair_used": tau_pair_used,
        "K_image1": K1,
        "K_image2": K2,
        "valid_masks_image1": valid_K1,
        "valid_masks_image2": valid_K2,
        "num_masks": len(mask_rows),
        "image1_path": str(image1_path),
        "image2_path": str(image2_path),
        "overall_path": str(overall_path),
        "status": "ok",
        "error": "",
    }
    if args.save_debug:
        pair_score["debug_dir"] = str(pair_out_dir)
        pair_score["mask_scores_csv"] = str(mask_scores_csv)
    return pair_score, mask_rows


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def apply_internal_defaults(args) -> None:
    """Attach internal algorithm defaults to argparse namespace."""
    args.model_size = DEFAULT_MODEL_SIZE
    args.image_size = DEFAULT_IMAGE_SIZE
    args.svd_comps = DEFAULT_SVD_COMPS
    args.semantic_layer = DEFAULT_SEMANTIC_LAYER
    args.texture_layers = DEFAULT_TEXTURE_LAYERS
    args.texture_weights = DEFAULT_TEXTURE_WEIGHTS
    args.tau = DEFAULT_TAU
    args.adaptive_tau_step = DEFAULT_ADAPTIVE_TAU_STEP
    args.adaptive_tau_max = DEFAULT_ADAPTIVE_TAU_MAX
    args.merge_thresh = DEFAULT_MERGE_THRESH
    args.batch_size = DEFAULT_BATCH_SIZE
    args.nn_chunk_size = DEFAULT_NN_CHUNK_SIZE
    args.texture_nn_chunk_size = DEFAULT_TEXTURE_NN_CHUNK_SIZE
    args.texture_stat = DEFAULT_TEXTURE_STAT
    args.texture_tol = DEFAULT_TEXTURE_TOL
    args.texture_cap = DEFAULT_TEXTURE_CAP
    args.lowfreq_sigma = DEFAULT_LOWFREQ_SIGMA
    args.lowfreq_cap = DEFAULT_LOWFREQ_CAP
    args.min_patches = DEFAULT_MIN_PATCHES
    args.max_ref_masks = DEFAULT_MAX_REF_MASKS
    args.area_mode = DEFAULT_AREA_MODE
    args.semantic_tol = DEFAULT_SEMANTIC_TOL
    args.semantic_cap = DEFAULT_SEMANTIC_CAP
    args.semantic_over_weight = DEFAULT_SEMANTIC_OVER_WEIGHT
    args.semantic_under_weight = DEFAULT_SEMANTIC_UNDER_WEIGHT
    args.min_support_ratio = DEFAULT_MIN_SUPPORT_RATIO
    args.lambda_lowfreq = DEFAULT_LAMBDA_LOWFREQ
    args.lambda_semantic = DEFAULT_LAMBDA_SEMANTIC
    args.lambda_texture = DEFAULT_LAMBDA_TEXTURE
    args.lambda_unmatched = DEFAULT_LAMBDA_UNMATCHED
    args.overview_tile_height = DEFAULT_OVERVIEW_TILE_HEIGHT
    args.heatmap_alpha = DEFAULT_HEATMAP_ALPHA
    args.empty_cache_each_pair = False
    # "fewer_valid_masks" uses the side with fewer valid self-masks as semantic ref.
    # Set to "image2" to recover the original fixed SR->LR behavior.
    args.semantic_ref_mode = "fewer_valid_masks"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image1-dir", required=True, help="Root folder for image1, e.g. LR/Bicubic folder.")
    parser.add_argument("--image2-dir", required=True, help="Root folder for image2, e.g. SR folder.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default=None, help="Dataset name. If provided, image1-dir/dataset is used.")
    parser.add_argument("--method", default=None, help="Method name. If provided, image2-dir/method/dataset is used.")

    parser.add_argument("--recursive", action="store_true", help="Match recursively by relative path.")
    parser.add_argument("--max-pairs", default=None, type=int)
    parser.add_argument("--skip-existing", action="store_true", help="Skip run if all_scores.csv already exists.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")

    # Only two user-facing scoring knobs.
    parser.add_argument("--min-base-masks", default=6, type=int)
    parser.add_argument("--texture-spatial-radius", default=2, type=int)

    parser.add_argument("--save-debug", action="store_true", help="Save per-image texture overlay, mask_scores.csv and meta.json. overall.png is always saved.")

    args = parser.parse_args()
    apply_internal_defaults(args)

    if args.dataset:
        image1_dir = Path(os.path.join(args.image1_dir, args.dataset))
    else:
        image1_dir = Path(args.image1_dir)

    if args.method or args.dataset:
        method_part = args.method if args.method else ""
        dataset_part = args.dataset if args.dataset else ""
        image2_dir = Path(os.path.join(args.image2_dir, method_part, dataset_part))
    else:
        image2_dir = Path(args.image2_dir)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.dataset:
        args.dataset = image1_dir.name
    if not args.method:
        args.method = image2_dir.parent.name

    all_scores_csv = out_dir / "all_scores.csv"
    field_order = [
        "dataset", "method", "relative_image_name",
        "consistency_score", "final_penalty",
        "lowfreq_penalty",
        "semantic_penalty", "semantic_over_penalty", "semantic_under_penalty",
        "texture_penalty", "unmatched_penalty", "unmatched_area_ratio",
        "tau_pair_used", "K_image1", "K_image2", "valid_masks_image1", "valid_masks_image2", "num_masks",
        "image1_path", "image2_path", "overall_path", "status", "error",
    ]
    if args.skip_existing and all_scores_csv.exists():
        print(f"Skip existing output: {all_scores_csv}")
        return

    # Create output_dir/all_scores.csv immediately so it exists even if the run
    # later stops on an error before reaching the final write.
    write_ordered_csv(all_scores_csv, [], field_order)

    pairs, missing_in_image2, missing_in_image1 = find_pairs(
        image1_dir=image1_dir,
        image2_dir=image2_dir,
        recursive=args.recursive,
    )
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]

    print(f"Matched pairs: {len(pairs)}")
    print(f"Missing in image2 dir: {len(missing_in_image2)}")
    print(f"Missing in image1 dir: {len(missing_in_image1)}")

    if not pairs:
        print("No matched image pairs found.")
        write_ordered_csv(all_scores_csv, [], field_order)
        return

    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Building INSID3...")
    model = build_insid3(
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

    global_scores = []

    for rel_key, image1_path, image2_path in tqdm(pairs, desc="Processing image pairs"):
        pair_dir = out_dir / safe_pair_dir_name(rel_key)

        try:
            pair_score, mask_rows = process_one_pair(
                model=model,
                image1_path=image1_path,
                image2_path=image2_path,
                rel_key=rel_key,
                pair_out_dir=pair_dir,
                args=args,
            )
            global_scores.append(pair_score)
            write_ordered_csv(all_scores_csv, global_scores, field_order)

            print(
                f"[OK] {rel_key} | score={pair_score['consistency_score']:.4f} "
                f"penalty={pair_score['final_penalty']:.4f} "
                f"low={pair_score['lowfreq_penalty']:.4f} "
                f"sem={pair_score['semantic_penalty']:.4f} "
                f"tex={pair_score['texture_penalty']:.4f} "
                f"unmatched_area={pair_score['unmatched_area_ratio']:.4f}"
            )

        except Exception as e:
            err = traceback.format_exc()
            print(f"[ERROR] {rel_key}: {e}")
            print(err)
            global_scores.append({
                "dataset": args.dataset,
                "method": args.method,
                "relative_image_name": str(rel_key),
                "consistency_score": 0.0,
                "final_penalty": 1.0,
                "lowfreq_penalty": 1.0,
                "semantic_penalty": 1.0,
                "semantic_over_penalty": 0.0,
                "semantic_under_penalty": 1.0,
                "texture_penalty": 1.0,
                "unmatched_penalty": 1.0,
                "unmatched_area_ratio": 1.0,
                "tau_pair_used": -1,
                "K_image1": -1,
                "K_image2": -1,
                "valid_masks_image1": -1,
                "valid_masks_image2": -1,
                "num_masks": 0,
                "image1_path": str(image1_path),
                "image2_path": str(image2_path),
                "overall_path": "",
                "status": "error",
                "error": str(e),
            })
            write_ordered_csv(all_scores_csv, global_scores, field_order)
            if not args.continue_on_error:
                raise

        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # One compact CSV for this run. Re-write once at the end to ensure final order.
    write_ordered_csv(all_scores_csv, global_scores, field_order)

    print("Done.")
    print(f"Output dir: {out_dir}")
    print(f"All scores CSV: {all_scores_csv}")
    print("Per-image overall PNGs: output_dir/<image_name>/overall.png")



if __name__ == "__main__":
    main()
