#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a point usefulness/value predictor on SAM2 teacher NPZ.

This variant follows a "SAM-style" idea:
- frozen SAM2 image encoder
- frozen SAM2 prompt encoder for BOX and POINTS (no pure-geo box encoding by default)
- optional baseline state: SAM2 box-only mask prob sampled at each candidate point
- a lightweight head (MLP or a Transformer "decoder") predicts:
    (1) usefulness u in {0,1}  where u = 1[v > useful_thresh]
    (2) value v = max(G_pos, G_neg)  (teacher from NPZ), regressed (in per-image z-score space)

Selection metric: top-k mean teacher value at predicted top-k points.

Run example:

CUDA_VISIBLE_DEVICES=0 python train_point_useful_value_samstyle.py \
  --train_npz_dir sam2_teacher_npz_iou/refcoco_train_seed2026_N64_pert0.05-0.15_base_plus_dIoU/ \
  --val_npz_dir   sam2_teacher_npz_iou/refcoco_testA_seed2026_N64_pert0.05-0.15_base_plus_dIoU/ \
  --sam2_ckpt ../checkpoints/sam2.1_hiera_base_plus.pt \
  --sam2_cfg  configs/sam2.1/sam2.1_hiera_b+.yaml \
  --epochs 3 --batch_size 8 --k_top 5 --seed 3 \
  --save_dir ./ckpt_useful_valuev5 \
  --head_type decoder \
  --decoder_dim 256 --decoder_layers 2 --decoder_nhead 8 --decoder_ffn_mult 4 \
  --decoder_mem_downsample 1 \
  --use_prompt_embed --use_box_prompt --use_base_mask \
  --useful_thresh 0.0 \
  --w_cls 1.0 --w_val 0.2 --w_rank 0.2 --rank_T 0.5 \
  -val_transform log --val_eps 1e-6 --val_clip 1.0 \
  --lr_scheduler cosine_warmup --warmup_steps 500 --lr_min 1e-6 \
  --use_ema --ema_decay 0.999 --ema_eval
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import glob
import platform
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# SAM2 imports (same as your sam2 repo layout)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# -------------------------
# Repro / misc
# -------------------------

def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _safe_run(cmd: Sequence[str]) -> str:
    import subprocess
    try:
        out = subprocess.check_output(list(cmd), stderr=subprocess.STDOUT).decode("utf-8", errors="ignore").strip()
        return out
    except Exception as e:
        return f"<err:{type(e).__name__}:{e}>"


def make_split_indices(n: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * val_ratio))
    va = idx[:n_val]
    tr = idx[n_val:]
    return tr, va


# -------------------------
# Dataset
# -------------------------

class TeacherNPZDataset(Dataset):
    """Teacher NPZ dataset.

    Supports both `img_path` (teacher builder default) and `image_path` (older scripts).
    If the stored absolute path does not exist, it can be re-resolved by basename under --data_root.
    """

    def __init__(self, npz_dir: str, data_root: str = ""):
        self.npz_dir = npz_dir
        self.data_root = data_root
        self.files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz found under: {npz_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        data = np.load(path, allow_pickle=True)

        out: Dict[str, Any] = {k: data[k] for k in data.files}

        # unify image path key
        if "image_path" not in out and "img_path" in out:
            out["image_path"] = out["img_path"]

        image_path = _to_py_str(out.get("image_path", ""))

        # If npz stores absolute path from another machine, optionally re-resolve via data_root.
        if self.data_root:
            file_name = _to_py_str(out.get("file_name", ""))
            key = file_name if file_name else (os.path.basename(image_path) if image_path else "")
            if (not image_path) or (not os.path.exists(image_path)):
                if key:
                    image_path = find_image_path(self.data_root, key)

        if not image_path or (not os.path.exists(image_path)):
            raise FileNotFoundError(
                f"NPZ image path not found. npz='{path}' image_path='{image_path}' data_root='{self.data_root}'"
            )

        out["image_path"] = image_path
        out["_npz_path"] = path
        return out
def _load_image_rgb(path: str) -> np.ndarray:
    """Load an RGB image from disk."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.array(img)



# ---------------------------
# Image path resolver (for portability)
# ---------------------------

def _to_py_str(v) -> str:
    """Convert numpy/bytes/py objects to a clean python str."""
    if isinstance(v, np.ndarray):
        v = v.item()
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="ignore")
    return str(v)

def find_image_path(data_root: str, file_name: str) -> str:
    """Locate RefCOCO/COCO image by file name (robust across repos)."""
    if not data_root:
        return file_name

    if file_name and os.path.exists(file_name):
        return file_name

    base = os.path.basename(str(file_name))
    candidates = [
        os.path.join(data_root, "images", base),
        os.path.join(data_root, "images", "train2014", base),
        os.path.join(data_root, "images", "val2014", base),
        os.path.join(data_root, "train2014", base),
        os.path.join(data_root, "val2014", base),
        os.path.join(data_root, "coco", "images", base),
        os.path.join(data_root, base),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # fallback: recursive search (slower but robust)
    hits = glob.glob(os.path.join(data_root, "**", base), recursive=True)
    for p in hits:
        if os.path.isfile(p):
            return p

    raise FileNotFoundError(f"Cannot find image '{base}' under data_root='{data_root}'")


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # images
    image_paths = [_to_py_str(b.get("image_path", b.get("img_path", ""))) for b in batch]
    images_np = [_load_image_rgb(p) for p in image_paths]
    # NPZ versions may not store orig_h/orig_w. Prefer stored metadata when available,
    # otherwise fall back to the loaded image resolution.
    def _to_int(v):
        if isinstance(v, np.ndarray):
            try:
                v = v.item()
            except Exception:
                pass
        return int(v)

    def _get_orig_hw(b, img):
        if "orig_h" in b and "orig_w" in b:
            return (_to_int(b["orig_h"]), _to_int(b["orig_w"]))
        if "H" in b and "W" in b:
            return (_to_int(b["H"]), _to_int(b["W"]))
        if "orig_hw" in b:
            hw = b["orig_hw"]
            try:
                return (_to_int(hw[0]), _to_int(hw[1]))
            except Exception:
                pass
        if "image_shape" in b:
            hw = b["image_shape"]
            try:
                return (_to_int(hw[0]), _to_int(hw[1]))
            except Exception:
                pass
        return (int(img.shape[0]), int(img.shape[1]))

    orig_hw = [_get_orig_hw(b, img) for b, img in zip(batch, images_np)]

    # tensors (robust key names across NPZ versions)
    def _get_box(b):
        if "box" in b:
            return b["box"]
        if "boxes_xyxy" in b:
            return b["boxes_xyxy"]
        if "boxes" in b:
            return b["boxes"]
        raise KeyError("Missing 'box'/'boxes_xyxy' in npz.")

    def _get_points(b):
        if "points" in b:
            return b["points"]
        if "points_xy" in b:
            return b["points_xy"]
        raise KeyError("Missing 'points'/'points_xy' in npz.")

    def _get_labels(b):
        if "labels" in b:
            return b["labels"]
        if "y" in b:
            return b["y"]
        raise KeyError("Missing 'labels'/'y' in npz.")

    boxes = torch.from_numpy(np.stack([_get_box(b).astype(np.float32) for b in batch], axis=0))   # (B,4)
    points = torch.from_numpy(np.stack([_get_points(b).astype(np.float32) for b in batch], axis=0)) # (B,N,2)
    y = torch.from_numpy(np.stack([_get_labels(b).astype(np.float32) for b in batch], axis=0))     # (B,N)

    G_pos = torch.from_numpy(np.stack([b["G_pos"].astype(np.float32) for b in batch], axis=0))    # (B,N)
    G_neg = torch.from_numpy(np.stack([b["G_neg"].astype(np.float32) for b in batch], axis=0))    # (B,N)

    if "valid_mask" in batch[0]:
        valid = torch.from_numpy(np.stack([b["valid_mask"].astype(bool) for b in batch], axis=0))
    else:
        valid = torch.ones_like(y, dtype=torch.bool)

    g_exp = None
    if "g_exp" in batch[0]:
        g_exp = torch.from_numpy(np.stack([b["g_exp"].astype(np.float32) for b in batch], axis=0))

    # keep meta if needed
    ref_id = [int(b.get("ref_id", -1)) for b in batch]
    file_name = [_to_py_str(b.get("file_name", "")) for b in batch]
    image_id = [int(b.get("image_id", -1)) for b in batch]

    return {
        "image_paths": image_paths,
        "images_np": images_np,
        "orig_hw": orig_hw,
        "boxes": boxes,
        "points": points,
        "y": y,
        "G_pos": G_pos,
        "G_neg": G_neg,
        "g_exp": g_exp,
        "valid_mask": valid,
        "ref_id": ref_id,
        "file_name": file_name,
        "image_id": image_id,
    }



# -------------------------
# SAM2 helpers
# -------------------------

def _get_model_image_size(predictor: SAM2ImagePredictor) -> int:
    # best-effort across forks
    if hasattr(predictor, "model") and hasattr(predictor.model, "image_size"):
        return int(predictor.model.image_size)
    if hasattr(predictor, "_image_size"):
        return int(predictor._image_size)
    # common default in SAM
    return 1024


def _get_image_embedding(predictor: SAM2ImagePredictor) -> torch.Tensor:
    """
    Try to fetch the image embedding tensor from predictor after set_image(_batch).
    Supports common SAM2 predictor internals.
    """
    # Most forks keep features under predictor._features or predictor._image_embedding
    for name in ["_image_embedding", "image_embedding", "_features", "_image_features", "_cached_features"]:
        if hasattr(predictor, name):
            obj = getattr(predictor, name)
            if torch.is_tensor(obj):
                return obj
            # sometimes dict with "image_embed"
            if isinstance(obj, dict):
                for k in ["image_embed", "image_embedding", "img_embed", "x"]:
                    if k in obj and torch.is_tensor(obj[k]):
                        return obj[k]
    # fallback: try model attribute
    for name in ["image_embed", "image_embedding", "img_embed"]:
        if hasattr(predictor.model, name):
            t = getattr(predictor.model, name)
            if torch.is_tensor(t):
                return t
    raise RuntimeError("Cannot find image embedding on SAM2 predictor. Inspect predictor attributes.")


def _find_prompt_encoder(model: torch.nn.Module):
    for name in ["prompt_encoder", "sam_prompt_encoder", "prompt_encoder_model"]:
        if hasattr(model, name):
            return getattr(model, name)
    raise RuntimeError("Cannot find prompt encoder on SAM2 model. Inspect model attributes.")


@torch.no_grad()
def extract_point_features_sam2(
    predictor: SAM2ImagePredictor,
    images_np: List[np.ndarray],
    points_xy: torch.Tensor,  # (B,N,2)
    orig_hw: List[Tuple[int, int]],
    device: torch.device,
    return_embed: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Sample per-point features from SAM2 *image embedding* via grid_sample.
    Optionally returns the full image embedding (B,C,Hf,Wf).
    """
    # Fast path
    if hasattr(predictor, "set_image_batch"):
        predictor.set_image_batch(images_np)
        img_embed = _get_image_embedding(predictor)
        if img_embed.dim() == 3:
            img_embed = img_embed.unsqueeze(0)
    else:
        # Robust fallback: do per-image and stack (IMPORTANT when batch_size>1)
        embeds = []
        for img in images_np:
            predictor.set_image(img)
            emb = _get_image_embedding(predictor)
            if emb.dim() == 3:
                emb = emb.unsqueeze(0)
            embeds.append(emb)
        img_embed = torch.cat(embeds, dim=0)

    img_embed = img_embed.to(device)
    B, C, Hf, Wf = img_embed.shape
    image_size = _get_model_image_size(predictor)

    grids = []
    for i in range(B):
        if hasattr(predictor, "_transforms") and hasattr(predictor._transforms, "transform_coords"):
            coords = predictor._transforms.transform_coords(
                points_xy[i].to(device), normalize=True, orig_hw=orig_hw[i]
            )
            x_ = coords[:, 0] / float(image_size) * 2.0 - 1.0
            y_ = coords[:, 1] / float(image_size) * 2.0 - 1.0
        else:
            H0, W0 = orig_hw[i]
            x_ = (points_xy[i, :, 0].to(device) / max(W0 - 1, 1)) * 2.0 - 1.0
            y_ = (points_xy[i, :, 1].to(device) / max(H0 - 1, 1)) * 2.0 - 1.0

        grid = torch.stack([x_, y_], dim=-1)
        grids.append(grid)

    grid = torch.stack(grids, dim=0).view(B, -1, 1, 2)
    sampled = F.grid_sample(img_embed, grid, mode="bilinear", align_corners=True)
    sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()  # (B,N,C)

    if return_embed:
        return sampled, img_embed
    return sampled


@torch.no_grad()
def extract_point_prompt_embeddings_sam2(
    predictor: SAM2ImagePredictor,
    points_xy: torch.Tensor,                 # (B,N,2) in original pixel coords
    point_labels: torch.Tensor,              # (B,N) int/float
    orig_hw: List[Tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    """Best-effort: get SAM2 prompt-encoder sparse embeddings for points."""
    model = predictor.model
    pe = _find_prompt_encoder(model)
    image_size = _get_model_image_size(predictor)

    coords_list = []
    for i in range(points_xy.shape[0]):
        if hasattr(predictor, "_transforms") and hasattr(predictor._transforms, "transform_coords"):
            coords = predictor._transforms.transform_coords(
                points_xy[i].to(device), normalize=True, orig_hw=orig_hw[i]
            )
        else:
            H0, W0 = orig_hw[i]
            coords = points_xy[i].to(device).clone().float()
            coords[:, 0] = coords[:, 0] / max(W0 - 1, 1) * float(image_size)
            coords[:, 1] = coords[:, 1] / max(H0 - 1, 1) * float(image_size)
        coords_list.append(coords)

    coords_b = torch.stack(coords_list, dim=0)  # (B,N,2)
    labels_b = point_labels.to(device).long()

    try:
        sparse, dense = pe(points=(coords_b, labels_b), boxes=None, masks=None)
    except TypeError:
        sparse, dense = pe(points=(coords_b, labels_b), boxes=None)

    if sparse.dim() != 3:
        raise RuntimeError(f"Unexpected sparse embedding shape: {tuple(sparse.shape)}")

    B, Toks, D = sparse.shape
    N = coords_b.shape[1]
    if Toks == N:
        out = sparse
    elif Toks > N:
        out = sparse[:, :N, :]
    else:
        pad = torch.zeros((B, N - Toks, D), device=sparse.device, dtype=sparse.dtype)
        out = torch.cat([sparse, pad], dim=1)

    return out.to(device)


@torch.no_grad()
def extract_box_prompt_embeddings_sam2(
    predictor: SAM2ImagePredictor,
    boxes_xyxy: torch.Tensor,                # (B,4) in original pixel coords
    orig_hw: List[Tuple[int, int]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get SAM2 prompt-encoder (sparse, dense) embeddings for boxes."""
    model = predictor.model
    pe = _find_prompt_encoder(model)
    image_size = _get_model_image_size(predictor)

    B = boxes_xyxy.shape[0]
    boxes_list = []
    for i in range(B):
        b = boxes_xyxy[i].to(device).float()
        if hasattr(predictor, "_transforms") and hasattr(predictor._transforms, "transform_boxes"):
            # some forks provide this
            bb = predictor._transforms.transform_boxes(b[None, :], normalize=True, orig_hw=orig_hw[i]).squeeze(0)
        else:
            H0, W0 = orig_hw[i]
            bb = b.clone()
            bb[0] = bb[0] / max(W0 - 1, 1) * float(image_size)
            bb[2] = bb[2] / max(W0 - 1, 1) * float(image_size)
            bb[1] = bb[1] / max(H0 - 1, 1) * float(image_size)
            bb[3] = bb[3] / max(H0 - 1, 1) * float(image_size)
        boxes_list.append(bb)

    boxes_b = torch.stack(boxes_list, dim=0)  # (B,4)

    # SAM-style expects (B,2,2) corners
    boxes_corners = boxes_b.view(B, 2, 2)

    try:
        sparse, dense = pe(points=None, boxes=boxes_corners, masks=None)
    except TypeError:
        sparse, dense = pe(points=None, boxes=boxes_corners)

    if sparse.dim() != 3:
        raise RuntimeError(f"Unexpected box sparse embedding shape: {tuple(sparse.shape)}")
    if dense.dim() != 4:
        raise RuntimeError(f"Unexpected box dense embedding shape: {tuple(dense.shape)}")

    return sparse.to(device), dense.to(device)


@torch.no_grad()
def predict_box_only_mask_probs_sam2(
    predictor: SAM2ImagePredictor,
    images_np: List[np.ndarray],
    boxes_xyxy: torch.Tensor,                # (B,4) original pixel coords
    orig_hw: List[Tuple[int, int]],
    device: torch.device,
) -> List[torch.Tensor]:
    """
    Run SAM2 mask prediction with box prompt only (baseline state).
    Returns a list of per-image mask probability tensors in original image resolution (H,W), float32 in [0,1].

    Note: different SAM2 forks expose different predict APIs; we try several fallbacks.
    """
    probs: List[torch.Tensor] = []
    B = boxes_xyxy.shape[0]

    # try batch API first
    if hasattr(predictor, "set_image_batch") and hasattr(predictor, "predict_batch"):
        predictor.set_image_batch(images_np)
        boxes_np = boxes_xyxy.detach().cpu().numpy().astype(np.float32)
        try:
            out = predictor.predict_batch(box=boxes_np, multimask_output=False, return_logits=True)
        except TypeError:
            try:
                out = predictor.predict_batch(box=boxes_np, multimask_output=False)
            except Exception:
                out = None
        if out is not None:
            # common returns: masks, scores, logits or masks, scores
            if isinstance(out, (list, tuple)) and len(out) >= 1:
                masks = out[0]  # (B,1,H,W) or list
                logits = out[2] if (len(out) >= 3 and out[2] is not None) else None
                for i in range(B):
                    if logits is not None:
                        lg = logits[i]
                        if isinstance(lg, np.ndarray):
                            lg = torch.from_numpy(lg)
                        if lg.dim() == 3:
                            lg = lg[0]
                        prob = torch.sigmoid(lg.float())
                    else:
                        mk = masks[i]
                        if isinstance(mk, np.ndarray):
                            mk = torch.from_numpy(mk)
                        if mk.dim() == 3:
                            mk = mk[0]
                        prob = mk.float()
                    probs.append(prob.to(device))
            if len(probs) == B:
                return probs

    # safe per-image fallback
    for i in range(B):
        predictor.set_image(images_np[i])
        box0 = boxes_xyxy[i].detach().cpu().numpy().astype(np.float32)

        # try a few common box shapes expected by different forks
        box_candidates = [
            box0,
            box0[None, :],
            box0.reshape(2, 2),
            box0.reshape(1, 2, 2),
        ]

        prob = None
        last_err: Optional[Exception] = None

        for bc in box_candidates:
            # try with logits
            try:
                out = predictor.predict(
                    box=bc, point_coords=None, point_labels=None, multimask_output=False, return_logits=True
                )
                if isinstance(out, (list, tuple)) and len(out) >= 3:
                    masks, scores, logits = out[0], out[1], out[2]
                else:
                    masks, scores, logits = out
                lg = logits
                if isinstance(lg, (list, tuple)):
                    lg = lg[0]
                if isinstance(lg, np.ndarray):
                    lg = torch.from_numpy(lg)
                if torch.is_tensor(lg) and lg.dim() == 3:
                    lg = lg[0]
                prob = torch.sigmoid(lg.float())
                break
            except TypeError as e:
                last_err = e
            except Exception as e:
                last_err = e

            # try without return_logits
            try:
                out = predictor.predict(box=bc, point_coords=None, point_labels=None, multimask_output=False)
                if isinstance(out, (list, tuple)) and len(out) >= 1:
                    masks = out[0]
                else:
                    masks = out
                mk = masks
                if isinstance(mk, (list, tuple)):
                    mk = mk[0]
                if isinstance(mk, np.ndarray):
                    mk = torch.from_numpy(mk)
                if torch.is_tensor(mk) and mk.dim() == 3:
                    mk = mk[0]
                prob = mk.float()
                break
            except Exception as e:
                last_err = e

            # last resort: some forks accept box as positional arg
            try:
                out = predictor.predict(bc, multimask_output=False)
                if isinstance(out, (list, tuple)) and len(out) >= 1:
                    masks = out[0]
                else:
                    masks = out
                mk = masks
                if isinstance(mk, (list, tuple)):
                    mk = mk[0]
                if isinstance(mk, np.ndarray):
                    mk = torch.from_numpy(mk)
                if torch.is_tensor(mk) and mk.dim() == 3:
                    mk = mk[0]
                prob = mk.float()
                break
            except Exception as e:
                last_err = e

        if prob is None:
            raise RuntimeError(f"predict_box_only_mask_probs_sam2 failed on sample {i}: {last_err}")

        probs.append(prob.to(device))
    return probs


@torch.no_grad()
def sample_mask_at_points(
    mask_probs: List[torch.Tensor],          # list of (H,W) in original res
    points_xy: torch.Tensor,                 # (B,N,2) original pixel coords
    device: torch.device,
) -> torch.Tensor:
    """Bilinear sample mask prob at points. Returns (B,N,1)."""
    B, N, _ = points_xy.shape
    out_list = []
    for i in range(B):
        m = mask_probs[i].to(device).float()
        if m.dim() != 2:
            raise RuntimeError(f"mask_probs[{i}] must be (H,W), got {tuple(m.shape)}")
        H0, W0 = m.shape
        # grid_sample wants (1,1,H,W)
        m4 = m.view(1, 1, H0, W0)

        x = points_xy[i, :, 0].to(device)
        y = points_xy[i, :, 1].to(device)
        gx = (x / max(W0 - 1, 1)) * 2.0 - 1.0
        gy = (y / max(H0 - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(1, N, 1, 2)
        samp = F.grid_sample(m4, grid, mode="bilinear", align_corners=True)  # (1,1,N,1)
        samp = samp.view(N, 1)
        out_list.append(samp)

    return torch.stack(out_list, dim=0)  # (B,N,1)


# -------------------------
# Metrics
# -------------------------

def topk_mean_teacher_value(
    teacher_v: torch.Tensor,   # (B,N)
    score: torch.Tensor,       # (B,N)
    valid: torch.Tensor,       # (B,N) bool
    k: int,
) -> float:
    """Mean teacher value at top-k by predicted score (per image, then average)."""
    B, N = teacher_v.shape
    vals = []
    neg_inf = -1e9
    for i in range(B):
        sc = score[i].masked_fill(~valid[i], neg_inf)
        idx = torch.topk(sc, k=min(k, int(valid[i].sum().item())), dim=0).indices
        vals.append(teacher_v[i, idx].mean().item() if idx.numel() > 0 else 0.0)
    return float(np.mean(vals)) if len(vals) else 0.0


def topk_random_teacher_value(
    teacher_v: torch.Tensor,
    valid: torch.Tensor,
    k: int,
    seed: int = 123,
) -> float:
    rng = np.random.RandomState(seed)
    B, N = teacher_v.shape
    vals = []
    for i in range(B):
        valid_idx = torch.nonzero(valid[i], as_tuple=False).squeeze(1).cpu().numpy()
        if valid_idx.size == 0:
            vals.append(0.0)
            continue
        choose = rng.choice(valid_idx, size=min(k, valid_idx.size), replace=False)
        vals.append(float(teacher_v[i, torch.from_numpy(choose).to(teacher_v.device)].mean().item()))
    return float(np.mean(vals)) if len(vals) else 0.0


def binary_auc_roc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute AUROC in pure torch, labels in {0,1}.
    Returns nan if only one class present.
    """
    probs = probs.detach().float().flatten()
    labels = labels.detach().float().flatten()
    pos = labels > 0.5
    neg = ~pos
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # rank probs
    order = torch.argsort(probs)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(order.numel(), device=probs.device, dtype=torch.float32)

    # AUROC via Mann–Whitney U
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos - 1) / 2.0) / (n_pos * n_neg)
    return float(auc.item())


# -------------------------
# Heads
# -------------------------

class UsefulValueMLPHead(nn.Module):
    def __init__(self, point_in_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(point_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_u = nn.Linear(hidden_dim, 1)     # usefulness logit
        self.head_v = nn.Linear(hidden_dim, 1)     # value in z-space

    def forward(self, x_point: torch.Tensor, **kwargs):
        h = self.trunk(x_point)
        u_logit = self.head_u(h).squeeze(-1)
        v_hat_z = self.head_v(h).squeeze(-1)
        return u_logit, v_hat_z


def _make_2d_sincos_pos_embed(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    """(H*W, dim) 2D sincos pos embed."""
    if dim % 4 != 0:
        # fall back to zeros if awkward
        return torch.zeros((h * w, dim), device=device)

    y = torch.arange(h, device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    # normalize
    yy = yy / max(h - 1, 1)
    xx = xx / max(w - 1, 1)

    dim_half = dim // 2
    dim_quarter = dim_half // 2

    omega = torch.arange(dim_quarter, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(dim_quarter - 1, 1)))

    out_y = yy.reshape(-1, 1) * omega.reshape(1, -1)
    out_x = xx.reshape(-1, 1) * omega.reshape(1, -1)

    pos_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)
    pos_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)
    return torch.cat([pos_y, pos_x], dim=1)  # (H*W, dim)


class UsefulValueDecoderHead(nn.Module):
    """
    A SAM-flavored "decoder":
    - memory tokens from image embedding (and optional dense box prompt)
    - target tokens: [box_sparse_tokens, point_tokens]
    - outputs per-point: usefulness logit + value z

    This is NOT the exact SAM2 mask decoder, but follows the same "tokens + cross-attn to image" pattern.
    """
    def __init__(
        self,
        point_in_dim: int,
        img_in_dim: int,
        box_sparse_dim: int,
        box_dense_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        mem_downsample: int = 1,
        use_mem_pos: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.mem_downsample = max(1, int(mem_downsample))
        self.use_mem_pos = bool(use_mem_pos)

        self.point_proj = nn.Linear(point_in_dim, d_model)
        self.img_proj = nn.Linear(img_in_dim, d_model)
        self.box_sparse_proj = nn.Linear(box_sparse_dim, d_model)
        self.box_dense_proj = nn.Linear(box_dense_dim, d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_mult * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

        self.head_u = nn.Linear(d_model, 1)
        self.head_v = nn.Linear(d_model, 1)

    def forward(
        self,
        x_point: torch.Tensor,                          # (B,N,point_in_dim)
        *,
        img_embed: torch.Tensor,                        # (B,C,Hf,Wf)
        box_sparse: Optional[torch.Tensor] = None,      # (B,Tb,Db)
        box_dense: Optional[torch.Tensor] = None,       # (B,Db,Hf,Wf) or (B,Db,Hd,Wd)
        tgt_key_padding_mask: Optional[torch.Tensor] = None,  # (B, N) for points only; we will expand for box tokens
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x_point.shape
        device = x_point.device

        # memory: image embedding (optionally downsample)
        mem = img_embed
        if self.mem_downsample > 1:
            mem = F.avg_pool2d(mem, kernel_size=self.mem_downsample, stride=self.mem_downsample)

        Bm, C, Hm, Wm = mem.shape
        mem_tokens = mem.permute(0, 2, 3, 1).reshape(Bm, Hm * Wm, C)  # (B,HW,C)
        mem_tokens = self.img_proj(mem_tokens)

        # add dense box embedding into memory if provided
        if box_dense is not None:
            bd = box_dense
            if bd.dim() != 4:
                raise RuntimeError(f"box_dense must be (B,Db,H,W), got {tuple(bd.shape)}")
            if self.mem_downsample > 1:
                bd = F.avg_pool2d(bd, kernel_size=self.mem_downsample, stride=self.mem_downsample)
            if bd.shape[-2:] != (Hm, Wm):
                bd = F.interpolate(bd, size=(Hm, Wm), mode="bilinear", align_corners=False)
            bd_tokens = bd.permute(0, 2, 3, 1).reshape(Bm, Hm * Wm, bd.shape[1])
            mem_tokens = mem_tokens + self.box_dense_proj(bd_tokens)

        if self.use_mem_pos:
            pos = _make_2d_sincos_pos_embed(Hm, Wm, self.d_model, device).unsqueeze(0)  # (1,HW,D)
            mem_tokens = mem_tokens + pos

        # target tokens: box_sparse + point tokens
        tgt_point = self.point_proj(x_point)

        Tb = 0
        tgt = tgt_point
        key_pad = None

        if box_sparse is not None:
            if box_sparse.dim() != 3:
                raise RuntimeError(f"box_sparse must be (B,Tb,Db), got {tuple(box_sparse.shape)}")
            Tb = box_sparse.shape[1]
            tgt_box = self.box_sparse_proj(box_sparse)
            tgt = torch.cat([tgt_box, tgt_point], dim=1)  # (B,Tb+N,D)

            if tgt_key_padding_mask is not None:
                # expand with False for box tokens
                if tgt_key_padding_mask.shape != (B, N):
                    raise RuntimeError(f"tgt_key_padding_mask must be (B,N), got {tuple(tgt_key_padding_mask.shape)}")
                pad_box = torch.zeros((B, Tb), device=device, dtype=torch.bool)
                key_pad = torch.cat([pad_box, tgt_key_padding_mask], dim=1)  # (B,Tb+N)
        else:
            if tgt_key_padding_mask is not None:
                key_pad = tgt_key_padding_mask

        # decode
        out = self.decoder(tgt=tgt, memory=mem_tokens, tgt_key_padding_mask=key_pad)  # (B,Tb+N,D) or (B,N,D)
        if Tb > 0:
            out_p = out[:, Tb:, :]
        else:
            out_p = out

        u_logit = self.head_u(out_p).squeeze(-1)  # (B,N)
        v_hat_z = self.head_v(out_p).squeeze(-1)  # (B,N)
        return u_logit, v_hat_z


# -------------------------
# Train/eval
# -------------------------

def _zscore_per_image(x: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    x: (B,N), valid: (B,N) bool
    returns x_z, mu, std each (B,1)
    """
    vt = valid.float()
    denom = vt.sum(dim=1, keepdim=True).clamp(min=1.0)
    mu = (x * vt).sum(dim=1, keepdim=True) / denom
    var = ((x - mu) ** 2 * vt).sum(dim=1, keepdim=True) / denom
    std = var.sqrt().clamp(min=1e-6)
    x_z = ((x - mu) / std).clamp(-5.0, 5.0)
    return x_z, mu, std



def apply_value_transform(v: torch.Tensor, mode: str = "none", eps: float = 1e-6) -> torch.Tensor:
    """Monotonic transform for heavy-tailed non-negative values."""
    if mode is None:
        mode = "none"
    mode = str(mode).lower()
    if mode == "none":
        return v
    v = v.clamp(min=0.0)
    if mode == "log":
        return torch.log(v + float(eps))
    if mode == "sqrt":
        return torch.sqrt(v + 1e-12)
    raise ValueError(f"Unknown val_transform: {mode}")


def invert_value_transform(v_t: torch.Tensor, mode: str = "none", eps: float = 1e-6) -> torch.Tensor:
    """Inverse of apply_value_transform (approx), used for scoring/metrics."""
    if mode is None:
        mode = "none"
    mode = str(mode).lower()
    if mode == "none":
        return v_t
    if mode == "log":
        return torch.exp(v_t) - float(eps)
    if mode == "sqrt":
        return v_t * v_t
    raise ValueError(f"Unknown val_transform: {mode}")


class EMA:
    """Exponential Moving Average for model parameters (fp32 shadow)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().float().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = p.detach().float().clone()
            else:
                self.shadow[name].mul_(d).add_(p.detach().float(), alpha=(1.0 - d))

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name].to(device=p.device, dtype=p.dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {k: v.detach().cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: Dict[str, torch.Tensor]):
        self.shadow = {k: v.detach().float().clone() for k, v in sd.items()}


def forward_and_loss(
    sam2_pred: SAM2ImagePredictor,
    head: nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    k_top: int,
    useful_thresh: float,
    # loss weights
    w_cls: float = 1.0,
    w_val: float = 0.2,
    w_rank: float = 0.2,
    T: float = 0.5,
    # value transform for heavy-tailed labels
    val_transform: str = "none",
    val_eps: float = 1e-6,
    val_clip: float = 1.0,
    # feature toggles
    use_prompt_embed: bool = True,
    use_box_prompt: bool = True,
    use_base_mask: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:

    points = batch["points"].to(device)      # (B,N,2)
    boxes = batch["boxes"].to(device)        # (B,4)
    G_pos = batch["G_pos"].to(device)        # (B,N)
    G_neg = batch["G_neg"].to(device)        # (B,N)
    valid = batch["valid_mask"].to(device)   # (B,N)

    # teacher targets
    v = torch.maximum(G_pos, G_neg).clamp(min=0.0)  # (B,N) non-negative value
    u = (v > useful_thresh).float()          # (B,N)

    # (A) image embedding + per-point visual features
    vis, img_embed = extract_point_features_sam2(
        predictor=sam2_pred,
        images_np=batch["images_np"],
        points_xy=points,
        orig_hw=batch["orig_hw"],
        device=device,
        return_embed=True,
    )  # vis: (B,N,C), img_embed: (B,C,Hf,Wf)

    # (B) point prompt embeddings (two hypotheses)
    if use_prompt_embed:
        ones = torch.ones_like(u)
        zeros = torch.zeros_like(u)
        pe_pos = extract_point_prompt_embeddings_sam2(sam2_pred, points, ones, batch["orig_hw"], device)
        pe_neg = extract_point_prompt_embeddings_sam2(sam2_pred, points, zeros, batch["orig_hw"], device)
        point_feat = torch.cat([vis, pe_pos, pe_neg], dim=-1)  # (B,N, C + 2*Dpe)
    else:
        point_feat = vis

    # (C) baseline mask state (box-only)
    mask_at_points = None
    if use_base_mask:
        mask_probs = predict_box_only_mask_probs_sam2(sam2_pred, batch["images_np"], boxes, batch["orig_hw"], device)
        mask_at_points = sample_mask_at_points(mask_probs, points, device)  # (B,N,1)
        point_feat = torch.cat([point_feat, mask_at_points], dim=-1)

    # (D) box prompt embeddings (sparse+dense) for decoder memory + tokens
    box_sparse = None
    box_dense = None
    if use_box_prompt:
        box_sparse, box_dense = extract_box_prompt_embeddings_sam2(sam2_pred, boxes, batch["orig_hw"], device)

    # forward
    # Note: decoder head expects img_embed + (optional) box_sparse/box_dense and point padding mask.
    # For MLP head, extra kwargs are ignored.
    u_logit, v_hat_z = head(
        point_feat,
        img_embed=img_embed,
        box_sparse=box_sparse,
        box_dense=box_dense,
        tgt_key_padding_mask=(~valid),
    )

    # losses
    # classification (usefulness)
    u_valid = u[valid]
    pos = u_valid.sum().clamp(min=1.0)
    neg = (1.0 - u_valid).sum().clamp(min=1.0)
    pos_weight = (neg / pos).detach()
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    loss_cls = bce(u_logit, u)[valid].mean()

    # value regression in z-space
    v_t = apply_value_transform(v, mode=val_transform, eps=val_eps)
    v_z, mu, std = _zscore_per_image(v_t, valid)
    loss_val = F.smooth_l1_loss(v_hat_z[valid], v_z[valid], reduction="mean")

    # listwise ranking on value (optional)
    neg_inf = -1e9
    p_log = F.log_softmax(v_hat_z.masked_fill(~valid, neg_inf) / T, dim=1)
    q = F.softmax(v_z.masked_fill(~valid, neg_inf) / T, dim=1)
    loss_rank = F.kl_div(p_log, q, reduction="batchmean")

    loss = w_cls * loss_cls + w_val * loss_val + w_rank * loss_rank

    # metrics
    with torch.no_grad():
        u_prob = torch.sigmoid(u_logit)
        # recover raw value scale for scoring/metrics (monotonic transform)
        v_hat_t = (v_hat_z * std + mu)
        v_hat_raw = invert_value_transform(v_hat_t, mode=val_transform, eps=val_eps).clamp(min=0.0, max=val_clip)
        score = u_prob * v_hat_raw

        topk_pred = topk_mean_teacher_value(v, score, valid, k_top)
        topk_teacher = topk_mean_teacher_value(v, v, valid, k_top)
        topk_rand = topk_random_teacher_value(v, valid, k_top, seed=123)

        acc = float(((u_prob > 0.5).float()[valid] == u[valid]).float().mean().item())
        auc = binary_auc_roc(u_prob[valid], u[valid])

        # mean predicted useful prob and teacher useful rate
        u_rate = float(u[valid].mean().item())
        u_prob_m = float(u_prob[valid].mean().item())

    return loss, {
        "topk_pred": topk_pred,
        "topk_teacher": topk_teacher,
        "topk_rand": topk_rand,
        "acc": acc,
        "auc": auc,
        "u_rate": u_rate,
        "u_prob": u_prob_m,
        "loss_cls": float(loss_cls.detach().cpu()),
        "loss_val": float(loss_val.detach().cpu()),
        "loss_rank": float(loss_rank.detach().cpu()),
    }


@dataclasses.dataclass
class EvalResult:
    loss: float
    topk_pred: float
    topk_teacher: float
    topk_rand: float
    acc: float
    auc: float
    u_rate: float
    u_prob: float


@torch.no_grad()
def evaluate(
    sam2_pred: SAM2ImagePredictor,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    k_top: int,
    useful_thresh: float,
    w_cls: float,
    w_val: float,
    w_rank: float,
    T: float,
    # value transform for heavy-tailed labels
    val_transform: str = "none",
    val_eps: float = 1e-6,
    val_clip: float = 1.0,
    use_prompt_embed: bool = True,
    use_box_prompt: bool = True,
    use_base_mask: bool = False,
) -> EvalResult:
    head.eval()
    losses = []
    topk_p = []
    topk_t = []
    topk_r = []
    accs = []
    aucs = []
    u_rates = []
    u_probs = []

    for batch in loader:
        loss, m = forward_and_loss(
            sam2_pred, head, batch, device, k_top, useful_thresh,
            w_cls=w_cls, w_val=w_val, w_rank=w_rank, T=T,
                val_transform=val_transform,
                val_eps=val_eps,
                val_clip=val_clip,
            use_prompt_embed=use_prompt_embed, use_box_prompt=use_box_prompt, use_base_mask=use_base_mask,
        )
        losses.append(loss.item())
        topk_p.append(m["topk_pred"])
        topk_t.append(m["topk_teacher"])
        topk_r.append(m["topk_rand"])
        accs.append(m["acc"])
        if not math.isnan(m["auc"]):
            aucs.append(m["auc"])
        u_rates.append(m["u_rate"])
        u_probs.append(m["u_prob"])

    return EvalResult(
        loss=float(np.mean(losses)) if losses else 0.0,
        topk_pred=float(np.mean(topk_p)) if topk_p else 0.0,
        topk_teacher=float(np.mean(topk_t)) if topk_t else 0.0,
        topk_rand=float(np.mean(topk_r)) if topk_r else 0.0,
        acc=float(np.mean(accs)) if accs else 0.0,
        auc=float(np.mean(aucs)) if aucs else float("nan"),
        u_rate=float(np.mean(u_rates)) if u_rates else 0.0,
        u_prob=float(np.mean(u_probs)) if u_probs else 0.0,
    )


# -------------------------
# Args / main
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_npz_dir", type=str, required=True)
    p.add_argument("--val_npz_dir", type=str, default="")
    p.add_argument("--val_ratio", type=float, default=0.02)
    p.add_argument("--data_root", type=str, default="", help="RefCOCO dataset root (used to re-resolve image paths when NPZ paths are not valid).")

    p.add_argument("--sam2_ckpt", type=str, required=True)
    p.add_argument("--sam2_cfg", type=str, required=True)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--k_top", type=int, default=5)
    p.add_argument("--save_dir", type=str, default="./ckpt_useful_value")
    p.add_argument("--log_every", type=int, default=50)

    # features
    p.add_argument("--use_prompt_embed", action="store_true", help="use SAM2 point prompt sparse embeddings (pos+neg)")
    p.add_argument("--use_box_prompt", action="store_true", help="use SAM2 box prompt (sparse+dense)")
    p.add_argument("--use_base_mask", action="store_true", help="compute SAM2 box-only mask prob and sample at points")

    # targets
    p.add_argument("--useful_thresh", type=float, default=0.0, help="u=1[v>thresh], v=max(G_pos,G_neg)")

    # losses
    p.add_argument("--w_cls", type=float, default=1.0)
    p.add_argument("--w_val", type=float, default=0.2)
    p.add_argument("--w_rank", type=float, default=0.2)
    p.add_argument("--rank_T", type=float, default=0.5)

    p.add_argument("--val_transform", type=str, default="none", choices=["none", "log", "sqrt"],
                   help="Monotonic transform applied to non-negative value targets before z-scoring (helps heavy-tailed labels).")
    p.add_argument("--val_eps", type=float, default=1e-6, help="Epsilon for log transform: log(eps + v).")
    p.add_argument("--val_clip", type=float, default=1.0, help="Clamp predicted raw value into [0, val_clip] for scoring/metrics.")

    # LR scheduler / training tricks
    p.add_argument("--lr_scheduler", type=str, default="cosine",
                   choices=["none", "cosine", "cosine_warmup", "step"],
                   help="Learning-rate scheduler (recommended: cosine or cosine_warmup).")
    p.add_argument("--lr_min", type=float, default=1e-6, help="Minimum LR for cosine schedulers.")
    p.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps for cosine_warmup.")
    p.add_argument("--step_size", type=int, default=1, help="StepLR step_size (in epochs).")
    p.add_argument("--step_gamma", type=float, default=0.5, help="StepLR gamma.")

    # EMA
    p.add_argument("--use_ema", action="store_true", help="Use exponential moving average (EMA) of head parameters.")
    p.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay.")
    p.add_argument("--ema_eval", action="store_true", help="Evaluate/snapshot using EMA weights.")

    # head
    p.add_argument("--head_type", type=str, default="decoder", choices=["mlp", "decoder"])
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)

    # decoder params
    p.add_argument("--decoder_dim", type=int, default=256)
    p.add_argument("--decoder_layers", type=int, default=2)
    p.add_argument("--decoder_nhead", type=int, default=8)
    p.add_argument("--decoder_ffn_mult", type=int, default=4)
    p.add_argument("--decoder_mem_downsample", type=int, default=1)
    p.add_argument("--decoder_use_mem_pos", action="store_true", help="add 2D sincos pos embed to memory tokens")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed, deterministic=True)

    # logger
    log_path = os.path.join(args.save_dir, "train.log")

    def log(msg: str):
        line = f"[{_now_str()}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log("=" * 80)
    log("Run started.")
    log(f"save_dir = {os.path.abspath(args.save_dir)}")
    log(f"cmdline  = {' '.join(sys.argv)}")
    log(f"python   = {sys.version.replace(os.linesep, ' ')}")
    log(f"platform = {platform.platform()}")
    log(f"cwd      = {os.getcwd()}")

    git_hash = _safe_run(["bash", "-lc", "git rev-parse HEAD"])
    git_stat = _safe_run(["bash", "-lc", "git status --porcelain | head"])
    log(f"git_hash = {git_hash}")
    log(f"git_dirty(head) = {git_stat}")

    log(f"torch    = {torch.__version__}")
    log(f"cuda_avail = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"cuda_device_count = {torch.cuda.device_count()}")
        log(f"cuda_current = {torch.cuda.current_device()}")
        log(f"cuda_name    = {torch.cuda.get_device_name(torch.cuda.current_device())}")

    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    log("Saved config.json")

    device = torch.device(args.device if (torch.cuda.is_available() and args.device.startswith("cuda")) else "cpu")
    log(f"device = {device}")

    # build SAM2 (frozen)
    sam2_model = build_sam2(args.sam2_cfg, args.sam2_ckpt, device=device)
    sam2_pred = SAM2ImagePredictor(sam2_model)
    sam2_pred.model.eval()
    for p_ in sam2_pred.model.parameters():
        p_.requires_grad = False
    log(f"SAM2 cfg  = {args.sam2_cfg}")
    log(f"SAM2 ckpt = {args.sam2_ckpt}")

    # datasets
    train_full = TeacherNPZDataset(args.train_npz_dir, data_root=args.data_root)
    if args.val_npz_dir and os.path.isdir(args.val_npz_dir):
        train_ds = train_full
        val_ds = TeacherNPZDataset(args.val_npz_dir, data_root=args.data_root)
        log("Dataset split: explicit val_npz_dir used.")
        log(f"train_npz_dir = {args.train_npz_dir}  (#files={len(train_full)})")
        log(f"val_npz_dir   = {args.val_npz_dir}    (#files={len(val_ds)})")
    else:
        tr_idx, va_idx = make_split_indices(len(train_full), args.val_ratio, args.seed)
        train_ds = torch.utils.data.Subset(train_full, tr_idx)
        val_ds = torch.utils.data.Subset(train_full, va_idx)
        log(f"Dataset split: random val_ratio={args.val_ratio} seed={args.seed}")
        log(f"train size = {len(train_ds)} | val size = {len(val_ds)} | total = {len(train_full)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )

    # infer dims
    first = next(iter(val_loader))
    with torch.no_grad():
        vis, img_embed = extract_point_features_sam2(
            sam2_pred, first["images_np"], first["points"].to(device), first["orig_hw"], device, return_embed=True
        )
        point_dim = vis.shape[-1]
        if args.use_prompt_embed:
            ones = torch.ones_like(first["y"]).to(device)
            zeros = torch.zeros_like(first["y"]).to(device)
            pe_pos = extract_point_prompt_embeddings_sam2(sam2_pred, first["points"].to(device), ones, first["orig_hw"], device)
            pe_neg = extract_point_prompt_embeddings_sam2(sam2_pred, first["points"].to(device), zeros, first["orig_hw"], device)
            point_dim = point_dim + pe_pos.shape[-1] + pe_neg.shape[-1]
        if args.use_base_mask:
            point_dim = point_dim + 1  # mask prob at point
        img_in_dim = img_embed.shape[1]

        box_sparse_dim = 1
        box_dense_dim = 1
        if args.use_box_prompt:
            bs, bd = extract_box_prompt_embeddings_sam2(sam2_pred, first["boxes"].to(device), first["orig_hw"], device)
            box_sparse_dim = bs.shape[-1]
            box_dense_dim = bd.shape[1]

    log(f"point_in_dim = {point_dim} (prompt={args.use_prompt_embed} base_mask={args.use_base_mask})")
    log(f"img_in_dim   = {img_in_dim} (SAM2 image embedding channels)")
    if args.use_box_prompt:
        log(f"box_sparse_dim = {box_sparse_dim} | box_dense_dim = {box_dense_dim}")
    else:
        log("box prompt disabled")

    # build head
    if args.head_type == "mlp":
        head: nn.Module = UsefulValueMLPHead(point_in_dim=point_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
        log(f"head_type=mlp | hidden_dim={args.hidden_dim} dropout={args.dropout}")
    else:
        head = UsefulValueDecoderHead(
            point_in_dim=point_dim,
            img_in_dim=img_in_dim,
            box_sparse_dim=box_sparse_dim,
            box_dense_dim=box_dense_dim,
            d_model=args.decoder_dim,
            nhead=args.decoder_nhead,
            num_layers=args.decoder_layers,
            ffn_mult=args.decoder_ffn_mult,
            dropout=args.dropout,
            mem_downsample=args.decoder_mem_downsample,
            use_mem_pos=args.decoder_use_mem_pos,
        ).to(device)
        log(
            f"head_type=decoder | decoder_dim={args.decoder_dim} layers={args.decoder_layers} "
            f"nhead={args.decoder_nhead} ffn_mult={args.decoder_ffn_mult} mem_downsample={args.decoder_mem_downsample} "
            f"use_mem_pos={args.decoder_use_mem_pos}"
        )

    # AdamW with no-decay for bias/norm (usually improves generalization)
    decay, no_decay = [], []
    for n, p in head.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.wd},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # LR scheduler
    total_steps = args.epochs * max(1, len(train_loader))
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=args.lr_min)
        sched_per_step = True
    elif args.lr_scheduler == "cosine_warmup":
        warm = max(0, int(args.warmup_steps))

        def lr_lambda(step: int):
            if warm > 0 and step < warm:
                return float(step + 1) / float(warm)
            t = step - warm
            T = max(1, total_steps - warm)
            return 0.5 * (1.0 + math.cos(math.pi * float(t) / float(T)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        sched_per_step = True
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.step_size), gamma=float(args.step_gamma))
        sched_per_step = False
    else:
        scheduler = None
        sched_per_step = False

    # EMA
    ema = EMA(head, decay=args.ema_decay) if args.use_ema else None

    log(f"targets: v=max(G_pos,G_neg), u=1[v>{args.useful_thresh}]")
    log(f"loss weights: w_cls={args.w_cls} w_val={args.w_val} w_rank={args.w_rank} T={args.rank_T}")
    log("Metrics columns: epoch step loss topk_pred topk_teacher topk_rand acc auc u_rate u_prob (and loss terms)")

    best_score = None
    best_path = None

    for epoch in range(1, args.epochs + 1):
        head.train()
        t0 = time.time()
        running = 0.0

        for step, batch in enumerate(train_loader, start=1):
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                loss, m = forward_and_loss(
                    sam2_pred, head, batch, device, args.k_top, args.useful_thresh,
                    w_cls=args.w_cls, w_val=args.w_val, w_rank=args.w_rank, T=args.rank_T,
                    val_transform=args.val_transform,
                    val_eps=args.val_eps,
                    val_clip=args.val_clip,
                    use_prompt_embed=args.use_prompt_embed,
                    use_box_prompt=args.use_box_prompt,
                    use_base_mask=args.use_base_mask,
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if ema is not None:
                ema.update(head)
            if scheduler is not None and sched_per_step:
                scheduler.step()

            running += loss.item()
            if step % args.log_every == 0:
                avg = running / args.log_every
                running = 0.0
                log(
                    f"[Train] E{epoch} S{step} "
                    f"loss={avg:.4f} "
                    f"topk_pred={m['topk_pred']:.4f} "
                    f"topk_teacher={m['topk_teacher']:.4f} "
                    f"topk_rand={m['topk_rand']:.4f} "
                    f"acc={m['acc']:.3f} "
                    f"auc={m['auc']:.3f} "
                    f"u_rate={m['u_rate']:.3f} "
                    f"u_prob={m['u_prob']:.3f} "
                    f"lr={opt.param_groups[0]['lr']:.2e} "
                    f"| lcls={m['loss_cls']:.3f} lval={m['loss_val']:.3f} lrank={m['loss_rank']:.3f}"
                )

        if scheduler is not None and (not sched_per_step):
            scheduler.step()

        if ema is not None and args.ema_eval:
            ema.apply_to(head)
            val_res = evaluate(
                sam2_pred, head, val_loader, device, args.k_top, args.useful_thresh,
                w_cls=args.w_cls, w_val=args.w_val, w_rank=args.w_rank, T=args.rank_T,
                val_transform=args.val_transform, val_eps=args.val_eps, val_clip=args.val_clip,
                use_prompt_embed=args.use_prompt_embed,
                use_box_prompt=args.use_box_prompt,
                use_base_mask=args.use_base_mask,
            )
            ema.restore(head)
        else:
            val_res = evaluate(
                sam2_pred, head, val_loader, device, args.k_top, args.useful_thresh,
                w_cls=args.w_cls, w_val=args.w_val, w_rank=args.w_rank, T=args.rank_T,
                val_transform=args.val_transform, val_eps=args.val_eps, val_clip=args.val_clip,
                use_prompt_embed=args.use_prompt_embed,
                use_box_prompt=args.use_box_prompt,
                use_base_mask=args.use_base_mask,
            )
        dt = time.time() - t0
        log(
            f"[Val]   E{epoch} time_min={dt/60:.2f} "
            f"val_loss={val_res.loss:.4f} "
            f"topk_pred={val_res.topk_pred:.4f} "
            f"topk_teacher={val_res.topk_teacher:.4f} "
            f"topk_rand={val_res.topk_rand:.4f} "
            f"acc={val_res.acc:.3f} "
            f"auc={val_res.auc:.3f} "
            f"u_rate={val_res.u_rate:.3f} "
            f"u_prob={val_res.u_prob:.3f}"
        )

        score = val_res.topk_pred
        if best_score is None or (not math.isnan(score) and score > best_score):
            best_score = score
            best_path = os.path.join(args.save_dir, f"best_epoch{epoch}.pt")
            save_obj = {"epoch": epoch, "head": head.state_dict(), "best_topk_pred": best_score}
            if ema is not None:
                save_obj["head_ema"] = ema.state_dict()
            if scheduler is not None:
                save_obj["scheduler"] = scheduler.state_dict()
            torch.save(save_obj, best_path)
            log(f"Saved best: {best_path} (best_topk_pred={best_score:.6f})")

        last_path = os.path.join(args.save_dir, "last.pt")
        save_obj = {"epoch": epoch, "head": head.state_dict()}
        if ema is not None:
            save_obj["head_ema"] = ema.state_dict()
        if scheduler is not None:
            save_obj["scheduler"] = scheduler.state_dict()
        torch.save(save_obj, last_path)
        log(f"Saved last: {last_path}")

    log(f"Training finished. Best: {best_path}")
    log("Run ended.")
    log("=" * 80)


if __name__ == "__main__":
    main()
