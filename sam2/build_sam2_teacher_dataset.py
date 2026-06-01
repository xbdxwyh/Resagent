#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build SAM2 disagreement/entropy teacher dataset for learned EBD (RefCOCO family).

Outputs one .npz per ref_id, containing:
- img_path, ref_id, image_id
- box (xyxy, after perturbation)
- points (N,2) in (x,y)
- y (N,) point membership from GT mask (for pi-head only)
- U0: entropy for box-only prompt
- U_pos/U_neg: entropy for box + (p, fg/bg)
- G_pos/G_neg: entropy reduction (U0 - U_pos/U_neg)
- g_exp: expected gain using y as fg prob proxy (y*G_pos + (1-y)*G_neg)

Reproducibility:
- global seed controls per-ref RNG via seed + ref_id (stable regardless of order)
- bbox perturbation is deterministic given (seed, ref_id, perturb params)
"""

import os
import sys
import json
import argparse
from typing import Tuple

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch

# ---- Adjust python path so that `refer.py` is importable ----
# If this script is in e.g. /amax/wangyh/GRES/scripts/, project_root might be ../../
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from refer import REFER  # noqa: E402

from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402


# -----------------------------
# Basic utils
# -----------------------------
def coco_xywh_to_xyxy(box_xywh):
    """COCO xywh -> xyxy"""
    x, y, w, h = box_xywh
    return [x, y, x + w, y + h]


def clamp_box_xyxy(box, W, H):
    x1, y1, x2, y2 = box
    x1 = float(np.clip(x1, 0, W - 1))
    y1 = float(np.clip(y1, 0, H - 1))
    x2 = float(np.clip(x2, 0, W - 1))
    y2 = float(np.clip(y2, 0, H - 1))
    # ensure valid
    if x2 <= x1:
        x2 = min(W - 1.0, x1 + 1.0)
    if y2 <= y1:
        y2 = min(H - 1.0, y1 + 1.0)
    return [x1, y1, x2, y2]


def perturb_box_all_directions(
    box_xyxy,
    W: int,
    H: int,
    rng: np.random.Generator,
    min_ratio: float = 0.05,
    max_ratio: float = 0.15,
):
    """
    Deterministic (via rng) bbox perturbation in all directions.
    Each side expands or shrinks by a random amount in [min_ratio, max_ratio] * side_length,
    with independent directions.
    """
    x1, y1, x2, y2 = map(float, box_xyxy)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    # sample signed deltas (expand/shrink)
    # positive delta -> expand outward; negative -> shrink inward
    def signed_delta(length):
        mag = rng.uniform(min_ratio, max_ratio) * length
        sign = rng.choice([-1.0, 1.0])
        return sign * mag

    dl = signed_delta(w)  # left
    dr = signed_delta(w)  # right
    dt = signed_delta(h)  # top
    db = signed_delta(h)  # bottom

    x1_new = x1 - dl
    x2_new = x2 + dr
    y1_new = y1 - dt
    y2_new = y2 + db

    box_new = clamp_box_xyxy([x1_new, y1_new, x2_new, y2_new], W, H)
    return box_new


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def entropy_from_scores(scores: np.ndarray) -> float:
    """
    Mask-level entropy from SAM2 multimask scores.
    scores: (M,)
    """
    scores = np.asarray(scores).reshape(-1)
    if scores.size <= 1:
        return 0.0
    q = softmax_np(scores)
    return float(-(q * np.log(q + 1e-12)).sum())


def sample_points_in_box_xyxy(
    box_xyxy,
    N: int,
    W: int,
    H: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniform integer sampling inside box; returns float32 (x,y)."""
    x1, y1, x2, y2 = box_xyxy
    x1i = int(max(0, min(W - 1, np.floor(x1))))
    x2i = int(max(0, min(W - 1, np.floor(x2))))
    y1i = int(max(0, min(H - 1, np.floor(y1))))
    y2i = int(max(0, min(H - 1, np.floor(y2))))
    if x2i <= x1i:
        x2i = min(W - 1, x1i + 1)
    if y2i <= y1i:
        y2i = min(H - 1, y1i + 1)

    xs = rng.integers(x1i, x2i + 1, size=(N,))
    ys = rng.integers(y1i, y2i + 1, size=(N,))
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    return pts


def find_image_path(data_root: str, file_name: str) -> str:
    """
    RefCOCO family commonly uses COCO train2014 images, but add a fallback to val2014.
    """
    p1 = os.path.join(data_root, "images", "train2014", file_name)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(data_root, "images", "val2014", file_name)
    if os.path.exists(p2):
        return p2
    return p1  # default (will error later if truly missing)


def run_sam_entropy(
    predictor: SAM2ImagePredictor,
    box_xyxy,
    point_xy=None,
    point_label=None,
    multimask_output=True,
) -> float:
    """
    Calls SAM2 predictor and returns entropy over multimask scores.
    """
    box_in = np.array(box_xyxy, dtype=np.float32)[None, :]  # (1,4)

    if point_xy is None:
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_in,
            multimask_output=multimask_output,
        )
    else:
        pc = np.array(point_xy, dtype=np.float32)[None, :]  # (1,2)
        pl = np.array([int(point_label)], dtype=np.int32)   # (1,)
        masks, scores, logits = predictor.predict(
            point_coords=pc,
            point_labels=pl,
            box=box_in,
            multimask_output=multimask_output,
        )

    # scores: (M,) if multimask_output=True
    return entropy_from_scores(np.asarray(scores))


# -----------------------------
# Main
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser("Build SAM2 teacher dataset for learned EBD.")
    parser.add_argument("--task", type=str, default="refcoco",
                        choices=["refcoco", "refcoco+", "refcocog"])
    parser.add_argument("--split", type=str, default="train",
                        help="train/val/testA/testB (depends on dataset)")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Global seed for reproducibility.")
    parser.add_argument("--num_points", type=int, default=64,
                        help="Number of candidate points sampled inside box.")
    parser.add_argument("--bbox_min_ratio", type=float, default=0.05,
                        help="Min ratio for bbox perturbation magnitude.")
    parser.add_argument("--bbox_max_ratio", type=float, default=0.15,
                        help="Max ratio for bbox perturbation magnitude.")
    parser.add_argument("--data_root", type=str, default="../../../DETRIS-main/datasets/",
                        help="Root folder containing RefCOCO data + COCO images.")
    parser.add_argument("--out_dir", type=str, default="./sam2_teacher_npz",
                        help="Output directory to save .npz files.")
    parser.add_argument("--sam2_variant", type=str, default="base_plus",
                        choices=["base_plus", "large"],
                        help="SAM2.1 checkpoint variant.")
    parser.add_argument("--sam2_ckpt_base_plus", type=str,
                        default="../checkpoints/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--sam2_ckpt_large", type=str,
                        default="../checkpoints/sam2.1_hiera_large.pt")
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- seeds (best-effort reproducibility) ----
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- device ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Info] using device: {device}")

    # ---- build SAM2 ----
    if args.sam2_variant == "base_plus":
        sam2_checkpoint = args.sam2_ckpt_base_plus
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    else:
        sam2_checkpoint = args.sam2_ckpt_large
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    print(f"[Info] SAM2 cfg: {model_cfg}")
    print(f"[Info] SAM2 ckpt: {sam2_checkpoint}")
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    # ---- build REFER ----
    if args.task in ["refcoco", "refcoco+"]:
        source = "unc"
    else:
        source = "umd"

    refer = REFER(args.data_root, args.task, source)
    ref_ids = refer.getRefIds(split=args.split)
    print(f"[Info] task={args.task}, split={args.split}, #refs={len(ref_ids)}")

    # ---- out dir (include seed + perturb params for traceability) ----
    tag = f"{args.task}_{args.split}_seed{args.seed}_N{args.num_points}_pert{args.bbox_min_ratio}-{args.bbox_max_ratio}_{args.sam2_variant}"
    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Info] saving to: {out_dir}")

    # ---- main loop ----
    for ref_id in tqdm(ref_ids):
        refs = refer.Refs[ref_id]
        img_meta = refer.loadImgs(image_ids=refs["image_id"])[0]
        file_name = img_meta["file_name"]
        img_path = find_image_path(args.data_root, file_name)

        # load image + gt mask
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        mask = refer.getMask(refs)["mask"].astype(np.uint8)  # [H,W], 0/1

        # gold box -> xyxy -> deterministic perturb
        box_xywh = refer.getRefBox(ref_id)
        box_xyxy = coco_xywh_to_xyxy(box_xywh)

        # per-ref rng ensures deterministic even if iteration order changes
        rng = np.random.default_rng(args.seed + int(ref_id) * 1000003)

        box_xyxy = perturb_box_all_directions(
            box_xyxy,
            W, H,
            rng=rng,
            min_ratio=args.bbox_min_ratio,
            max_ratio=args.bbox_max_ratio,
        )

        # sample candidate points inside box
        pts = sample_points_in_box_xyxy(box_xyxy, args.num_points, W, H, rng=rng)  # (N,2)

        # point membership y (only for pi-head training later)
        ys = np.zeros((args.num_points,), dtype=np.int64)
        for i, (x, y) in enumerate(pts.astype(int)):
            ys[i] = 1 if mask[y, x] > 0 else 0

        # run SAM2 entropies
        predictor.set_image(np.array(img))

        # baseline: box-only
        U0 = run_sam_entropy(predictor, box_xyxy, point_xy=None, point_label=None, multimask_output=True)

        U_pos = np.zeros((args.num_points,), dtype=np.float32)
        U_neg = np.zeros((args.num_points,), dtype=np.float32)

        for i, (x, y) in enumerate(pts):
            U_pos[i] = run_sam_entropy(predictor, box_xyxy, point_xy=(x, y), point_label=1, multimask_output=True)
            U_neg[i] = run_sam_entropy(predictor, box_xyxy, point_xy=(x, y), point_label=0, multimask_output=True)

        G_pos = (U0 - U_pos).astype(np.float32)
        G_neg = (U0 - U_neg).astype(np.float32)
        g_exp = (ys * G_pos + (1 - ys) * G_neg).astype(np.float32)

        # save
        save_file = os.path.join(out_dir, f"{ref_id}.npz")
        np.savez_compressed(
            save_file,
            ref_id=np.int64(ref_id),
            image_id=np.int64(refs["image_id"]),
            img_path=img_path,
            box=np.array(box_xyxy, dtype=np.float32),
            points=pts.astype(np.float32),
            y=ys,
            U0=np.float32(U0),
            U_pos=U_pos,
            U_neg=U_neg,
            G_pos=G_pos,
            G_neg=G_neg,
            g_exp=g_exp,
        )

    print("[Done] Teacher dataset saved.")


if __name__ == "__main__":
    main()
