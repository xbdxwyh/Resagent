#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build SAM2 ΔIoU teacher dataset for learned EBD (RefCOCO family),
using *spiral* candidate points (internal/external) instead of random points.

Teacher definition (A, realistic):
- multimask_output=False (single mask)
- baseline mask M0 from box-only (with perturbed box)
- for each point p:
    M_pos(p): SAM2 with (box + fg click at p)
    M_neg(p): SAM2 with (box + bg click at p)
    IoU0 = IoU(M0, GT)
    IoU_pos(p) = IoU(M_pos(p), GT)
    IoU_neg(p) = IoU(M_neg(p), GT)
    G_pos(p) = IoU_pos(p) - IoU0
    G_neg(p) = IoU_neg(p) - IoU0
    g_exp(p) = y*G_pos(p) + (1-y)*G_neg(p)  (y from GT membership)

Outputs one .npz per ref_id, containing:
- img_path, ref_id, image_id
- box (xyxy, after perturbation)
- points (N,2) in (x,y)
- y (N,) point membership from GT mask (for pi-head)
- IoU0: scalar
- IoU_pos/IoU_neg: (N,)
- G_pos/G_neg: (N,) ΔIoU gains
- g_exp: (N,) expected gain

Reproducibility:
- per-ref seed = seed + ref_id * 1000003
- bbox perturbation deterministic under per-ref rng
- spiral point generation is made deterministic by setting np.random.seed & random.seed per ref

CUDA_VISIBLE_DEVICES=0 python  build_sam2_teacher_dataset_iou_spiral.py --task refcoco --split train   --seed 2026 --num_points 64 --num_internal 32 --num_external 32   --bbox_min_ratio 0.05 --bbox_max_ratio 0.15   --sam2_variant base_plus
"""

import os
import sys
import argparse
import random
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch

# ---- Adjust python path so that `refer.py` and `tools.py` are importable ----
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from refer import REFER  # noqa: E402
from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402

# your spiral samplers (as you used before)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
try:
    from tools import generate_internal_candidate_points, generate_external_candidate_points  # noqa: E402
except Exception as e:
    raise RuntimeError(
        "Failed to import spiral point generators from tools.py.\n"
        "Please make sure tools.py is under project_root and contains:\n"
        "  generate_internal_candidate_points, generate_external_candidate_points\n"
        f"Original error: {e}"
    )

EPS = 1e-6


def coco_xywh_to_xyxy(box_xywh):
    x, y, w, h = box_xywh
    return [x, y, x + w, y + h]


def clamp_box_xyxy(box, W, H):
    x1, y1, x2, y2 = box
    x1 = float(np.clip(x1, 0, W - 1))
    y1 = float(np.clip(y1, 0, H - 1))
    x2 = float(np.clip(x2, 0, W - 1))
    y2 = float(np.clip(y2, 0, H - 1))
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
    x1, y1, x2, y2 = map(float, box_xyxy)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    def signed_delta(length):
        mag = rng.uniform(min_ratio, max_ratio) * length
        sign = rng.choice([-1.0, 1.0])
        return sign * mag

    dl = signed_delta(w)
    dr = signed_delta(w)
    dt = signed_delta(h)
    db = signed_delta(h)

    x1_new = x1 - dl
    x2_new = x2 + dr
    y1_new = y1 - dt
    y2_new = y2 + db
    return clamp_box_xyxy([x1_new, y1_new, x2_new, y2_new], W, H)


def find_image_path(data_root: str, file_name: str) -> str:
    p1 = os.path.join(data_root, "images", "train2014", file_name)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(data_root, "images", "val2014", file_name)
    if os.path.exists(p2):
        return p2
    return p1


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / float(union + EPS)


def run_sam_single_mask(
    predictor: SAM2ImagePredictor,
    box_xyxy,
    point_xy=None,
    point_label=None,
    mask_threshold: float = 0.5,
) -> np.ndarray:
    box_in = np.array(box_xyxy, dtype=np.float32)[None, :]  # (1,4)

    if point_xy is None:
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_in,
            multimask_output=False,
        )
    else:
        pc = np.array(point_xy, dtype=np.float32)[None, :]  # (1,2)
        pl = np.array([int(point_label)], dtype=np.int32)
        masks, scores, logits = predictor.predict(
            point_coords=pc,
            point_labels=pl,
            box=box_in,
            multimask_output=False,
        )

    if masks.ndim == 3:
        m = masks[0]
    else:
        m = masks
    return (m > mask_threshold).astype(np.uint8)


def sample_points_uniform_in_box(box_xyxy, N: int, W: int, H: int, rng: np.random.Generator) -> List[Tuple[float, float]]:
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
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def build_spiral_candidates(
    box_xyxy,
    gt_mask: np.ndarray,
    num_internal: int,
    num_external: int,
    oversample_factor: int,
    per_ref_seed: int,
) -> List[Tuple[float, float]]:
    """
    Use your spiral sampler to generate internal/external candidates, then
    *re-assign* them by GT membership (like your original pipeline) to ensure
    internal candidates are truly inside and external candidates truly outside.

    If insufficient points after filtering, fallback with uniform samples to fill.
    """
    # Make spiral generation deterministic per ref_id
    random.seed(per_ref_seed)
    np.random.seed(per_ref_seed)

    # oversample first, then filter + truncate
    n_int_raw = max(num_internal * oversample_factor, num_internal)
    n_ext_raw = max(num_external * oversample_factor, num_external)

    pts_int_raw = generate_internal_candidate_points(box_xyxy, n_int_raw)
    pts_ext_raw = generate_external_candidate_points(box_xyxy, n_ext_raw)

    internal_candidate_points = []
    external_candidate_points = []
    int_appendix_points = []
    ext_appendix_points = []

    # external sampler may include inside points -> move to int_appendix
    for (x, y) in pts_ext_raw:
        xi, yi = int(x), int(y)
        inside = (gt_mask[yi, xi] > 0)
        if not inside:
            external_candidate_points.append((float(x), float(y)))
        else:
            int_appendix_points.append((float(x), float(y)))

    # internal sampler may include outside points -> move to ext_appendix
    for (x, y) in pts_int_raw:
        xi, yi = int(x), int(y)
        inside = (gt_mask[yi, xi] > 0)
        if inside:
            internal_candidate_points.append((float(x), float(y)))
        else:
            ext_appendix_points.append((float(x), float(y)))

    internal_candidate_points += int_appendix_points
    external_candidate_points += ext_appendix_points

    internal_candidate_points = internal_candidate_points[:num_internal]
    external_candidate_points = external_candidate_points[:num_external]

    # fill if not enough
    # NOTE: we intentionally don't require the filled points to be strictly in/out;
    # they are just candidates to keep N stable.
    need = (num_internal - len(internal_candidate_points)) + (num_external - len(external_candidate_points))
    if need > 0:
        # caller will provide rng if needed; here just use a deterministic rng
        rng = np.random.default_rng(per_ref_seed + 99991)
        # fill inside-box uniformly
        extra = sample_points_uniform_in_box(box_xyxy, need, gt_mask.shape[1], gt_mask.shape[0], rng)
        # distribute
        for p in extra:
            if len(internal_candidate_points) < num_internal:
                internal_candidate_points.append(p)
            else:
                external_candidate_points.append(p)

    pts = internal_candidate_points + external_candidate_points
    # ensure exact length
    return pts[: (num_internal + num_external)]


def parse_args():
    parser = argparse.ArgumentParser("Build SAM2 ΔIoU teacher dataset (spiral candidates).")
    parser.add_argument("--task", type=str, default="refcoco",
                        choices=["refcoco", "refcoco+", "refcocog"])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--num_points", type=int, default=64,
                        help="Total candidate points per sample (internal+external).")
    parser.add_argument("--num_internal", type=int, default=-1,
                        help="Internal candidates count. default = num_points//2")
    parser.add_argument("--num_external", type=int, default=-1,
                        help="External candidates count. default = num_points - num_internal")
    parser.add_argument("--oversample_factor", type=int, default=3,
                        help="Oversample spiral points then filter by GT membership.")

    parser.add_argument("--bbox_min_ratio", type=float, default=0.05)
    parser.add_argument("--bbox_max_ratio", type=float, default=0.15)

    parser.add_argument("--data_root", type=str, default="../../../DETRIS-main/datasets/")
    parser.add_argument("--out_dir", type=str, default="./sam2_teacher_npz_iou_spiral")

    parser.add_argument("--sam2_variant", type=str, default="base_plus",
                        choices=["base_plus", "large"])
    parser.add_argument("--sam2_ckpt_base_plus", type=str,
                        default="../checkpoints/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--sam2_ckpt_large", type=str,
                        default="../checkpoints/sam2.1_hiera_large.pt")

    parser.add_argument("--mask_threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()

    # derive internal/external counts
    if args.num_internal < 0:
        args.num_internal = args.num_points // 2
    if args.num_external < 0:
        args.num_external = args.num_points - args.num_internal
    assert args.num_internal + args.num_external == args.num_points

    # seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # device
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

    # build SAM2
    if args.sam2_variant == "base_plus":
        sam2_checkpoint = args.sam2_ckpt_base_plus
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    else:
        sam2_checkpoint = args.sam2_ckpt_large
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    print(f"[Info] SAM2 cfg:  {model_cfg}")
    print(f"[Info] SAM2 ckpt: {sam2_checkpoint}")
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    # build REFER
    source = "unc" if args.task in ["refcoco", "refcoco+"] else "umd"
    refer = REFER(args.data_root, args.task, source)
    ref_ids = refer.getRefIds(split=args.split)
    print(f"[Info] task={args.task}, split={args.split}, #refs={len(ref_ids)}")

    # out dir
    tag = (
        f"{args.task}_{args.split}_seed{args.seed}_N{args.num_points}"
        f"_int{args.num_internal}_ext{args.num_external}"
        f"_pert{args.bbox_min_ratio}-{args.bbox_max_ratio}"
        f"_{args.sam2_variant}_dIoU_spiral"
    )
    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Info] saving to: {out_dir}")

    for ref_id in tqdm(ref_ids):
        refs = refer.Refs[ref_id]
        img_meta = refer.loadImgs(image_ids=refs["image_id"])[0]
        file_name = img_meta["file_name"]
        img_path = find_image_path(args.data_root, file_name)

        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        gt_mask = refer.getMask(refs)["mask"].astype(np.uint8)

        # gold box -> xyxy -> deterministic perturb
        box_xywh = refer.getRefBox(ref_id)
        box_xyxy = coco_xywh_to_xyxy(box_xywh)

        per_ref_seed = int(args.seed + int(ref_id) * 1003)
        rng = np.random.default_rng(per_ref_seed)
        box_xyxy = perturb_box_all_directions(
            box_xyxy, W, H, rng=rng,
            min_ratio=args.bbox_min_ratio,
            max_ratio=args.bbox_max_ratio,
        )

        # spiral candidates (deterministic per ref_id)
        pts_list = build_spiral_candidates(
            box_xyxy=box_xyxy,
            gt_mask=gt_mask,
            num_internal=args.num_internal,
            num_external=args.num_external,
            oversample_factor=args.oversample_factor,
            per_ref_seed=per_ref_seed,
        )
        pts = np.array(pts_list, dtype=np.float32)  # (N,2)

        # y from GT membership
        ys = np.zeros((args.num_points,), dtype=np.int64)
        for i, (x, y) in enumerate(pts.astype(int)):
            # safe clamp
            x = int(np.clip(x, 0, W - 1))
            y = int(np.clip(y, 0, H - 1))
            ys[i] = 1 if gt_mask[y, x] > 0 else 0

        # run SAM2
        predictor.set_image(np.array(img))

        # baseline box-only
        m0 = run_sam_single_mask(
            predictor, box_xyxy,
            point_xy=None, point_label=None,
            mask_threshold=args.mask_threshold
        )
        IoU0 = compute_iou(m0, gt_mask)

        IoU_pos = np.zeros((args.num_points,), dtype=np.float32)
        IoU_neg = np.zeros((args.num_points,), dtype=np.float32)

        for i, (x, y) in enumerate(pts):
            m_pos = run_sam_single_mask(
                predictor, box_xyxy,
                point_xy=(float(x), float(y)), point_label=1,
                mask_threshold=args.mask_threshold
            )
            m_neg = run_sam_single_mask(
                predictor, box_xyxy,
                point_xy=(float(x), float(y)), point_label=0,
                mask_threshold=args.mask_threshold
            )
            IoU_pos[i] = compute_iou(m_pos, gt_mask)
            IoU_neg[i] = compute_iou(m_neg, gt_mask)

        G_pos = (IoU_pos - IoU0).astype(np.float32)
        G_neg = (IoU_neg - IoU0).astype(np.float32)
        g_exp = (ys * G_pos + (1 - ys) * G_neg).astype(np.float32)

        save_file = os.path.join(out_dir, f"{ref_id}.npz")
        np.savez_compressed(
            save_file,
            ref_id=np.int64(ref_id),
            image_id=np.int64(refs["image_id"]),
            img_path=img_path,
            box=np.array(box_xyxy, dtype=np.float32),
            points=pts.astype(np.float32),
            y=ys,
            IoU0=np.float32(IoU0),
            IoU_pos=IoU_pos,
            IoU_neg=IoU_neg,
            G_pos=G_pos,
            G_neg=G_neg,
            g_exp=g_exp,
            sampler=np.array("spiral", dtype=object),
            num_internal=np.int64(args.num_internal),
            num_external=np.int64(args.num_external),
            oversample_factor=np.int64(args.oversample_factor),
        )

    print("[Done] ΔIoU teacher dataset saved.")
    print(f"[Done] out_dir = {out_dir}")


if __name__ == "__main__":
    main()
