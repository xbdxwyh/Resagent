#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merged multi-expert router for RES box priors.

What it does:
1) Keeps the traditional IoU-clustering / fusion baselines:
   - oracle_best
   - all_mean
   - trimmed_mean
   - medoid
   - vote_mean
2) Adds CLIP-based routing/fusion:
   - clip_top1
   - clip_wmean
   - clip_filtered_mean
   - clip_weighted_vote_mean
3) Keeps VLM overlay routing in the same file:
   - vlm_router
4) Auto-discovers prediction jsonl files from the current folder (or input-dir)
   according to dataset / splitBy / split, instead of hard-coding all paths.
5) Preserves the old per-sample routed output style via routed_* fields, while also
   saving extra debugging info for all candidate methods.

Typical examples:
python merged_router_clip_vlm.py \
  --data-root ../../DETRIS-main/datasets \
  --dataset refcoco --split-by unc --split testA \
  --router-methods oracle_best,all_mean,trimmed_mean,medoid,vote_mean,clip_top1,clip_wmean,clip_filtered_mean,clip_weighted_vote_mean \
  --primary-router clip_weighted_vote_mean

python merged_router_clip_vlm.py \
  --data-root ../../DETRIS-main/datasets \
  --dataset refcoco+ --split-by unc --split testA \
  --router-methods oracle_best,all_mean,trimmed_mean,medoid,vote_mean,vlm_router \
  --primary-router vlm_router \
  --vlm-model-path ../../../pretrained/Qwen3-VL-8B-Instruct/
"""

import os
import re
import json
import math
import glob
import argparse
import random
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import torch

try:
    from transformers import CLIPModel, CLIPProcessor
    HAVE_CLIP = True
except Exception:
    HAVE_CLIP = False

try:
    from transformers import AutoProcessor, AutoConfig
    try:
        from transformers import AutoModelForVision2Seq
        HAVE_V2S = True
    except Exception:
        HAVE_V2S = False
    from transformers import AutoModelForCausalLM
    HAVE_VLM = True
except Exception:
    HAVE_VLM = False
    HAVE_V2S = False

from refer import REFER
try:
    from grefer import G_REFER
    HAVE_GREFER = True
except Exception:
    HAVE_GREFER = False


EPS = 1e-9
BOX_RE = re.compile(r"\[\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]\]")
LOOSE_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

COLOR_POOL = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 128, 255),
    (255, 165, 0),
    (255, 0, 255),
    (255, 255, 0),
]
COLOR_NAME_MAP = {
    (255, 0, 0): "red",
    (0, 255, 0): "green",
    (0, 128, 255): "blue",
    (255, 165, 0): "orange",
    (255, 0, 255): "magenta",
    (255, 255, 0): "yellow",
}
LABEL_POOL = list("ABCDEF")


# =========================================================
# Basic utils
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_box_from_raw(raw_text: str) -> Optional[List[float]]:
    if not raw_text:
        return None
    m = BOX_RE.search(raw_text)
    if m:
        return [float(m.group(i)) for i in range(1, 5)]
    nums = LOOSE_NUM_RE.findall(raw_text)
    if len(nums) >= 4:
        return [float(x) for x in nums[:4]]
    return None


def clamp_xyxy(b: List[float], W: int, H: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in b]
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1 = max(0.0, min(float(W - 1), x1))
    x2 = max(0.0, min(float(W - 1), x2))
    y1 = max(0.0, min(float(H - 1), y1))
    y2 = max(0.0, min(float(H - 1), y2))
    return [x1, y1, x2, y2]


def decode_to_pixel_xyxy(box_raw: List[float], W: int, H: int, coord_mode: str) -> Optional[List[float]]:
    if box_raw is None:
        return None
    b = [float(x) for x in box_raw]
    mx = max(b)

    if coord_mode == "pixel":
        return clamp_xyxy(b, W, H)

    if coord_mode == "force_1000":
        x1, y1, x2, y2 = b
        return clamp_xyxy([x1 / 1000.0 * W, y1 / 1000.0 * H, x2 / 1000.0 * W, y2 / 1000.0 * H], W, H)

    if coord_mode == "auto":
        if mx <= 1.5:
            x1, y1, x2, y2 = b
            return clamp_xyxy([x1 * W, y1 * H, x2 * W, y2 * H], W, H)
        if mx <= 1100:
            x1, y1, x2, y2 = b
            return clamp_xyxy([x1 / 1000.0 * W, y1 / 1000.0 * H, x2 / 1000.0 * W, y2 / 1000.0 * H], W, H)
        return clamp_xyxy(b, W, H)

    raise ValueError(f"Unknown coord_mode={coord_mode}")


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    areaA = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    areaB = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (areaA + areaB - inter + EPS))


def xywh_to_xyxy(box_xywh: List[float]) -> List[float]:
    x, y, w, h = [float(v) for v in box_xywh]
    return [x, y, x + w, y + h]


def mask_to_box_xyxy(mask: np.ndarray) -> Optional[List[float]]:
    m = (mask > 0).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    x1, y1 = float(xs.min()), float(ys.min())
    x2, y2 = float(xs.max() + 1), float(ys.max() + 1)
    return [x1, y1, x2, y2]


def box_mean(boxes: List[List[float]], weights: Optional[List[float]] = None) -> Optional[List[float]]:
    if not boxes:
        return None
    arr = np.asarray(boxes, dtype=np.float32)
    if weights is None:
        return arr.mean(axis=0).tolist()
    w = np.asarray(weights, dtype=np.float32)
    if w.sum() <= 0:
        return arr.mean(axis=0).tolist()
    w = w / (w.sum() + EPS)
    return np.sum(arr * w[:, None], axis=0).tolist()


def medoid_box(boxes: List[List[float]]) -> Optional[List[float]]:
    if not boxes:
        return None
    if len(boxes) == 1:
        return boxes[0]
    scores = []
    for i, bi in enumerate(boxes):
        s = 0.0
        for j, bj in enumerate(boxes):
            if i == j:
                continue
            s += iou_xyxy(bi, bj)
        scores.append(s)
    return boxes[int(np.argmax(scores))]


def vote_mean_box(boxes: List[List[float]], iou_thresh: float) -> Optional[List[float]]:
    if not boxes:
        return None
    n = len(boxes)
    if n == 1:
        return boxes[0]
    ious = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            v = 1.0 if i == j else iou_xyxy(boxes[i], boxes[j])
            ious[i, j] = v
            ious[j, i] = v
    best_support = []
    best_dense = -1.0
    for i in range(n):
        support = [j for j in range(n) if ious[i, j] >= iou_thresh]
        dense = float(ious[np.ix_(support, support)].mean()) if support else -1.0
        if len(support) > len(best_support) or (len(support) == len(best_support) and dense > best_dense):
            best_support = support
            best_dense = dense
    chosen = [boxes[j] for j in best_support] if best_support else boxes
    return box_mean(chosen)


def trimmed_mean_box(boxes: List[List[float]], trim_num: int) -> Optional[List[float]]:
    if not boxes:
        return None
    n = len(boxes)
    if n <= 2 or trim_num <= 0:
        return box_mean(boxes)
    keep = max(1, n - trim_num)
    scores = []
    for i, bi in enumerate(boxes):
        vals = [iou_xyxy(bi, bj) for j, bj in enumerate(boxes) if i != j]
        scores.append(float(np.mean(vals)) if vals else 1.0)
    order = np.argsort(scores)[::-1][:keep]
    chosen = [boxes[int(i)] for i in order]
    return box_mean(chosen)


def pairwise_iou_mean(boxes: List[List[float]]) -> float:
    n = len(boxes)
    if n <= 1:
        return 1.0
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(iou_xyxy(boxes[i], boxes[j]))
    return float(np.mean(vals)) if vals else 1.0


def pad_box_xyxy(b: List[float], W: int, H: int, pad_ratio: float) -> List[float]:
    x1, y1, x2, y2 = b
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px = bw * pad_ratio
    py = bh * pad_ratio
    return clamp_xyxy([x1 - px, y1 - py, x2 + px, y2 + py], W, H)


def crop_pil(img: Image.Image, box_xyxy: List[float]) -> Image.Image:
    x1, y1, x2, y2 = box_xyxy
    left = int(math.floor(x1))
    upper = int(math.floor(y1))
    right = max(left + 1, int(math.ceil(x2)))
    lower = max(upper + 1, int(math.ceil(y2)))
    return img.crop((left, upper, right, lower))


def softmax_np(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32) / max(temp, 1e-6)
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + EPS)


# =========================================================
# Expert discovery / loading
# =========================================================
def sanitize_name(name: str) -> str:
    return name.replace(".jsonl", "")


def sanitize_filename_token(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"[^A-Za-z0-9._+\-]+", "-", x)
    x = re.sub(r"-+", "-", x).strip("-._")
    return x or "unknown"


def box_to_pred_xyxy(box: Optional[List[float]]) -> Optional[List[float]]:
    if box is None:
        return None
    return [float(round(float(v), 4)) for v in box]


def box_to_raw_text(box: Optional[List[float]]) -> Optional[str]:
    if box is None:
        return None
    vals = [int(round(float(v))) for v in box]
    return f"[[{vals[0]},{vals[1]},{vals[2]},{vals[3]}]]"


def build_single_style_record(
    *,
    args,
    img_meta: Dict[str, Any],
    ref_id: int,
    sent_id: int,
    expr: str,
    routed_box: Optional[List[float]],
    error: Optional[str],
) -> Dict[str, Any]:
    image_id = int(img_meta["id"])
    return {
        "key": f"{args.dataset}:{args.split}:{image_id}:{int(ref_id)}:{int(sent_id)}",
        "dataset": args.dataset,
        "splitBy": args.split_by,
        "split": args.split,
        "image_id": image_id,
        "ref_id": int(ref_id),
        "sent_id": int(sent_id),
        "expr": expr,
        "img_file": img_meta["file_name"],
        "w": int(img_meta["width"]),
        "h": int(img_meta["height"]),
        "model": args.primary_router,
        "pred_box_xyxy": box_to_pred_xyxy(routed_box),
        "raw_text": box_to_raw_text(routed_box),
        "error": error,
    }


def build_default_output_names(args) -> Tuple[str, str]:
    dataset = sanitize_filename_token(args.dataset)
    split_by = sanitize_filename_token(args.split_by)
    split = sanitize_filename_token(args.split)
    primary_router = sanitize_filename_token(args.primary_router)
    stem = f"{dataset}_{split_by}_{split}_{primary_router}"
    return f"{stem}_choices.jsonl", f"{stem}_summary.json"


def infer_expert_name_from_path(path: str, dataset: str, split_by: str, split: str) -> str:
    base = os.path.basename(path)
    prefix = f"pred_{dataset}_{split_by}_{split}_"
    if base.startswith(prefix):
        return sanitize_name(base[len(prefix):])
    return sanitize_name(os.path.splitext(base)[0])


def infer_coord_mode(expert_name: str, obj: Optional[Dict[str, Any]]) -> str:
    lname = expert_name.lower()
    if obj is not None:
        pred = obj.get("pred_box_xyxy", None)
        if isinstance(pred, list) and len(pred) >= 4:
            mx = max(float(x) for x in pred[:4])
            if mx <= 1.5:
                return "auto"
            if mx <= 1100:
                return "force_1000"
            return "pixel"
        raw = parse_box_from_raw(obj.get("raw_text", "") or "")
        if raw is not None:
            mx = max(float(x) for x in raw[:4])
            if mx <= 1.5:
                return "auto"
            if mx <= 1100:
                return "force_1000"
            return "pixel"
    if "gpt" in lname and "norm1000" not in lname:
        return "pixel"
    return "force_1000"


def load_pred_jsonl(path: str, source: str = "auto") -> Tuple[Dict[Tuple[int, int], Dict[str, Any]], str]:
    mp: Dict[Tuple[int, int], Dict[str, Any]] = {}
    detected_mode = "auto"
    sample_obj = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if sample_obj is None:
                sample_obj = obj
            try:
                ref_id = int(obj["ref_id"])
                sent_id = int(obj["sent_id"])
            except Exception:
                continue

            box_raw = None
            if source in ["raw", "auto"]:
                box_raw = parse_box_from_raw(obj.get("raw_text", ""))
            if box_raw is None and source in ["pred", "auto", "raw"]:
                box_raw = obj.get("pred_box_xyxy", None)
            if box_raw is None:
                continue
            try:
                box_raw = [float(x) for x in box_raw]
            except Exception:
                continue

            mp[(ref_id, sent_id)] = {
                "box_raw": box_raw,
                "image_id": obj.get("image_id", None),
                "expr": obj.get("expr", None),
                "raw_text": obj.get("raw_text", None),
                "key": obj.get("key", f"{ref_id}:{sent_id}"),
                "model": obj.get("model", None),
            }
    if sample_obj is not None:
        detected_mode = infer_coord_mode(os.path.basename(path), sample_obj)
    return mp, detected_mode


def discover_expert_files(input_dir: str, dataset: str, split_by: str, split: str) -> List[str]:
    pat = os.path.join(input_dir, f"pred_{dataset}_{split_by}_{split}_*.jsonl")
    files = sorted(glob.glob(pat))
    return files


def build_expert_configs(args) -> List[Dict[str, Any]]:
    if args.expert_file:
        out = []
        for raw in args.expert_file:
            # format: /path/to/file.jsonl or name=/path/to/file.jsonl
            if "=" in raw:
                name, path = raw.split("=", 1)
                name = name.strip()
                path = path.strip()
            else:
                path = raw.strip()
                name = infer_expert_name_from_path(path, args.dataset, args.split_by, args.split)
            out.append({"name": name, "path": path, "source": "auto", "coord_mode": "auto"})
        return out

    files = discover_expert_files(args.input_dir, args.dataset, args.split_by, args.split)
    if not files:
        raise FileNotFoundError(
            f"No prediction files found under {args.input_dir} for pattern "
            f"pred_{args.dataset}_{args.split_by}_{args.split}_*.jsonl"
        )
    out = []
    for path in files:
        out.append({
            "name": infer_expert_name_from_path(path, args.dataset, args.split_by, args.split),
            "path": path,
            "source": "auto",
            "coord_mode": "auto",
        })
    return out


# =========================================================
# IoU clustering
# =========================================================
class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def cluster_boxes(boxes: List[List[float]], names: List[str], iou_thresh: float) -> List[Dict[str, Any]]:
    n = len(boxes)
    if n == 0:
        return []
    dsu = DSU(n)
    for i in range(n):
        for j in range(i + 1, n):
            if iou_xyxy(boxes[i], boxes[j]) >= iou_thresh:
                dsu.union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[dsu.find(i)].append(i)

    clusters = []
    for gid, idxs in groups.items():
        c_boxes = [boxes[i] for i in idxs]
        c_names = [names[i] for i in idxs]
        clusters.append({
            "gid": int(gid),
            "indices": [int(i) for i in idxs],
            "member_names": c_names,
            "support": len(idxs),
            "mean_box": box_mean(c_boxes),
            "medoid_box": medoid_box(c_boxes),
            "dense": pairwise_iou_mean(c_boxes),
        })
    clusters.sort(key=lambda c: (-c["support"], -c["dense"]))
    return clusters


# =========================================================
# CLIP router
# =========================================================
class ClipRouter:
    def __init__(self, model_name: str, device: str, dtype: torch.dtype, pad_ratio: float,
                 use_ctx_diff: bool, clip_temp: float, clip_filter_margin: float):
        if not HAVE_CLIP:
            raise RuntimeError("transformers CLIP is not available in this environment.")
        self.device = device
        self.dtype = dtype
        self.pad_ratio = pad_ratio
        self.use_ctx_diff = use_ctx_diff
        self.clip_temp = clip_temp
        self.clip_filter_margin = clip_filter_margin

        print(f"[CLIP] Loading {model_name} on {device}")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()
        if device == "cuda":
            self.model = self.model.to(dtype=dtype)

    @torch.no_grad()
    def image_text_sims(self, images: List[Image.Image], texts: List[str]) -> np.ndarray:
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        for k in inputs:
            inputs[k] = inputs[k].to(self.device)
        out = self.model(**inputs)
        img = out.image_embeds
        txt = out.text_embeds
        img = img / (img.norm(dim=-1, keepdim=True) + 1e-12)
        txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-12)
        sims = (img * txt).sum(dim=-1)
        return sims.detach().float().cpu().numpy()

    def score_boxes(self, img: Image.Image, expr: str, boxes: List[List[float]]) -> List[float]:
        if not boxes:
            return []
        W, H = img.size
        crops = [crop_pil(img, pad_box_xyxy(b, W, H, self.pad_ratio)) for b in boxes]
        texts = [expr] * len(crops)
        crop_scores = self.image_text_sims(crops, texts)
        if self.use_ctx_diff:
            full_scores = self.image_text_sims([img] * len(crops), texts)
            crop_scores = crop_scores - full_scores
        return [float(x) for x in crop_scores]

    def route(self, boxes: List[List[float]], names: List[str], scores: List[float],
              iou_thresh: float) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        if not boxes:
            return out
        arr_scores = np.asarray(scores, dtype=np.float32)
        weights = softmax_np(arr_scores, temp=self.clip_temp)

        # clip_top1
        top1_idx = int(np.argmax(arr_scores))
        out["clip_top1"] = {
            "box": boxes[top1_idx],
            "picked_names": [names[top1_idx]],
            "score": float(arr_scores[top1_idx]),
        }

        # clip_wmean
        out["clip_wmean"] = {
            "box": box_mean(boxes, weights.tolist()),
            "picked_names": list(names),
            "score": float(arr_scores.max()),
        }

        # clip_filtered_mean
        keep_idx = np.where(arr_scores >= arr_scores.max() - self.clip_filter_margin)[0].tolist()
        if not keep_idx:
            keep_idx = [top1_idx]
        keep_boxes = [boxes[i] for i in keep_idx]
        keep_names = [names[i] for i in keep_idx]
        keep_scores = arr_scores[keep_idx]
        keep_weights = softmax_np(keep_scores, temp=self.clip_temp)
        out["clip_filtered_mean"] = {
            "box": box_mean(keep_boxes, keep_weights.tolist()),
            "picked_names": keep_names,
            "score": float(keep_scores.max()) if len(keep_scores) else float(arr_scores[top1_idx]),
        }

        # clip_weighted_vote_mean
        clusters = cluster_boxes(boxes, names, iou_thresh)
        if clusters:
            best_cluster = None
            best_cluster_weight = -1.0
            for c in clusters:
                c_w = float(weights[c["indices"]].sum())
                if c_w > best_cluster_weight:
                    best_cluster_weight = c_w
                    best_cluster = c
            assert best_cluster is not None
            idxs = best_cluster["indices"]
            c_boxes = [boxes[i] for i in idxs]
            c_names = [names[i] for i in idxs]
            c_weights = weights[idxs]
            c_weights = c_weights / (c_weights.sum() + EPS)
            out["clip_weighted_vote_mean"] = {
                "box": box_mean(c_boxes, c_weights.tolist()),
                "picked_names": c_names,
                "score": float(best_cluster_weight),
                "cluster_support": int(best_cluster["support"]),
            }
        return out


# =========================================================
# VLM router
# =========================================================
def _pick_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    try:
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        return x1 - x0, y1 - y0
    except Exception:
        return draw.textsize(text, font=font)


def render_overlay(img: Image.Image, candidates: List[Dict[str, Any]]) -> Image.Image:
    out = img.copy().convert("RGBA")
    W, H = out.size
    line_width_ratio = 0.008
    legend_pad_ratio = 0.012
    legend_bg_alpha = 220
    lw = max(2, int(min(W, H) * line_width_ratio))
    font_big = _pick_font(max(16, int(min(W, H) * 0.035)))
    font_small = _pick_font(max(13, int(min(W, H) * 0.025)))

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for item in candidates:
        box = item["box"]
        color = item["color"]
        label = item["label"]
        x1, y1, x2, y2 = box
        for k in range(lw):
            draw.rectangle([x1 - k, y1 - k, x2 + k, y2 + k], outline=color + (255,))

        tw, th = _text_size(draw, label, font_big)
        pad = max(4, lw * 2)
        bw = tw + 2 * pad
        bh = th + 2 * pad
        bx = max(0, int(x1))
        by = max(0, int(y1 - bh - pad))
        if by < 0:
            by = min(H - bh, int(y1 + pad))
        draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=max(4, lw), fill=color + (235,))
        draw.text((bx + pad, by + pad - 1), label, fill=(0, 0, 0, 255), font=font_big)

    out = Image.alpha_composite(out, overlay)

    legend = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_leg = ImageDraw.Draw(legend)
    pad = max(6, int(min(W, H) * legend_pad_ratio))
    line_h = max(20, int(min(W, H) * 0.045))
    title = "Candidates"
    title_w, title_h = _text_size(draw_leg, title, font_big)

    row_ws = []
    row_texts = []
    for item in candidates:
        color_name = COLOR_NAME_MAP.get(tuple(item["color"]), "colored")
        txt = f"{item['label']} = {color_name} (support {item['support']})"
        row_texts.append(txt)
        row_ws.append(_text_size(draw_leg, txt, font_small)[0])

    panel_w = max([title_w] + row_ws + [120]) + pad * 4 + line_h
    panel_h = title_h + pad * 3 + len(candidates) * line_h
    x0, y0 = pad, pad
    draw_leg.rounded_rectangle(
        [x0, y0, x0 + panel_w, y0 + panel_h],
        radius=max(6, pad),
        fill=(255, 255, 255, legend_bg_alpha),
        outline=(20, 20, 20, 220),
        width=2,
    )
    draw_leg.text((x0 + pad * 2, y0 + pad), title, fill=(0, 0, 0, 255), font=font_big)
    cy = y0 + pad * 2 + title_h
    for item, txt in zip(candidates, row_texts):
        c = item["color"]
        sw = line_h - 6
        sx = x0 + pad * 2
        sy = cy + 3
        draw_leg.rounded_rectangle([sx, sy, sx + sw, sy + sw], radius=4, fill=c + (255,))
        draw_leg.text((sx + sw + pad, cy), txt, fill=(0, 0, 0, 255), font=font_small)
        cy += line_h

    out = Image.alpha_composite(out, legend)
    return out.convert("RGB")


def build_user_prompt(expr: str, items: List[Dict[str, Any]]) -> str:
    descs = []
    labels = []
    for item in items:
        label = item["label"]
        color_name = COLOR_NAME_MAP.get(tuple(item["color"]), "colored")
        labels.append(label)
        descs.append(f"{label} is the {color_name} box")
    label_set = "/".join(labels)
    desc_text = "; ".join(descs)
    prompt = (
        f'Referring expression: "{expr}".\n'
        f"In the image, {desc_text}.\n"
        "Choose the single best candidate that matches the referring expression.\n"
        f"Answer with exactly one character from {{{label_set}}}."
    )
    return prompt


@dataclass
class JudgeResult:
    choice: str
    raw_text: str


class QwenVLLocalJudge:
    def __init__(self, model_path: str, device: str, dtype: torch.dtype, max_new_tokens: int = 6):
        if not HAVE_VLM:
            raise RuntimeError("transformers VLM dependencies are not available.")
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.system_prompt = (
            "You are a strict judge for referring expression grounding. "
            "You must choose the single candidate box that best matches the referring expression."
        )

        print(f"[Judge] Loading processor from: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        arch = (config.architectures[0] if getattr(config, "architectures", None) else "")
        print(f"[Judge] Detected architecture: {arch}")

        if HAVE_V2S:
            try:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_path, trust_remote_code=True, torch_dtype=dtype, device_map=None
                )
                print("[Judge] Loaded with AutoModelForVision2Seq")
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, trust_remote_code=True, torch_dtype=dtype, device_map=None
                )
                print("[Judge] Fallback loaded with AutoModelForCausalLM")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=dtype, device_map=None
            )
            print("[Judge] Loaded with AutoModelForCausalLM (Vision2Seq unavailable)")

        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def judge_overlay(self, expr: str, annotated_img: Image.Image, items: List[Dict[str, Any]]) -> JudgeResult:
        label_list = [x["label"] for x in items]
        prompt = build_user_prompt(expr, items)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]},
        ]
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=prompt_text, images=[annotated_img], return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(self.device)

        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out[:, prompt_len:]
        txt = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip().upper()
        label_pattern = r"\b(" + "|".join(re.escape(x) for x in label_list) + r")\b"
        m = re.search(label_pattern, txt)
        choice = m.group(1) if m else label_list[0]
        return JudgeResult(choice=choice, raw_text=txt)


# =========================================================
# Metrics
# =========================================================
class BoxMetrics:
    def __init__(self):
        self.ious: List[float] = []
        self.n = 0
        self.missing_pred = 0
        self.missing_gt = 0

    def add(self, iou: Optional[float], missing_pred: bool = False, missing_gt: bool = False):
        self.n += 1
        if missing_pred:
            self.missing_pred += 1
        if missing_gt:
            self.missing_gt += 1
        if iou is not None:
            self.ious.append(float(iou))

    def to_dict(self) -> Dict[str, float]:
        arr = np.array(self.ious, dtype=np.float32)
        return {
            "total": int(self.n),
            "valid": int(arr.size),
            "missing_pred": int(self.missing_pred),
            "missing_gt": int(self.missing_gt),
            "mean_iou": float(arr.mean()) if arr.size else 0.0,
            "acc@0.5": float((arr >= 0.5).mean()) if arr.size else 0.0,
            "acc@0.75": float((arr >= 0.75).mean()) if arr.size else 0.0,
            "acc@0.9": float((arr >= 0.9).mean()) if arr.size else 0.0,
        }


def add_metric(metrics_map: Dict[str, BoxMetrics], method: str, iou: Optional[float],
               missing_pred: bool = False, missing_gt: bool = False):
    if method not in metrics_map:
        metrics_map[method] = BoxMetrics()
    metrics_map[method].add(iou=iou, missing_pred=missing_pred, missing_gt=missing_gt)


# =========================================================
# Dataset helpers
# =========================================================
def build_refer(data_root: str, dataset: str, split_by: str):
    if dataset == "grefcoco":
        if not HAVE_GREFER:
            raise RuntimeError("grefer.py is not available, cannot load grefcoco.")
        return G_REFER(data_root, dataset, split_by)
    return REFER(data_root, dataset, split_by)


def get_gt_ann_xyxy(refer_obj, ref_id: int, ref: Dict[str, Any], dataset: str) -> Optional[List[float]]:
    if dataset == "grefcoco":
        anns = refer_obj.getRefBox(ref_id)
        if not anns:
            return None
        # merge multiple gt boxes as a loose outer box
        xs1, ys1, xs2, ys2 = [], [], [], []
        for a in anns:
            x, y, w, h = [float(v) for v in a]
            xs1.append(x)
            ys1.append(y)
            xs2.append(x + w)
            ys2.append(y + h)
        return [min(xs1), min(ys1), max(xs2), max(ys2)]
    return xywh_to_xyxy(refer_obj.getRefBox(ref_id))


def get_gt_mask_xyxy(refer_obj, ref: Dict[str, Any], dataset: str) -> Optional[List[float]]:
    if dataset == "grefcoco":
        m = refer_obj.getMaskByRef(ref=ref, merge=True)
        if m is None or m.get("empty", False):
            return None
        return mask_to_box_xyxy(m["mask"])
    m = refer_obj.getMask(ref)
    return mask_to_box_xyxy(m["mask"]) if m is not None else None


# =========================================================
# Core evaluation
# =========================================================
def resolve_primary_record(primary_method: str, method_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    rec = method_results.get(primary_method, None)
    if rec is not None and rec.get("box", None) is not None:
        return rec
    for fallback in ["vote_mean", "medoid", "trimmed_mean", "all_mean", "oracle_best"]:
        if fallback in method_results and method_results[fallback].get("box", None) is not None:
            r = dict(method_results[fallback])
            r["fallback_from"] = primary_method
            return r
    return {"box": None, "label": None, "picked_names": [], "judge_raw": "", "skip_reason": "no_valid_box"}


def parse_methods(raw: str) -> List[str]:
    out = []
    for x in raw.split(","):
        x = x.strip()
        if x:
            out.append(x)
    return out


def evaluate(args):
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[INFO] Loading dataset={args.dataset}, splitBy={args.split_by}, split={args.split}")
    refer = build_refer(args.data_root, args.dataset, args.split_by)
    ref_ids = refer.getRefIds(split=args.split)
    print(f"[INFO] #refs in split: {len(ref_ids)}")

    expert_cfgs = build_expert_configs(args)
    print("[INFO] Loading expert predictions...")
    expert_preds: Dict[str, Dict[Tuple[int, int], Dict[str, Any]]] = {}
    for cfg in expert_cfgs:
        if not os.path.exists(cfg["path"]):
            raise FileNotFoundError(f"Expert file not found: {cfg['path']}")
        mp, detected_mode = load_pred_jsonl(cfg["path"], source=cfg.get("source", "auto"))
        cfg["coord_mode"] = detected_mode if cfg.get("coord_mode", "auto") == "auto" else cfg["coord_mode"]
        expert_preds[cfg["name"]] = mp
        print(f"  - {cfg['name']}: {len(mp)} entries | {cfg['path']} | coord_mode={cfg['coord_mode']}")

    method_order = [cfg["name"] for cfg in expert_cfgs]
    method_order += ["oracle_best", "all_mean", "trimmed_mean", "medoid", "vote_mean"]
    router_methods = parse_methods(args.router_methods)
    for m in router_methods:
        if m not in method_order:
            method_order.append(m)

    need_clip = any(m.startswith("clip_") for m in router_methods) or args.primary_router.startswith("clip_")
    need_vlm = ("vlm_router" in router_methods) or (args.primary_router == "vlm_router")

    clip_router = None
    if need_clip:
        clip_router = ClipRouter(
            model_name=args.clip_name,
            device=device,
            dtype=dtype,
            pad_ratio=args.clip_pad_ratio,
            use_ctx_diff=args.clip_use_ctx_diff,
            clip_temp=args.clip_temp,
            clip_filter_margin=args.clip_filter_margin,
        )

    vlm_judge = None
    if need_vlm:
        if not args.vlm_model_path:
            raise ValueError("--vlm-model-path is required when vlm_router is enabled.")
        vlm_judge = QwenVLLocalJudge(args.vlm_model_path, device, dtype, max_new_tokens=args.vlm_max_new_tokens)

    os.makedirs(args.output_dir, exist_ok=True)
    auto_jsonl_name, auto_summary_name = build_default_output_names(args)
    out_jsonl_name = args.out_jsonl.strip() if str(args.out_jsonl).strip() else auto_jsonl_name
    out_summary_name = args.out_summary_json.strip() if str(args.out_summary_json).strip() else auto_summary_name
    if out_jsonl_name == "merged_router_choices.jsonl":
        out_jsonl_name = auto_jsonl_name
    if out_summary_name == "merged_router_summary.json":
        out_summary_name = auto_summary_name
    out_jsonl = os.path.join(args.output_dir, out_jsonl_name)
    out_summary = os.path.join(args.output_dir, out_summary_name)

    metrics_ann: Dict[str, BoxMetrics] = {}
    metrics_msk: Dict[str, BoxMetrics] = {}
    oracle_win_by_expert_ann: Dict[str, int] = defaultdict(int)
    oracle_win_by_expert_msk: Dict[str, int] = defaultdict(int)
    clip_choice_counter = Counter()
    vlm_choice_counter = Counter()
    skip_reason_counter = Counter()
    valid_expert_count_hist = defaultdict(int)
    cluster_count_hist = defaultdict(int)
    img_cache: Dict[str, Image.Image] = {}

    judge_called = 0
    judge_skipped = 0

    with open(out_jsonl, "w", encoding="utf-8") as f_out:
        for ref_id in tqdm(ref_ids, desc=f"Router Eval {args.dataset}-{args.split_by}-{args.split}"):
            ref = refer.loadRefs(ref_id)[0]
            img_meta = refer.loadImgs(ref["image_id"])[0]
            W, H = int(img_meta["width"]), int(img_meta["height"])
            img_path = os.path.join(refer.IMAGE_DIR, img_meta["file_name"])
            gt_ann = get_gt_ann_xyxy(refer, ref_id, ref, args.dataset)
            gt_msk = get_gt_mask_xyxy(refer, ref, args.dataset)

            if img_path not in img_cache:
                img_cache[img_path] = Image.open(img_path).convert("RGB")
            img_full = img_cache[img_path]

            for sent in ref["sentences"]:
                sent_id = int(sent["sent_id"])
                expr = sent["sent"]
                key = (int(ref_id), int(sent_id))

                valid_boxes: List[List[float]] = []
                valid_names: List[str] = []
                expert_boxes: Dict[str, Optional[List[float]]] = {}
                expert_ious_ann: Dict[str, Optional[float]] = {}
                expert_ious_msk: Dict[str, Optional[float]] = {}
                method_results: Dict[str, Dict[str, Any]] = {}

                # single experts
                for cfg in expert_cfgs:
                    name = cfg["name"]
                    item = expert_preds[name].get(key, None)
                    if item is None:
                        expert_boxes[name] = None
                        expert_ious_ann[name] = None
                        expert_ious_msk[name] = None
                        add_metric(metrics_ann, name, None, missing_pred=True)
                        add_metric(metrics_msk, name, None, missing_pred=True, missing_gt=(gt_msk is None))
                        method_results[name] = {"box": None, "label": None, "picked_names": [], "judge_raw": ""}
                        continue

                    box = decode_to_pixel_xyxy(item["box_raw"], W, H, str(cfg.get("coord_mode", "auto")))
                    if box is None:
                        expert_boxes[name] = None
                        expert_ious_ann[name] = None
                        expert_ious_msk[name] = None
                        add_metric(metrics_ann, name, None, missing_pred=True)
                        add_metric(metrics_msk, name, None, missing_pred=True, missing_gt=(gt_msk is None))
                        method_results[name] = {"box": None, "label": None, "picked_names": [], "judge_raw": ""}
                        continue

                    box = clamp_xyxy(box, W, H)
                    expert_boxes[name] = box
                    valid_boxes.append(box)
                    valid_names.append(name)

                    if gt_ann is None:
                        iou_ann = None
                        add_metric(metrics_ann, name, None, missing_gt=True)
                    else:
                        iou_ann = iou_xyxy(box, gt_ann)
                        add_metric(metrics_ann, name, iou_ann)
                    expert_ious_ann[name] = iou_ann

                    if gt_msk is None:
                        iou_msk = None
                        add_metric(metrics_msk, name, None, missing_gt=True)
                    else:
                        iou_msk = iou_xyxy(box, gt_msk)
                        add_metric(metrics_msk, name, iou_msk)
                    expert_ious_msk[name] = iou_msk

                    method_results[name] = {
                        "box": box,
                        "label": name,
                        "picked_names": [name],
                        "judge_raw": "",
                    }

                valid_expert_count_hist[len(valid_boxes)] += 1

                # oracle
                if gt_ann is not None:
                    best_name_ann, best_iou_ann = None, None
                    for name in valid_names:
                        v = expert_ious_ann[name]
                        if v is not None and (best_iou_ann is None or v > best_iou_ann):
                            best_name_ann, best_iou_ann = name, v
                    if best_name_ann is not None:
                        oracle_win_by_expert_ann[best_name_ann] += 1
                        add_metric(metrics_ann, "oracle_best", best_iou_ann)
                        method_results["oracle_best"] = {
                            "box": expert_boxes[best_name_ann],
                            "label": best_name_ann,
                            "picked_names": [best_name_ann],
                        }
                    else:
                        add_metric(metrics_ann, "oracle_best", None, missing_pred=True)
                        method_results["oracle_best"] = {"box": None, "label": None, "picked_names": []}
                else:
                    add_metric(metrics_ann, "oracle_best", None, missing_gt=True)
                    method_results["oracle_best"] = {"box": None, "label": None, "picked_names": []}

                if gt_msk is not None:
                    best_name_msk, best_iou_msk = None, None
                    for name in valid_names:
                        v = expert_ious_msk[name]
                        if v is not None and (best_iou_msk is None or v > best_iou_msk):
                            best_name_msk, best_iou_msk = name, v
                    if best_name_msk is not None:
                        oracle_win_by_expert_msk[best_name_msk] += 1
                        add_metric(metrics_msk, "oracle_best", best_iou_msk)
                    else:
                        add_metric(metrics_msk, "oracle_best", None, missing_pred=True)
                else:
                    add_metric(metrics_msk, "oracle_best", None, missing_gt=True)

                # classical fusion
                all_mean_box = clamp_xyxy(box_mean(valid_boxes), W, H) if valid_boxes else None
                trimmed_mean = clamp_xyxy(trimmed_mean_box(valid_boxes, args.trim_num), W, H) if valid_boxes else None
                medoid = clamp_xyxy(medoid_box(valid_boxes), W, H) if valid_boxes else None
                vote_mean = clamp_xyxy(vote_mean_box(valid_boxes, args.cluster_iou_thresh), W, H) if valid_boxes else None
                classical = {
                    "all_mean": all_mean_box,
                    "trimmed_mean": trimmed_mean,
                    "medoid": medoid,
                    "vote_mean": vote_mean,
                }
                for method, box in classical.items():
                    method_results[method] = {"box": box, "label": method, "picked_names": list(valid_names)}
                    if box is None:
                        add_metric(metrics_ann, method, None, missing_pred=True)
                        add_metric(metrics_msk, method, None, missing_pred=True, missing_gt=(gt_msk is None))
                    else:
                        add_metric(metrics_ann, method, iou_xyxy(box, gt_ann) if gt_ann is not None else None, missing_gt=(gt_ann is None))
                        add_metric(metrics_msk, method, iou_xyxy(box, gt_msk) if gt_msk is not None else None, missing_gt=(gt_msk is None))

                clusters = cluster_boxes(valid_boxes, valid_names, args.cluster_iou_thresh)
                cluster_count_hist[len(clusters)] += 1
                agree_iou = pairwise_iou_mean(valid_boxes) if valid_boxes else 0.0

                # clip methods
                clip_scores_map = None
                if clip_router is not None and valid_boxes:
                    clip_scores = clip_router.score_boxes(img_full, expr, valid_boxes)
                    clip_scores_map = {name: float(score) for name, score in zip(valid_names, clip_scores)}
                    clip_results = clip_router.route(valid_boxes, valid_names, clip_scores, args.cluster_iou_thresh)
                    for method, rec in clip_results.items():
                        box = clamp_xyxy(rec["box"], W, H) if rec.get("box") is not None else None
                        method_results[method] = {
                            "box": box,
                            "label": method,
                            "picked_names": rec.get("picked_names", []),
                            "score": rec.get("score", None),
                        }
                        if method == "clip_top1" and rec.get("picked_names"):
                            clip_choice_counter[rec["picked_names"][0]] += 1
                        if box is None:
                            add_metric(metrics_ann, method, None, missing_pred=True)
                            add_metric(metrics_msk, method, None, missing_pred=True, missing_gt=(gt_msk is None))
                        else:
                            add_metric(metrics_ann, method, iou_xyxy(box, gt_ann) if gt_ann is not None else None, missing_gt=(gt_ann is None))
                            add_metric(metrics_msk, method, iou_xyxy(box, gt_msk) if gt_msk is not None else None, missing_gt=(gt_msk is None))
                else:
                    for method in ["clip_top1", "clip_wmean", "clip_filtered_mean", "clip_weighted_vote_mean"]:
                        if method in router_methods or args.primary_router == method:
                            method_results[method] = {"box": None, "label": None, "picked_names": []}
                            add_metric(metrics_ann, method, None, missing_pred=True)
                            add_metric(metrics_msk, method, None, missing_pred=True, missing_gt=(gt_msk is None))

                # vlm router
                judge_called_flag = False
                judge_raw = ""
                skip_reason = None
                routed_label = None
                routed_cluster = None
                if vlm_judge is not None:
                    if not valid_boxes:
                        judge_skipped += 1
                        skip_reason = "no_valid_box"
                        skip_reason_counter[skip_reason] += 1
                        method_results["vlm_router"] = {
                            "box": None, "label": None, "picked_names": [], "judge_raw": "", "skip_reason": skip_reason,
                        }
                        add_metric(metrics_ann, "vlm_router", None, missing_pred=True)
                        add_metric(metrics_msk, "vlm_router", None, missing_pred=True, missing_gt=(gt_msk is None))
                    else:
                        do_judge = True
                        top_cluster = clusters[0]
                        second_support = clusters[1]["support"] if len(clusters) > 1 else 0
                        top_support_ratio = top_cluster["support"] / max(len(valid_boxes), 1)

                        if args.vlm_gate_enable:
                            if len(clusters) <= 1:
                                do_judge = False
                                skip_reason = "single_cluster"
                            elif agree_iou >= args.vlm_gate_pairwise_high:
                                do_judge = False
                                skip_reason = "high_agreement"
                            elif top_support_ratio >= args.vlm_gate_top_support_ratio and (top_cluster["support"] - second_support) >= args.vlm_gate_top_support_gap:
                                do_judge = False
                                skip_reason = "dominant_top_cluster"

                        if not do_judge:
                            judge_skipped += 1
                            skip_reason_counter[skip_reason] += 1
                            if args.vlm_skip_policy == "top_cluster_mean":
                                routed_box = top_cluster["mean_box"]
                            elif args.vlm_skip_policy == "vote_mean":
                                routed_box = vote_mean if vote_mean is not None else top_cluster["mean_box"]
                            else:
                                routed_box = medoid if medoid is not None else top_cluster["mean_box"]
                            routed_cluster = top_cluster
                            routed_label = "SKIP"
                            judge_raw = ""
                        else:
                            judge_called += 1
                            judge_called_flag = True
                            cand_clusters = clusters[:args.vlm_max_judge_candidates]
                            items = []
                            for i, c in enumerate(cand_clusters):
                                items.append({
                                    "cluster": c,
                                    "box": clamp_xyxy(c["mean_box"], W, H),
                                    "support": c["support"],
                                    "dense": c["dense"],
                                    "label": LABEL_POOL[i],
                                    "color": COLOR_POOL[i % len(COLOR_POOL)],
                                })
                            annotated = render_overlay(img_full, items)
                            res = vlm_judge.judge_overlay(expr, annotated, items)
                            judge_raw = res.raw_text
                            routed_label = res.choice
                            vlm_choice_counter[routed_label] += 1
                            chosen = None
                            for item in items:
                                if item["label"] == routed_label:
                                    chosen = item
                                    break
                            if chosen is None:
                                chosen = items[0]
                            routed_cluster = chosen["cluster"]
                            routed_box = chosen["box"]

                        routed_box = clamp_xyxy(routed_box, W, H) if routed_box is not None else None
                        picked_names = routed_cluster["member_names"] if routed_cluster is not None else []
                        method_results["vlm_router"] = {
                            "box": routed_box,
                            "label": routed_label,
                            "picked_names": picked_names,
                            "judge_raw": judge_raw,
                            "skip_reason": skip_reason,
                            "routed_cluster_support": int(routed_cluster["support"]) if routed_cluster is not None else None,
                            "routed_cluster_members": picked_names,
                        }
                        if routed_box is None:
                            add_metric(metrics_ann, "vlm_router", None, missing_pred=True)
                            add_metric(metrics_msk, "vlm_router", None, missing_pred=True, missing_gt=(gt_msk is None))
                        else:
                            add_metric(metrics_ann, "vlm_router", iou_xyxy(routed_box, gt_ann) if gt_ann is not None else None, missing_gt=(gt_ann is None))
                            add_metric(metrics_msk, "vlm_router", iou_xyxy(routed_box, gt_msk) if gt_msk is not None else None, missing_gt=(gt_msk is None))

                primary = resolve_primary_record(args.primary_router, method_results)
                routed_box = primary.get("box", None)
                routed_iou_ann = None if routed_box is None or gt_ann is None else iou_xyxy(routed_box, gt_ann)
                routed_iou_msk = None if routed_box is None or gt_msk is None else iou_xyxy(routed_box, gt_msk)

                error = None
                if routed_box is None:
                    error = primary.get("skip_reason") or primary.get("fallback_from") or "no_valid_box"

                rec = build_single_style_record(
                    args=args,
                    img_meta=img_meta,
                    ref_id=int(ref_id),
                    sent_id=int(sent_id),
                    expr=expr,
                    routed_box=routed_box,
                    error=error,
                )
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "dataset": args.dataset,
        "splitBy": args.split_by,
        "split": args.split,
        "experts": expert_cfgs,
        "num_refs": int(len(ref_ids)),
        "primary_router": args.primary_router,
        "router_methods": router_methods,
        "judge_called": int(judge_called),
        "judge_skipped": int(judge_skipped),
        "judge_call_rate": float(judge_called / (judge_called + judge_skipped + 1e-9)),
        "skip_reason_counter": dict(skip_reason_counter),
        "clip_choice_counter": dict(clip_choice_counter),
        "vlm_choice_counter": dict(vlm_choice_counter),
        "valid_expert_count_hist": {str(k): int(v) for k, v in sorted(valid_expert_count_hist.items())},
        "cluster_count_hist": {str(k): int(v) for k, v in sorted(cluster_count_hist.items())},
        "oracle_win_by_expert_ann": {k: int(v) for k, v in sorted(oracle_win_by_expert_ann.items())},
        "oracle_win_by_expert_mask": {k: int(v) for k, v in sorted(oracle_win_by_expert_msk.items())},
        "metrics_ann": {},
        "metrics_mask": {},
    }
    for m in method_order:
        summary["metrics_ann"][m] = metrics_ann.get(m, BoxMetrics()).to_dict()
        summary["metrics_mask"][m] = metrics_msk.get(m, BoxMetrics()).to_dict()

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 108)
    print("AnnBox metrics")
    for m in method_order:
        met = summary["metrics_ann"][m]
        print(
            f"{m:>24s} | total={met['total']:6d} valid={met['valid']:6d} "
            f"| miou={met['mean_iou']:.4f} | acc@0.5={met['acc@0.5']:.4f} | acc@0.75={met['acc@0.75']:.4f}"
        )
    print("\nMaskBox metrics")
    for m in method_order:
        met = summary["metrics_mask"][m]
        print(
            f"{m:>24s} | total={met['total']:6d} valid={met['valid']:6d} "
            f"| miou={met['mean_iou']:.4f} | acc@0.5={met['acc@0.5']:.4f} | acc@0.75={met['acc@0.75']:.4f}"
        )
    print(f"\n[INFO] Summary saved to: {out_summary}")
    print(f"[INFO] Per-sample details saved to: {out_jsonl}")
    print("=" * 108)


# =========================================================
# CLI
# =========================================================
def build_argparser():
    parser = argparse.ArgumentParser(description="Merged CLIP + VLM multi-expert router for RES box priors")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="refcoco", choices=["refcoco", "refcoco+", "refcocog", "grefcoco"])
    parser.add_argument("--split-by", type=str, default="unc")
    parser.add_argument("--split", type=str, default="testA")
    parser.add_argument("--input-dir", type=str, default=".")
    parser.add_argument("--expert-file", action="append", default=[], help="Optional explicit expert file(s): path or name=path")
    parser.add_argument("--output-dir", type=str, default="router_out")
    parser.add_argument(
        "--out-jsonl",
        type=str,
        default="",
        help="Optional explicit per-sample jsonl filename. Empty means auto naming: {dataset}_{split_by}_{split}_{primary_router}_choices.jsonl",
    )
    parser.add_argument(
        "--out-summary-json",
        type=str,
        default="",
        help="Optional explicit summary json filename. Empty means auto naming: {dataset}_{split_by}_{split}_{primary_router}_summary.json",
    )
    parser.add_argument(
        "--router-methods",
        type=str,
        default="oracle_best,all_mean,trimmed_mean,medoid,vote_mean,clip_top1,clip_wmean,clip_filtered_mean,clip_weighted_vote_mean",
    )
    parser.add_argument("--primary-router", type=str, default="clip_weighted_vote_mean")

    # fusion / clustering
    parser.add_argument("--cluster-iou-thresh", type=float, default=0.60)
    parser.add_argument("--trim-num", type=int, default=2, help="How many low-consensus boxes to trim in trimmed_mean")

    # clip
    parser.add_argument("--clip-name", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--clip-pad-ratio", type=float, default=0.15)
    parser.add_argument("--clip-use-ctx-diff", action="store_true", default=True)
    parser.add_argument("--no-clip-use-ctx-diff", action="store_false", dest="clip_use_ctx_diff")
    parser.add_argument("--clip-temp", type=float, default=0.07)
    parser.add_argument("--clip-filter-margin", type=float, default=0.02)

    # vlm
    parser.add_argument("--vlm-model-path", type=str, default="")
    parser.add_argument("--vlm-max-new-tokens", type=int, default=6)
    parser.add_argument("--vlm-max-judge-candidates", type=int, default=2)
    parser.add_argument("--vlm-gate-enable", action="store_true", default=True)
    parser.add_argument("--no-vlm-gate-enable", action="store_false", dest="vlm_gate_enable")
    parser.add_argument("--vlm-gate-pairwise-high", type=float, default=0.78)
    parser.add_argument("--vlm-gate-top-support-ratio", type=float, default=0.60)
    parser.add_argument("--vlm-gate-top-support-gap", type=int, default=2)
    parser.add_argument("--vlm-skip-policy", type=str, default="top_cluster_mean", choices=["top_cluster_mean", "vote_mean", "medoid"])

    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
