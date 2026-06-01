#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-expert VLM router with overlay judging (improved version).

Main improvements over the previous version:
1) Explicit color-to-label mapping in the prompt, e.g. "A is the red box".
2) Judge only top-2 clusters by default (closer to the original two-expert setting).
3) Bigger, clearer labels and a legend panel rendered into the image.
4) Random permutation disabled by default for stability.
5) Minor prompt tightening and more robust label parsing.
"""

import os
import re
import json
import random
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import torch
from transformers import AutoProcessor, AutoConfig
try:
    from transformers import AutoModelForVision2Seq
    HAVE_V2S = True
except Exception:
    HAVE_V2S = False
from transformers import AutoModelForCausalLM

from refer import REFER

# =========================================================
# Config (EDIT HERE)
# =========================================================
DATA_ROOT = "../../DETRIS-main/datasets"
DATASET = "refcoco"
SPLITBY = "unc"
SPLIT = "testB"

EXPERTS = [
    {"name": "qwen",       "path": "pred_refcoco_unc_testB_qwen3-vl-flash.jsonl",           "source": "raw",  "coord_mode": "force_1000"},
    {"name": "qwen3-plus", "path": "pred_refcoco_unc_testB_qwen3-vl-plus.jsonl",            "source": "raw",  "coord_mode": "force_1000"},
    {"name": "glm",        "path": "pred_refcoco_unc_testB_glm-4.6v-flash.jsonl",            "source": "raw",  "coord_mode": "force_1000"},
    {"name": "glm-4.6v",   "path": "pred_refcoco_unc_testB_glm-4.6v.jsonl",                  "source": "raw",  "coord_mode": "force_1000"},
    # {"name": "gpt5",       "path": "pred_refcoco_unc_testB_gpt-5-2025-08-07.jsonl",         "source": "pred", "coord_mode": "pixel"},
    # {"name": "gpt5.2",     "path": "pred_refcoco_unc_testB_gpt-5.2-2025-12-11_norm1000.jsonl","source": "raw",  "coord_mode": "force_1000"},
    {"name": "qwen30b",    "path": "pred_refcoco_unc_testB_qwen3-vl-30b-a3b-instruct.jsonl", "source": "raw",  "coord_mode": "force_1000"},
]

MODEL_PATH = "../../../pretrained/Qwen3-VL-8B-Instruct/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

CLUSTER_IOU_THRESH = 0.60
MAX_JUDGE_CANDIDATES = 2          # changed: top-2 only by default

GATE_ENABLE = True
GATE_PAIRWISE_HIGH = 0.78
GATE_TOP_SUPPORT_RATIO = 0.60
GATE_TOP_SUPPORT_GAP = 2
DEFAULT_SKIP_POLICY = "top_cluster_mean"  # top_cluster_mean | vote_mean | medoid

LINE_WIDTH_RATIO = 0.008          # slightly thicker
LABEL_BOX_SCALE = 3.0             # larger label patch
LEGEND_PAD_RATIO = 0.012
LEGEND_BG_ALPHA = 220
COLOR_POOL = [
    (255, 0, 0),      # red
    (0, 255, 0),      # green
    (0, 128, 255),    # blue
    (255, 165, 0),    # orange
    (255, 0, 255),    # magenta
    (255, 255, 0),    # yellow
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
ENABLE_RANDOM_PERMUTE = False     # changed: disabled for stability

MAX_NEW_TOKENS = 6
SYSTEM_PROMPT = (
    "You are a strict judge for referring expression grounding. "
    "You must choose the single candidate box that best matches the referring expression."
)

OUT_JSONL = "qwen_vlm_multi_expert_router_choices_overlay_v2.jsonl"
OUT_SUMMARY_JSON = "qwen_vlm_multi_expert_router_summary_v2.json"
SAVE_PER_SAMPLE = True
PRINT_TOPLINE = True

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================================================
# Utils
# =========================================================
EPS = 1e-9
BOX_RE = re.compile(r"\[\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]\]")
LOOSE_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


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


def box_mean(boxes: List[List[float]]) -> Optional[List[float]]:
    if not boxes:
        return None
    return np.asarray(boxes, dtype=np.float32).mean(axis=0).tolist()


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


def pairwise_iou_mean(boxes: List[List[float]]) -> float:
    n = len(boxes)
    if n <= 1:
        return 1.0
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(iou_xyxy(boxes[i], boxes[j]))
    return float(np.mean(vals)) if vals else 1.0


def load_pred_jsonl(path: str, source: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    mp: Dict[Tuple[int, int], Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            try:
                ref_id = int(obj["ref_id"])
                sent_id = int(obj["sent_id"])
            except Exception:
                continue

            if source == "raw":
                box_raw = parse_box_from_raw(obj.get("raw_text", ""))
                if box_raw is None:
                    box_raw = obj.get("pred_box_xyxy", None)
            elif source == "pred":
                box_raw = obj.get("pred_box_xyxy", None)
                if box_raw is None:
                    box_raw = parse_box_from_raw(obj.get("raw_text", ""))
            else:
                raise ValueError(f"Unknown source={source}")

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
            }
    return mp


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
    """
    Draw candidate boxes plus a legend panel.
    Each candidate must include: box, color, label, support.
    """
    out = img.copy().convert("RGBA")
    W, H = out.size
    lw = max(2, int(min(W, H) * LINE_WIDTH_RATIO))
    font_big = _pick_font(max(16, int(min(W, H) * 0.035)))
    font_small = _pick_font(max(13, int(min(W, H) * 0.025)))

    # draw boxes on a transparent overlay first
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for item in candidates:
        box = item["box"]
        color = item["color"]
        label = item["label"]
        x1, y1, x2, y2 = box

        for k in range(lw):
            draw.rectangle([x1 - k, y1 - k, x2 + k, y2 + k], outline=color + (255,))

        # large label patch near top-left of the box
        label_text = label
        tw, th = _text_size(draw, label_text, font_big)
        pad = max(4, lw * 2)
        bw = tw + 2 * pad
        bh = th + 2 * pad
        bx = max(0, int(x1))
        by = max(0, int(y1 - bh - pad))
        if by < 0:
            by = min(H - bh, int(y1 + pad))
        draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=max(4, lw), fill=color + (235,))
        draw.text((bx + pad, by + pad - 1), label_text, fill=(0, 0, 0, 255), font=font_big)

    out = Image.alpha_composite(out, overlay)

    # legend panel
    legend = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_leg = ImageDraw.Draw(legend)
    pad = max(6, int(min(W, H) * LEGEND_PAD_RATIO))
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
        fill=(255, 255, 255, LEGEND_BG_ALPHA),
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


# =========================================================
# Local Qwen-VL Judge
# =========================================================
@dataclass
class JudgeResult:
    choice: str
    raw_text: str


class QwenVLLocalJudge:
    def __init__(self, model_path: str, device: str, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype

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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]},
        ]

        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=prompt_text, images=[annotated_img], return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(self.device)

        out = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

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


def add_metric(metrics_map: Dict[str, BoxMetrics], method: str, iou: Optional[float], missing_pred: bool = False, missing_gt: bool = False):
    if method not in metrics_map:
        metrics_map[method] = BoxMetrics()
    metrics_map[method].add(iou=iou, missing_pred=missing_pred, missing_gt=missing_gt)


# =========================================================
# Main
# =========================================================
def main():
    print(f"[INFO] Loading REFER: dataset={DATASET}, splitBy={SPLITBY}, split={SPLIT}")
    refer = REFER(DATA_ROOT, DATASET, SPLITBY)

    print("[INFO] Loading expert predictions...")
    expert_preds: Dict[str, Dict[Tuple[int, int], Dict[str, Any]]] = {}
    for cfg in EXPERTS:
        name = cfg["name"]
        path = cfg["path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert file not found: {path}")
        expert_preds[name] = load_pred_jsonl(path, source=cfg.get("source", "pred"))
        print(
            f"  - {name}: {len(expert_preds[name])} entries from {path} "
            f"| source={cfg.get('source', 'pred')} | coord_mode={cfg.get('coord_mode', 'auto')}"
        )

    judge = QwenVLLocalJudge(MODEL_PATH, DEVICE, DTYPE)
    ref_ids = refer.getRefIds(split=SPLIT)
    print(f"[INFO] #refs in split: {len(ref_ids)}")

    metrics_ann: Dict[str, BoxMetrics] = {}
    metrics_msk: Dict[str, BoxMetrics] = {}
    oracle_win_by_expert_ann: Dict[str, int] = defaultdict(int)
    oracle_win_by_expert_msk: Dict[str, int] = defaultdict(int)

    judge_called = 0
    judge_skipped = 0
    skip_reason_counter = Counter()
    choice_counter = Counter()
    valid_expert_count_hist = defaultdict(int)
    cluster_count_hist = defaultdict(int)

    per_sample_f = open(OUT_JSONL, "w", encoding="utf-8") if SAVE_PER_SAMPLE else open(os.devnull, "w", encoding="utf-8")
    img_cache: Dict[str, Image.Image] = {}

    with per_sample_f as f_out:
        for ref_id in tqdm(ref_ids, desc=f"VLM Router Eval {DATASET}-{SPLITBY}-{SPLIT}"):
            ref = refer.loadRefs(ref_id)[0]
            img_meta = refer.loadImgs(ref["image_id"])[0]
            W, H = int(img_meta["width"]), int(img_meta["height"])
            img_path = os.path.join(refer.IMAGE_DIR, img_meta["file_name"])

            gt_ann = xywh_to_xyxy(refer.getRefBox(ref_id))
            M = refer.getMask(ref)
            gt_msk = mask_to_box_xyxy(M["mask"])

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

                for cfg in EXPERTS:
                    name = cfg["name"]
                    item = expert_preds[name].get(key, None)
                    if item is None:
                        expert_boxes[name] = None
                        expert_ious_ann[name] = None
                        expert_ious_msk[name] = None
                        add_metric(metrics_ann, name, None, missing_pred=True)
                        add_metric(metrics_msk, name, None, missing_pred=True, missing_gt=(gt_msk is None))
                        continue

                    box = decode_to_pixel_xyxy(item["box_raw"], W, H, str(cfg.get("coord_mode", "auto")))
                    if box is None:
                        expert_boxes[name] = None
                        expert_ious_ann[name] = None
                        expert_ious_msk[name] = None
                        add_metric(metrics_ann, name, None, missing_pred=True)
                        add_metric(metrics_msk, name, None, missing_pred=True, missing_gt=(gt_msk is None))
                        continue

                    box = clamp_xyxy(box, W, H)
                    expert_boxes[name] = box
                    valid_boxes.append(box)
                    valid_names.append(name)

                    iou_ann = iou_xyxy(box, gt_ann)
                    expert_ious_ann[name] = iou_ann
                    add_metric(metrics_ann, name, iou_ann)

                    if gt_msk is None:
                        expert_ious_msk[name] = None
                        add_metric(metrics_msk, name, None, missing_gt=True)
                    else:
                        iou_msk = iou_xyxy(box, gt_msk)
                        expert_ious_msk[name] = iou_msk
                        add_metric(metrics_msk, name, iou_msk)

                valid_expert_count_hist[len(valid_boxes)] += 1

                best_name_ann, best_iou_ann = None, None
                for name in valid_names:
                    v = expert_ious_ann[name]
                    if v is not None and (best_iou_ann is None or v > best_iou_ann):
                        best_name_ann, best_iou_ann = name, v
                if best_name_ann is not None:
                    oracle_win_by_expert_ann[best_name_ann] += 1
                    add_metric(metrics_ann, "oracle_best", best_iou_ann)
                else:
                    add_metric(metrics_ann, "oracle_best", None, missing_pred=True)

                best_name_msk, best_iou_msk = None, None
                if gt_msk is None:
                    add_metric(metrics_msk, "oracle_best", None, missing_gt=True)
                else:
                    for name in valid_names:
                        v = expert_ious_msk[name]
                        if v is not None and (best_iou_msk is None or v > best_iou_msk):
                            best_name_msk, best_iou_msk = name, v
                    if best_name_msk is not None:
                        oracle_win_by_expert_msk[best_name_msk] += 1
                        add_metric(metrics_msk, "oracle_best", best_iou_msk)
                    else:
                        add_metric(metrics_msk, "oracle_best", None, missing_pred=True)

                if valid_boxes:
                    all_mean_box = clamp_xyxy(box_mean(valid_boxes), W, H)
                    medoid = clamp_xyxy(medoid_box(valid_boxes), W, H)
                    vote_mean = clamp_xyxy(vote_mean_box(valid_boxes, CLUSTER_IOU_THRESH), W, H)
                else:
                    all_mean_box = medoid = vote_mean = None

                for method, box in [("all_mean", all_mean_box), ("medoid", medoid), ("vote_mean", vote_mean)]:
                    if box is None:
                        add_metric(metrics_ann, method, None, missing_pred=True)
                        add_metric(metrics_msk, method, None, missing_pred=True, missing_gt=(gt_msk is None))
                    else:
                        add_metric(metrics_ann, method, iou_xyxy(box, gt_ann))
                        if gt_msk is None:
                            add_metric(metrics_msk, method, None, missing_gt=True)
                        else:
                            add_metric(metrics_msk, method, iou_xyxy(box, gt_msk))

                clusters = cluster_boxes(valid_boxes, valid_names, CLUSTER_IOU_THRESH)
                cluster_count_hist[len(clusters)] += 1

                routed_box = None
                routed_cluster = None
                routed_label = None
                judge_raw = ""
                judge_called_flag = False
                skip_reason = None
                agree_iou = pairwise_iou_mean(valid_boxes) if valid_boxes else 0.0

                if not valid_boxes:
                    skip_reason = "no_valid_box"
                    judge_skipped += 1
                    skip_reason_counter[skip_reason] += 1
                else:
                    do_judge = True
                    top_cluster = clusters[0]
                    second_support = clusters[1]["support"] if len(clusters) > 1 else 0
                    top_support_ratio = top_cluster["support"] / max(len(valid_boxes), 1)

                    if GATE_ENABLE:
                        if len(clusters) <= 1:
                            do_judge = False
                            skip_reason = "single_cluster"
                        elif agree_iou >= GATE_PAIRWISE_HIGH:
                            do_judge = False
                            skip_reason = "high_agreement"
                        elif top_support_ratio >= GATE_TOP_SUPPORT_RATIO and (top_cluster["support"] - second_support) >= GATE_TOP_SUPPORT_GAP:
                            do_judge = False
                            skip_reason = "dominant_top_cluster"

                    if not do_judge:
                        judge_skipped += 1
                        skip_reason_counter[skip_reason] += 1
                        if DEFAULT_SKIP_POLICY == "top_cluster_mean":
                            routed_box = top_cluster["mean_box"]
                        elif DEFAULT_SKIP_POLICY == "vote_mean":
                            routed_box = vote_mean if vote_mean is not None else top_cluster["mean_box"]
                        else:
                            routed_box = medoid if medoid is not None else top_cluster["mean_box"]
                        routed_cluster = top_cluster
                        routed_label = "SKIP"
                    else:
                        judge_called += 1
                        judge_called_flag = True

                        cand_clusters = clusters[:MAX_JUDGE_CANDIDATES]
                        items = []
                        for i, c in enumerate(cand_clusters):
                            items.append({
                                "cluster": c,
                                "box": clamp_xyxy(c["mean_box"], W, H),
                                "support": c["support"],
                                "dense": c["dense"],
                            })

                        if ENABLE_RANDOM_PERMUTE:
                            random.shuffle(items)

                        for i, item in enumerate(items):
                            item["label"] = LABEL_POOL[i]
                            item["color"] = COLOR_POOL[i % len(COLOR_POOL)]

                        annotated = render_overlay(img_full, items)
                        res = judge.judge_overlay(expr, annotated, items)
                        judge_raw = res.raw_text
                        routed_label = res.choice
                        choice_counter[routed_label] += 1

                        chosen = None
                        for item in items:
                            if item["label"] == routed_label:
                                chosen = item
                                break
                        if chosen is None:
                            chosen = items[0]
                        routed_cluster = chosen["cluster"]
                        routed_box = chosen["box"]

                if routed_box is None:
                    add_metric(metrics_ann, "vlm_router", None, missing_pred=True)
                    add_metric(metrics_msk, "vlm_router", None, missing_pred=True, missing_gt=(gt_msk is None))
                    routed_iou_ann = None
                    routed_iou_msk = None
                else:
                    routed_box = clamp_xyxy(routed_box, W, H)
                    routed_iou_ann = iou_xyxy(routed_box, gt_ann)
                    add_metric(metrics_ann, "vlm_router", routed_iou_ann)
                    if gt_msk is None:
                        routed_iou_msk = None
                        add_metric(metrics_msk, "vlm_router", None, missing_gt=True)
                    else:
                        routed_iou_msk = iou_xyxy(routed_box, gt_msk)
                        add_metric(metrics_msk, "vlm_router", routed_iou_msk)

                if SAVE_PER_SAMPLE:
                    rec = {
                        "ref_id": int(ref_id),
                        "sent_id": int(sent_id),
                        "image_id": int(ref["image_id"]),
                        "expr": expr,
                        "gt_ann_xyxy": [float(x) for x in gt_ann],
                        "gt_mask_xyxy": [float(x) for x in gt_msk] if gt_msk is not None else None,
                        "expert_boxes": expert_boxes,
                        "expert_ious_ann": expert_ious_ann,
                        "expert_ious_mask": expert_ious_msk,
                        "clusters": [
                            {
                                "support": int(c["support"]),
                                "member_names": c["member_names"],
                                "dense": float(c["dense"]),
                                "mean_box": [float(x) for x in c["mean_box"]],
                            }
                            for c in clusters
                        ],
                        "agree_iou": float(agree_iou),
                        "judge_called": bool(judge_called_flag),
                        "skip_reason": skip_reason,
                        "routed_label": routed_label,
                        "routed_cluster_support": int(routed_cluster["support"]) if routed_cluster is not None else None,
                        "routed_cluster_members": routed_cluster["member_names"] if routed_cluster is not None else None,
                        "routed_box_xyxy": [float(x) for x in routed_box] if routed_box is not None else None,
                        "routed_iou_ann": float(routed_iou_ann) if routed_iou_ann is not None else None,
                        "routed_iou_mask": float(routed_iou_msk) if routed_iou_msk is not None else None,
                        "judge_raw": judge_raw,
                        "num_valid_experts": len(valid_boxes),
                        "num_clusters": len(clusters),
                    }
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    method_order = [cfg["name"] for cfg in EXPERTS] + ["oracle_best", "all_mean", "medoid", "vote_mean", "vlm_router"]
    summary = {
        "dataset": DATASET,
        "splitBy": SPLITBY,
        "split": SPLIT,
        "experts": EXPERTS,
        "num_refs": int(len(ref_ids)),
        "judge_called": int(judge_called),
        "judge_skipped": int(judge_skipped),
        "judge_call_rate": float(judge_called / (judge_called + judge_skipped + 1e-9)),
        "skip_reason_counter": dict(skip_reason_counter),
        "choice_counter": dict(choice_counter),
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

    with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if PRINT_TOPLINE:
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
        print("\n[INFO] Oracle win by expert (AnnBox):")
        for k, v in sorted(summary["oracle_win_by_expert_ann"].items(), key=lambda x: (-x[1], x[0])):
            print(f"  {k}: {v}")
        print("\n[INFO] Oracle win by expert (MaskBox):")
        for k, v in sorted(summary["oracle_win_by_expert_mask"].items(), key=lambda x: (-x[1], x[0])):
            print(f"  {k}: {v}")
        print(
            f"\n[INFO] judge_called={summary['judge_called']} skipped={summary['judge_skipped']} "
            f"call_rate={summary['judge_call_rate']:.3f}"
        )
        print(f"[INFO] skip reasons: {summary['skip_reason_counter']}")
        print(f"[INFO] label choices: {summary['choice_counter']}")
        print(f"[INFO] Summary saved to: {OUT_SUMMARY_JSON}")
        if SAVE_PER_SAMPLE:
            print(f"[INFO] Per-sample details saved to: {OUT_JSONL}")
        print("=" * 108)


if __name__ == "__main__":
    main()
