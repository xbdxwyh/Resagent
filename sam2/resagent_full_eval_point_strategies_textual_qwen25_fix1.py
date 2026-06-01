import argparse
import ast
import base64
import io
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from refer import REFER
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception:
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import Qwen3VLMoeForConditionalGeneration
except Exception:
    Qwen3VLMoeForConditionalGeneration = None

try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (THIS_DIR / "../..").resolve()
for _p in [str(THIS_DIR), str(PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# local helpers from your uploaded tools.py
from tools import (
    analyze_generation_logits,
    compute_ciou,
    compute_iou,
    convert_from_qwen2vl_format,
    extract_single_bounding_box,
    generate_external_candidate_points,
    generate_internal_candidate_points,
)

RAW_BOX_RE = re.compile(r"\[\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]\]")


# -----------------------------
# basic utils
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def image_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def clip_box_xyxy(box: Sequence[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(x1, float(w - 1)))
    y1 = max(0.0, min(y1, float(h - 1)))
    x2 = max(0.0, min(x2, float(w - 1)))
    y2 = max(0.0, min(y2, float(h - 1)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def is_valid_box_xyxy(box: Sequence[float], min_size: float = 2.0) -> bool:
    if box is None or len(box) != 4:
        return False
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x2 - x1) >= min_size and (y2 - y1) >= min_size


def xywh_to_xyxy(box_xywh: Sequence[float]) -> List[float]:
    x, y, w, h = [float(v) for v in box_xywh]
    return [x, y, x + w, y + h]


def maybe_normalized_box_to_absolute(box: Sequence[float], w: int, h: int, coord_mode: str = "auto") -> List[float]:
    box = [float(v) for v in box]
    if coord_mode == "absolute":
        return clip_box_xyxy(box, w, h)
    if coord_mode == "qwen1000":
        return convert_from_qwen2vl_format(box, h, w)

    # smarter auto:
    # 1) [0,1] normalized coords
    # 2) if the box already fits inside the image canvas, treat it as absolute pixel coords
    # 3) otherwise, if values are in [0,1000], treat it as Qwen-style 1000-scale coords
    mx = max(box)
    mn = min(box)
    x1, y1, x2, y2 = box
    if mx <= 1.5 and mn >= 0.0:
        return clip_box_xyxy([x1 * w, y1 * h, x2 * w, y2 * h], w, h)
    if mn >= 0.0 and x2 <= float(w) + 1.0 and y2 <= float(h) + 1.0:
        return clip_box_xyxy(box, w, h)
    if mn >= 0.0 and mx <= 1100.0:
        return convert_from_qwen2vl_format(box, h, w)
    return clip_box_xyxy(box, w, h)


def get_model_device(model: torch.nn.Module, fallback: torch.device) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return fallback


def parse_boxes_from_text(text: str) -> List[List[float]]:
    if not isinstance(text, str) or not text.strip():
        return []
    text = text.strip()

    def collect(o):
        out = []
        if isinstance(o, (list, tuple)) and len(o) == 4 and all(isinstance(x, (int, float)) for x in o):
            out.append([float(x) for x in o])
        elif isinstance(o, list):
            for item in o:
                out.extend(collect(item))
        elif isinstance(o, dict):
            for v in o.values():
                out.extend(collect(v))
        return out

    # 1) bare [[x1,y1,x2,y2]] or [x1,y1,x2,y2] without calling helper first,
    # to avoid noisy helper-side JSON parse logs.
    if text.startswith('[') or text.startswith('{'):
        try:
            obj = json.loads(text)
            boxes = collect(obj)
            if boxes:
                return boxes
        except Exception:
            pass

    # 2) direct regex fallback aligned with the old eval script
    m = RAW_BOX_RE.search(text)
    if m:
        return [[float(m.group(i)) for i in range(1, 5)]]

    nums = re.findall(r"-?\d+\.?\d*", text)
    if len(nums) >= 4:
        return [[float(nums[i]) for i in range(4)]]

    # 3) finally try the uploaded helper for nested dict/json cases with bbox keys.
    boxes: List[List[float]] = []
    try:
        parsed_boxes = extract_single_bounding_box(text)
        if parsed_boxes:
            for b in parsed_boxes:
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    boxes.append([float(x) for x in b])
    except Exception:
        pass
    return boxes


def infer_coord_mode_for_source(source_name: str, global_mode: str) -> str:
    if global_mode != "auto":
        return global_mode
    s = source_name.lower()
    if any(k in s for k in ["raw_text", "output_text"]):
        return "qwen1000"
    if any(k in s for k in ["pred_box_xyxy", "actual_bboxes", "gt_box", "box_xyxy"]):
        return "absolute"
    return "auto"


# -----------------------------
# model loading / generation
# -----------------------------


def identify_vlm_family(model: Any = None, processor: Any = None, model_path: str = "") -> str:
    names: List[str] = []
    for obj in [model, processor]:
        if obj is None:
            continue
        for attr in ["name_or_path", "model_name_or_path"]:
            try:
                v = getattr(obj, attr, None)
                if isinstance(v, str) and v:
                    names.append(v)
            except Exception:
                pass
        try:
            cfg = getattr(obj, "config", None)
            if cfg is not None:
                for attr in ["model_type", "architectures"]:
                    v = getattr(cfg, attr, None)
                    if isinstance(v, str) and v:
                        names.append(v)
                    elif isinstance(v, (list, tuple)):
                        names.extend([str(x) for x in v if x])
        except Exception:
            pass
        names.append(obj.__class__.__name__)
    if model_path:
        names.append(model_path)
    joined = " ".join(str(x).lower() for x in names if x)
    if "qwen3" in joined and "vl" in joined:
        return "qwen3_vl"
    if "qwen2.5" in joined and "vl" in joined:
        return "qwen2_5_vl"
    if "qwen2_5" in joined and "vl" in joined:
        return "qwen2_5_vl"
    if "qwen2" in joined and "vl" in joined:
        return "qwen2_vl"
    return "generic_vlm"


def build_chat_messages(image: Image.Image, text: str) -> List[Dict[str, Any]]:
    image_data_uri = f"data:image/jpeg;base64,{image_to_base64(image)}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_data_uri},
                {"type": "text", "text": text},
            ],
        }
    ]


def prepare_multimodal_inputs(
    model,
    processor,
    image: Image.Image,
    text: str,
) -> Tuple[Dict[str, Any], str]:
    family = identify_vlm_family(model=model, processor=processor)
    messages = build_chat_messages(image=image, text=text)
    model_device = get_model_device(model, torch.device("cpu"))
    last_err = None

    if family in {"qwen2_5_vl", "qwen3_vl"} and process_vision_info is not None:
        try:
            prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if family == "qwen3_vl":
                image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
            else:
                image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[prompt_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                do_resize=False,
                return_tensors="pt",
            )
            inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
            return inputs, family + "_qwen_vl_utils"
        except Exception as e:
            last_err = e

    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        return inputs, family + "_templated"
    except Exception as e:
        last_err = e

    try:
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
        inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        return inputs, family + "_processor"
    except Exception as e:
        last_err = e

    raise RuntimeError(f"Failed to prepare multimodal inputs for family={family}: {last_err}")


def load_qwen_model(model_path: str, device: torch.device):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model = None
    last_err = None
    family_hint = identify_vlm_family(model=None, processor=processor, model_path=model_path)
    path_l = str(model_path).lower()
    is_qwen3_moe = ("a3b" in path_l) or ("moe" in path_l)

    if family_hint == "qwen3_vl":
        # Important: regular Qwen3-VL checkpoints (2B/4B/8B) must try the non-MoE class first.
        # Trying the MoE loader on a dense checkpoint triggers the exact warning the user saw and can stall loading.
        load_order = [Qwen3VLForConditionalGeneration, AutoModelForVision2Seq]
        if is_qwen3_moe:
            load_order = [Qwen3VLMoeForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoModelForVision2Seq]
    elif family_hint == "qwen2_5_vl":
        load_order = [Qwen2_5_VLForConditionalGeneration, AutoModelForVision2Seq]
    else:
        load_order = [AutoModelForVision2Seq, Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration]
        if is_qwen3_moe:
            load_order = [Qwen3VLMoeForConditionalGeneration] + load_order

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float16 if device.type == "mps" else torch.float32
    device_map = "auto" if device.type == "cuda" else None

    load_errors = []
    for cls in load_order:
        if cls is None:
            continue
        try:
            print(f"Trying VLM loader: {cls.__name__} (family_hint={family_hint}, device_map={device_map})")
            model = cls.from_pretrained(
                model_path,
                dtype=dtype,
                trust_remote_code=True,
                device_map=device_map,
            )
            print(f"Loaded VLM with: {cls.__name__}")
            break
        except Exception as e:
            last_err = e
            load_errors.append(f"{getattr(cls, '__name__', str(cls))}: {repr(e)}")

    if model is None:
        joined = " | ".join(load_errors) if load_errors else str(last_err)
        raise RuntimeError(f"Failed to load VLM from {model_path}. Load attempts: {joined}")

    if device.type != "cuda":
        model = model.to(device)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("processor.tokenizer is missing; please adapt the loader to your local transformers version.")
    return model.eval(), processor, tokenizer


def run_chat_query_with_details(
    model,
    processor,
    tokenizer,
    image: Image.Image,
    text: str,
    max_new_tokens: int = 256,
    do_sample: bool = False,
) -> Dict[str, Any]:
    inputs, input_mode = prepare_multimodal_inputs(
        model=model,
        processor=processor,
        image=image,
        text=text,
    )
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        output_scores=True,
        return_dict_in_generate=True,
    )
    generated_ids = outputs.sequences
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    text_out = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    try:
        analysis_inputs = inputs
        if isinstance(inputs, dict):
            analysis_inputs = SimpleNamespace(input_ids=inputs["input_ids"])
        analysis_results = analyze_generation_logits(outputs, analysis_inputs, tokenizer)
    except Exception:
        analysis_results = []
    token_text = "".join([str(x.get("token", "")) for x in analysis_results])
    return {
        "text": text_out[0],
        "analysis_results": analysis_results,
        "token_text": token_text,
        "input_mode": input_mode,
    }


def run_chat_query(
    model,
    processor,
    image: Image.Image,
    text: str,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    tokenizer=None,
) -> str:
    if tokenizer is None:
        tokenizer = getattr(processor, "tokenizer", None)
    return run_chat_query_with_details(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        image=image,
        text=text,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )["text"]



# -----------------------------
# sentence / dataset helpers
# -----------------------------

def choose_sentence_entry(refs: Dict[str, Any], sentence_mode: str) -> Optional[Dict[str, Any]]:
    sent_entries = [s for s in refs.get("sentences", []) if isinstance(s, dict) and s.get("sent")]
    if not sent_entries:
        return None
    if sentence_mode == "first":
        return sent_entries[0]
    if sentence_mode == "longest":
        return max(sent_entries, key=lambda x: len(x.get("sent", "").strip()))
    if sentence_mode == "shortest":
        return min(sent_entries, key=lambda x: len(x.get("sent", "").strip()))
    if sentence_mode == "random":
        return random.choice(sent_entries)
    return sent_entries[0]


def resolve_sentence(refs: Dict[str, Any], predict_info: Optional[Dict[str, Any]], sentence_mode: str) -> str:
    candidate_keys = [
        "sentence",
        "sent",
        "expr",
        "expression",
        "query",
        "text",
        "referring_expression",
    ]
    if predict_info is not None:
        for k in candidate_keys:
            if k in predict_info and isinstance(predict_info[k], str) and predict_info[k].strip():
                return predict_info[k].strip()

    sent_entry = choose_sentence_entry(refs, sentence_mode)
    if sent_entry is not None:
        return sent_entry.get("sent", "").strip()

    sents = [s["sent"].strip() for s in refs.get("sentences", []) if s.get("sent")]
    if not sents:
        return ""
    if sentence_mode == "join":
        return " | ".join(sents)
    return sents[0]


def load_refer(data_root: str, task: str) -> REFER:
    if task in {"refcoco", "refcoco+"}:
        source = "unc"
    elif task == "refcocog":
        source = "umd"
    else:
        raise ValueError(f"Unsupported task: {task}")
    return REFER(data_root, task, source)


# -----------------------------
# bbox prior resolution
# -----------------------------

def _collect_boxes_recursive(obj: Any, key_path: str = "") -> List[Tuple[str, List[float]]]:
    results: List[Tuple[str, List[float]]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{key_path}.{k}" if key_path else k
            k_lower = k.lower()
            if isinstance(v, (list, tuple)) and len(v) == 4 and any(tag in k_lower for tag in ["box", "bbox"]):
                try:
                    vv = [float(x) for x in v]
                    results.append((path, vv))
                except Exception:
                    pass
            else:
                results.extend(_collect_boxes_recursive(v, path))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            results.extend(_collect_boxes_recursive(item, f"{key_path}[{i}]"))
    return results


def gather_box_candidates(
    predict_info: Optional[Dict[str, Any]],
    image_size: Tuple[int, int],
    box_fields: Sequence[str],
    coord_mode: str,
) -> List[Dict[str, Any]]:
    w, h = image_size
    candidates: List[Dict[str, Any]] = []
    seen = set()
    if predict_info is None:
        return candidates

    # 1) explicit fields
    for field in box_fields:
        if field in predict_info:
            value = predict_info[field]
            field_coord_mode = infer_coord_mode_for_source(field, coord_mode)
            if isinstance(value, (list, tuple)) and len(value) == 4 and all(isinstance(x, (int, float)) for x in value):
                box = maybe_normalized_box_to_absolute(value, w, h, field_coord_mode)
                key = tuple(round(v, 2) for v in box)
                if is_valid_box_xyxy(box) and key not in seen:
                    seen.add(key)
                    candidates.append({"source": field, "box": box})
            elif isinstance(value, str):
                parsed = parse_boxes_from_text(value)
                for j, box_raw in enumerate(parsed):
                    box = maybe_normalized_box_to_absolute(box_raw, w, h, field_coord_mode)
                    key = tuple(round(v, 2) for v in box)
                    if is_valid_box_xyxy(box) and key not in seen:
                        seen.add(key)
                        candidates.append({"source": f"{field}[{j}]", "box": box})
            elif isinstance(value, (list, tuple)):
                for j, item in enumerate(value):
                    if isinstance(item, (list, tuple)) and len(item) == 4 and all(isinstance(x, (int, float)) for x in item):
                        item_coord_mode = infer_coord_mode_for_source(f"{field}[{j}]", coord_mode)
                        box = maybe_normalized_box_to_absolute(item, w, h, item_coord_mode)
                        key = tuple(round(v, 2) for v in box)
                        if is_valid_box_xyxy(box) and key not in seen:
                            seen.add(key)
                            candidates.append({"source": f"{field}[{j}]", "box": box})
                    elif isinstance(item, str):
                        parsed = parse_boxes_from_text(item)
                        for k, box_raw in enumerate(parsed):
                            item_coord_mode = infer_coord_mode_for_source(f"{field}[{j}]", coord_mode)
                            box = maybe_normalized_box_to_absolute(box_raw, w, h, item_coord_mode)
                            key = tuple(round(v, 2) for v in box)
                            if is_valid_box_xyxy(box) and key not in seen:
                                seen.add(key)
                                candidates.append({"source": f"{field}[{j}][{k}]", "box": box})

    # 2) recursive search over keys containing box/bbox
    for source, box_raw in _collect_boxes_recursive(predict_info):
        source_coord_mode = infer_coord_mode_for_source(source, coord_mode)
        box = maybe_normalized_box_to_absolute(box_raw, w, h, source_coord_mode)
        key = tuple(round(v, 2) for v in box)
        if is_valid_box_xyxy(box) and key not in seen:
            seen.add(key)
            candidates.append({"source": source, "box": box})

    return candidates


def choose_voted_box(candidates: List[Dict[str, Any]], metric: str = "iou") -> Dict[str, Any]:
    assert candidates, "No bbox candidates for voting."
    n = len(candidates)
    if n == 1:
        return {
            "chosen_box": candidates[0]["box"],
            "chosen_source": candidates[0]["source"],
            "vote_scores": [1.0],
            "pairwise": [[1.0]],
        }

    pairwise = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            b1 = candidates[i]["box"]
            b2 = candidates[j]["box"]
            if metric == "ciou":
                score = float(compute_ciou(b1, b2))
            else:
                score = float(compute_iou(b1, b2))
            pairwise[i, j] = score
            pairwise[j, i] = score

    vote_scores = pairwise.mean(axis=1)
    best_idx = int(np.argmax(vote_scores))
    return {
        "chosen_box": candidates[best_idx]["box"],
        "chosen_source": candidates[best_idx]["source"],
        "vote_scores": vote_scores.tolist(),
        "pairwise": pairwise.tolist(),
    }


def resolve_box_prior(
    args,
    predict_info: Optional[Dict[str, Any]],
    image: Image.Image,
    sentence: str,
    model_bundle: Optional[Tuple[Any, Any, Any]],
) -> Dict[str, Any]:
    w, h = image.size
    fields = [x.strip() for x in args.box_fields.split(",") if x.strip()]
    candidates = gather_box_candidates(predict_info, (w, h), fields, args.box_coord_mode)

    # Optional live bbox inference as final fallback.
    live_box = None
    live_raw = None
    if not candidates and model_bundle is not None and args.allow_live_bbox_fallback:
        model, processor, tokenizer = model_bundle
        live_prompt = (
            f"Identify the target referred to by '{sentence}' in the image and return its bounding box in JSON format. "
            f"Use one box only."
        )
        live_raw = run_chat_query(model, processor, image, live_prompt, max_new_tokens=128, tokenizer=getattr(processor, "tokenizer", None))
        live_boxes = extract_single_bounding_box(live_raw)
        if live_boxes:
            live_box = maybe_normalized_box_to_absolute(live_boxes[0], w, h, args.box_coord_mode)
            if is_valid_box_xyxy(live_box):
                candidates.append({"source": "live_bbox", "box": live_box})

    if not candidates:
        # hard fallback: full image
        chosen_box = [0.0, 0.0, float(w - 1), float(h - 1)]
        return {
            "mode": "full_image_fallback",
            "chosen_box": chosen_box,
            "chosen_source": "full_image_fallback",
            "candidates": [],
            "vote_scores": None,
            "pairwise": None,
            "live_raw": live_raw,
        }

    if args.box_prior_mode == "fixed":
        fixed_field = args.fixed_box_field.strip()
        chosen = None
        if fixed_field:
            for item in candidates:
                if item["source"] == fixed_field or item["source"].startswith(fixed_field + "."):
                    chosen = item
                    break
        if chosen is None:
            chosen = candidates[0]
        return {
            "mode": "fixed",
            "chosen_box": chosen["box"],
            "chosen_source": chosen["source"],
            "candidates": candidates,
            "vote_scores": None,
            "pairwise": None,
            "live_raw": live_raw,
        }

    if args.box_prior_mode == "precomputed_vote":
        pre_field = args.precomputed_vote_field.strip()
        chosen = None
        if predict_info is not None and pre_field and pre_field in predict_info:
            box = maybe_normalized_box_to_absolute(predict_info[pre_field], w, h, args.box_coord_mode)
            if is_valid_box_xyxy(box):
                chosen = {"source": pre_field, "box": box}
        if chosen is None:
            chosen = candidates[0]
        return {
            "mode": "precomputed_vote",
            "chosen_box": chosen["box"],
            "chosen_source": chosen["source"],
            "candidates": candidates,
            "vote_scores": None,
            "pairwise": None,
            "live_raw": live_raw,
        }

    # vote / auto / vote_or_fixed
    if args.box_prior_mode in {"vote", "auto", "vote_or_fixed"}:
        if len(candidates) >= 2:
            vote_info = choose_voted_box(candidates, metric=args.box_vote_metric)
            return {
                "mode": "vote",
                "chosen_box": vote_info["chosen_box"],
                "chosen_source": vote_info["chosen_source"],
                "candidates": candidates,
                "vote_scores": vote_info["vote_scores"],
                "pairwise": vote_info["pairwise"],
                "live_raw": live_raw,
            }
        chosen = candidates[0]
        return {
            "mode": "single_candidate_fixed",
            "chosen_box": chosen["box"],
            "chosen_source": chosen["source"],
            "candidates": candidates,
            "vote_scores": [1.0],
            "pairwise": [[1.0]],
            "live_raw": live_raw,
        }

    raise ValueError(f"Unsupported box prior mode: {args.box_prior_mode}")


# -----------------------------
# point strategy helpers (spiral / random / grid / learned)
# -----------------------------

def make_per_ref_rng(global_seed: int, ref_id: Any, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng(int(global_seed) + int(ref_id) * 1000003 + int(salt))


def dedupe_points(points: Sequence[Sequence[float]], w: int, h: int) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    seen = set()
    for p in points:
        if p is None or len(p) != 2:
            continue
        x = float(np.clip(float(p[0]), 0.0, float(w - 1)))
        y = float(np.clip(float(p[1]), 0.0, float(h - 1)))
        key = (int(round(x)), int(round(y)))
        if key in seen:
            continue
        seen.add(key)
        out.append((x, y))
    return out


def sample_points_uniform_in_box(box_xyxy: Sequence[float], n: int, w: int, h: int, rng: np.random.Generator) -> List[Tuple[float, float]]:
    if n <= 0:
        return []
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    x1i = int(max(0, min(w - 1, np.floor(x1))))
    x2i = int(max(0, min(w - 1, np.ceil(x2))))
    y1i = int(max(0, min(h - 1, np.floor(y1))))
    y2i = int(max(0, min(h - 1, np.ceil(y2))))
    if x2i <= x1i:
        x2i = min(w - 1, x1i + 1)
    if y2i <= y1i:
        y2i = min(h - 1, y1i + 1)
    xs = rng.integers(x1i, x2i + 1, size=(n,))
    ys = rng.integers(y1i, y2i + 1, size=(n,))
    return dedupe_points([(float(x), float(y)) for x, y in zip(xs, ys)], w, h)


def sample_points_random_outside_box(
    box_xyxy: Sequence[float],
    n: int,
    w: int,
    h: int,
    rng: np.random.Generator,
    expand_ratio: float = 0.25,
) -> List[Tuple[float, float]]:
    if n <= 0:
        return []
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    ex1 = max(0.0, x1 - bw * expand_ratio)
    ey1 = max(0.0, y1 - bh * expand_ratio)
    ex2 = min(float(w - 1), x2 + bw * expand_ratio)
    ey2 = min(float(h - 1), y2 + bh * expand_ratio)
    pts: List[Tuple[float, float]] = []
    trials = 0
    max_trials = max(64, n * 40)
    while len(pts) < n and trials < max_trials:
        trials += 1
        px = float(rng.uniform(ex1, ex2))
        py = float(rng.uniform(ey1, ey2))
        if (x1 <= px <= x2) and (y1 <= py <= y2):
            continue
        pts.append((px, py))
    if len(pts) < n:
        pts.extend(build_grid_points_outside_box(box_xyxy, n - len(pts), w, h, expand_ratio=expand_ratio))
    return dedupe_points(pts, w, h)[:n]


def build_grid_points_in_box(box_xyxy: Sequence[float], n: int, w: int, h: int) -> List[Tuple[float, float]]:
    if n <= 0:
        return []
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cols = max(1, int(math.ceil(math.sqrt(n * (bw / max(bh, 1.0))))))
    rows = max(1, int(math.ceil(n / cols)))
    xs = np.linspace(x1 + bw / (cols + 1), x2 - bw / (cols + 1), num=cols) if cols > 1 else np.array([(x1 + x2) / 2.0])
    ys = np.linspace(y1 + bh / (rows + 1), y2 - bh / (rows + 1), num=rows) if rows > 1 else np.array([(y1 + y2) / 2.0])
    pts = [(float(x), float(y)) for y in ys for x in xs]
    return dedupe_points(pts, w, h)[:n]


def build_grid_points_outside_box(box_xyxy: Sequence[float], n: int, w: int, h: int, expand_ratio: float = 0.15) -> List[Tuple[float, float]]:
    if n <= 0:
        return []
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    mx = max(2.0, bw * expand_ratio)
    my = max(2.0, bh * expand_ratio)
    top_y = max(0.0, y1 - my)
    bot_y = min(float(h - 1), y2 + my)
    left_x = max(0.0, x1 - mx)
    right_x = min(float(w - 1), x2 + mx)
    per_edge = max(1, int(math.ceil(n / 4.0)))
    xs = np.linspace(left_x, right_x, num=per_edge)
    ys = np.linspace(top_y, bot_y, num=per_edge)
    pts: List[Tuple[float, float]] = []
    pts.extend((float(x), float(top_y)) for x in xs)
    pts.extend((float(x), float(bot_y)) for x in xs)
    pts.extend((float(left_x), float(y)) for y in ys)
    pts.extend((float(right_x), float(y)) for y in ys)
    filtered = []
    for px, py in pts:
        if (x1 <= px <= x2) and (y1 <= py <= y2):
            continue
        filtered.append((px, py))
    return dedupe_points(filtered, w, h)[:n]


def build_candidates_by_strategy(
    strategy: str,
    box_xyxy: Sequence[float],
    image_size: Tuple[int, int],
    internal_n: int,
    external_n: int,
    rng: np.random.Generator,
    random_external_expand: float = 0.25,
    grid_external_expand: float = 0.15,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], Dict[str, Any]]:
    w, h = image_size
    strategy = str(strategy).lower()
    if strategy == "spiral":
        internal = dedupe_points(generate_internal_candidate_points(box_xyxy, internal_n), w, h)[:internal_n]
        external = dedupe_points(generate_external_candidate_points(box_xyxy, external_n), w, h)[:external_n]
    elif strategy == "random":
        internal = sample_points_uniform_in_box(box_xyxy, internal_n, w, h, rng)
        if len(internal) < internal_n:
            internal.extend(sample_points_uniform_in_box(box_xyxy, internal_n - len(internal), w, h, rng))
            internal = dedupe_points(internal, w, h)[:internal_n]
        external = sample_points_random_outside_box(box_xyxy, external_n, w, h, rng, expand_ratio=random_external_expand)
    elif strategy == "grid":
        internal = build_grid_points_in_box(box_xyxy, internal_n, w, h)
        external = build_grid_points_outside_box(box_xyxy, external_n, w, h, expand_ratio=grid_external_expand)
    else:
        raise ValueError(f"Unsupported point strategy: {strategy}")
    return internal, external, {
        "strategy": strategy,
        "internal_count": len(internal),
        "external_count": len(external),
    }


def _get_model_image_size(predictor: SAM2ImagePredictor) -> int:
    if hasattr(predictor, "model") and hasattr(predictor.model, "image_size"):
        return int(predictor.model.image_size)
    if hasattr(predictor, "_image_size"):
        return int(predictor._image_size)
    return 1024


def _get_image_embedding(predictor: SAM2ImagePredictor) -> torch.Tensor:
    for name in ["_image_embedding", "image_embedding", "_features", "_image_features", "_cached_features"]:
        if hasattr(predictor, name):
            obj = getattr(predictor, name)
            if torch.is_tensor(obj):
                return obj
            if isinstance(obj, dict):
                for k in ["image_embed", "image_embedding", "img_embed", "x"]:
                    if k in obj and torch.is_tensor(obj[k]):
                        return obj[k]
    for name in ["image_embed", "image_embedding", "img_embed"]:
        if hasattr(predictor.model, name):
            t = getattr(predictor.model, name)
            if torch.is_tensor(t):
                return t
    raise RuntimeError("Cannot find image embedding on SAM2 predictor.")


def _find_prompt_encoder(model: torch.nn.Module):
    for name in ["prompt_encoder", "sam_prompt_encoder", "prompt_encoder_model"]:
        if hasattr(model, name):
            return getattr(model, name)
    raise RuntimeError("Cannot find prompt encoder on SAM2 model.")


@torch.no_grad()
def extract_point_features_sam2(
    predictor: SAM2ImagePredictor,
    images_np: List[np.ndarray],
    points_xy: torch.Tensor,
    orig_hw: List[Tuple[int, int]],
    device: torch.device,
    return_embed: bool = False,
):
    if hasattr(predictor, "set_image_batch"):
        predictor.set_image_batch(images_np)
        img_embed = _get_image_embedding(predictor)
        if img_embed.dim() == 3:
            img_embed = img_embed.unsqueeze(0)
    else:
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
            coords = predictor._transforms.transform_coords(points_xy[i].to(device), normalize=True, orig_hw=orig_hw[i])
            x_ = coords[:, 0] / float(image_size) * 2.0 - 1.0
            y_ = coords[:, 1] / float(image_size) * 2.0 - 1.0
        else:
            H0, W0 = orig_hw[i]
            x_ = (points_xy[i, :, 0].to(device) / max(W0 - 1, 1)) * 2.0 - 1.0
            y_ = (points_xy[i, :, 1].to(device) / max(H0 - 1, 1)) * 2.0 - 1.0
        grids.append(torch.stack([x_, y_], dim=-1))

    grid = torch.stack(grids, dim=0).view(B, -1, 1, 2)
    sampled = F.grid_sample(img_embed, grid, mode="bilinear", align_corners=True)
    sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()
    if return_embed:
        return sampled, img_embed
    return sampled


@torch.no_grad()
def extract_point_prompt_embeddings_sam2(
    predictor: SAM2ImagePredictor,
    points_xy: torch.Tensor,
    point_labels: torch.Tensor,
    orig_hw: List[Tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    pe = _find_prompt_encoder(predictor.model)
    image_size = _get_model_image_size(predictor)
    coords_list = []
    for i in range(points_xy.shape[0]):
        if hasattr(predictor, "_transforms") and hasattr(predictor._transforms, "transform_coords"):
            coords = predictor._transforms.transform_coords(points_xy[i].to(device), normalize=True, orig_hw=orig_hw[i])
        else:
            H0, W0 = orig_hw[i]
            coords = points_xy[i].to(device).clone().float()
            coords[:, 0] = coords[:, 0] / max(W0 - 1, 1) * float(image_size)
            coords[:, 1] = coords[:, 1] / max(H0 - 1, 1) * float(image_size)
        coords_list.append(coords)
    coords_b = torch.stack(coords_list, dim=0)
    labels_b = point_labels.to(device).long()
    try:
        sparse, dense = pe(points=(coords_b, labels_b), boxes=None, masks=None)
    except TypeError:
        sparse, dense = pe(points=(coords_b, labels_b), boxes=None)
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
    boxes_xyxy: torch.Tensor,
    orig_hw: List[Tuple[int, int]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pe = _find_prompt_encoder(predictor.model)
    image_size = _get_model_image_size(predictor)
    B = boxes_xyxy.shape[0]
    boxes_list = []
    for i in range(B):
        b = boxes_xyxy[i].to(device).float()
        if hasattr(predictor, "_transforms") and hasattr(predictor._transforms, "transform_boxes"):
            bb = predictor._transforms.transform_boxes(b[None, :], normalize=True, orig_hw=orig_hw[i]).squeeze(0)
        else:
            H0, W0 = orig_hw[i]
            bb = b.clone()
            bb[0] = bb[0] / max(W0 - 1, 1) * float(image_size)
            bb[2] = bb[2] / max(W0 - 1, 1) * float(image_size)
            bb[1] = bb[1] / max(H0 - 1, 1) * float(image_size)
            bb[3] = bb[3] / max(H0 - 1, 1) * float(image_size)
        boxes_list.append(bb)
    boxes_b = torch.stack(boxes_list, dim=0)
    boxes_corners = boxes_b.view(B, 2, 2)
    try:
        sparse, dense = pe(points=None, boxes=boxes_corners, masks=None)
    except TypeError:
        sparse, dense = pe(points=None, boxes=boxes_corners)
    return sparse.to(device), dense.to(device)


@torch.no_grad()
def predict_box_only_mask_probs_sam2(
    predictor: SAM2ImagePredictor,
    images_np: List[np.ndarray],
    boxes_xyxy: torch.Tensor,
    orig_hw: List[Tuple[int, int]],
    device: torch.device,
) -> List[torch.Tensor]:
    probs: List[torch.Tensor] = []
    B = boxes_xyxy.shape[0]

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
        if out is not None and isinstance(out, (list, tuple)) and len(out) >= 1:
            masks = out[0]
            logits = out[2] if (len(out) >= 3 and out[2] is not None) else None
            for i in range(B):
                if logits is not None:
                    lg = logits[i]
                    lg = torch.from_numpy(lg) if isinstance(lg, np.ndarray) else lg
                    if torch.is_tensor(lg) and lg.dim() == 3:
                        lg = lg[0]
                    prob = torch.sigmoid(lg.float())
                else:
                    mk = masks[i]
                    mk = torch.from_numpy(mk) if isinstance(mk, np.ndarray) else mk
                    if torch.is_tensor(mk) and mk.dim() == 3:
                        mk = mk[0]
                    prob = mk.float()
                probs.append(prob.to(device))
            if len(probs) == B:
                return probs

    for i in range(B):
        predictor.set_image(images_np[i])
        box0 = boxes_xyxy[i].detach().cpu().numpy().astype(np.float32)
        box_candidates = [box0, box0[None, :], box0.reshape(2, 2), box0.reshape(1, 2, 2)]
        prob = None
        last_err = None
        for bc in box_candidates:
            try:
                out = predictor.predict(box=bc, point_coords=None, point_labels=None, multimask_output=False, return_logits=True)
                if isinstance(out, (list, tuple)) and len(out) >= 3:
                    logits = out[2]
                else:
                    logits = out
                lg = logits[0] if isinstance(logits, (list, tuple)) else logits
                lg = torch.from_numpy(lg) if isinstance(lg, np.ndarray) else lg
                if torch.is_tensor(lg) and lg.dim() == 3:
                    lg = lg[0]
                prob = torch.sigmoid(lg.float())
                break
            except Exception as e:
                last_err = e
            try:
                out = predictor.predict(box=bc, point_coords=None, point_labels=None, multimask_output=False)
                masks = out[0] if isinstance(out, (list, tuple)) else out
                mk = masks[0] if isinstance(masks, (list, tuple)) else masks
                mk = torch.from_numpy(mk) if isinstance(mk, np.ndarray) else mk
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
def sample_mask_at_points(mask_probs: List[torch.Tensor], points_xy: torch.Tensor, device: torch.device) -> torch.Tensor:
    B, N, _ = points_xy.shape
    out_list = []
    for i in range(B):
        m = mask_probs[i].to(device).float()
        H0, W0 = m.shape
        m4 = m.view(1, 1, H0, W0)
        x = points_xy[i, :, 0].to(device)
        y = points_xy[i, :, 1].to(device)
        gx = (x / max(W0 - 1, 1)) * 2.0 - 1.0
        gy = (y / max(H0 - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(1, N, 1, 2)
        out_list.append(F.grid_sample(m4, grid, mode="bilinear", align_corners=True).view(N, 1))
    return torch.stack(out_list, dim=0)


def _make_2d_sincos_pos_embed(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    if dim % 4 != 0:
        return torch.zeros((h * w, dim), device=device)
    y = torch.arange(h, device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
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
    return torch.cat([pos_y, pos_x], dim=1)


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
        self.head_u = nn.Linear(hidden_dim, 1)
        self.head_v = nn.Linear(hidden_dim, 1)

    def forward(self, x_point: torch.Tensor, **kwargs):
        h = self.trunk(x_point)
        return self.head_u(h).squeeze(-1), self.head_v(h).squeeze(-1)


class UsefulValueDecoderHead(nn.Module):
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

    def forward(self, x_point: torch.Tensor, *, img_embed: torch.Tensor, box_sparse: Optional[torch.Tensor] = None, box_dense: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None, **kwargs):
        B, N, _ = x_point.shape
        device = x_point.device
        mem = img_embed
        if self.mem_downsample > 1:
            mem = F.avg_pool2d(mem, kernel_size=self.mem_downsample, stride=self.mem_downsample)
        Bm, C, Hm, Wm = mem.shape
        mem_tokens = self.img_proj(mem.permute(0, 2, 3, 1).reshape(Bm, Hm * Wm, C))
        if box_dense is not None:
            bd = box_dense
            if self.mem_downsample > 1:
                bd = F.avg_pool2d(bd, kernel_size=self.mem_downsample, stride=self.mem_downsample)
            if bd.shape[-2:] != (Hm, Wm):
                bd = F.interpolate(bd, size=(Hm, Wm), mode="bilinear", align_corners=False)
            bd_tokens = bd.permute(0, 2, 3, 1).reshape(Bm, Hm * Wm, bd.shape[1])
            mem_tokens = mem_tokens + self.box_dense_proj(bd_tokens)
        if self.use_mem_pos:
            mem_tokens = mem_tokens + _make_2d_sincos_pos_embed(Hm, Wm, self.d_model, device).unsqueeze(0)
        tgt_point = self.point_proj(x_point)
        Tb = 0
        tgt = tgt_point
        key_pad = tgt_key_padding_mask
        if box_sparse is not None:
            Tb = box_sparse.shape[1]
            tgt_box = self.box_sparse_proj(box_sparse)
            tgt = torch.cat([tgt_box, tgt_point], dim=1)
            if tgt_key_padding_mask is not None:
                pad_box = torch.zeros((B, Tb), device=device, dtype=torch.bool)
                key_pad = torch.cat([pad_box, tgt_key_padding_mask], dim=1)
        out = self.decoder(tgt=tgt, memory=mem_tokens, tgt_key_padding_mask=key_pad)
        out_p = out[:, Tb:, :] if Tb > 0 else out
        return self.head_u(out_p).squeeze(-1), self.head_v(out_p).squeeze(-1)


class LearnedPointScorer:
    def __init__(self, predictor: SAM2ImagePredictor, device: torch.device, ckpt_path: str, config_json: str = "", use_ema: bool = False, score_mode: str = "useful_value"):
        self.predictor = predictor
        self.device = device
        self.ckpt_path = ckpt_path
        self.config_json = config_json or os.path.join(os.path.dirname(ckpt_path), "config.json")
        self.use_ema = use_ema
        self.score_mode = score_mode
        self.cfg: Dict[str, Any] = {}
        if self.config_json and os.path.exists(self.config_json):
            try:
                with open(self.config_json, "r", encoding="utf-8") as f:
                    self.cfg = json.load(f)
            except Exception:
                self.cfg = {}
        self.use_prompt_embed = bool(self.cfg.get("use_prompt_embed", False))
        self.use_box_prompt = bool(self.cfg.get("use_box_prompt", False))
        self.use_base_mask = bool(self.cfg.get("use_base_mask", False))
        self.head_type = str(self.cfg.get("head_type", "decoder"))
        self.hidden_dim = int(self.cfg.get("hidden_dim", 512))
        self.dropout = float(self.cfg.get("dropout", 0.1))
        self.decoder_dim = int(self.cfg.get("decoder_dim", 256))
        self.decoder_layers = int(self.cfg.get("decoder_layers", 2))
        self.decoder_nhead = int(self.cfg.get("decoder_nhead", 8))
        self.decoder_ffn_mult = int(self.cfg.get("decoder_ffn_mult", 4))
        self.decoder_mem_downsample = int(self.cfg.get("decoder_mem_downsample", 1))
        self.decoder_use_mem_pos = bool(self.cfg.get("decoder_use_mem_pos", False))
        self.head: Optional[nn.Module] = None
        self.load_info: Dict[str, Any] = {}

    def _build_head(self, point_dim: int, img_in_dim: int, box_sparse_dim: int, box_dense_dim: int) -> nn.Module:
        if self.head_type == "mlp":
            return UsefulValueMLPHead(point_in_dim=point_dim, hidden_dim=self.hidden_dim, dropout=self.dropout).to(self.device)
        return UsefulValueDecoderHead(
            point_in_dim=point_dim,
            img_in_dim=img_in_dim,
            box_sparse_dim=box_sparse_dim,
            box_dense_dim=box_dense_dim,
            d_model=self.decoder_dim,
            nhead=self.decoder_nhead,
            num_layers=self.decoder_layers,
            ffn_mult=self.decoder_ffn_mult,
            dropout=self.dropout,
            mem_downsample=self.decoder_mem_downsample,
            use_mem_pos=self.decoder_use_mem_pos,
        ).to(self.device)

    def _ensure_head(self, point_dim: int, img_in_dim: int, box_sparse_dim: int, box_dense_dim: int) -> None:
        if self.head is not None:
            return
        self.head = self._build_head(point_dim, img_in_dim, box_sparse_dim, box_dense_dim)
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        if self.use_ema and isinstance(ckpt, dict) and "head_ema" in ckpt:
            state = ckpt["head_ema"]
        elif isinstance(ckpt, dict) and "head" in ckpt:
            state = ckpt["head"]
        elif isinstance(ckpt, dict):
            state = ckpt
        else:
            raise RuntimeError(f"Unexpected learned checkpoint format: {type(ckpt)}")
        missing, unexpected = self.head.load_state_dict(state, strict=False)
        self.head.eval()
        self.load_info = {
            "ckpt_path": self.ckpt_path,
            "config_json": self.config_json if self.config_json and os.path.exists(self.config_json) else None,
            "use_ema": self.use_ema,
            "missing": list(missing),
            "unexpected": list(unexpected),
        }

    @torch.no_grad()
    def score_points(self, image: Image.Image, box_xyxy: Sequence[float], points: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
        if not points:
            return []
        image_np = np.array(image)
        orig_hw = [(image.height, image.width)]
        points_tensor = torch.tensor(points, dtype=torch.float32, device=self.device).unsqueeze(0)
        boxes_tensor = torch.tensor(box_xyxy, dtype=torch.float32, device=self.device).unsqueeze(0)
        vis, img_embed = extract_point_features_sam2(self.predictor, [image_np], points_tensor, orig_hw, self.device, return_embed=True)
        point_feat = vis
        if self.use_prompt_embed:
            ones = torch.ones((1, points_tensor.shape[1]), dtype=torch.float32, device=self.device)
            zeros = torch.zeros((1, points_tensor.shape[1]), dtype=torch.float32, device=self.device)
            pe_pos = extract_point_prompt_embeddings_sam2(self.predictor, points_tensor, ones, orig_hw, self.device)
            pe_neg = extract_point_prompt_embeddings_sam2(self.predictor, points_tensor, zeros, orig_hw, self.device)
            point_feat = torch.cat([point_feat, pe_pos, pe_neg], dim=-1)
        if self.use_base_mask:
            mask_probs = predict_box_only_mask_probs_sam2(self.predictor, [image_np], boxes_tensor, orig_hw, self.device)
            point_feat = torch.cat([point_feat, sample_mask_at_points(mask_probs, points_tensor, self.device)], dim=-1)
        box_sparse = None
        box_dense = None
        box_sparse_dim = 1
        box_dense_dim = 1
        if self.use_box_prompt:
            box_sparse, box_dense = extract_box_prompt_embeddings_sam2(self.predictor, boxes_tensor, orig_hw, self.device)
            box_sparse_dim = int(box_sparse.shape[-1])
            box_dense_dim = int(box_dense.shape[1])
        self._ensure_head(int(point_feat.shape[-1]), int(img_embed.shape[1]), box_sparse_dim, box_dense_dim)
        valid = torch.ones((1, points_tensor.shape[1]), dtype=torch.bool, device=self.device)
        u_logit, v_hat_z = self.head(point_feat, img_embed=img_embed, box_sparse=box_sparse, box_dense=box_dense, tgt_key_padding_mask=(~valid))
        u_prob = torch.sigmoid(u_logit[0])
        value_term = F.softplus(v_hat_z[0])
        if self.score_mode == "useful":
            score = u_prob
        elif self.score_mode == "value":
            score = value_term
        else:
            score = u_prob * value_term
        out = []
        for i, pt in enumerate(points):
            out.append({
                "point": [float(pt[0]), float(pt[1])],
                "score": float(score[i].detach().cpu()),
                "u_prob": float(u_prob[i].detach().cpu()),
                "value_term": float(value_term[i].detach().cpu()),
            })
        out.sort(key=lambda z: z["score"], reverse=True)
        return out


def build_point_candidates(args, ref_id: Any, image: Image.Image, box_xyxy: Sequence[float], learned_scorer: Optional[LearnedPointScorer] = None) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], Dict[str, Any]]:
    w, h = image.size
    strategy = str(args.point_strategy).lower()
    rng = make_per_ref_rng(args.seed, ref_id, salt=97)
    if strategy in {"spiral", "random", "grid"}:
        ints, exts, dbg = build_candidates_by_strategy(
            strategy=strategy,
            box_xyxy=box_xyxy,
            image_size=(w, h),
            internal_n=args.internal_candidates,
            external_n=args.external_candidates,
            rng=rng,
            random_external_expand=args.random_external_expand,
            grid_external_expand=args.grid_external_expand,
        )
        dbg.update({"strategy_mode": strategy})
        return ints, exts, dbg
    if strategy != "learned":
        raise ValueError(f"Unsupported point strategy: {strategy}")
    if learned_scorer is None:
        raise RuntimeError("point_strategy=learned requires --learned_ckpt_path")
    base_strategy = str(args.learned_base_strategy).lower()
    pool_int_n = int(args.learned_internal_pool) if int(args.learned_internal_pool) > 0 else max(args.internal_candidates * int(args.learned_pool_scale), args.internal_candidates)
    pool_ext_n = int(args.learned_external_pool) if int(args.learned_external_pool) > 0 else max(args.external_candidates * int(args.learned_pool_scale), args.external_candidates)
    pool_int, pool_ext, pool_dbg = build_candidates_by_strategy(
        strategy=base_strategy,
        box_xyxy=box_xyxy,
        image_size=(w, h),
        internal_n=pool_int_n,
        external_n=pool_ext_n,
        rng=rng,
        random_external_expand=args.random_external_expand,
        grid_external_expand=args.grid_external_expand,
    )
    int_scored = learned_scorer.score_points(image, box_xyxy, pool_int) if pool_int else []
    ext_scored = learned_scorer.score_points(image, box_xyxy, pool_ext) if pool_ext else []
    ints = [tuple(x["point"]) for x in int_scored[: args.internal_candidates]]
    exts = [tuple(x["point"]) for x in ext_scored[: args.external_candidates]]
    dbg = {
        "strategy_mode": "learned",
        "base_strategy": base_strategy,
        "pool_internal_n": pool_int_n,
        "pool_external_n": pool_ext_n,
        "pool_debug": pool_dbg,
        "internal_pool": [[float(a), float(b)] for a, b in pool_int],
        "external_pool": [[float(a), float(b)] for a, b in pool_ext],
        "internal_scores": int_scored,
        "external_scores": ext_scored,
        "load_info": getattr(learned_scorer, "load_info", {}),
    }
    return ints, exts, dbg


# -----------------------------
# point validation (new one-shot mode)
# -----------------------------

COLOR_PALETTE = [
    ((230, 57, 70), "red"),
    ((29, 78, 216), "blue"),
    ((34, 197, 94), "green"),
    ((250, 204, 21), "yellow"),
    ((249, 115, 22), "orange"),
    ((168, 85, 247), "purple"),
    ((236, 72, 153), "pink"),
    ((6, 182, 212), "cyan"),
    ((244, 63, 94), "rose"),
    ((14, 165, 233), "sky blue"),
    ((132, 204, 22), "lime"),
    ((245, 158, 11), "amber"),
]

SHAPES = ["circle", "square", "triangle", "diamond"]


def marker_token(i: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if i < len(alphabet):
        return alphabet[i]
    return f"P{i+1}"


def load_marker_font(font_size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(8, int(font_size)))
    except Exception:
        return ImageFont.load_default()


def resolve_effective_marker_size(
    bbox: Sequence[float],
    image_size: Tuple[int, int],
    base_size: int,
    min_size: int,
    max_size: int,
    adaptive: bool = True,
) -> int:
    if not adaptive:
        return max(min_size, min(max_size, int(base_size)))
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    img_w, img_h = image_size
    ref_scale = min(bw, bh, img_w * 0.18, img_h * 0.18)
    auto_size = int(round(ref_scale * 0.10))
    if auto_size <= 0:
        auto_size = int(base_size)
    mixed = int(round((float(base_size) + auto_size) / 2.0))
    return max(int(min_size), min(int(max_size), mixed))


def draw_single_marker(
    draw: ImageDraw.ImageDraw,
    point: Tuple[float, float],
    color: Tuple[int, int, int],
    shape: str,
    text: str,
    radius: int,
    font,
) -> None:
    x, y = point
    x = float(x)
    y = float(y)
    outline = (255, 255, 255)
    width = max(1, radius // 7)

    if shape == "circle":
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=outline, width=width)
    elif shape == "square":
        draw.rounded_rectangle((x - radius, y - radius, x + radius, y + radius), radius=max(3, radius // 4), fill=color, outline=outline, width=width)
    elif shape == "triangle":
        pts = [(x, y - radius), (x - 0.9 * radius, y + 0.8 * radius), (x + 0.9 * radius, y + 0.8 * radius)]
        draw.polygon(pts, fill=color, outline=outline)
    elif shape == "diamond":
        pts = [(x, y - radius), (x - radius, y), (x, y + radius), (x + radius, y)]
        draw.polygon(pts, fill=color, outline=outline)
    else:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=outline, width=width)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = x - tw / 2
    ty = y - th / 2 - 1
    draw.text((tx + 1, ty + 1), text, font=font, fill=(0, 0, 0))
    draw.text((tx, ty), text, font=font, fill=(255, 255, 255))



def compute_context_crop_bbox(
    image_size: Tuple[int, int],
    bbox: Sequence[float],
    points: Sequence[Tuple[float, float]],
    marker_radius: int,
    pad_scale: float,
) -> Tuple[int, int, int, int]:
    img_w, img_h = image_size
    x1, y1, x2, y2 = [float(v) for v in bbox]
    pad = int(round(marker_radius * pad_scale))
    crop_x1 = int(max(0, min([x1] + [p[0] - marker_radius - pad for p in points])))
    crop_y1 = int(max(0, min([y1] + [p[1] - marker_radius - pad for p in points])))
    crop_x2 = int(min(img_w, max([x2] + [p[0] + marker_radius + pad for p in points])))
    crop_y2 = int(min(img_h, max([y2] + [p[1] + marker_radius + pad for p in points])))
    crop_x2 = max(crop_x1 + 1, crop_x2)
    crop_y2 = max(crop_y1 + 1, crop_y2)
    return crop_x1, crop_y1, crop_x2, crop_y2


def format_textual_point_prompt(sentence: str, coord_x: Any, coord_y: Any, prompt_format: str) -> str:
    pf = str(prompt_format).lower()
    if pf in {"paren_xy", "xy", "paren"}:
        coord_text = f"({coord_x},{coord_y})"
        return f"Answer strictly yes or no: Is the Point {coord_text} on the object referred to by `{sentence}' in the picture?"
    if pf in {"x_is_y_is", "xisyis"}:
        return f"Answer strictly yes or no: Is the Point x is {coord_x}, y is {coord_y} on the object referred to by `{sentence}' in the picture?"
    if pf in {"x_eq_y_eq", "xeqy", "legacy"}:
        return f"Answer strictly yes or no: Is the Point x = {coord_x}, y = {coord_y} on the object referred to by `{sentence}' in the picture?"
    raise ValueError(f"Unsupported textual prompt format: {prompt_format}")


def resolve_textual_point_coordinates(
    point_xy: Sequence[float],
    crop_bbox: Sequence[int],
    image_size: Tuple[int, int],
    coord_space: str,
    coord_format: str,
) -> Tuple[Any, Any, Dict[str, Any]]:
    x, y = float(point_xy[0]), float(point_xy[1])
    img_w, img_h = image_size
    cx1, cy1, cx2, cy2 = [float(v) for v in crop_bbox]
    crop_w = max(1.0, cx2 - cx1)
    crop_h = max(1.0, cy2 - cy1)

    if coord_space == "crop":
        rx = x - cx1
        ry = y - cy1
        ref_w, ref_h = crop_w, crop_h
    elif coord_space == "image":
        rx = x
        ry = y
        ref_w, ref_h = float(img_w), float(img_h)
    else:
        raise ValueError(f"Unsupported textual coordinate space: {coord_space}")

    if coord_format == "qwen1000":
        px = int(round(rx / max(ref_w, 1.0) * 1000.0))
        py = int(round(ry / max(ref_h, 1.0) * 1000.0))
        px = int(np.clip(px, 0, 1000))
        py = int(np.clip(py, 0, 1000))
    elif coord_format == "absolute":
        px = int(round(rx))
        py = int(round(ry))
    elif coord_format == "normalized01":
        px = round(float(np.clip(rx / max(ref_w, 1.0), 0.0, 1.0)), 4)
        py = round(float(np.clip(ry / max(ref_h, 1.0), 0.0, 1.0)), 4)
    else:
        raise ValueError(f"Unsupported textual coordinate format: {coord_format}")

    return px, py, {
        "coord_space": coord_space,
        "coord_format": coord_format,
        "raw_point_xy": [x, y],
        "crop_bbox": [int(v) for v in crop_bbox],
        "rendered_coord": [px, py],
        "reference_size": [ref_w, ref_h],
    }


def parse_binary_yes_no(raw_text: str) -> Tuple[Optional[bool], Optional[str]]:
    if not isinstance(raw_text, str):
        return None, None
    m = re.search(r"\b(yes|no|true|false|1|0)\b", raw_text, flags=re.I)
    if m is None:
        return None, None
    pred = normalize_answer(m.group(1))
    return pred, "yes" if pred is True else "no" if pred is False else None


def extract_binary_token_confidence(analysis_results: Sequence[Dict[str, Any]], pred_answer: Optional[str]) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    if not analysis_results:
        return None, None
    best_step = None
    for step in analysis_results:
        yes_prob, no_prob, answer_conf = _summarize_yes_no_probs(step, pred_answer)
        if answer_conf is not None:
            step_info = dict(step)
            step_info["yes_prob"] = yes_prob
            step_info["no_prob"] = no_prob
            best_step = step_info
            return float(answer_conf), best_step
    first = dict(analysis_results[0])
    yes_prob, no_prob, answer_conf = _summarize_yes_no_probs(first, pred_answer)
    first["yes_prob"] = yes_prob
    first["no_prob"] = no_prob
    return (float(answer_conf) if answer_conf is not None else None), first


def render_multi_marker_query(
    image: Image.Image,
    bbox: Sequence[float],
    points: Sequence[Tuple[float, float]],
    marker_size: int,
    pad_scale: float,
    use_crop: bool,
    start_idx: int = 0,
    adaptive_marker_size: bool = True,
    marker_min_size: int = 8,
    marker_max_size: int = 16,
) -> Tuple[Image.Image, List[Dict[str, Any]], Tuple[int, int, int, int]]:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    effective_marker_size = resolve_effective_marker_size(
        bbox=bbox,
        image_size=img.size,
        base_size=marker_size,
        min_size=marker_min_size,
        max_size=marker_max_size,
        adaptive=adaptive_marker_size,
    )
    font = load_marker_font(max(8, int(round(effective_marker_size * 0.95))))

    x1, y1, x2, y2 = [float(v) for v in bbox]
    marker_meta: List[Dict[str, Any]] = []

    colors = COLOR_PALETTE.copy()
    random.shuffle(colors)
    shapes = SHAPES.copy()
    random.shuffle(shapes)

    for i, (px, py) in enumerate(points):
        token = marker_token(start_idx + i)
        color_rgb, color_name = colors[i % len(colors)]
        shape = shapes[i % len(shapes)]
        draw_single_marker(draw, (px, py), color_rgb, shape, token, effective_marker_size, font)
        marker_meta.append(
            {
                "marker_id": token,
                "point": [float(px), float(py)],
                "color": color_name,
                "shape": shape,
                "marker_size": effective_marker_size,
            }
        )

    crop_bbox = compute_context_crop_bbox(
        image_size=img.size,
        bbox=bbox,
        points=points,
        marker_radius=effective_marker_size,
        pad_scale=pad_scale,
    )

    if use_crop:
        img = img.crop(crop_bbox)

    return img, marker_meta, crop_bbox


def build_multi_marker_prompt(sentence: str, marker_meta: Sequence[Dict[str, Any]]) -> str:
    ids = [m["marker_id"] for m in marker_meta]
    format_text = "\n".join([f"{mid}: yes" for mid in ids])
    return (
        f"The image contains several marked points labeled {', '.join(ids)}.\n"
        f"For each label, decide whether the CENTER of that marked point lies on the object referred to by: '{sentence}'.\n"
        f"Return only one line per label in exactly this format:\n{format_text}\n"
        f"Replace yes with no when needed. Use only yes or no. Do not output JSON. Do not explain."
    )


def _parse_multi_marker_lines(raw_text: str) -> Dict[str, Optional[bool]]:
    normalized = raw_text.replace("；", "\n").replace(";", "\n")
    normalized = normalized.replace("，", ",")
    lines = [ln.strip() for ln in normalized.splitlines() if ln.strip()]
    line_map: Dict[str, Optional[bool]] = {}
    patterns = [
        r'^\s*([A-Z]\d*|P\d+)\s*[:=\-]\s*(yes|no|true|false|1|0)\b',
        r'^\s*\(?([A-Z]\d*|P\d+)\)?\s*(?:is|->)\s*(yes|no|true|false|1|0)\b',
    ]
    for line in lines:
        for pat in patterns:
            m = re.match(pat, line, flags=re.I)
            if m:
                line_map[m.group(1).upper()] = normalize_answer(m.group(2))
                break
    return line_map


def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        text = m.group(1)
    else:
        m = re.search(r"(\{.*\})", text, flags=re.S)
        if m:
            text = m.group(1)
        else:
            return None

    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def normalize_answer(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if not isinstance(x, str):
        return None
    xl = x.strip().lower()
    if xl in {"yes", "true", "1", "y"}:
        return True
    if xl in {"no", "false", "0", "n"}:
        return False
    return None


def parse_multi_marker_predictions(raw_text: str, marker_meta: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}

    # Prefer strict line-based parsing first. This is the default path for the new prompt
    # and avoids spurious JSON parsing attempts.
    line_map = _parse_multi_marker_lines(raw_text)
    if line_map:
        for m in marker_meta:
            mid = m["marker_id"]
            pred = line_map.get(mid.upper())
            result[mid] = {
                "pred_bool": pred,
                "pred_answer": "yes" if pred is True else "no" if pred is False else None,
                "confidence": None,
            }
        return result

    parsed = parse_json_from_text(raw_text)
    if parsed is not None:
        for m in marker_meta:
            mid = m["marker_id"]
            payload = parsed.get(mid)
            pred = normalize_answer(payload.get("answer")) if isinstance(payload, dict) else normalize_answer(payload)
            result[mid] = {
                "pred_bool": pred,
                "pred_answer": "yes" if pred is True else "no" if pred is False else None,
                "confidence": None,
            }
        return result

    for m in marker_meta:
        mid = m["marker_id"]
        pred = None
        pat = rf'(?im)(?:^|[{{,\n\s])"?{re.escape(mid)}"?\s*[:=\-]\s*"?(yes|no|true|false|1|0)\b"?'
        mm = re.search(pat, raw_text, flags=re.I)
        if mm:
            pred = normalize_answer(mm.group(1))
        else:
            # last-resort search around the marker id in plain text
            mm = re.search(rf'(?im)\b{re.escape(mid)}\b[^\n:=-]{{0,12}}[:=\-]?[^\nA-Za-z0-9]{{0,4}}(yes|no)\b', raw_text, flags=re.I)
            if mm:
                pred = normalize_answer(mm.group(1))
        result[mid] = {
            "pred_bool": pred,
            "pred_answer": "yes" if pred is True else "no" if pred is False else None,
            "confidence": None,
        }
    return result


def _find_step_for_answer(token_text: str, analysis_results: Sequence[Dict[str, Any]], marker_id: str) -> Optional[Tuple[int, str]]:
    patterns = [
        rf'(?im)^\s*{re.escape(marker_id)}\s*[:=\-]\s*(yes|no)\b',
        rf'"{re.escape(marker_id)}"\s*:\s*"?(yes|no)"?',
    ]
    match = None
    for pat in patterns:
        match = re.search(pat, token_text, flags=re.I | re.M)
        if match:
            break
    if match is None:
        return None
    answer_start = match.span(1)[0]
    cur = 0
    for idx, step in enumerate(analysis_results):
        tok = str(step.get("token", ""))
        nxt = cur + len(tok)
        if answer_start < nxt:
            return idx, match.group(1).lower()
        cur = nxt
    return None


def _summarize_yes_no_probs(step_info: Optional[Dict[str, Any]], pred_answer: Optional[str]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not step_info:
        return None, None, None
    yes_prob = 0.0
    no_prob = 0.0
    for tok, prob in step_info.get("top_candidates") or []:
        tok_norm = str(tok).strip().lower()
        if tok_norm == "yes":
            yes_prob += float(prob)
        elif tok_norm == "no":
            no_prob += float(prob)
    answer_conf = None
    if pred_answer == "yes" and yes_prob > 0:
        answer_conf = yes_prob
    elif pred_answer == "no" and no_prob > 0:
        answer_conf = no_prob
    return yes_prob if yes_prob > 0 else None, no_prob if no_prob > 0 else None, answer_conf


def attach_token_confidence(
    token_text: str,
    analysis_results: Sequence[Dict[str, Any]],
    parsed: Dict[str, Dict[str, Any]],
    marker_meta: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    enriched: Dict[str, Dict[str, Any]] = {}
    for meta in marker_meta:
        mid = meta["marker_id"]
        pred_info = dict(parsed.get(mid, {}))
        pred_answer = pred_info.get("pred_answer")
        step_match = _find_step_for_answer(token_text, analysis_results, mid)
        analysis_step = None
        if step_match is not None:
            step_idx, matched_answer = step_match
            step_info = dict(analysis_results[step_idx])
            pred_answer = pred_answer or matched_answer
            yes_prob, no_prob, answer_conf = _summarize_yes_no_probs(step_info, pred_answer)
            step_info["yes_prob"] = yes_prob
            step_info["no_prob"] = no_prob
            analysis_step = step_info
            if answer_conf is not None:
                pred_info["confidence"] = float(answer_conf)
        pred_info["pred_answer"] = pred_answer
        pred_info["pred_bool"] = normalize_answer(pred_answer)
        pred_info["analysis_step"] = analysis_step
        enriched[mid] = pred_info
    return enriched


def validate_points_multimarker(
    model,
    processor,
    tokenizer,
    image: Image.Image,
    sentence: str,
    bbox: Sequence[float],
    mask_gt: np.ndarray,
    point_candidates: Sequence[Tuple[float, float]],
    args,
    save_dir: Optional[str] = None,
    prefix: str = "chunk",
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    if not point_candidates:
        return outputs

    chunk_size = max(1, args.max_markers_per_query)
    w, h = image.size

    for chunk_idx in range(0, len(point_candidates), chunk_size):
        chunk_points = point_candidates[chunk_idx: chunk_idx + chunk_size]
        render_img, marker_meta, crop_bbox = render_multi_marker_query(
            image=image,
            bbox=bbox,
            points=chunk_points,
            marker_size=args.marker_size,
            pad_scale=args.marker_pad_scale,
            use_crop=args.use_crop_for_points,
            start_idx=chunk_idx,
            adaptive_marker_size=not args.disable_adaptive_marker_size,
            marker_min_size=args.marker_min_size,
            marker_max_size=args.marker_max_size,
        )
        prompt = build_multi_marker_prompt(sentence, marker_meta)
        generation = run_chat_query_with_details(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            image=render_img,
            text=prompt,
            max_new_tokens=args.point_max_new_tokens,
        )
        raw_text = generation["text"]
        analysis_results = generation["analysis_results"]
        token_text = generation["token_text"]
        parsed = parse_multi_marker_predictions(raw_text, marker_meta)
        parsed = attach_token_confidence(token_text, analysis_results, parsed, marker_meta)

        render_path = None
        if save_dir is not None:
            ensure_dir(save_dir)
            render_path = os.path.join(save_dir, f"{prefix}_{chunk_idx//chunk_size:02d}.jpg")
            render_img.save(render_path)
            save_json(
                os.path.join(save_dir, f"{prefix}_{chunk_idx//chunk_size:02d}.json"),
                {
                    "prompt": prompt,
                    "raw_output": raw_text,
                    "token_text": token_text,
                    "analysis_results": analysis_results,
                    "crop_bbox": list(crop_bbox),
                    "marker_meta": marker_meta,
                    "parsed": parsed,
                },
            )

        for meta in marker_meta:
            x, y = meta["point"]
            gt_label = bool(mask_gt[min(h - 1, int(y)), min(w - 1, int(x))] > 0)
            pred_info = parsed.get(meta["marker_id"], {})
            outputs.append(
                {
                    "point": [float(x), float(y)],
                    "marker_id": meta["marker_id"],
                    "color": meta["color"],
                    "shape": meta["shape"],
                    "model_output": raw_text,
                    "pred_label": pred_info.get("pred_bool"),
                    "pred_answer": pred_info.get("pred_answer"),
                    "confidence": pred_info.get("confidence"),
                    "analysis_results": pred_info.get("analysis_step"),
                    "label": gt_label,
                    "crop_bbox": list(crop_bbox),
                    "render_path": render_path,
                }
            )

    return outputs



def validate_points_textual(
    model,
    processor,
    tokenizer,
    image: Image.Image,
    sentence: str,
    bbox: Sequence[float],
    mask_gt: np.ndarray,
    point_candidates: Sequence[Tuple[float, float]],
    args,
    save_dir: Optional[str] = None,
    prefix: str = "chunk",
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    if not point_candidates:
        return outputs

    marker_radius = resolve_effective_marker_size(
        bbox=bbox,
        image_size=image.size,
        base_size=args.marker_size,
        min_size=args.marker_min_size,
        max_size=args.marker_max_size,
        adaptive=not args.disable_adaptive_marker_size,
    )
    crop_bboxes = [
        compute_context_crop_bbox(
            image_size=image.size,
            bbox=bbox,
            points=[pt],
            marker_radius=marker_radius,
            pad_scale=args.marker_pad_scale,
        )
        for pt in point_candidates
    ]

    for idx, (pt, crop_bbox) in enumerate(zip(point_candidates, crop_bboxes)):
        render_img = image.crop(crop_bbox) if args.use_crop_for_points else image.copy()
        coord_x, coord_y, coord_debug = resolve_textual_point_coordinates(
            point_xy=pt,
            crop_bbox=crop_bbox,
            image_size=image.size,
            coord_space=args.textual_coord_space,
            coord_format=args.textual_coord_format,
        )
        prompt = format_textual_point_prompt(sentence, coord_x, coord_y, args.textual_prompt_format)
        generation = run_chat_query_with_details(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            image=render_img,
            text=prompt,
            max_new_tokens=args.point_max_new_tokens,
        )
        raw_text = generation["text"]
        pred_bool, pred_answer = parse_binary_yes_no(raw_text)
        confidence, analysis_step = extract_binary_token_confidence(generation["analysis_results"], pred_answer)
        render_path = None
        if save_dir is not None:
            ensure_dir(save_dir)
            render_path = os.path.join(save_dir, f"{prefix}_{idx:02d}.jpg")
            render_img.save(render_path)
            save_json(
                os.path.join(save_dir, f"{prefix}_{idx:02d}.json"),
                {
                    "prompt": prompt,
                    "raw_output": raw_text,
                    "token_text": generation["token_text"],
                    "analysis_results": generation["analysis_results"],
                    "crop_bbox": list(crop_bbox),
                    "coord_debug": coord_debug,
                    "input_mode": generation.get("input_mode"),
                },
            )
        x, y = pt
        h, w = mask_gt.shape[:2]
        gt_label = bool(mask_gt[min(h - 1, int(y)), min(w - 1, int(x))] > 0)
        outputs.append(
            {
                "point": [float(x), float(y)],
                "marker_id": None,
                "color": None,
                "shape": None,
                "model_output": raw_text,
                "pred_label": pred_bool,
                "pred_answer": pred_answer,
                "confidence": confidence,
                "analysis_results": analysis_step,
                "label": gt_label,
                "crop_bbox": list(crop_bbox),
                "render_path": render_path,
                "coord_debug": coord_debug,
                "input_mode": generation.get("input_mode"),
            }
        )
    return outputs


# -----------------------------
# point selection / SAM2 decode
# -----------------------------

def select_points_from_queues(
    point_outputs_int: Sequence[Dict[str, Any]],
    point_outputs_ext: Sequence[Dict[str, Any]],
    conf_thresh: float,
    max_internal_points: int,
    max_external_points: int,
    selection_mode: str = "order",
) -> Tuple[List[List[float]], List[List[float]], Dict[str, Any]]:
    def keep(item: Dict[str, Any], positive: bool) -> bool:
        pred = item.get("pred_label")
        conf = item.get("confidence")
        if pred is None:
            return False
        if conf is not None and conf < conf_thresh:
            return False
        return pred is positive

    int_main = [x for x in point_outputs_int if keep(x, True)]
    ext_appendix = [x for x in point_outputs_int if keep(x, False)]
    ext_main = [x for x in point_outputs_ext if keep(x, False)]
    int_appendix = [x for x in point_outputs_ext if keep(x, True)]

    if selection_mode == "confidence":
        key_fn = lambda z: (z.get("confidence") is not None, z.get("confidence") or -1.0)
        int_main = sorted(int_main, key=key_fn, reverse=True)
        ext_main = sorted(ext_main, key=key_fn, reverse=True)
        int_appendix = sorted(int_appendix, key=key_fn, reverse=True)
        ext_appendix = sorted(ext_appendix, key=key_fn, reverse=True)
    else:
        int_appendix = list(reversed(int_appendix))
        ext_appendix = list(reversed(ext_appendix))

    internal_points = [x["point"] for x in (int_main + int_appendix)[:max_internal_points]]
    external_points = [x["point"] for x in (ext_main + ext_appendix)[:max_external_points]]
    debug_info = {
        "int_main_ids": [x.get("marker_id") for x in int_main],
        "ext_main_ids": [x.get("marker_id") for x in ext_main],
        "int_appendix_ids": [x.get("marker_id") for x in int_appendix],
        "ext_appendix_ids": [x.get("marker_id") for x in ext_appendix],
    }
    return internal_points, external_points, debug_info


def sam2_decode(
    predictor: SAM2ImagePredictor,
    image: Image.Image,
    input_box: Optional[Sequence[float]],
    internal_points: Sequence[Sequence[float]],
    external_points: Sequence[Sequence[float]],
    result_type: str,
    finer_step: int,
    finer_use_box: bool,
) -> Dict[str, Any]:
    predictor.set_image(np.array(image))

    input_point = [list(p) for p in internal_points] + [list(p) for p in external_points]
    input_label = [1] * len(internal_points) + [0] * len(external_points)

    if len(input_point) == 0 and input_box is not None:
        x1, y1, x2, y2 = input_box
        input_point = [[(x1 + x2) / 2.0, (y1 + y2) / 2.0]]
        input_label = [1]

    point_coords = np.array(input_point, dtype=np.float32) if len(input_point) else None
    point_labels = np.array(input_label, dtype=np.int32) if len(input_label) else None
    box = np.array(input_box, dtype=np.float32) if input_box is not None else None

    if "box" not in result_type:
        box = None
    if "point" not in result_type:
        point_coords = None
        point_labels = None
        if box is not None:
            box = box[None, :]

    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=False,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    mask_input = logits[np.argmax(scores), :, :]

    for _ in range(finer_step):
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box if finer_use_box else None,
            multimask_output=False,
            mask_input=mask_input[None, :, :],
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        mask_input = logits[np.argmax(scores), :, :]

    return {
        "masks": masks,
        "scores": scores,
        "logits": logits,
        "point_coords": None if point_coords is None else point_coords.tolist(),
        "point_labels": None if point_labels is None else point_labels.tolist(),
        "input_box": None if box is None else np.array(box).reshape(-1).tolist(),
    }


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred > 0, gt > 0).sum()
    union = np.logical_or(pred > 0, gt > 0).sum()
    return float(inter / (union + 1e-6))


def summarize_decode_branch(name: str, decode_out: Dict[str, Any], pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, Any]:
    score = float(np.max(decode_out["scores"])) if decode_out.get("scores") is not None and len(decode_out["scores"]) else None
    return {
        "name": name,
        "score": score,
        "point_coords": decode_out.get("point_coords"),
        "point_labels": decode_out.get("point_labels"),
        "input_box": decode_out.get("input_box"),
        "mask_iou": mask_iou(pred_mask, gt_mask),
    }


def choose_decode_branch(
    box_branch: Optional[Dict[str, Any]],
    point_branch: Dict[str, Any],
    mode: str = "sam_score",
    score_margin: float = 0.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if box_branch is None or mode == "none":
        return point_branch, {
            "mode": mode,
            "chosen": point_branch["name"],
            "box_score": None if box_branch is None else box_branch.get("score"),
            "point_score": point_branch.get("score"),
            "score_margin": score_margin,
        }

    box_score = -1e9 if box_branch.get("score") is None else float(box_branch["score"])
    point_score = -1e9 if point_branch.get("score") is None else float(point_branch["score"])
    if mode == "sam_score":
        chosen = point_branch if point_score > box_score + float(score_margin) else box_branch
    else:
        chosen = point_branch
    return chosen, {
        "mode": mode,
        "chosen": chosen["name"],
        "box_score": None if box_branch.get("score") is None else float(box_branch["score"]),
        "point_score": None if point_branch.get("score") is None else float(point_branch["score"]),
        "score_margin": float(score_margin),
    }


# -----------------------------
# main
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full ResAgent evaluation from bbox prior -> point discovery -> one-shot validation -> SAM2")
    parser.add_argument("--task", type=str, default="refcoco", choices=["refcoco", "refcoco+", "refcocog"])
    parser.add_argument("--split", type=str, default="testA")
    parser.add_argument("--data_root", type=str, default="../../../DETRIS-main/datasets/")

    parser.add_argument("--model_name", type=str, default="8B-Instruct")
    parser.add_argument("--model_root", type=str, default="/amax/wangyh/pretrained/")
    parser.add_argument("--skip_vlm_loading", action="store_true", help="Use this only when bbox priors are fully precomputed and point validation is disabled.")

    parser.add_argument("--sam2_checkpoint", type=str, default="base_plus", choices=["large", "base_plus", "checkpoint", "checkpoint_large"])
    parser.add_argument("--sam2_checkpoint_path", type=str, default="", help="Optional explicit SAM2 checkpoint path. If set, it overrides --sam2_checkpoint.")
    parser.add_argument("--sam2_config_path", type=str, default="", help="Optional explicit SAM2 config yaml path. If set, it overrides the config resolved from --sam2_checkpoint.")
    parser.add_argument("--result_type", type=str, default="point_box", choices=["point_box", "point", "box"])
    parser.add_argument("--finer_step", type=int, default=0)
    parser.add_argument("--finer_use_box", action="store_true")

    parser.add_argument("--box_input_json", type=str, default="", help="Optional detection / bbox prior json/jsonl.")
    parser.add_argument("--box_prior_mode", type=str, default="auto", choices=["auto", "fixed", "vote", "precomputed_vote", "vote_or_fixed"])
    parser.add_argument("--box_fields", type=str, default="raw_text,raw_text_list,pred_box_xyxy,pred_box_xyxy_list,voted_box,actual_bboxes,bbox,bbox_2d,pred_bbox,box,output_text")
    parser.add_argument("--fixed_box_field", type=str, default="raw_text")
    parser.add_argument("--precomputed_vote_field", type=str, default="voted_box")
    parser.add_argument("--box_coord_mode", type=str, default="auto", choices=["auto", "absolute", "qwen1000"])
    parser.add_argument("--box_vote_metric", type=str, default="iou", choices=["iou", "ciou"])
    parser.add_argument("--allow_live_bbox_fallback", action="store_true")
    parser.add_argument("--prefer_sentence_level_prior", action="store_true", help="When the prior file is sentence-level JSONL, prefer the sentence matched by --sentence_mode.")

    parser.add_argument("--sentence_mode", type=str, default="first", choices=["first", "longest", "shortest", "random", "join"])


    parser.add_argument("--point_strategy", type=str, default="spiral", choices=["spiral", "random", "grid", "learned"])
    parser.add_argument("--learned_ckpt_path", type=str, default="")
    parser.add_argument("--learned_config_json", type=str, default="")
    parser.add_argument("--learned_base_strategy", type=str, default="spiral", choices=["spiral", "random", "grid"])
    parser.add_argument("--learned_pool_scale", type=int, default=4)
    parser.add_argument("--learned_internal_pool", type=int, default=0)
    parser.add_argument("--learned_external_pool", type=int, default=0)
    parser.add_argument("--learned_use_ema", action="store_true")
    parser.add_argument("--learned_score_mode", type=str, default="useful_value", choices=["useful_value", "useful", "value"])
    parser.add_argument("--random_external_expand", type=float, default=0.25)
    parser.add_argument("--grid_external_expand", type=float, default=0.15)

    parser.add_argument("--internal_candidates", type=int, default=4)
    parser.add_argument("--external_candidates", type=int, default=4)
    parser.add_argument("--max_internal_points", type=int, default=2)
    parser.add_argument("--max_external_points", type=int, default=1)
    parser.add_argument("--point_conf_thresh", type=float, default=0.5)
    parser.add_argument("--point_selection_mode", type=str, default="order", choices=["order", "confidence"])

    parser.add_argument("--validation_mode", type=str, default="multi_marker", choices=["multi_marker", "textual_point"])
    parser.add_argument("--use_crop_for_points", action="store_true")
    parser.add_argument("--max_markers_per_query", type=int, default=8)
    parser.add_argument("--marker_size", type=int, default=12)
    parser.add_argument("--marker_min_size", type=int, default=8)
    parser.add_argument("--marker_max_size", type=int, default=16)
    parser.add_argument("--disable_adaptive_marker_size", action="store_true")
    parser.add_argument("--marker_pad_scale", type=float, default=1.2)
    parser.add_argument("--point_max_new_tokens", type=int, default=256)
    parser.add_argument("--textual_prompt_format", type=str, default="x_eq_y_eq", choices=["paren_xy", "x_is_y_is", "x_eq_y_eq"])
    parser.add_argument("--textual_coord_space", type=str, default="crop", choices=["crop", "image"])
    parser.add_argument("--textual_coord_format", type=str, default="qwen1000", choices=["qwen1000", "absolute", "normalized01"])

    parser.add_argument("--decode_fallback_mode", type=str, default="sam_score", choices=["none", "sam_score"], help="Compare box-only and box+points branches and choose one with higher SAM2 score.")
    parser.add_argument("--decode_score_margin", type=float, default=0.0, help="Require point branch score to exceed box-only by this margin before selecting it.")

    parser.add_argument("--output_root", type=str, default="./resagent_eval_outputs")
    parser.add_argument("--save_debug_images", action="store_true")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=3)
    return parser.parse_args()


def resolve_model_path(model_root: str, model_name: str) -> str:
    mapping = {
        "8B-Instruct": "Qwen3-VL-8B-Instruct",
        "30B-A3B-Instruct": "Qwen3-VL-30B-A3B-Instruct",
        "4B-Instruct": "Qwen3-VL-4B-Instruct",
        "2B-Instruct": "Qwen3-VL-2B-Instruct",
        "Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B-Instruct",
        "Qwen2.5-VL-72B-Instruct": "Qwen2.5-VL-72B-Instruct",
        "2.5-3B-Instruct": "Qwen2.5-VL-3B-Instruct",
        "2.5-7B-Instruct": "Qwen2.5-VL-7B-Instruct",
        "2.5-72B-Instruct": "Qwen2.5-VL-72B-Instruct",
    }
    resolved = mapping.get(model_name, model_name)
    if os.path.isabs(resolved) or os.path.isdir(resolved) or resolved.startswith("Qwen/"):
        return resolved
    return os.path.join(model_root, resolved)


def resolve_sam2_checkpoint(name: str, checkpoint_path: str = "", config_path: str = "") -> Tuple[str, str]:
    if checkpoint_path and config_path:
        return checkpoint_path, config_path
    if name == "large":
        ckpt, cfg = "../checkpoints/sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"
    elif name == "base_plus":
        ckpt, cfg = "../checkpoints/sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"
    elif name == "checkpoint":
        ckpt, cfg = "../checkpoints/checkpoint.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"
    elif name == "checkpoint_large":
        ckpt, cfg = "../checkpoints/checkpoint_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"
    else:
        raise ValueError(name)
    if checkpoint_path:
        ckpt = checkpoint_path
    if config_path:
        cfg = config_path
    return ckpt, cfg


def _load_json_or_jsonl(path: str) -> Any:
    if not path:
        return None
    if path.endswith('.jsonl'):
        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read().strip()
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        rows = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows


def load_prior_json(path: str) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    data = _load_json_or_jsonl(path)
    by_ref: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            ref_id = item.get('segment_id', item.get('ref_id'))
            if ref_id is None:
                key = item.get('key')
                if isinstance(key, str):
                    parts = key.split(':')
                    if len(parts) >= 4:
                        ref_id = parts[3]
            if ref_id is None:
                continue
            ref_key = str(ref_id)
            if ref_key not in by_ref:
                by_ref[ref_key] = {
                    'segment_id': ref_id,
                    'ref_id': ref_id,
                    'sent_level_items': [],
                    'sent_id_to_item': {},
                    'key_to_item': {},
                    'expr_to_item': {},
                }
            group = by_ref[ref_key]
            group['sent_level_items'].append(item)
            if item.get('sent_id') is not None:
                group['sent_id_to_item'][str(item['sent_id'])] = item
            if item.get('key') is not None:
                group['key_to_item'][str(item['key'])] = item
            expr = item.get('expr') or item.get('sent') or item.get('sentence')
            if isinstance(expr, str) and expr.strip():
                group['expr_to_item'][expr.strip()] = item

            # old-style per-ref json should still be directly usable
            for k, v in item.items():
                if k not in group:
                    group[k] = v

        for ref_key, group in by_ref.items():
            items = group.get('sent_level_items', [])
            if len(items) == 1:
                only_item = items[0]
                for k, v in only_item.items():
                    group[k] = v
                group['selected_sent_record'] = only_item
            else:
                pred_boxes = [it.get('pred_box_xyxy') for it in items if isinstance(it.get('pred_box_xyxy'), (list, tuple)) and len(it.get('pred_box_xyxy')) == 4]
                raw_texts = [it.get('raw_text') for it in items if isinstance(it.get('raw_text'), str) and it.get('raw_text').strip()]
                if pred_boxes:
                    group['pred_box_xyxy_list'] = pred_boxes
                if raw_texts:
                    group['raw_text_list'] = raw_texts
        return by_ref

    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if isinstance(v, dict):
                out[str(k)] = v
        return out
    return {}


def resolve_predict_info(ref_id: Any, refs: Dict[str, Any], prior_json: Dict[str, Dict[str, Any]], sentence_mode: str, prefer_sentence_level: bool = False) -> Optional[Dict[str, Any]]:
    predict_info = prior_json.get(str(ref_id))
    if predict_info is None:
        return None
    sent_items = predict_info.get('sent_level_items') if isinstance(predict_info, dict) else None
    if not sent_items:
        return predict_info

    sent_entry = choose_sentence_entry(refs, sentence_mode)
    chosen_item = None
    if sent_entry is not None:
        sent_id = sent_entry.get('sent_id')
        if sent_id is not None:
            chosen_item = predict_info.get('sent_id_to_item', {}).get(str(sent_id))
        if chosen_item is None:
            sent_text = sent_entry.get('sent', '').strip()
            if sent_text:
                chosen_item = predict_info.get('expr_to_item', {}).get(sent_text)
    if chosen_item is None and sent_items:
        chosen_item = sent_items[0]

    merged = dict(predict_info)
    if isinstance(chosen_item, dict):
        for k, v in chosen_item.items():
            merged[k] = v
        merged['selected_sent_record'] = chosen_item
    if prefer_sentence_level and isinstance(chosen_item, dict):
        # let fixed mode naturally hit the sentence-level top fields first
        pass
    return merged


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    ensure_dir(args.output_root)
    records_dir = os.path.join(args.output_root, "records")
    mask_dir = os.path.join(args.output_root, "masks")
    debug_dir = os.path.join(args.output_root, "debug")
    ensure_dir(records_dir)
    ensure_dir(mask_dir)
    if args.save_debug_images:
        ensure_dir(debug_dir)

    model_bundle = None
    if not args.skip_vlm_loading:
        model_path = resolve_model_path(args.model_root, args.model_name)
        print(f"Loading VLM from: {model_path}")
        model_bundle = load_qwen_model(model_path, device)
    else:
        print("Skip VLM loading is ON.")

    sam2_checkpoint, model_cfg = resolve_sam2_checkpoint(
        args.sam2_checkpoint,
        checkpoint_path=args.sam2_checkpoint_path,
        config_path=args.sam2_config_path,
    )

    print(f"Loading SAM2 from: {sam2_checkpoint}")
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    learned_scorer = None
    if args.point_strategy == "learned":
        if not args.learned_ckpt_path:
            raise ValueError("point_strategy=learned requires --learned_ckpt_path")
        learned_scorer = LearnedPointScorer(
            predictor=predictor,
            device=device,
            ckpt_path=args.learned_ckpt_path,
            config_json=args.learned_config_json,
            use_ema=args.learned_use_ema,
            score_mode=args.learned_score_mode,
        )
        print(f"Prepared learned point scorer from: {args.learned_ckpt_path}")

    refer = load_refer(args.data_root, args.task)
    ref_ids = refer.getRefIds(split=args.split)
    if args.limit > 0:
        ref_ids = ref_ids[: args.limit]
    print(f"Total refs to evaluate: {len(ref_ids)}")

    prior_json = load_prior_json(args.box_input_json)
    summary_jsonl = os.path.join(args.output_root, "records.jsonl")
    if os.path.exists(summary_jsonl):
        os.remove(summary_jsonl)

    metrics_iou: List[float] = []
    num_50 = 0
    num_75 = 0
    valid_count = 0

    for idx, ref_id in enumerate(tqdm(ref_ids, desc="Evaluating"), 1):
        refs = refer.Refs[ref_id]
        image_info = refer.loadImgs(image_ids=refs["image_id"])[0]
        image_name = image_info["file_name"]
        image_path = os.path.join(args.data_root, "images", "train2014", image_name)
        image = Image.open(image_path).convert("RGB")
        mask_gt = refer.getMask(refs)["mask"].astype(np.uint8)
        gt_box_xywh = refer.getRefBox(ref_id)
        gt_box_xyxy = xywh_to_xyxy(gt_box_xywh)

        predict_info = resolve_predict_info(
            ref_id=ref_id,
            refs=refs,
            prior_json=prior_json,
            sentence_mode=args.sentence_mode,
            prefer_sentence_level=args.prefer_sentence_level_prior,
        )
        sentence = resolve_sentence(refs, predict_info, args.sentence_mode)

        box_info = resolve_box_prior(args, predict_info, image, sentence, model_bundle)
        chosen_box = clip_box_xyxy(box_info["chosen_box"], image.size[0], image.size[1])


        internal_candidates, external_candidates, candidate_strategy_debug = build_point_candidates(
            args=args,
            ref_id=ref_id,
            image=image,
            box_xyxy=chosen_box,
            learned_scorer=learned_scorer,
        )

        sample_debug_dir = None
        if args.save_debug_images:
            sample_debug_dir = os.path.join(debug_dir, str(ref_id))
            ensure_dir(sample_debug_dir)

        if model_bundle is None:
            raise RuntimeError("Point validation needs the VLM. Disable --skip_vlm_loading or provide another validator.")
        model, processor, tokenizer = model_bundle
        if args.validation_mode == "multi_marker":
            point_outputs_int = validate_points_multimarker(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                image=image,
                sentence=sentence,
                bbox=chosen_box,
                mask_gt=mask_gt,
                point_candidates=internal_candidates,
                args=args,
                save_dir=None if sample_debug_dir is None else os.path.join(sample_debug_dir, "int"),
                prefix="int",
            )
            point_outputs_ext = validate_points_multimarker(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                image=image,
                sentence=sentence,
                bbox=chosen_box,
                mask_gt=mask_gt,
                point_candidates=external_candidates,
                args=args,
                save_dir=None if sample_debug_dir is None else os.path.join(sample_debug_dir, "ext"),
                prefix="ext",
            )
        elif args.validation_mode == "textual_point":
            point_outputs_int = validate_points_textual(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                image=image,
                sentence=sentence,
                bbox=chosen_box,
                mask_gt=mask_gt,
                point_candidates=internal_candidates,
                args=args,
                save_dir=None if sample_debug_dir is None else os.path.join(sample_debug_dir, "int"),
                prefix="int",
            )
            point_outputs_ext = validate_points_textual(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                image=image,
                sentence=sentence,
                bbox=chosen_box,
                mask_gt=mask_gt,
                point_candidates=external_candidates,
                args=args,
                save_dir=None if sample_debug_dir is None else os.path.join(sample_debug_dir, "ext"),
                prefix="ext",
            )
        else:
            raise ValueError(f"Unsupported validation_mode: {args.validation_mode}")

        internal_points, external_points, selection_debug = select_points_from_queues(
            point_outputs_int=point_outputs_int,
            point_outputs_ext=point_outputs_ext,
            conf_thresh=args.point_conf_thresh,
            max_internal_points=args.max_internal_points,
            max_external_points=args.max_external_points,
            selection_mode=args.point_selection_mode,
        )

        point_decode_out = sam2_decode(
            predictor=predictor,
            image=image,
            input_box=chosen_box,
            internal_points=internal_points,
            external_points=external_points,
            result_type=args.result_type,
            finer_step=args.finer_step,
            finer_use_box=args.finer_use_box,
        )
        point_pred_mask = point_decode_out["masks"][0].astype(np.uint8)
        point_branch = summarize_decode_branch("point_branch", point_decode_out, point_pred_mask, mask_gt)

        box_branch = None
        box_decode_out = None
        box_pred_mask = None
        if args.decode_fallback_mode != "none" and chosen_box is not None and "box" in args.result_type:
            box_decode_out = sam2_decode(
                predictor=predictor,
                image=image,
                input_box=chosen_box,
                internal_points=[],
                external_points=[],
                result_type="box",
                finer_step=args.finer_step,
                finer_use_box=args.finer_use_box,
            )
            box_pred_mask = box_decode_out["masks"][0].astype(np.uint8)
            box_branch = summarize_decode_branch("box_branch", box_decode_out, box_pred_mask, mask_gt)

        chosen_branch, decode_choice = choose_decode_branch(
            box_branch=box_branch,
            point_branch=point_branch,
            mode=args.decode_fallback_mode,
            score_margin=args.decode_score_margin,
        )
        if chosen_branch["name"] == "box_branch":
            decode_out = box_decode_out
            pred_mask = box_pred_mask
        else:
            decode_out = point_decode_out
            pred_mask = point_pred_mask

        iou = mask_iou(pred_mask, mask_gt)
        metrics_iou.append(iou)
        valid_count += 1
        num_50 += int(iou >= 0.50)
        num_75 += int(iou >= 0.75)

        mask_path = os.path.join(mask_dir, f"mask_{ref_id}.png")
        Image.fromarray((pred_mask * 255).astype(np.uint8)).save(mask_path)

        box_only_mask_path = None
        point_mask_path = None
        if box_pred_mask is not None:
            box_only_mask_path = os.path.join(mask_dir, f"mask_{ref_id}_box.png")
            Image.fromarray((box_pred_mask * 255).astype(np.uint8)).save(box_only_mask_path)
        if point_pred_mask is not None:
            point_mask_path = os.path.join(mask_dir, f"mask_{ref_id}_point.png")
            Image.fromarray((point_pred_mask * 255).astype(np.uint8)).save(point_mask_path)

        record = {
            "segment_id": ref_id,
            "image_name": image_name,
            "image_path": image_path,
            "sentence": sentence,
            "gt_box_xywh": [float(x) for x in gt_box_xywh],
            "gt_box_xyxy": [float(x) for x in gt_box_xyxy],
            "box_info": {
                "mode": box_info["mode"],
                "chosen_box": [float(x) for x in chosen_box],
                "chosen_source": box_info["chosen_source"],
                "candidates": box_info["candidates"],
                "vote_scores": box_info["vote_scores"],
                "pairwise": box_info["pairwise"],
                "live_raw": box_info["live_raw"],
            },
            "internal_candidates": [[float(a), float(b)] for a, b in internal_candidates],
            "external_candidates": [[float(a), float(b)] for a, b in external_candidates],
            "candidate_strategy": candidate_strategy_debug,
            "point_outputs_int": point_outputs_int,
            "point_outputs_ext": point_outputs_ext,
            "selected_internal_points": internal_points,
            "selected_external_points": external_points,
            "selection_debug": selection_debug,
            "sam2": {
                "chosen_branch": decode_choice,
                "chosen_scores": [float(x) for x in decode_out["scores"]],
                "chosen_point_coords": decode_out["point_coords"],
                "chosen_point_labels": decode_out["point_labels"],
                "chosen_input_box": decode_out["input_box"],
                "mask_path": mask_path,
                "box_branch": None if box_decode_out is None else {
                    "scores": [float(x) for x in box_decode_out["scores"]],
                    "point_coords": box_decode_out["point_coords"],
                    "point_labels": box_decode_out["point_labels"],
                    "input_box": box_decode_out["input_box"],
                    "mask_path": box_only_mask_path,
                    "summary": box_branch,
                },
                "point_branch": {
                    "scores": [float(x) for x in point_decode_out["scores"]],
                    "point_coords": point_decode_out["point_coords"],
                    "point_labels": point_decode_out["point_labels"],
                    "input_box": point_decode_out["input_box"],
                    "mask_path": point_mask_path,
                    "summary": point_branch,
                },
            },
            "metrics": {
                "iou": iou,
                "acc50": bool(iou >= 0.50),
                "acc75": bool(iou >= 0.75),
                "chosen_branch": decode_choice.get("chosen"),
                "box_branch_iou": None if box_branch is None else box_branch.get("mask_iou"),
                "point_branch_iou": point_branch.get("mask_iou"),
            },
        }
        save_json(os.path.join(records_dir, f"{ref_id}.json"), record)
        append_jsonl(summary_jsonl, record)

        if idx % args.save_every == 0 or idx == len(ref_ids):
            summary = {
                "task": args.task,
                "split": args.split,
                "total": len(ref_ids),
                "valid": valid_count,
                "miou": float(np.mean(metrics_iou)) if metrics_iou else 0.0,
                "acc@0.5": float(num_50 / max(valid_count, 1)),
                "acc@0.75": float(num_75 / max(valid_count, 1)),
                "config": vars(args),
            }
            save_json(os.path.join(args.output_root, "summary.json"), summary)

    final_summary = {
        "task": args.task,
        "split": args.split,
        "total": len(ref_ids),
        "valid": valid_count,
        "miou": float(np.mean(metrics_iou)) if metrics_iou else 0.0,
        "acc@0.5": float(num_50 / max(valid_count, 1)),
        "acc@0.75": float(num_75 / max(valid_count, 1)),
        "config": vars(args),
    }
    save_json(os.path.join(args.output_root, "summary.json"), final_summary)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
