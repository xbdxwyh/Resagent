#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"
SCRIPT="${SCRIPT:-resagent_full_eval_point_strategies_textual_qwen25_fix1.py}"
MODEL_NAME="${MODEL_NAME:-8B-Instruct}"
MODEL_TAG="${MODEL_TAG:-qwen3-vl-8b-instruct}"
BOX_JSON_ROOT="${BOX_JSON_ROOT:-../../ResAgentv2}"
OUTPUT_BASE="${OUTPUT_BASE:-./resagent-baseline}"
SAM2_CKPT="${SAM2_CKPT:-../checkpoints/sam2.1_hiera_base_plus.pt}"
SAM2_CFG="${SAM2_CFG:-configs/sam2.1/sam2.1_hiera_b+.yaml}"
TEXTUAL_PROMPT_FORMAT="${TEXTUAL_PROMPT_FORMAT:-x_eq_y_eq}"
POINT_STRATEGY="${POINT_STRATEGY:-random}"
INTERNAL_CANDIDATES="${INTERNAL_CANDIDATES:-10}"
MAX_INTERNAL_POINTS="${MAX_INTERNAL_POINTS:-4}"
MAX_EXTERNAL_POINTS="${MAX_EXTERNAL_POINTS:-0}"
POINT_SELECTION_MODE="${POINT_SELECTION_MODE:-confidence}"
TEXTUAL_COORD_SPACE="${TEXTUAL_COORD_SPACE:-crop}"
TEXTUAL_COORD_FORMAT="${TEXTUAL_COORD_FORMAT:-qwen1000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

build_json_path() {
  local task="$1"
  local split="$2"
  if [[ "$task" == "refcocog" ]]; then
    echo "${BOX_JSON_ROOT}/pred_refcocog_umd_${split}_${MODEL_TAG}.jsonl"
  else
    echo "${BOX_JSON_ROOT}/pred_${task}_unc_${split}_${MODEL_TAG}.jsonl"
  fi
}

run_one() {
  local task="$1"
  local split="$2"
  local box_json
  box_json="$(build_json_path "$task" "$split")"

  if [[ ! -f "$box_json" ]]; then
    echo "[WARN] Missing box_input_json, skip: $box_json"
    return 0
  fi

  local out_dir="${OUTPUT_BASE}/resagent_${MODEL_TAG}_${POINT_STRATEGY}_${task}_${split}_${TEXTUAL_PROMPT_FORMAT}"
  mkdir -p "$out_dir"
  local log_file="${out_dir}/run.log"

  echo "============================================================"
  echo "Model : ${MODEL_NAME} (${MODEL_TAG})"
  echo "Task  : ${task}"
  echo "Split : ${split}"
  echo "JSON  : ${box_json}"
  echo "Out   : ${out_dir}"
  echo "Log   : ${log_file}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$SCRIPT" \
    --task "$task" \
    --split "$split" \
    --model_name "$MODEL_NAME" \
    --box_input_json "$box_json" \
    --box_prior_mode fixed \
    --fixed_box_field raw_text \
    --box_coord_mode qwen1000 \
    --prefer_sentence_level_prior \
    --sam2_checkpoint_path "$SAM2_CKPT" \
    --sam2_config_path "$SAM2_CFG" \
    --point_strategy "$POINT_STRATEGY" \
    --internal_candidates "$INTERNAL_CANDIDATES" \
    --max_internal_points "$MAX_INTERNAL_POINTS" \
    --max_external_points "$MAX_EXTERNAL_POINTS" \
    --point_selection_mode "$POINT_SELECTION_MODE" \
    --validation_mode textual_point \
    --textual_prompt_format "$TEXTUAL_PROMPT_FORMAT" \
    --textual_coord_space "$TEXTUAL_COORD_SPACE" \
    --textual_coord_format "$TEXTUAL_COORD_FORMAT" \
    --output_root "$out_dir" \
    $EXTRA_ARGS 2>&1 | tee "$log_file"
}

for split in val testA testB; do
  run_one refcoco "$split"
done

for split in val testA testB; do
  run_one refcoco+ "$split"
done

for split in val test; do
  run_one refcocog "$split"
done

