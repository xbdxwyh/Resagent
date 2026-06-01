#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-7}"
PYTHON="${PYTHON:-python}"
SCRIPT="${SCRIPT:-resagent_full_eval_point_strategies_textual_qwen25_fix1.py}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-VL-3B-Instruct}"
ROUTER_NAME="${ROUTER_NAME:-medoid}"
BOX_JSON_ROOT="${BOX_JSON_ROOT:-../../ResAgentv2/router_out}"
OUTPUT_BASE="${OUTPUT_BASE:-./resagent_qwen25_3b}"
SAM2_CKPT="${SAM2_CKPT:-../checkpoints/checkpoint.pt}"
SAM2_CFG="${SAM2_CFG:-configs/sam2.1/sam2.1_hiera_b+.yaml}"
POINT_STRATEGY="${POINT_STRATEGY:-learned}"
LEARNED_CKPT="${LEARNED_CKPT:-./ckpt_useful_value_mix/best_epoch5.pt}"
LEARNED_CONFIG_JSON="${LEARNED_CONFIG_JSON:-./ckpt_useful_value_mix/config.json}"
LEARNED_BASE_STRATEGY="${LEARNED_BASE_STRATEGY:-random}"
LEARNED_POOL_SCALE="${LEARNED_POOL_SCALE:-4}"
INTERNAL_CANDIDATES="${INTERNAL_CANDIDATES:-10}"
MAX_INTERNAL_POINTS="${MAX_INTERNAL_POINTS:-4}"
MAX_EXTERNAL_POINTS="${MAX_EXTERNAL_POINTS:-0}"
POINT_SELECTION_MODE="${POINT_SELECTION_MODE:-confidence}"
LIMIT="${LIMIT:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

build_json_path() {
  local task="$1"
  local split="$2"
  if [[ "$task" == "refcocog" ]]; then
    echo "${BOX_JSON_ROOT}/refcocog_umd_${split}_${ROUTER_NAME}_choices.jsonl"
  else
    echo "${BOX_JSON_ROOT}/${task}_unc_${split}_${ROUTER_NAME}_choices.jsonl"
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

  local out_dir="${OUTPUT_BASE}/${task}_${split}_${ROUTER_NAME}"
  mkdir -p "$out_dir"
  local log_file="${out_dir}/run.log"

  echo "============================================================"
  echo "Model : ${MODEL_NAME}"
  echo "Task  : ${task}"
  echo "Split : ${split}"
  echo "Router: ${ROUTER_NAME}"
  echo "JSON  : ${box_json}"
  echo "Out   : ${out_dir}"
  echo "Log   : ${log_file}"
  echo "============================================================"

  local cmd=(
    "$PYTHON" "$SCRIPT"
    --task "$task"
    --split "$split"
    --model_name "$MODEL_NAME"
    --box_input_json "$box_json"
    --box_prior_mode fixed
    --fixed_box_field raw_text
    --box_coord_mode absolute
    --prefer_sentence_level_prior
    --sam2_checkpoint_path "$SAM2_CKPT"
    --sam2_config_path "$SAM2_CFG"
    --point_strategy "$POINT_STRATEGY"
    --learned_ckpt_path "$LEARNED_CKPT"
    --learned_config_json "$LEARNED_CONFIG_JSON"
    --learned_base_strategy "$LEARNED_BASE_STRATEGY"
    --learned_pool_scale "$LEARNED_POOL_SCALE"
    --internal_candidates "$INTERNAL_CANDIDATES"
    --max_internal_points "$MAX_INTERNAL_POINTS"
    --max_external_points "$MAX_EXTERNAL_POINTS"
    --point_selection_mode "$POINT_SELECTION_MODE"
    # --save_debug_images
    --output_root "$out_dir"
  )

  if [[ -n "$LIMIT" ]]; then
    cmd+=(--limit "$LIMIT")
  fi

  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    local extra_arr=( $EXTRA_ARGS )
    cmd+=("${extra_arr[@]}")
  fi

  CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" 2>&1 | tee "$log_file"
}

# for split in val testA testB; do
#   run_one refcoco "$split"
# done


# for split in val testA testB; do
#   run_one refcoco+ "$split"
# done



for split in val test; do
  run_one refcocog "$split"
done