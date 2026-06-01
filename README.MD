# ResAgent

This repository contains the research code for **ResAgent: Entropy-based Point Discovery and Visual Reasoning for Referring Expression Segmentation**.

ResAgent follows a coarse-to-fine **discover-and-validate** pipeline for referring expression segmentation (RES). Given an image and a referring expression, it first uses routed coarse box priors, then discovers informative point candidates with EBD, verifies them through marker-grounded VBR, and finally decodes the mask with SAM2.

## Overview

The main pipeline contains four stages:

1. **Coarse Box Prior Acquisition**  
   Multiple upstream box predictions are routed into a single coarse prior. We provide precomputed routed box jsonl files under `router_eval_medoid/`.

2. **Entropy-Based Point Discovery (EBD)**  
   Candidate points are selected inside the coarse prior. The code supports random, grid, spiral/geometric, and learned point discovery strategies.

3. **Vision-Based Reasoning (VBR)**  
   Candidate points are rendered as visual markers and verified by a frozen MLLM through structured multi-marker queries.

4. **SAM2 Mask Decoding**  
   The routed box and validated positive/negative point prompts are passed to SAM2 for final mask prediction.

## Repository Structure

```text
Resagent/
├── README.MD
├── LICENSE
├── .gitignore
├── .gitattributes
│
├── run_resagent_router_qwen2_5.sh
├── run_textual_baseline_qwen3.sh
│
├── merged_router_clip_vlm_single_style.py
├── router_by_qwen3vl_overlay_multi_expert_v2.py
├── tools.py
├── transfercoco2SA1Bformat.ipynb
│
├── router_eval_medoid/
│   ├── refcoco_unc_val_medoid_choices.jsonl
│   ├── refcoco_unc_testA_medoid_choices.jsonl
│   ├── refcoco_unc_testB_medoid_choices.jsonl
│   ├── refcoco+_unc_val_medoid_choices.jsonl
│   ├── refcoco+_unc_testA_medoid_choices.jsonl
│   ├── refcoco+_unc_testB_medoid_choices.jsonl
│   ├── refcocog_umd_val_medoid_choices.jsonl
│   ├── refcocog_umd_test_medoid_choices.jsonl
│   └── *_summary.json
│
├── ckpt_useful_value_mix/
│   └── config.json
│
└── sam2/
    ├── build_sam2_teacher_dataset.py
    ├── build_sam2_teacher_dataset_iou.py
    ├── build_sam2_teacher_dataset_iou_spiral.py
    ├── train_point_useful_value_samstyle_imgpathfix_v5.py
    ├── train_point_useful_value_samstyle_imgpathfix_v6_mix.py
    ├── resagent_full_eval_point_strategies_final.py
    ├── resagent_full_eval_point_strategies_textual_qwen25_fix1.py
    ├── refer.py
    ├── grefer.py
    ├── sam2_hiera_b+.yaml
    ├── sam2_hiera_l.yaml
    ├── sam2_hiera_s.yaml
    ├── sam2_hiera_t.yaml
    └── ...
````

## Data Preparation

Please prepare the RefCOCO-family datasets in the common REFER format:

```text
datasets/
├── images/
│   ├── train2014/
│   └── val2014/
├── refcoco/
├── refcoco+/
└── refcocog/
```

The code uses `refer.py` and `grefer.py` for loading RefCOCO, RefCOCO+, and RefCOCOg annotations.

## Coarse Box Priors

We provide routed coarse box priors in `router_eval_medoid/`. Each jsonl file corresponds to one dataset split, for example:

```text
router_eval_medoid/refcoco_unc_testA_medoid_choices.jsonl
router_eval_medoid/refcoco+_unc_testA_medoid_choices.jsonl
router_eval_medoid/refcocog_umd_test_medoid_choices.jsonl
```

These files are used by the evaluation script through `--box_input_json`.

The router code is provided in:

```text
merged_router_clip_vlm_single_style.py
```

It supports several routing and fusion strategies, including:

```text
oracle_best
all_mean
trimmed_mean
medoid
vote_mean
clip_top1
clip_wmean
clip_filtered_mean
clip_weighted_vote_mean
vlm_router
```

The paper mainly uses the **Medoid Router** as the default coarse prior strategy.

## Running ResAgent Evaluation

An example script is provided:

```bash
bash run_resagent_router_qwen2_5.sh
```

The core command calls:

```bash
python sam2/resagent_full_eval_point_strategies_textual_qwen25_fix1.py \
  --task refcoco \
  --split testA \
  --model_name Qwen2.5-VL-3B-Instruct \
  --box_input_json router_eval_medoid/refcoco_unc_testA_medoid_choices.jsonl \
  --box_prior_mode fixed \
  --fixed_box_field raw_text \
  --box_coord_mode absolute \
  --prefer_sentence_level_prior \
  --sam2_checkpoint_path path/to/sam2_checkpoint.pt \
  --sam2_config_path sam2/sam2_hiera_b+.yaml \
  --point_strategy learned \
  --learned_ckpt_path ckpt_useful_value_mix/best_epoch5.pt \
  --learned_config_json ckpt_useful_value_mix/config.json \
  --learned_base_strategy random \
  --learned_pool_scale 4 \
  --internal_candidates 10 \
  --max_internal_points 4 \
  --max_external_points 0 \
  --point_selection_mode confidence \
  --output_root ./outputs/resagent_eval
```

The shell scripts are example commands. Some default values reflect intermediate experiments and can be overridden by environment variables. Please use the released routed box jsonl files, learned EBD checkpoint, and SAM2 checkpoint/config corresponding to your target setting.

## Running Textual Coordinate Baselines

To evaluate textualized coordinate verification baselines, use:

```bash
bash run_textual_baseline_qwen3.sh
```

This runs the same evaluation pipeline but switches point verification to textual coordinate prompting:

```bash
--validation_mode textual_point
--textual_prompt_format x_eq_y_eq
--textual_coord_space crop
--textual_coord_format qwen1000
```

Other textual formats can be specified by changing `TEXTUAL_PROMPT_FORMAT`.

## Learned EBD Proxy

The learned EBD proxy is trained from SAM2 teacher data. The teacher data stores point-wise segmentation gains under positive and negative point prompts.

### Build SAM2 Teacher Data

Random-point teacher data:

```bash
python sam2/build_sam2_teacher_dataset_iou.py \
  --task refcoco \
  --split train \
  --seed 2026 \
  --num_points 64 \
  --bbox_min_ratio 0.05 \
  --bbox_max_ratio 0.15 \
  --sam2_variant base_plus \
  --out_dir ./sam2_teacher_npz_iou
```

Spiral/geometric-point teacher data:

```bash
python sam2/build_sam2_teacher_dataset_iou_spiral.py \
  --task refcoco \
  --split train \
  --seed 2026 \
  --num_points 64 \
  --num_internal 32 \
  --num_external 32 \
  --bbox_min_ratio 0.05 \
  --bbox_max_ratio 0.15 \
  --sam2_variant base_plus \
  --out_dir ./sam2_teacher_npz_iou
```

Each `.npz` file stores the perturbed box, candidate points, point membership, box-only IoU, positive/negative prompted IoU, and the corresponding point-wise gain.

### Train Learned Point Utility Predictor

```bash
python sam2/train_point_useful_value_samstyle_imgpathfix_v6_mix.py \
  --train_npz_dirs ./sam2_teacher_npz_iou/refcoco_train_seed2026_N64_pert0.05-0.15_base_plus_dIoU/ \
  --train_tags refcoco \
  --val_npz_dirs ./sam2_teacher_npz_iou/refcoco_testA_seed2026_N64_pert0.05-0.15_base_plus_dIoU/ \
  --val_tags refcoco_testA \
  --sam2_ckpt path/to/sam2_checkpoint.pt \
  --sam2_cfg sam2/sam2_hiera_b+.yaml \
  --epochs 5 \
  --batch_size 8 \
  --k_top 5 \
  --seed 3 \
  --save_dir ./ckpt_useful_value_mix \
  --head_type decoder \
  --decoder_dim 256 \
  --decoder_layers 2 \
  --decoder_nhead 8 \
  --decoder_ffn_mult 4 \
  --decoder_mem_downsample 1 \
  --use_prompt_embed \
  --use_box_prompt \
  --use_base_mask \
  --useful_thresh 0.0 \
  --w_cls 1.0 \
  --w_val 0.2 \
  --w_rank 0.2 \
  --rank_T 0.5 \
  --val_transform log \
  --val_eps 1e-6 \
  --val_clip 1.0 \
  --lr_scheduler cosine_warmup \
  --warmup_steps 500 \
  --lr_min 1e-6 \
  --use_ema \
  --ema_decay 0.999 \
  --ema_eval \
  --mix_strategy uniform
```

## Optional SAM2 Adaptation

For the full model, SAM2 can be lightly adapted on COCO train2014. We first convert COCO instance annotations into the SA-1B-style annotation format used by the SAM2 training pipeline.

The conversion notebook is:

```text
transfercoco2SA1Bformat.ipynb
```

It converts COCO instance annotations into per-image JSON files compatible with SAM2-style raw dataset loading. After conversion, SAM2 can be fine-tuned using the official SAM2 training recipe and the corresponding config.

The adapted SAM2 checkpoint is used for the full model, while the official SAM2 checkpoint can be used for the no-adaptation variant.

## Important Notes

* The released shell scripts are intended as runnable examples.
* Some paths and default hyperparameters may need to be adjusted according to your local environment.
* The paper results are obtained with the released routed box priors, learned EBD checkpoint, and the corresponding SAM2 checkpoint/config.
* The MLLM backbone is kept frozen in all ResAgent variants.
* The learned EBD proxy and optional SAM2 adaptation do not fine-tune the MLLM.

## Citation

If you find this code useful, please cite our paper:

```bibtex
@article{resagent2026,
  title={ResAgent: Entropy-based Point Discovery and Visual Reasoning for Referring Expression Segmentation},
  author={Wang, Yihao and Zhang, Jusheng and Tang, Ziyi and He, Zijian and Wang, Keze and Yang, Meng},
  journal={arXiv preprint arXiv:2601.16394},
  year={2026}
}
```
