#!/bin/bash
# =============================================================================
# Stream-R1 — Reliability-Complexity Aware Reward Distillation
#
# Combines (all driven by a single video reward model):
#   1. Inter-Reliability Weighting   — exp(beta * r_final) per-rollout multiplier
#                                       (reward_mode: BalancedOverall)
#   2. Intra-Complexity (spatial)    — per-pixel reward-gradient saliency,
#                                       adaptively combined across VQ/MQ/TA
#   3. Intra-Complexity (temporal)   — per-frame importance from the same
#                                       saliency volume
#   4. Adaptive Reward Balancing     — penalty on std of per-dim improvement
#
# After training: generates 20 videos from the final checkpoint.
# =============================================================================
set -e

# ========================== User-tunable settings ==========================
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}   # GPU IDs
NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-29710}

NUM_INFERENCE_PROMPTS=20
NUM_OUTPUT_FRAMES=21
OUTPUT_BASE="output"
PROMPT_FILE="prompts/MovieGenVideoBench.txt"

# Unified hyperparameter overrides
FULL_TRAINING_STEPS=1000
GRAD_ACCUM_STEPS=8
LOG_ITERS=200
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$OUTPUT_BASE"
mkdir -p exp_configs

# ---- Build the 20-prompt eval file ----------------------------------------
EVAL_PROMPT_FILE="prompts/eval_${NUM_INFERENCE_PROMPTS}.txt"
head -n "$NUM_INFERENCE_PROMPTS" "$PROMPT_FILE" > "$EVAL_PROMPT_FILE"

# ---- Helper: create a merged temp config with common + extra overrides -----
make_config() {
    local base_yaml="$1"     # path to original yaml
    local out_yaml="$2"      # where to write merged yaml
    local extra_overrides="${3:-}"   # optional python snippet to apply extras
    python3 - <<PYEOF
from omegaconf import OmegaConf

cfg = OmegaConf.load("${base_yaml}")

# Common overrides
cfg.full_training_steps       = ${FULL_TRAINING_STEPS}
cfg.gradient_accumulation_steps = ${GRAD_ACCUM_STEPS}
cfg.log_iters                 = ${LOG_ITERS}

# Extra per-experiment overrides
${extra_overrides}

OmegaConf.save(cfg, "${out_yaml}")
print(f"[make_config] Saved merged config to ${out_yaml}")
PYEOF
}

# ---- Helper: extract generator weights from checkpoint ---------------------
extract_weights() {
    local ckpt_path="$1"
    local out_path="$2"
    python3 - <<PYEOF
import torch
ckpt = torch.load("${ckpt_path}", map_location="cpu")
if isinstance(ckpt, dict) and "generator_ema" in ckpt:
    print("Extracting generator_ema weights...")
    torch.save(ckpt["generator_ema"], "${out_path}")
elif isinstance(ckpt, dict) and "generator" in ckpt:
    print("Extracting generator weights...")
    torch.save(ckpt["generator"], "${out_path}")
else:
    print("Checkpoint is already raw weights, copying...")
    torch.save(ckpt, "${out_path}")
print(f"Saved to ${out_path}")
PYEOF
}

# ---- Main runner: train + infer one experiment ----------------------------
run_experiment() {
    local exp_name="$1"
    local tmp_config="$2"      # pre-built temp config path

    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local exp_dir="${OUTPUT_BASE}/${timestamp}_${exp_name}"
    mkdir -p "$exp_dir"

    echo ""
    echo "======================================================================"
    echo " Experiment : ${exp_name}"
    echo " Config     : ${tmp_config}"
    echo " Output     : ${exp_dir}"
    echo " GPUs       : ${NUM_GPUS}  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
    echo " Steps      : full_training_steps=${FULL_TRAINING_STEPS}  grad_accum=${GRAD_ACCUM_STEPS}"
    echo "              → ${FULL_TRAINING_STEPS}×${GRAD_ACCUM_STEPS}=$(( FULL_TRAINING_STEPS * GRAD_ACCUM_STEPS )) raw steps"
    echo "======================================================================"

    # -------------------- Training --------------------
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun \
        --nnodes=1 \
        --nproc_per_node="$NUM_GPUS" \
        --rdzv_id=$RANDOM \
        --rdzv_backend=c10d \
        --rdzv_endpoint="localhost:${MASTER_PORT}" \
        train.py \
        --config_path "$tmp_config" \
        --logdir "$exp_dir" \
        --disable-wandb

    echo ""
    echo "[${exp_name}] Training complete. Checkpoints in: ${exp_dir}"

    # -------------------- Find last checkpoint --------------------
    local last_ckpt_dir
    last_ckpt_dir=$(ls -d "${exp_dir}"/checkpoint_model_* 2>/dev/null | sort | tail -1)
    if [ -z "$last_ckpt_dir" ]; then
        echo "[${exp_name}] ERROR: No checkpoint found under ${exp_dir}. Skipping inference."
        return 1
    fi

    local ckpt_file="${last_ckpt_dir}/model.pt"
    local gen_file="${last_ckpt_dir}/generator.pt"

    echo "[${exp_name}] Using checkpoint: ${ckpt_file}"
    extract_weights "$ckpt_file" "$gen_file"

    # -------------------- Inference --------------------
    local video_dir="${exp_dir}/videos"
    mkdir -p "$video_dir"

    echo ""
    echo "[${exp_name}] Generating ${NUM_INFERENCE_PROMPTS} videos..."
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python3 inference.py \
        --config_path "$tmp_config" \
        --checkpoint_path "$gen_file" \
        --data_path "$EVAL_PROMPT_FILE" \
        --output_folder "$video_dir" \
        --num_output_frames "$NUM_OUTPUT_FRAMES"

    echo ""
    echo "[${exp_name}] Done. Videos saved to: ${video_dir}"
    echo "======================================================================"

    # Bump port to avoid c10d conflicts between sequential runs
    MASTER_PORT=$(( MASTER_PORT + 1 ))
}

# ===========================================================================
# Build temp config for the Stream-R1 experiment
# ===========================================================================

CFG_STREAM_R1="exp_configs/stream_r1.yaml"

echo ""
echo "########################################################################"
echo "#  Building merged config (full_training_steps=${FULL_TRAINING_STEPS}, grad_accum=${GRAD_ACCUM_STEPS})"
echo "########################################################################"
echo ""

make_config "configs/exp_stream_r1.yaml" "$CFG_STREAM_R1" ""

# ===========================================================================
# Run the experiment
# ===========================================================================

echo ""
echo "########################################################################"
echo "#  Starting Stream-R1 experiment"
echo "########################################################################"
echo ""

run_experiment "stream_r1" "$CFG_STREAM_R1"

# ===========================================================================
# Summary
# ===========================================================================

echo ""
echo "########################################################################"
echo "#  Stream-R1 experiment complete!"
echo "########################################################################"
echo ""
echo "Results under: ${OUTPUT_BASE}/"
ls -lhd "${OUTPUT_BASE}"/*/
