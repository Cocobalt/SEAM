#!/usr/bin/env bash
set -x

# ========= Args =========
# Usage: bash scripts/seam_sft.sh <nproc_per_node> <save_path> [other_configs...]
if [ "$#" -lt 2 ]; then
  echo "Usage: seam_sft.sh <nproc_per_node> <save_path> [other_configs...]"
  exit 1
fi

nproc_per_node=$1
save_path=$2
shift 2   # now $@ are extra hydra overrides

# ========= Paths / Vars (fill in) =========
PROJ_ROOT=xxx
VERL_ROOT=xxx
WANDB_DIR=xxx
WANDB_MODE=${WANDB_MODE:-online}   # or: offline / disabled

DATA_ROOT=xxx
SFT_TRAIN_FILE=$DATA_ROOT/xxx_success_sft_train.parquet
SFT_VAL_FILE=$DATA_ROOT/xxx_success_sft_val.parquet

# SEAM checkpoint to continue SFT from (e.g., your latest GRPO-trained SEAM)
SEAM_MODEL=xxx   # e.g., /path/to/seam_ckpt or HF repo id
SEAM_TOKENIZER_PATH=$SEAM_MODEL

PROJECT_NAME=seam-sft
EXPERIMENT_NAME=seam-periodic-sft

# ========= Env =========
export PYTHONPATH=$VERL_ROOT:$PYTHONPATH
export WANDB_MODE=$WANDB_MODE
export WANDB_DIR=$WANDB_DIR
export SRC_TOKENIZER_PATH=$SEAM_TOKENIZER_PATH

# ========= Launch SFT =========
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files=$SFT_TRAIN_FILE \
  data.val_files=$SFT_VAL_FILE \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.max_prompt_length=2048 \
  data.max_response_length=4096 \
  \
  data.prompt_key=extra_info \
  data.response_key=extra_info \
  data.prompt_dict_keys="['problem']" \
  +data.response_dict_keys="['experience']" \
  \
  data.micro_batch_size=4 \
  optim.lr=1e-5 \
  \
  model.partial_pretrain=$SEAM_MODEL \
  use_remove_padding=true \
  ulysses_sequence_parallel_size=1 \
  \
  trainer.default_local_dir=$save_path \
  trainer.project_name=$PROJECT_NAME \
  trainer.experiment_name=$EXPERIMENT_NAME \
  trainer.logger=['wandb'] \
  trainer.total_training_steps=500 \
  trainer.save_freq=20 \
  trainer.eval_freq=20 \
  "$@"
