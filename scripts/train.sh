#!/usr/bin/env bash
set -x

PROJ_ROOT=xxx
VERL_ROOT=xxx
WANDB_DIR=xxx

DATA_ROOT=xxx
TRAIN_FILE=$DATA_ROOT/xxx
VAL_FILE=$DATA_ROOT/xxx
VALID_DIR=$PROJ_ROOT/xxx

ACTOR_MODEL=xxx
SRC_TOKENIZER_PATH=$ACTOR_MODEL

GRM_MODEL=xxx
GRM_TOKENIZER_PATH=$GRM_MODEL
GRM_TEMPLATE=/templates/slove_qwen.txt

CUDA_VISIBLE_DEVICES=0,1,2,3
N_GPUS_PER_NODE=2
N_NODES=1
GRM_NNODES=1
PROJECT_NAME=xxx
EXPERIMENT_NAME=xxx
TOTAL_EPOCHS=10

# ====== 环境变量 ======
export PYTHONPATH=$VERL_ROOT:$PYTHONPATH
export WANDB_MODE=$WANDB_MODE
export WANDB_DIR=$WANDB_DIR
export SRC_TOKENIZER_PATH=$SRC_TOKENIZER_PATH

# ====== 启动训练 ======
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$ACTOR_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=20 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=5 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=5 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=lpem \
    +reward_model.rm_tokenizer_path=$GRM_TOKENIZER_PATH \
    +reward_model.grm.enable=True \
    +reward_model.grm.model.path=$GRM_MODEL \
    +reward_model.grm.rollout.tokenizer_path=$GRM_TOKENIZER_PATH \
    +reward_model.grm.rollout.src_tokenizer_path=$ACTOR_MODEL \
    +reward_model.grm.rollout.prompt_length=5120 \
    +reward_model.grm.rollout.response_length=8192 \
    +reward_model.grm.rollout.m=1 \
    +reward_model.grm.rollout.temperature=0 \
    +reward_model.grm.rollout.max_num_batched_tokens=16384 \
    +reward_model.grm.rollout.ulysses_sequence_parallel_size=1 \
    +reward_model.grm.rollout.tensor_model_parallel_size=1 \
    +reward_model.grm.rollout.fsdp_config.fsdp_size=-1 \
    +reward_model.grm.rollout.template_file=$GRM_TEMPLATE \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.validation_data_dir=$VALID_DIR \
    trainer.log_val_generations=5 \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$N_NODES \
    +trainer.grm_nnodes=$GRM_NNODES \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    "$@"
