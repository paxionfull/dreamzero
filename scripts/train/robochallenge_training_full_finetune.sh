#!/bin/bash
# DreamZero DROID Full Fine-Tuning Script (8x H100, ZeRO-2 + CPU Offload)
#
# Usage:
#   bash scripts/train/droid_training_full_finetune.sh
#
# Prerequisites:
#   - DROID dataset in LeRobot format at DROID_DATA_ROOT
#     Download: huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir ./data/droid_lerobot
#     Or convert from scratch: see scripts/data/convert_droid.py
#   - Wan2.1-I2V-14B-480P weights (auto-downloaded or pre-downloaded from HuggingFace)
#     Download: huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
#   - umt5-xxl tokenizer (auto-downloaded or pre-downloaded from HuggingFace)
#     Download: huggingface-cli download google/umt5-xxl --local-dir ./checkpoints/umt5-xxl

export HYDRA_FULL_ERROR=1

# ============ USER CONFIGURATION ============
# Dataset path (DROID in LeRobot format)
ROBOCHALLENGE_DATA_ROOT=${ROBOCHALLENGE_DATA_ROOT:-"/mnt/public/algm/datasets/Table30_lerobot_0.3.4/put_cup_on_coaster"}

# Output directory for training checkpoints
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/robochallenge_put_cup_on_coaster_ft"}

# Model weight paths (download from HuggingFace if not already present)
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"/mnt/public/algm/models/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"/mnt/public/algm/models/umt5-xxl"}

# Number of GPUs to use (8x H100 for ZeRO-2 + CPU offload full fine-tuning)
NUM_GPUS=${NUM_GPUS:-8}
PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH:-"/mnt/public/algm/models/DreamZero-DROID"}

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/robochallenge_relative \
    wandb_project=dreamzero \
    train_architecture=full \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2_offload.json" \
    training_args.gradient_accumulation_steps=4 \
    save_steps=100 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=3000 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=2 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=false \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=no \
    robochallenge_data_root=$ROBOCHALLENGE_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=$PRETRAIN_MODEL_PATH
