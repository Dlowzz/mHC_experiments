#!/bin/bash
# L 档 LoRA 系两个变体，串行跑在 GPU 2,3 上。
#
#   L 档原始 lr：3e-4 -> min 3e-5（config/large_model.py）
#   bs=8, grad_accum=16  -> 16 x 8 x 1024 = 131,072 tokens/iter（DDP 会把 ga 除以卡数，与卡数无关）
#   25,000 步 = 3.28B tokens
#   warmup 2,000；余弦 decay 到第 20,000 步；最后 5,000 步恒定在 min_lr
#   （train.py 的 get_lr 在 it > lr_decay_iters 时返回 min_lr）
#
# 用 `;` 而非 `&&` 串联：第一个 run 挂了第二个也要跑，好把两个变体的速度/显存都量到。
set -u
cd /home/work/data/guotianzizhe/project/mhc-lite

COMMON="--batch_size=8 --gradient_accumulation_steps=16 \
--max_iters=25000 --lr_decay_iters=20000 --warmup_iters=2000 --compile=True"

echo "############ 1/2  mhc-lora (mhc_lora_residual_midnorm)  $(date) ############"
WANDB_MODE=online CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 train.py \
    config/train_owt.py config/large_model.py config/with_mhc_lora_residual_midnorm.py \
    $COMMON --wandb_run_name=L-lora-midnorm-owt-bs8ga16-25kstep \
    2>&1 | tee logs/L-lora-midnorm-bs8ga16-25kstep.log
echo "############ 1/2 结束，exit=$?  $(date) ############"

echo "############ 2/2  mhc-group-lora (mhc_group_lora_midnorm)  $(date) ############"
WANDB_MODE=online CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 train.py \
    config/train_owt.py config/large_model.py config/with_mhc_group_lora_midnorm.py \
    $COMMON --wandb_run_name=L-group-lora-midnorm-owt-bs8ga16-25kstep \
    2>&1 | tee logs/L-group-lora-midnorm-bs8ga16-25kstep.log
echo "############ 2/2 结束，exit=$?  $(date) ############"
