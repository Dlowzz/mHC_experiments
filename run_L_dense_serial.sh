#!/bin/bash
# dense H_pre 消融：L 档两个变体串行跑在 GPU 2,3 上，设置与之前的 L 档 LoRA 系实验一致。
#
#   mhc_group_dense_embedding      <- mhc_group_embedding 的 dense (n^3*C) 读对照
#   mhc_group_lora_dense_midnorm   <- mhc_group_lora_midnorm 的 dense (n^3*C) 读对照
#
#   L 档原始 lr：3e-4 -> min 3e-5（config/large_model.py）
#   bs=8, grad_accum=16 -> 16 x 8 x 1024 = 131,072 tokens/iter（DDP 会把 ga 除以卡数）
#   30,000 步 = 3.93B tokens
#   warmup 2,000；余弦 decay 到第 20,000 步；最后 10,000 步恒定在 min_lr
#   （train.py 的 get_lr 在 it > lr_decay_iters 时返回 min_lr）
#
# 这里是一次性跑满 30,000 步，不是先 25k 再续跑，所以 lr 曲线与 25k+resume 的两个
# 基线完全重合（decay 段只由 warmup_iters / lr_decay_iters 决定，与 max_iters 无关）。
#
# 用 `;` 而非 `&&` 串联：第一个 run 挂了第二个也要跑。
set -u
cd /home/work/data/guotianzizhe/project/mhc-lite

COMMON="--batch_size=8 --gradient_accumulation_steps=16 \
--max_iters=30000 --lr_decay_iters=20000 --warmup_iters=2000 --compile=True"

echo "############ 1/2  mhc-group-dense (mhc_group_dense_embedding)  $(date) ############"
WANDB_MODE=online CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 train.py \
    config/train_owt.py config/large_model.py config/with_mhc_group_dense_embedding.py \
    $COMMON --wandb_run_name=L-group-dense-owt-bs8ga16-30kstep \
    2>&1 | tee logs/L-group-dense-bs8ga16-30kstep.log
echo "############ 1/2 结束，exit=$?  $(date) ############"

echo "############ 2/2  mhc-group-lora-dense (mhc_group_lora_dense_midnorm)  $(date) ############"
WANDB_MODE=online CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 train.py \
    config/train_owt.py config/large_model.py config/with_mhc_group_lora_dense_midnorm.py \
    $COMMON --wandb_run_name=L-group-lora-dense-midnorm-owt-bs8ga16-30kstep \
    2>&1 | tee logs/L-group-lora-dense-midnorm-bs8ga16-30kstep.log
echo "############ 2/2 结束，exit=$?  $(date) ############"
