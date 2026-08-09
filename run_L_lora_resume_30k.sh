#!/bin/bash
# L 档 LoRA 系两个变体：从 25000 步续跑到 30000 步，lr 恒定 3e-5。
#
# lr：沿用原配置 lr_decay_iters=20000 / min_lr=3e-5。train.py 的 get_lr 在
#     it > lr_decay_iters 时直接返回 min_lr，所以 25000->30000 全程恒定 3e-5，
#     不需要额外参数，也不能用 --decay_lr=False（那会退回 learning_rate=3e-4）。
#
# 断点：train.py 的 init_from=resume 只读 out_dir/ckpt.pt，而 out_dir 名字里带
#     max_iters。所以先建 ...-bs8-30000step/ 并把 25000step 的 **ckpt_last.pt**
#     （iter 25000 的真实末态；ckpt.pt 是 best-val，停在 21500 / 24500）硬链接成
#     新目录的 ckpt.pt。硬链接不额外占盘，原目录不动。
#
# wandb：用 WANDB_RUN_ID + WANDB_RESUME=allow 续写原来的 run，曲线接在一起而不是
#     新开一条。原 run 已记到 step 25000，续跑第一次 eval 也在 25000，会被 wandb
#     当作非单调 step 丢弃（一条警告），从 25010 开始正常追加。
set -u
cd /home/work/data/guotianzizhe/project/mhc-lite

COMMON="--batch_size=8 --gradient_accumulation_steps=16 \
--max_iters=30000 --lr_decay_iters=20000 --warmup_iters=2000 \
--compile=True --init_from=resume"

prep () {   # prep <method-dir-name>
    local old="out-owt-large-$1-bs8-25000step" new="out-owt-large-$1-bs8-30000step"
    mkdir -p "$new"
    if [ ! -f "$new/ckpt.pt" ]; then
        ln "$old/ckpt_last.pt" "$new/ckpt.pt"
        echo "### $new/ckpt.pt <- $old/ckpt_last.pt (hardlink)"
    else
        echo "### $new/ckpt.pt 已存在，跳过"
    fi
    ls -l "$new"
}

prep mhc-lora-residual-midnorm
prep mhc-group-lora-midnorm

echo "############ 1/2  mhc-lora 25000->30000  $(date) ############"
WANDB_MODE=online WANDB_RUN_ID=xb58eiyb WANDB_RESUME=allow \
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 train.py \
    config/train_owt.py config/large_model.py config/with_mhc_lora_residual_midnorm.py \
    $COMMON --wandb_run_name=L-lora-midnorm-owt-bs8ga16-30kstep \
    2>&1 | tee logs/L-lora-midnorm-bs8ga16-30kstep-resume.log
echo "############ 1/2 结束 exit=$?  $(date) ############"

echo "############ 2/2  mhc-group-lora 25000->30000  $(date) ############"
WANDB_MODE=online WANDB_RUN_ID=z5whemei WANDB_RESUME=allow \
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 train.py \
    config/train_owt.py config/large_model.py config/with_mhc_group_lora_midnorm.py \
    $COMMON --wandb_run_name=L-group-lora-midnorm-owt-bs8ga16-30kstep \
    2>&1 | tee logs/L-group-lora-midnorm-bs8ga16-30kstep-resume.log
echo "############ 2/2 结束 exit=$?  $(date) ############"
