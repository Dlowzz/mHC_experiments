#!/bin/bash
# mhc-lite 四档（S/M/L/XL）串行训练，GPU 2,3 + venv。
# 每档的 batch_size / grad_accum / max_iters / warmup / lr_decay 与 round2 各档
# 完全一致（取自 round2 ckpt 里保存的 config，逐档核对）：
#   S  bs32 ga8  10000 步  warmup200  decay10000  lr1e-3   (config/small_model.py)
#   M  bs24 ga8  18000 步  warmup200  decay18000  lr6e-4   (config/medium_model.py)
#   L  bs8  ga16 30000 步  warmup2000 decay20000  lr3e-4   (config/large_model.py)
#   XL bs6  ga10 80000 步  warmup2000 decay50000  lr2e-4   (config/xl_model_5b.py)
# 变体统一 config/with_mhc_lite.py（hyper_conn_type=mhc_lite, n=4）。
# 数据/日志间隔/always_save/dtype 由 config/train_owt.py 提供（eval_interval=500,
# always_save_checkpoint=False, dataset=openwebtext, out_prefix_dataset=owt, bf16）。
# 输出目录: ./out-owt-<档>-mhc-lite-bs<bs>-<步数>step（项目目录下，与 round2 L 档一致）。
# 用 `;` 串联：某一档失败后面仍继续。日志各档 tee 到 logs/。
set -u
cd /home/work/data/guotianzizhe/project/mhc-lite
PY=/home/work/data/guotianzizhe/venv/bin/python
export WANDB_MODE=online
RUN="$PY -m torch.distributed.run --standalone --nproc_per_node=2 train.py config/train_owt.py"

echo "######## 1/4 S mhc-lite  $(date) ########"
CUDA_VISIBLE_DEVICES=2,3 $RUN config/small_model.py config/with_mhc_lite.py \
  --batch_size=32 --gradient_accumulation_steps=8 --max_iters=10000 --lr_decay_iters=10000 --warmup_iters=200 --compile=True \
  --wandb_run_name=S-mhc-lite-owt-bs32ga8-10kstep 2>&1 | tee logs/S-mhc-lite-bs32-10kstep.log
echo "######## 1/4 done exit=${PIPESTATUS[0]}  $(date) ########"

echo "######## 2/4 M mhc-lite  $(date) ########"
CUDA_VISIBLE_DEVICES=2,3 $RUN config/medium_model.py config/with_mhc_lite.py \
  --batch_size=24 --gradient_accumulation_steps=8 --max_iters=18000 --lr_decay_iters=18000 --warmup_iters=200 --compile=True \
  --wandb_run_name=M-mhc-lite-owt-bs24ga8-18kstep 2>&1 | tee logs/M-mhc-lite-bs24-18kstep.log
echo "######## 2/4 done exit=${PIPESTATUS[0]}  $(date) ########"

echo "######## 3/4 L mhc-lite  $(date) ########"
CUDA_VISIBLE_DEVICES=2,3 $RUN config/large_model.py config/with_mhc_lite.py \
  --batch_size=8 --gradient_accumulation_steps=16 --max_iters=30000 --lr_decay_iters=20000 --warmup_iters=2000 --compile=True \
  --wandb_run_name=L-mhc-lite-owt-bs8ga16-30kstep 2>&1 | tee logs/L-mhc-lite-bs8-30kstep.log
echo "######## 3/4 done exit=${PIPESTATUS[0]}  $(date) ########"

echo "######## 4/4 XL mhc-lite  $(date) ########"
CUDA_VISIBLE_DEVICES=2,3 $RUN config/xl_model_5b.py config/with_mhc_lite.py \
  --batch_size=6 --gradient_accumulation_steps=10 --max_iters=80000 --lr_decay_iters=50000 --warmup_iters=2000 --compile=True \
  --wandb_run_name=XL-mhc-lite-owt-bs6ga10-80kstep 2>&1 | tee logs/XL-mhc-lite-bs6-80kstep.log
echo "######## 4/4 done exit=${PIPESTATUS[0]}  $(date) ########"
echo "ALL DONE $(date)"
