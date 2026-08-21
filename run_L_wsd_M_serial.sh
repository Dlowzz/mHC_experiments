#!/bin/bash
# 串行：L 档 group-lora-midnorm(in-beta) + mhc-lite（均 WSD），最后补 M 档 mhc-lite（cosine）。
# GPU 2,3 + venv。某一档失败后面仍继续（`;` 串联 + 每档 tee 日志）。
#
# L 档（两个变体一致）—— 与 wandb run L-mhc-owt-bs8ga32-30kstep-wsd 完全一致：
#   bs8 ga32 (tokens/iter=8*32*1024=262,144, 30k 步共 7.86B)
#   WSD: warmup 0->3e-4 (0-500) -> 恒定 3e-4 (500-25500) -> 余弦降到 3e-5 (25500-30000)
#   group-lora = config/with_mhc_group_lora_midnorm.py（mhc_group_lora.depth_connection 里
#     u_s = beta_s(h + lambda*delta_s)，lambda 默认 1.0 —— 即 in-beta）
# M 档 —— 沿用原 M 配置，仅把 bs24 ga8 换成 bs16 ga12（同 196,608 tokens/iter）修 OOM：
#   18000 步 / warmup 200 / cosine / lr 6e-4->6e-5（config/medium_model.py 默认）
set -u
cd /home/work/data/guotianzizhe/project/mhc-lite
PY=/home/work/data/guotianzizhe/venv/bin/python
export WANDB_MODE=online
RUN="$PY -m torch.distributed.run --standalone --nproc_per_node=2 train.py config/train_owt.py"

# L 档公共 WSD 覆盖项
LWSD="--batch_size=8 --gradient_accumulation_steps=32 --max_iters=30000 \
--warmup_iters=500 --lr_schedule=wsd --lr_decay_start_iters=25500 --lr_decay_iters=30000 \
--learning_rate=3e-4 --min_lr=3e-5 --compile=True"

echo "######## 1/3 L group-lora-midnorm (in-beta) WSD  $(date) ########"
CUDA_VISIBLE_DEVICES=2,3 $RUN config/large_model.py config/with_mhc_group_lora_midnorm.py $LWSD \
  --out_prefix_method=mhc-group-lora-midnorm-wsd \
  --wandb_run_name=L-group-lora-midnorm-owt-bs8ga32-30kstep-wsd \
  2>&1 | tee logs/L-group-lora-midnorm-bs8ga32-30kstep-wsd.log
echo "######## 1/3 done exit=${PIPESTATUS[0]}  $(date) ########"
sleep 20   # 让上一档的 CUDA 显存彻底释放，避免串行启动时的抢显存

echo "######## 2/3 L mhc-lite WSD  $(date) ########"
CUDA_VISIBLE_DEVICES=2,3 $RUN config/large_model.py config/with_mhc_lite.py $LWSD \
  --out_prefix_method=mhc-lite-wsd \
  --wandb_run_name=L-mhc-lite-owt-bs8ga32-30kstep-wsd \
  2>&1 | tee logs/L-mhc-lite-bs8ga32-30kstep-wsd.log
echo "######## 2/3 done exit=${PIPESTATUS[0]}  $(date) ########"
sleep 20

echo "######## 3/3 M mhc-lite (cosine, bs16 ga12)  $(date) ########"
CUDA_VISIBLE_DEVICES=2,3 $RUN config/medium_model.py config/with_mhc_lite.py \
  --batch_size=16 --gradient_accumulation_steps=12 --max_iters=18000 \
  --lr_decay_iters=18000 --warmup_iters=200 --compile=True \
  --wandb_run_name=M-mhc-lite-owt-bs16ga12-18kstep \
  2>&1 | tee logs/M-mhc-lite-bs16ga12-18kstep.log
echo "######## 3/3 done exit=${PIPESTATUS[0]}  $(date) ########"
echo "ALL DONE $(date)"
