#!/bin/bash
# 评测 3 个模型（都用 best-val ckpt.pt，cuda:1，因为 2,3 在训练 M mhc-lite）：
#   L group-embedding : data/test/out-owt-large-mhc-group-embedding-bs8-30000step (远端最新 08/19)
#   S mhc-lite        : <repo>/out-owt-small-mhc-lite-bs32-10000step
#   L mhc-lite (WSD)  : <repo>/out-owt-large-mhc-lite-wsd-bs8-30000step
# Flow1: run_all.py(owt+wt103 PPL) + run_tasks.py(lambada + 下游 acc)
# Flow2: eval_paloma.py(C4/Dolma/Falcon/RedPajama/WikiText)
# 结果追加进 eval/results/{bulk_new,tasks_new,paloma_new}（与之前 2 个 L 结果同目录，不覆盖）。
set -u
cd /home/work/data/guotianzizhe/project/mhc-lite
PY=/home/work/data/guotianzizhe/venv/bin/python
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_TRUST_REMOTE_CODE=1
DEV=cuda:1
TROOT=/home/work/data/guotianzizhe/data/test        # group-embedding 在这
LROOT=/home/work/data/guotianzizhe/project/mhc-lite  # 两个 mhc-lite 在这
GRP=out-owt-large-mhc-group-embedding-bs8-30000step
LITES=out-owt-small-mhc-lite-bs32-10000step,out-owt-large-mhc-lite-wsd-bs8-30000step

echo "############ FLOW1a run_all (owt + wt103 PPL)  $(date) ############"
$PY eval/run_all.py --branch ext --device "$DEV" --root "$TROOT" --dirs "$GRP" --wt103 \
  --out_dir eval/results/bulk_new --ckpt_name ckpt.pt ; echo "rc=$?"
$PY eval/run_all.py --branch ext --device "$DEV" --root "$LROOT" --dirs "$LITES" --wt103 \
  --out_dir eval/results/bulk_new --ckpt_name ckpt.pt ; echo "rc=$?  $(date)"

echo "############ FLOW1b run_tasks (lambada + 下游 acc)  $(date) ############"
$PY eval/run_tasks.py --branch ext --device "$DEV" --root "$TROOT" --dirs "$GRP" --batch_size 8 \
  --out_dir eval/results/tasks_new --ckpt_name ckpt.pt ; echo "rc=$?"
$PY eval/run_tasks.py --branch ext --device "$DEV" --root "$LROOT" --dirs "$LITES" --batch_size 8 \
  --out_dir eval/results/tasks_new --ckpt_name ckpt.pt ; echo "rc=$?  $(date)"

echo "############ FLOW2 eval_paloma (5 corpora)  $(date) ############"
$PY eval/eval_paloma.py --checkpoints \
  "$TROOT/$GRP" \
  "$LROOT/out-owt-small-mhc-lite-bs32-10000step" \
  "$LROOT/out-owt-large-mhc-lite-wsd-bs8-30000step" \
  --device "$DEV" --batch-size 8 --output-dir eval/results/paloma_new --ckpt-name ckpt.pt ; echo "rc=$?  $(date)"
echo "############ ALL EVAL DONE  $(date) ############"
