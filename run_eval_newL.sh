#!/bin/bash
# 两个 L 档模型的两套评测（都用 best-val ckpt.pt，cuda:1，因为 2,3 在训练）：
#   MHC = out-owt-large-mhc-bs8-30000step               (纯 mHC, cosine, 7.86B)  —— 从远端拉来的
#   GLM = out-owt-large-mhc-group-lora-midnorm-wsd-bs8-30000step (in-beta, wsd, 7.86B) —— 本地新训
# Flow1: run_all.py(owt+wt103 PPL) + run_tasks.py(lambada + 下游 acc)
# Flow2: eval_paloma.py(C4/Dolma/Falcon/RedPajama/WikiText 的 word_ppl/bpb)
# 结果写到全新目录 *_new，不覆盖 round2 旧结果。
set -u
cd /home/work/data/guotianzizhe/project/mhc-lite
PY=/home/work/data/guotianzizhe/venv/bin/python
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_TRUST_REMOTE_CODE=1
DEV=cuda:1
ROOT=/home/work/data/guotianzizhe/data/test
MHC=out-owt-large-mhc-bs8-30000step
GLM=out-owt-large-mhc-group-lora-midnorm-wsd-bs8-30000step
DIRS="$MHC,$GLM"

echo "############ FLOW1a run_all (owt + wt103 PPL)  $(date) ############"
$PY eval/run_all.py --branch newL --device "$DEV" --root "$ROOT" --dirs "$DIRS" --wt103 \
  --out_dir eval/results/bulk_new --ckpt_name ckpt.pt
echo "rc=$?  $(date)"

echo "############ FLOW1b run_tasks (lambada + 下游 acc)  $(date) ############"
$PY eval/run_tasks.py --branch newL --device "$DEV" --root "$ROOT" --dirs "$DIRS" --batch_size 16 \
  --out_dir eval/results/tasks_new --ckpt_name ckpt.pt
echo "rc=$?  $(date)"

echo "############ FLOW2 eval_paloma (5 corpora bpb/ppl)  $(date) ############"
$PY eval/eval_paloma.py --checkpoints "$ROOT/$MHC" "$ROOT/$GLM" --device "$DEV" --batch-size 8 \
  --output-dir eval/results/paloma_new --ckpt-name ckpt.pt
echo "rc=$?  $(date)"
echo "############ ALL EVAL DONE  $(date) ############"
