"""Run Tier-2 lm_eval tasks (lambada + zero-shot MC) on a list of checkpoints, on the
CURRENTLY checked-out git branch. Writes one JSON per ckpt to <out_dir>/<branch>__<dir>.json.

Run from the project root (or a worktree root). Datasets pulled via the HF mirror.
  HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_TRUST_REMOTE_CODE=1 \
  python eval/run_tasks.py --branch inbeta --dirs a,b,c --device cuda:1
"""
import os
import sys
import json
import argparse
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from lm_adapter import run as run_ckpt

DEF_TASKS = "lambada_openai,hellaswag,piqa,arc_easy,winogrande,sciq,openbookqa,copa"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/work/data/guotianzizhe/data/test")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--branch", required=True)
    ap.add_argument("--out_dir", default="eval/results/tasks")
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--tasks", default=DEF_TASKS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tasks = args.tasks.split(",")
    print(f"[run_tasks] branch={args.branch} device={args.device} tasks={tasks}", flush=True)
    for d in args.dirs.split(","):
        ckpt = os.path.join(args.root, d, "ckpt.pt")
        out = os.path.join(args.out_dir, f"{args.branch}__{d}.json")
        if not os.path.exists(ckpt):
            json.dump({"dir": d, "branch": args.branch, "error": "missing ckpt.pt"}, open(out, "w"), indent=2)
            print(f"SKIP(missing) {d}", flush=True)
            continue
        try:
            res = run_ckpt(ckpt, tasks, device=args.device, limit=args.limit, batch_size=args.batch_size)
            res["dir"] = d
            res["branch"] = args.branch
            json.dump(res, open(out, "w"), indent=2)
            summ = " ".join(f"{k}={v}" for k, v in res.items() if "::acc" in k or k in ("lambada_acc", "lambada_ppl"))
            print(f"OK {d}: {summ}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            json.dump({"dir": d, "branch": args.branch, "error": f"{type(e).__name__}: {e}"}, open(out, "w"), indent=2)
            print(f"ERR {d}: {e}", flush=True)
    print("[run_tasks] done", flush=True)


if __name__ == "__main__":
    main()
