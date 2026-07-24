"""Bulk Tier-1 evaluation of the testckpt.md '1'-marked checkpoints, on the
CURRENTLY checked-out git branch.

Writes one JSON per ckpt to <out_dir>/<branch>__<dir>.json (incremental, so a
crash keeps finished results). WikiText-103 is tokenized once and reused.

IMPORTANT: a checkpoint is only scored correctly when the checked-out branch
matches the code it was trained with. Non-LoRA variants (mhc/lite/embedding/
group_embedding/hc/shc) are branch-invariant; LoRA/group-lora variants are not.
Run this per relevant branch and take the min-CE per ckpt (see make_report.py).

Run from the project root:
  HF_ENDPOINT=https://hf-mirror.com python eval/run_all.py --branch inbeta --wt103 --device cuda:1
"""
import os
import sys
import json
import math
import argparse
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from loader import load_ckpt, get_encoder, full_logits
from eval_ppl import owt_val_ppl, BLOCK

MARKED = {
    "L": [
        "out-owt-large-mhc",
        "out-owt-large-mhc-group-embedding",
        "out-owt-large-mhc-group-lora",
        "out-owt-large-mhc-group-lora-bs4-20000step",
        "out-owt-large-mhc-group-lora-bs8-20000step",
        "out-owt-large-mhc-group-lora-midnorm-bs4-20000step",
        "out-owt-large-mhc-lite",
        "out-owt-large-mhc-lora-residual",
        "out-owt-large-mhc-lora-residual-midnorm-bs8-20000step",
        "out-owt-large-mhc-lora-residual-midnorm-inbeta-bs8-20000step",
    ],
    "M": [
        "out-owt-medium-hc-bs16-10000step",
        "out-owt-medium-mhc",
        "out-owt-medium-mhc-group-embedding",
        "out-owt-medium-mhc-group-lora",
        "out-owt-medium-mhc-group-lora-midnorm-bs16-10000step",
        "out-owt-medium-mhc-group-lora-midnorm-inbeta-bs16-10000step",
        "out-owt-medium-mhc-lite",
        "out-owt-medium-mhc-lora-residual",
        "out-owt-medium-mhc-lora-residual-local-jul13",
        "out-owt-medium-mhc-lora-residual-midnorm-bs16-10000step",
        "out-owt-medium-mhc-lora-residual-scalar",
    ],
    "S": [
        "out--small-mhc-lite",
        "out-owt-small-mhc",
        "out-owt-small-mhc-group-embedding",
        "out-owt-small-mhc-group-lora",
        "out-owt-small-mhc-group-lora-bs24-10000step",
        "out-owt-small-mhc-lite",
        "out-owt-small-mhc-lora-residual-bs24-10000step",
        "smoke-owt-small-mhc-lite",
    ],
}


def wt103_ids():
    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    return torch.tensor(get_encoder().encode_ordinary("\n\n".join(ds["text"])), dtype=torch.long)


@torch.no_grad()
def wt103_from_ids(model, ids, window=BLOCK, stride=512, device="cuda"):
    N = ids.numel()
    tot_loss, tot_tok, prev = 0.0, 0, 0
    for begin in range(0, N, stride):
        end = min(begin + window, N)
        trg = end - prev
        chunk = ids[begin:end].to(device)
        lg = full_logits(model, chunk[None])[0].float()
        sl = lg[:-1]
        st = chunk[1:].clone()
        if trg < st.numel():
            st[:-trg] = -100
        tot_loss += F.cross_entropy(sl, st, ignore_index=-100, reduction="sum").item()
        tot_tok += int((st != -100).sum().item())
        prev = end
        if end == N:
            break
    ce = tot_loss / tot_tok
    return round(ce, 6), round(math.exp(ce), 4), tot_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/work/data/guotianzizhe/data/test")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--branch", required=True, help="tag for output filenames (e.g. inbeta, midnorm)")
    ap.add_argument("--out_dir", default="eval/results/bulk")
    ap.add_argument("--tiers", default="L,M,S")
    ap.add_argument("--dirs", default="", help="explicit comma-separated dir list (overrides tiers)")
    ap.add_argument("--filter", default="", help="only dirs containing this substring")
    ap.add_argument("--wt103", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.dirs:
        todo = [(None, d) for d in args.dirs.split(",")]
    else:
        todo = [(t, d) for t in args.tiers.split(",") for d in MARKED[t]]
    if args.filter:
        todo = [(t, d) for (t, d) in todo if args.filter in d]

    ids = wt103_ids() if args.wt103 else None
    print(f"[run_all] branch={args.branch} device={args.device} n={len(todo)} wt103={args.wt103}", flush=True)

    for tier, d in todo:
        ckpt = os.path.join(args.root, d, "ckpt.pt")
        out = os.path.join(args.out_dir, f"{args.branch}__{d}.json")
        rec = {"dir": d, "tier": tier, "branch": args.branch, "ckpt": ckpt}
        if not os.path.exists(ckpt):
            rec["error"] = "missing ckpt.pt"
            json.dump(rec, open(out, "w"), indent=2)
            print(f"SKIP(missing) {d}", flush=True)
            continue
        try:
            model, ck = load_ckpt(ckpt, device=args.device)
            ma = ck["model_args"]
            rec["hyper_conn_type"] = ma.get("hyper_conn_type")
            rec["n_layer"] = ma.get("n_layer")
            rec["n_embd"] = ma.get("n_embd")
            rec["iter"] = int(ck.get("iter_num", -1))
            blk = int(ma.get("block_size", BLOCK))   # some ckpts were trained at block_size != 1024
            rec["block_size"] = blk
            rec.update(owt_val_ppl(model, "data/openwebtext/val.bin", block=blk, batch=16, device=args.device))
            if ids is not None:
                ce, ppl, tk = wt103_from_ids(model, ids, window=blk, stride=max(1, blk // 2), device=args.device)
                rec.update(wt103_ce=ce, wt103_ppl=ppl, wt103_tokens=tk)
            print(f"OK [{tier}] {d}: owt_ppl={rec['owt_val_ppl']} wt103_ppl={rec.get('wt103_ppl','-')}", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"ERR {d}: {e}", flush=True)
        json.dump(rec, open(out, "w"), indent=2)

    print("[run_all] done", flush=True)


if __name__ == "__main__":
    main()
