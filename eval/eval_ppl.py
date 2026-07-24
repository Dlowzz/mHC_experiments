"""Deterministic continuous-PPL evaluator (Tier-1 discriminator for mHC variants).

Metrics (fp32, eval mode, no sampling; identical protocol across all variants):
  * OWT val          : full non-overlapping pass over data/openwebtext/val.bin  (in-distribution)
  * WikiText-103 test: strided rolling PPL (out-of-distribution)   [needs `datasets`]

Reports CE (nats) and PPL = exp(CE). This is the authoritative held-out metric;
the in-loop `estimate_loss` (200 random batches) is only for training monitoring.

Usage:
  python eval/eval_ppl.py --ckpt <dir_or_ckpt.pt> --device cuda:1 [--wt103] \
      [--owt_val data/openwebtext/val.bin] [--batch 16] [--out eval/results/<name>.json]
"""
import os
import sys
import json
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from loader import load_ckpt, get_encoder, full_logits

BLOCK = 1024


@torch.no_grad()
def owt_val_ppl(model, bin_path, block=BLOCK, batch=16, device="cuda"):
    """Full non-overlapping pass over val.bin (deterministic, in-distribution)."""
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    n_win = (len(data) - 1) // block  # full windows with a valid shifted target
    starts = [i * block for i in range(n_win)]
    tot_loss, tot_tok = 0.0, 0
    for b0 in range(0, n_win, batch):
        idxs = starts[b0:b0 + batch]
        x = torch.from_numpy(np.stack([np.asarray(data[s:s + block], dtype=np.int64) for s in idxs])).to(device)
        y = torch.from_numpy(np.stack([np.asarray(data[s + 1:s + 1 + block], dtype=np.int64) for s in idxs])).to(device)
        logits = full_logits(model, x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        tot_loss += loss.item()
        tot_tok += y.numel()
    ce = tot_loss / tot_tok
    return dict(owt_val_ce=round(ce, 6), owt_val_ppl=round(math.exp(ce), 4),
                owt_val_tokens=int(tot_tok), owt_val_windows=int(n_win))


@torch.no_grad()
def wt103_rolling_ppl(model, window=BLOCK, stride=512, device="cuda"):
    """Strided rolling PPL over the WikiText-103 test split (OOD).

    Each window scores only the last ``end - prev_end`` tokens so that every
    token is predicted with (up to) ``window - stride`` tokens of left context,
    avoiding the cold-start over-estimate of a naive non-overlapping pass.
    """
    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = torch.tensor(get_encoder().encode_ordinary(text), dtype=torch.long)
    N = ids.numel()
    tot_loss, tot_tok, prev_end = 0.0, 0, 0
    for begin in range(0, N, stride):
        end = min(begin + window, N)
        trg_len = end - prev_end
        chunk = ids[begin:end].to(device)
        logits = full_logits(model, chunk[None])
        logits = logits[0].float()
        sl = logits[:-1]              # position t predicts chunk[t+1]
        st = chunk[1:].clone()
        if trg_len < st.numel():      # mask the left-context part; score only new tokens
            st[:-trg_len] = -100
        loss = F.cross_entropy(sl, st, ignore_index=-100, reduction="sum")
        tot_loss += loss.item()
        tot_tok += int((st != -100).sum().item())
        prev_end = end
        if end == N:
            break
    ce = tot_loss / tot_tok
    return dict(wt103_ce=round(ce, 6), wt103_ppl=round(math.exp(ce), 4), wt103_tokens=int(tot_tok))


def resolve_ckpt(p):
    return os.path.join(p, "ckpt.pt") if os.path.isdir(p) else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint file or its out-* directory")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--owt_val", default="data/openwebtext/val.bin")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--wt103", action="store_true", help="also run WikiText-103 rolling PPL")
    ap.add_argument("--wt103_stride", type=int, default=512)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    ckpt = resolve_ckpt(args.ckpt)
    model, ck = load_ckpt(ckpt, device=args.device)
    res = {
        "ckpt": ckpt,
        "iter": int(ck.get("iter_num", -1)),
        "hyper_conn_type": ck["model_args"].get("hyper_conn_type"),
        "n_layer": ck["model_args"].get("n_layer"),
        "n_embd": ck["model_args"].get("n_embd"),
    }
    if args.owt_val and os.path.exists(args.owt_val):
        res.update(owt_val_ppl(model, args.owt_val, batch=args.batch, device=args.device))
    if args.wt103:
        try:
            res.update(wt103_rolling_ppl(model, stride=args.wt103_stride, device=args.device))
        except Exception as e:
            res["wt103_error"] = f"{type(e).__name__}: {e}"
            print(f"[warn] WikiText-103 skipped ({type(e).__name__}); needs offline dataset on this box")
    print(json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
