"""Windowing/context-length diagnostic for the mHC rolling-PPL discrepancy.

Question under test: is XL-group-lora's poor Paloma-WikiText score (vs XL-mhc) a
real effect, or an artifact of Paloma's canonical rolling giving each token almost
no left context? We hold the *text* fixed (Paloma paloma_wikitext_103 test docs)
and the *model* fixed, and vary ONLY the windowing:

  Protocol A  canonical context_len=1  (what Paloma / eval_paloma.py reports):
              lm_eval.utils.get_rolling_token_windows(max_seq_len=1024, context_len=1)
              -> tokens at window starts see ~1 token of left context.
  Protocol B  stride=512 rolling, per doc (the eval_ppl.py:wt103_rolling_ppl logic):
              window=1024, stride=512, left-context masked via ignore_index
              -> every token sees 512-1024 tokens of left context.

Both prepend <eot> as BOS and score EVERY token of each doc exactly once, so token
counts / bytes are identical across protocols -> token_ppl and bits_per_byte are
directly comparable; they differ only in how much context each token is scored with.

If group-lora improves MORE than mhc when going A->B, the "reversal" is a
context-length effect, not OOD generalization. fp32, eval, no grad, offline cache.

Usage:
  python eval/diag_ppl_protocol.py --device cuda:2 \
      --ckpts XL-mhc=/home/work/data/guotianzizhe/data/eval_round2/out-owt-xl-mhc-bs6-80000step \
              XL-grouplora=/home/work/data/guotianzizhe/data/test/out-owt-xl-mhc-group-lora-midnorm-bs6-80000step
"""
import os
import re
import sys
import math
import argparse

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from loader import load_ckpt, get_encoder, full_logits
from lm_eval.utils import get_rolling_token_windows

BLOCK = 1024

@torch.no_grad()
def nll_canonical(model, tokens, eot, device):
    """Protocol A: context_len=1 canonical rolling; sum NLL (nats) over all tokens."""
    nll = 0.0
    for inp, pred in get_rolling_token_windows(tokens, eot, BLOCK, 1):
        x = torch.tensor(inp, dtype=torch.long, device=device)
        logits = full_logits(model, x[None])[0].float()          # [Li, V]
        p = len(pred)
        sel = logits[len(inp) - p:]                              # rows predicting pred
        tgt = torch.tensor(pred, dtype=torch.long, device=device)
        nll += float(F.cross_entropy(sel, tgt, reduction="sum").item())
    return nll


@torch.no_grad()
def nll_stride(model, tokens, eot, device, window=BLOCK, stride=512):
    """Protocol B: eval_ppl.py-style stride rolling, per doc; sum NLL (nats)."""
    ids = torch.tensor([eot] + list(tokens), dtype=torch.long, device=device)
    n = ids.numel()
    nll, prev_end = 0.0, 0
    for begin in range(0, n, stride):
        end = min(begin + window, n)
        chunk = ids[begin:end]
        logits = full_logits(model, chunk[None])[0].float()
        sl = logits[:-1]                          # position t predicts chunk[t+1]
        st = chunk[1:].clone()
        trg = end - prev_end                      # new tokens this window
        if trg < st.numel():
            st[:-trg] = -100                      # score only the new tokens
        nll += float(F.cross_entropy(sl, st, ignore_index=-100, reduction="sum").item())
        prev_end = end
        if end == n:
            break
    return nll


def eval_model(ckpt, docs, device):
    model, ck = load_ckpt(ckpt, device=device)
    model.eval()
    enc = get_encoder()
    eot = enc.eot_token
    agg = {"nllA": 0.0, "nllB": 0.0, "tok": 0, "byt": 0, "wrd": 0}
    for text in docs:
        toks = enc.encode_ordinary(text)
        if not toks:
            continue
        agg["nllA"] += nll_canonical(model, toks, eot, device)
        agg["nllB"] += nll_stride(model, toks, eot, device)
        agg["tok"] += len(toks)
        agg["byt"] += len(text.encode("utf-8"))
        agg["wrd"] += len(re.split(r"\s+", text.strip())) if text.strip() else 0
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return agg

def metrics(agg):
    ln2 = math.log(2)
    return {
        "A_token_ppl": math.exp(agg["nllA"] / agg["tok"]),
        "A_word_ppl": math.exp(agg["nllA"] / agg["wrd"]),
        "A_bpb": agg["nllA"] / (agg["byt"] * ln2),
        "B_token_ppl": math.exp(agg["nllB"] / agg["tok"]),
        "B_word_ppl": math.exp(agg["nllB"] / agg["wrd"]),
        "B_bpb": agg["nllB"] / (agg["byt"] * ln2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--ckpt-name", default="ckpt_last.pt")
    ap.add_argument("--limit", type=int, default=None, help="first N docs (debug)")
    ap.add_argument("--ckpts", nargs="+", required=True, help="NAME=dir_or_ckpt pairs")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("allenai/paloma", "wikitext_103", split="test")
    docs = [str(t) for t in ds["text"]]
    if args.limit:
        docs = docs[:args.limit]
    print(f"[data] paloma_wikitext_103 test: {len(docs)} docs")

    out = {}
    for spec in args.ckpts:
        name, _, path = spec.partition("=")
        ckpt = os.path.join(path, args.ckpt_name) if os.path.isdir(path) else path
        print(f"[run] {name}  {ckpt}")
        m = metrics(eval_model(ckpt, docs, args.device))
        out[name] = m
        print(f"    A (canon ctx=1) : token_ppl={m['A_token_ppl']:.4f}  word_ppl={m['A_word_ppl']:.4f}  bpb={m['A_bpb']:.4f}")
        print(f"    B (stride=512)  : token_ppl={m['B_token_ppl']:.4f}  word_ppl={m['B_word_ppl']:.4f}  bpb={m['B_bpb']:.4f}")
        print(f"    A->B bpb drop   : {m['A_bpb'] - m['B_bpb']:.4f}  ({100*(m['A_bpb']-m['B_bpb'])/m['A_bpb']:.2f}%)")

    if len(out) == 2:
        (n1, a), (n2, b) = out.items()
        print(f"\n[gap] {n2} minus {n1} (positive => {n2} worse)")
        print(f"    protocol A (canonical) : d_bpb={b['A_bpb']-a['A_bpb']:+.4f}   d_word_ppl={b['A_word_ppl']-a['A_word_ppl']:+.3f}")
        print(f"    protocol B (stride=512): d_bpb={b['B_bpb']-a['B_bpb']:+.4f}   d_word_ppl={b['B_word_ppl']-a['B_word_ppl']:+.3f}")
        print(f"    => the {n2}-vs-{n1} gap changes by {(b['B_bpb']-a['B_bpb'])-(b['A_bpb']-a['A_bpb']):+.4f} bpb when adding context (A->B)")


if __name__ == "__main__":
    main()
