"""Custom lm-evaluation-harness adapter for the mHC GPT checkpoints (Tier-2).

The mHC model is a bespoke architecture (multi residual streams + beta write-back
+ depth-LoRA), so it cannot be loaded via `--model hf`. This wraps model.py's GPT
as an lm_eval `LM`, implementing token-level `loglikelihood` (and a rolling variant)
on top of `loader.full_logits` (full-sequence logits; the plain forward only returns
the last position).

Only loglikelihood-style tasks are supported (lambada_openai + the MC suite);
`generate_until` is intentionally unimplemented in v1.

Usage (datasets pulled through the HF mirror):
  HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_TRUST_REMOTE_CODE=1 \
  python eval/lm_adapter.py --ckpt <dir_or_ckpt.pt> --device cuda:1 \
      --tasks lambada_openai,piqa --limit 200 --out eval/results/<name>_lmeval.json
"""
import os
import sys
import json
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from loader import load_ckpt, get_encoder, full_logits

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import get_rolling_token_windows

BLOCK = 1024


@register_model("mhc_gpt")
class MHCLMAdapter(LM):
    def __init__(self, ckpt=None, device="cuda", batch_size=1, max_length=BLOCK, model=None, **kwargs):
        super().__init__()
        self._device = device
        self.enc = get_encoder()
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.model = model if model is not None else load_ckpt(ckpt, device=device)[0]
        self.model.eval()
        self._rank = 0
        self._world_size = 1

    @property
    def eot_token_id(self):
        return self.enc.eot_token

    def tok_encode(self, s):
        return self.enc.encode_ordinary(s)

    @torch.no_grad()
    def _score_pair(self, ctx_ids, cont_ids):
        """(sum log p(cont | ctx), is_greedy) with the full continuation kept."""
        keep_ctx = self.max_length + 1 - len(cont_ids)
        ctx_ids = ctx_ids[-keep_ctx:] if keep_ctx > 0 else []
        if not ctx_ids:
            ctx_ids = [self.eot_token_id]
        inp = ctx_ids + cont_ids
        ctx_len = len(ctx_ids)
        x = torch.tensor(inp[:-1], dtype=torch.long, device=self._device)[None]
        logits = full_logits(self.model, x)[0].float()          # [L-1, V]; logits[t] predicts inp[t+1]
        logprobs = F.log_softmax(logits, dim=-1)
        tgt = torch.tensor(inp[ctx_len:], dtype=torch.long, device=self._device)
        sel = logprobs[ctx_len - 1: ctx_len - 1 + tgt.numel()]   # rows predicting the continuation
        tok_ll = sel.gather(-1, tgt[:, None]).squeeze(-1)
        return float(tok_ll.sum().item()), bool((sel.argmax(-1) == tgt).all().item())

    @torch.no_grad()
    def loglikelihood(self, requests):
        # tokenize all pairs
        items = []
        out = [None] * len(requests)
        for i, req in enumerate(requests):
            ctx, cont = req.args
            ctx_ids = self.tok_encode(ctx) if ctx else [self.eot_token_id]
            cont_ids = self.tok_encode(cont)
            if not cont_ids:
                out[i] = (0.0, True)
                continue
            items.append((i, ctx_ids, cont_ids))
        # sort by length for efficient right-padded batching (causal LM: right-pad is safe,
        # padded positions come after the scored tokens and never influence them)
        items.sort(key=lambda t: len(t[1]) + len(t[2]))
        for b0 in range(0, len(items), self.batch_size):
            chunk = items[b0:b0 + self.batch_size]
            seqs, metas = [], []
            for oi, ctx_ids, cont_ids in chunk:
                keep_ctx = self.max_length + 1 - len(cont_ids)
                c = ctx_ids[-keep_ctx:] if keep_ctx > 0 else [self.eot_token_id]
                if not c:
                    c = [self.eot_token_id]
                inp = c + cont_ids
                seqs.append(inp[:-1])
                metas.append((oi, len(c), cont_ids))
            maxlen = max(len(s) for s in seqs)
            x = torch.full((len(seqs), maxlen), self.eot_token_id, dtype=torch.long, device=self._device)
            for k, s in enumerate(seqs):
                x[k, :len(s)] = torch.tensor(s, dtype=torch.long, device=self._device)
            lp = torch.log_softmax(full_logits(self.model, x).float(), dim=-1)   # [B, maxlen, V]
            for k, (oi, ctx_len, cont_ids) in enumerate(metas):
                n = len(cont_ids)
                sel = lp[k, ctx_len - 1: ctx_len - 1 + n]
                tgt = torch.tensor(cont_ids, dtype=torch.long, device=self._device)
                ll = float(sel.gather(-1, tgt[:, None]).squeeze(-1).sum().item())
                out[oi] = (ll, bool((sel.argmax(-1) == tgt).all().item()))
        return out

    @torch.no_grad()
    def loglikelihood_rolling(self, requests):
        # Rolling PPL (Paloma / wikitext, output_type=loglikelihood_rolling).
        # Windows come from lm_eval's OWN generator (context_len=1) so the windowing
        # matches every other model on these benchmarks, and each input is <= max_length
        # tokens (fits block_size). logits[t] predicts inp[t+1]; score the last len(pred)
        # rows. Windows are independent sequences, so we right-pad and batch them through
        # the model together -- for a causal LM the padding sits AFTER every scored token
        # and never influences it, so batched == unbatched exactly (verified in
        # eval/check_paloma_rolling.py).
        windows = []                                    # (req_idx, inp, pred)
        for ri, req in enumerate(requests):
            (string,) = req.args
            for inp, pred in get_rolling_token_windows(
                token_list=self.tok_encode(string),
                prefix_token=self.eot_token_id,
                max_seq_len=self.max_length,
                context_len=1,
            ):
                windows.append((ri, list(inp), list(pred)))
        totals = [0.0] * len(requests)
        order = sorted(range(len(windows)), key=lambda k: len(windows[k][1]))  # pack by length
        for b0 in range(0, len(order), self.batch_size):
            idxs = order[b0:b0 + self.batch_size]
            maxlen = max(len(windows[k][1]) for k in idxs)
            x = torch.full((len(idxs), maxlen), self.eot_token_id, dtype=torch.long, device=self._device)
            for j, k in enumerate(idxs):
                inp = windows[k][1]
                x[j, :len(inp)] = torch.tensor(inp, dtype=torch.long, device=self._device)
            logp = torch.log_softmax(full_logits(self.model, x).float(), dim=-1)   # [B, maxlen, V]
            for j, k in enumerate(idxs):
                ri, inp, pred = windows[k]
                li, p = len(inp), len(pred)
                sel = logp[j, li - p: li]                # rows predicting pred
                tgt = torch.tensor(pred, dtype=torch.long, device=self._device)
                totals[ri] += float(sel.gather(-1, tgt[:, None]).squeeze(-1).sum().item())
        return totals          # loglikelihood_rolling returns list[float]

    def generate_until(self, requests):
        raise NotImplementedError("v1 supports loglikelihood tasks only (no generation)")


def run(ckpt, tasks, device="cuda", limit=None, num_fewshot=0, batch_size=32):
    import lm_eval
    model, ck = load_ckpt(ckpt, device=device)
    adapter = MHCLMAdapter(model=model, device=device, batch_size=batch_size)
    res = lm_eval.simple_evaluate(model=adapter, tasks=tasks, num_fewshot=num_fewshot,
                                  limit=limit, bootstrap_iters=0)
    out = {
        "ckpt": ckpt,
        "iter": int(ck.get("iter_num", -1)),
        "hyper_conn_type": ck["model_args"].get("hyper_conn_type"),
        "n_layer": ck["model_args"].get("n_layer"),
        "n_embd": ck["model_args"].get("n_embd"),
        "num_fewshot": num_fewshot,
        "limit": limit,
    }
    for task, m in res["results"].items():
        for mk, v in m.items():
            if mk.endswith(",none") and isinstance(v, (int, float)) and not math.isnan(v):
                out[f"{task}::{mk[:-5]}"] = round(float(v), 4)
    lam = res["results"].get("lambada_openai", {})
    if lam:
        if "acc,none" in lam:
            out["lambada_acc"] = round(float(lam["acc,none"]), 4)
        if "perplexity,none" in lam:
            out["lambada_ppl"] = round(float(lam["perplexity,none"]), 4)
    return out


def resolve_ckpt(p):
    return os.path.join(p, "ckpt.pt") if os.path.isdir(p) else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tasks", default="lambada_openai,piqa")
    ap.add_argument("--limit", type=int, default=None, help="examples per task (smoke test)")
    ap.add_argument("--num_fewshot", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = run(resolve_ckpt(args.ckpt), args.tasks.split(","), device=args.device,
              limit=args.limit, num_fewshot=args.num_fewshot)
    print(json.dumps(out, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
