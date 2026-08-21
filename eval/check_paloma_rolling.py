"""Correctness check for the mHC lm-eval rolling log-likelihood (the Paloma path).

Paloma tasks use output_type=loglikelihood_rolling: the harness feeds each
document to MHCLMAdapter.loglikelihood_rolling and converts the returned total
log-prob into word/byte perplexity and bits-per-byte. Before any large Paloma
run we prove that total log-prob is correct by cross-checking the adapter against
references built on the project's OWN CE primitive (F.cross_entropy over
loader.full_logits -- the exact loss used by eval_ppl.owt_val_ppl / wt103_rolling_ppl):

  Test A  single window (short text): isolates the token shift -- logits[t] must
          predict ids[t+1]. An off-by-one shows up as a large disagreement.
  Test B  harness-equivalent rolling (long text): re-scores the document using
          lm_eval.utils.get_rolling_token_windows(context_len=1) -- the harness's
          OWN window generator -- and its prescribed scoring ("score only the last
          len(pred) logits"). Agreement proves the adapter matches the canonical
          Paloma windowing, so its perplexity is comparable to any other model's.

Data: cached WikiText-103 test split (offline; no network, no gradients).

Usage:
  python eval/check_paloma_rolling.py --ckpt data/test/out-owt-small-mhc --device cuda:0
"""
import os
import sys
import math
import argparse

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from loader import load_ckpt, get_encoder, full_logits
from lm_adapter import MHCLMAdapter
from lm_eval.utils import get_rolling_token_windows

BLOCK = 1024


class _Req:
    """Minimal stand-in for an lm_eval Instance (adapter only reads .args)."""
    def __init__(self, text):
        self.args = (text,)

@torch.inference_mode()
def plain_ll(model, ids, device):
    """Single-window reference: run the model on ids[:-1] and sum
    log p(ids[t+1] | ids[:t+1]).  Direct, no windowing -- isolates the shift."""
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device)[None]
    logits = full_logits(model, x)[0].float()          # [L-1, V]; logits[t] -> ids[t+1]
    logp = F.log_softmax(logits, dim=-1)
    tgt = torch.tensor(ids[1:], dtype=torch.long, device=device)
    return float(logp.gather(-1, tgt[:, None]).squeeze(-1).sum().item()), len(ids) - 1


@torch.inference_mode()
def harness_ll(model, token_list, eot, device, max_seq_len=BLOCK):
    """Reference over lm_eval's OWN canonical windows (context_len=1), but scored
    INDEPENDENTLY of the adapter: F.cross_entropy with ignore_index masking (the
    eval_ppl loss formulation) and hand-built targets, instead of the adapter's
    log_softmax+gather over the last len(pred) rows. Sharing only the window
    partition (the scheme every model uses on these benchmarks) makes an exact
    equality check possible; the scoring math is derived separately."""
    total_nll, n = 0.0, 0
    for inp, pred in get_rolling_token_windows(
        token_list=token_list, prefix_token=eot, max_seq_len=max_seq_len, context_len=1
    ):
        x = torch.tensor(inp, dtype=torch.long, device=device)
        logits = full_logits(model, x[None])[0].float()        # [Li, V]; logits[j]->inp[j+1]
        li, p = len(inp), len(pred)
        tgt = torch.full((li,), -100, dtype=torch.long, device=device)
        if p >= 2:                                             # inner scored positions
            tgt[li - p: li - 1] = x[li - p + 1: li]            #   predict inp[li-p+1 .. li-1]
        tgt[li - 1] = int(pred[-1])                            # last logit predicts pred[-1]
        total_nll += float(F.cross_entropy(logits, tgt, ignore_index=-100, reduction="sum"))
        n += p
    return -total_nll, n



@torch.inference_mode()
def adapter_ll(adapter, text):
    """The value Paloma actually consumes (loglikelihood_rolling returns list[float])."""
    return float(adapter.loglikelihood_rolling([_Req(text)])[0])


def _wikitext_text():
    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    return "\n\n".join(t for t in ds["text"] if t and t.strip())

def _fmt(ll, n):
    return f"ll={ll:.4f}  tokens={n}  token_ppl={math.exp(-ll / n):.4f}"


def main():
    ap = argparse.ArgumentParser(description="Verify the Paloma rolling log-likelihood")
    ap.add_argument("--ckpt", required=True, help="ckpt file or its out-* directory")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max |adapter - reference| in nats over the whole doc")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="adapter batch size (>1 exercises right-padded window batching)")
    args = ap.parse_args()

    ckpt = os.path.join(args.ckpt, "ckpt.pt") if os.path.isdir(args.ckpt) else args.ckpt
    model, ck = load_ckpt(ckpt, device=args.device)
    model.eval()
    enc = get_encoder()
    eot = enc.eot_token
    adapter = MHCLMAdapter(model=model, device=args.device, batch_size=args.batch_size)
    print(f"[ckpt] {ckpt}  type={ck['model_args'].get('hyper_conn_type')} "
          f"n_layer={ck['model_args'].get('n_layer')}")

    full = _wikitext_text()
    short = full[:3000]
    while len(enc.encode_ordinary(short)) > BLOCK - 8:      # keep to a single window
        short = short[: int(len(short) * 0.9)]
    long = full[:16000]                                     # forces several windows

    # ---- Test A: single window -> isolates the token shift ----
    a_ad = adapter_ll(adapter, short)
    a_ref, a_n = plain_ll(model, [eot] + enc.encode_ordinary(short), args.device)
    print("\n[Test A] single window (token shift)")
    print("  adapter  :", _fmt(a_ad, a_n))
    print("  ref (CE) :", _fmt(a_ref, a_n))
    print(f"  |diff|   = {abs(a_ad - a_ref):.6e} nats")

    # ---- Test B: multi-window rolling vs the harness's own window generator ----
    tl = enc.encode_ordinary(long)
    b_ad = adapter_ll(adapter, long)
    b_ref, b_n = harness_ll(model, tl, eot, args.device)
    nwin = 1 + max(0, (len(tl) - 1) // BLOCK)
    print(f"\n[Test B] rolling, {len(tl)} tokens (~{nwin} windows, context_len=1)")
    print("  adapter        :", _fmt(b_ad, b_n))
    print("  ref (CE/canon) :", _fmt(b_ref, b_n))
    print(f"  |diff|         = {abs(b_ad - b_ref):.6e} nats")
    # illustrative Paloma-style metrics for the long sample
    import re
    words = len(re.split(r"\s+", long))
    nbytes = len(long.encode("utf-8"))
    print(f"  word_ppl={math.exp(-b_ad / words):.4f}  byte_ppl={math.exp(-b_ad / nbytes):.4f}"
          f"  bits_per_byte={-b_ad / nbytes / math.log(2):.4f}")

    ok = abs(a_ad - a_ref) < args.tol and abs(b_ad - b_ref) < args.tol
    same_n = (a_n == len(enc.encode_ordinary(short))) and (b_n == len(tl))
    print(f"\n[RESULT] shift+rolling agree within {args.tol} nats: {ok}; "
          f"token coverage exact: {same_n}")
    sys.exit(0 if (ok and same_n) else 1)


if __name__ == "__main__":
    main()
