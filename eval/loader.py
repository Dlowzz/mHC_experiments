"""Shared checkpoint loader + GPT-2 tokenizer for offline evaluation.

Import from any eval script. Adds the project root to sys.path so that
`import model` resolves regardless of the current working directory.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import tiktoken

from model import GPTConfig, GPT

_ENC = None


def get_encoder():
    """Cached GPT-2 BPE encoder (matches data/openwebtext/prepare.py)."""
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("gpt2")
    return _ENC


def load_ckpt(ckpt_path, device="cuda", dtype=torch.float32):
    """Rebuild a GPT from a training checkpoint, in fp32 eval mode.

    Strips the torch.compile ``_orig_mod.`` prefix and rebuilds from the
    stored ``model_args``. fp32 + eval() gives a deterministic forward that
    is applied identically to every variant. Returns ``(model, ckpt_dict)``.

    IMPORTANT: evaluate a checkpoint with the *code version it was trained
    with* (checkout the matching branch). ``hyper_conn_type`` alone does not
    distinguish e.g. LoRA-outside-beta vs LoRA-in-beta, so loading a checkpoint
    under a mismatched branch silently produces a wrong forward.
    """
    ck = torch.load(ckpt_path, map_location="cpu")
    model = GPT(GPTConfig(**ck["model_args"]))
    sd = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in ck["model"].items()
    }
    model.load_state_dict(sd)
    model.to(device=device, dtype=dtype).eval()
    return model, ck


def full_logits(model, idx):
    """Full-sequence logits ``[B, T, V]`` for offline scoring.

    Uses the targets-path of ``GPT.forward`` (which returns logits at *every*
    position); the returned loss is unshifted and is discarded. This needs no
    model.py change and works unchanged on any branch -- the plain
    ``forward(idx)`` path only returns the last-position logits.
    """
    logits, _ = model(idx, targets=idx)
    return logits
