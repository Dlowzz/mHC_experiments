"""Isolate the gate-generation path's contribution to the embedding gradient.

In `width_connection`, `normed = self.norm(...)` feeds ONLY the alpha/beta logits
(`mix_h` contracts the raw `residuals`, not `normed`).  So detaching the input of
`self.norm` cuts exactly the "x -> gate value" path while leaving the gate values,
the value path and every parameter gradient intact.
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
from torch import nn


class DetachIn(nn.Module):
    """Wraps a module so its input carries no gradient back to x."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x.detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/work/data/guotianzizhe/project/mhc-lite")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hyper_conn_type", default="mhc")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data", default="/home/work/data/guotianzizhe/data/openwebtext/train.bin")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    os.chdir(args.repo)
    from model import GPT, GPTConfig

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(1337)
    random.seed(1337)
    conf = GPTConfig(n_layer=28, n_head=20, n_embd=1280, block_size=1024, bias=False,
                     vocab_size=50304, dropout=0.0, hyper_conn_n=4,
                     hyper_conn_type=args.hyper_conn_type)
    model = GPT(conf)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict({k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
                           for k, v in ck["model"].items()}, strict=True)
    print(f"ckpt {args.ckpt} iter={ck.get('iter_num')}")
    del ck
    model.to(args.device).train()

    hc = [(n, m) for n, m in model.named_modules() if hasattr(m, "static_beta")]
    original = {n: m.norm for n, m in hc}

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(20260804)
    ix = rng.integers(0, len(data) - 1025, size=args.batch_size)
    X = torch.from_numpy(np.stack([data[i:i + 1024].astype(np.int64) for i in ix])).to(args.device)
    Y = torch.from_numpy(np.stack([data[i + 1:i + 1025].astype(np.int64) for i in ix])).to(args.device)

    def run(tag, detach_names):
        for n, m in hc:
            m.norm = DetachIn(original[n]) if n in detach_names else original[n]
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(X, Y)
        loss.backward()
        wte = model.transformer.wte.weight.grad
        wpe = model.transformer.wpe.weight.grad
        tot = torch.norm(torch.stack([p.grad.float().norm() for p in model.parameters()
                                      if p.grad is not None]))
        l0 = torch.norm(torch.stack([p.grad.float().norm() for n, p in model.named_parameters()
                                     if p.grad is not None and n.startswith("transformer.h.0.hc_")]))
        print(f"{tag:34} loss={float(loss):.6f}  |g wte|={float(wte.float().norm()):9.4f}  "
              f"|g wpe|={float(wpe.float().norm()):9.4f}  |g layer0 gates|={float(l0):9.4f}  "
              f"|g total|={float(tot):9.4f}")

    all_names = [n for n, _ in hc]
    run("full model (baseline)", set())
    run("detach gate input @ layer0 only", {"transformer.h.0.hc_attn", "transformer.h.0.hc_mlp"})
    run("detach gate input @ all layers", set(all_names))
    for n, m in hc:
        m.norm = original[n]


if __name__ == "__main__":
    main()
