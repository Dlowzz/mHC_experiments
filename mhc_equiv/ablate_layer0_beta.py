"""Causal test: is the layer-0 beta write the control point for the grad-norm blowup?

Scales the beta gate returned by selected hyper-connection modules by a constant k
(everything else untouched) and re-measures where the gradient lives.  k=1 is the
baseline; k~0.1 puts layer-0's write at the level the group variants learned.
"""
import argparse
import os
import random
import sys

import numpy as np
import torch


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
    model = GPT(GPTConfig(n_layer=28, n_head=20, n_embd=1280, block_size=1024, bias=False,
                          vocab_size=50304, dropout=0.0, hyper_conn_n=4,
                          hyper_conn_type=args.hyper_conn_type))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict({k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
                           for k, v in ck["model"].items()}, strict=True)
    print(f"ckpt {args.ckpt} iter={ck.get('iter_num')}")
    del ck
    model.to(args.device).train()

    hc = dict((n, m) for n, m in model.named_modules() if hasattr(m, "static_beta"))
    orig = {n: m.width_connection for n, m in hc.items()}

    def patch(names, k):
        for n, m in hc.items():
            if n in names:
                def wrapped(residuals, _f=orig[n], _k=k):
                    bi, res, kw = _f(residuals)
                    if kw.get("beta") is not None:
                        kw = dict(kw)
                        kw["beta"] = kw["beta"] * _k
                    return bi, res, kw
                m.width_connection = wrapped
            else:
                m.width_connection = orig[n]

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(20260804)
    ix = rng.integers(0, len(data) - 1025, size=args.batch_size)
    X = torch.from_numpy(np.stack([data[i:i + 1024].astype(np.int64) for i in ix])).to(args.device)
    Y = torch.from_numpy(np.stack([data[i + 1:i + 1025].astype(np.int64) for i in ix])).to(args.device)

    def run(tag):
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(X, Y)
        loss.backward()
        g = lambda p: float(p.grad.float().norm())
        tot = float(torch.norm(torch.stack([p.grad.float().norm() for p in model.parameters()
                                           if p.grad is not None])))
        l0 = float(torch.norm(torch.stack([p.grad.float().norm() for n, p in model.named_parameters()
                                           if p.grad is not None and n.startswith("transformer.h.0.hc_")])))
        print(f"{tag:38} loss={float(loss):.6f}  |g wte|={g(model.transformer.wte.weight):9.3f} "
              f" |g wpe|={g(model.transformer.wpe.weight):9.3f}  |g L0 gates|={l0:9.3f}  |g total|={tot:9.3f}")

    L0 = {"transformer.h.0.hc_attn", "transformer.h.0.hc_mlp"}
    L0A = {"transformer.h.0.hc_attn"}
    patch(set(), 1.0);           run("baseline (k=1)")
    patch(L0A, 0.1);             run("layer0 attn  beta x0.1")
    patch(L0, 0.1);              run("layer0 a+m   beta x0.1")
    patch(L0, 0.01);             run("layer0 a+m   beta x0.01")
    patch(set(hc), 0.1);         run("all layers   beta x0.1")
    patch(set(), 1.0)


if __name__ == "__main__":
    main()
