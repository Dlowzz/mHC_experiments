"""Run one mHC forward/backward probe inside a given repo and dump a fingerprint.

The point is to be able to run the *same* probe against two checkouts (my repo vs
the pristine mhc-lite paper repo) and diff the fingerprints bit for bit.

  python run_one.py --repo /path/to/repo --out dump.pt [--ckpt ckpt.pt] ...

Everything that can perturb the result is pinned here rather than inherited from
the repo's train.py: seeds (torch *and* random -- mhc picks its "home" stream with
random.randrange), the token batches, the autocast dtype and the SDPA backend.
"""
import argparse
import hashlib
import os
import random
import sys

import numpy as np
import torch


def _bytes(t):
    return t.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()


def sha(t):
    return hashlib.sha256(_bytes(t)).hexdigest()[:32]


def fp(t):
    """dtype-independent fingerprint of a tensor: exact hash + f64 summaries."""
    f = t.detach().float()
    return dict(
        sha=sha(t),
        shape=tuple(t.shape),
        dtype=str(t.dtype),
        norm=float(f.norm(dtype=torch.float64)),
        absmax=float(f.abs().max()),
        mean=float(f.mean(dtype=torch.float64)),
    )


def get_batches(bin_path, n, bs, block, seed):
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ix = rng.integers(0, len(data) - block - 1, size=bs)
        x = torch.from_numpy(np.stack([data[i:i + block].astype(np.int64) for i in ix]))
        y = torch.from_numpy(np.stack([data[i + 1:i + 1 + block].astype(np.int64) for i in ix]))
        out.append((x, y))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None, help="load model weights from this ckpt")
    ap.add_argument("--hyper_conn_type", default="mhc")
    ap.add_argument("--hyper_conn_n", type=int, default=4)
    ap.add_argument("--n_layer", type=int, default=28)
    ap.add_argument("--n_head", type=int, default=20)
    ap.add_argument("--n_embd", type=int, default=1280)
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--vocab_size", type=int, default=50304)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--grad_accum", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--data_seed", type=int, default=20260804)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--sdpa", default="default", choices=["default", "math"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data", default="/home/work/data/guotianzizhe/data/openwebtext/train.bin")
    ap.add_argument("--grad_hash", type=int, default=1, help="hash full grads (slow but exact)")
    ap.add_argument("--save_grads", default=None, help="also dump raw grads (fp32) here")
    args = ap.parse_args()

    # import model.py / hyper_conn from the repo under test, nothing else
    sys.path.insert(0, args.repo)
    os.chdir(args.repo)
    from model import GPT, GPTConfig
    import hyper_conn
    import model as model_mod
    print(f"[{args.repo}] model.py -> {model_mod.__file__}", flush=True)
    print(f"[{args.repo}] hyper_conn -> {hyper_conn.__file__}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # PROBE_BODY
    # seed exactly the way train.py does, plus `random` (mhc's home-stream choice)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    conf = GPTConfig(
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        block_size=args.block_size, bias=False, vocab_size=args.vocab_size,
        dropout=0.0, hyper_conn_n=args.hyper_conn_n,
        hyper_conn_type=args.hyper_conn_type,
    )
    model = GPT(conf)

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = ck["model"]
        for k in list(sd):
            if k.startswith("_orig_mod."):
                sd[k[len("_orig_mod."):]] = sd.pop(k)
        missing, unexpected = model.load_state_dict(sd, strict=True), None
        print(f"loaded ckpt {args.ckpt} iter_num={ck.get('iter_num')} "
              f"best_val={ck.get('best_val_loss')} val={ck.get('val_loss')}", flush=True)
        del ck, sd

    model.to(args.device)
    model.train()

    # the "home" stream each layer picked -- part of the init, so worth fingerprinting
    home = []
    for n, m in model.named_modules():
        if hasattr(m, "static_beta"):
            home.append((n, int(m.static_beta.detach().argmax()),
                         getattr(m, "init_residual_index", None)))

    param_fp = {n: fp(p) for n, p in model.named_parameters()}

    batches = get_batches(args.data, args.grad_accum, args.batch_size,
                          args.block_size, args.data_seed)

    ptdtype = dict(bfloat16=torch.bfloat16, float32=torch.float32)[args.dtype]
    if args.dtype == "float32":
        ctx = torch.autocast("cuda", enabled=False)
    else:
        ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    from contextlib import nullcontext
    if args.sdpa == "math":
        from torch.nn.attention import sdpa_kernel, SDPBackend
        sdpa_ctx = sdpa_kernel(SDPBackend.MATH)
    else:
        sdpa_ctx = nullcontext()

    losses, logit_fps = [], []
    with sdpa_ctx:
        for i, (x, y) in enumerate(batches):
            x, y = x.to(args.device), y.to(args.device)
            with ctx:
                logits, loss = model(x, y)
                scaled = loss / args.grad_accum
            losses.append(float(loss.double()))
            logit_fps.append(fp(logits))
            scaled.backward()
            print(f"  micro {i}: loss={losses[-1]:.10f}", flush=True)

    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    total = torch.norm(torch.stack([g.detach().float().norm(2) for g in grads.values()]), 2)
    grad_fp = {}
    for n, g in grads.items():
        d = dict(shape=tuple(g.shape), dtype=str(g.dtype),
                 norm=float(g.detach().float().norm(dtype=torch.float64)),
                 absmax=float(g.detach().float().abs().max()),
                 mean=float(g.detach().float().mean(dtype=torch.float64)))
        if args.grad_hash:
            d["sha"] = sha(g)
        grad_fp[n] = d

    dump = dict(
        args=vars(args), losses=losses, logits=logit_fps, params=param_fp,
        grads=grad_fp, total_grad_norm=float(total.double()), home=home,
        torch=torch.__version__,
    )
    torch.save(dump, args.out)
    if args.save_grads:
        torch.save({n: g.detach().cpu().clone() for n, g in grads.items()}, args.save_grads)
    print(f"total grad norm = {float(total):.10f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
