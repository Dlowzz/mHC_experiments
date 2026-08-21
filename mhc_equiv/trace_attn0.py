"""Break the layer-0 attention branch into its internal stages, on two checkpoints.

Reports rms/absmax at: branch input -> LayerNorm out -> q,k,v -> attention out y ->
c_proj out (= h).  Also the operator norm of W_v and W_o, since a Frobenius norm can
hide a single huge direction.
"""
import argparse
import os
import random
import sys

import numpy as np
import torch


def s(t):
    f = t.detach().float()
    return f"rms={float(f.pow(2).mean().sqrt()):9.4f} amax={float(f.abs().max()):9.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/work/data/guotianzizhe/project/mhc-lite")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layer", type=int, default=0)
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
    it = ck.get("iter_num")
    del ck
    model.to(args.device).eval()

    blk = model.transformer.h[args.layer]
    ln, attn = blk.branch_attn[0], blk.branch_attn[1]
    cap = {}
    hs = [
        ln.register_forward_pre_hook(lambda m, i: cap.__setitem__("branch_in", i[0])),
        ln.register_forward_hook(lambda m, i, o: cap.__setitem__("ln_out", o)),
        attn.c_attn.register_forward_hook(lambda m, i, o: cap.__setitem__("qkv", o)),
        attn.c_proj.register_forward_pre_hook(lambda m, i: cap.__setitem__("y", i[0])),
        attn.c_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o)),
    ]

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(20260804)
    ix = rng.integers(0, len(data) - 1025, size=args.batch_size)
    X = torch.from_numpy(np.stack([data[i:i + 1024].astype(np.int64) for i in ix])).to(args.device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model(X)
    for h in hs:
        h.remove()

    print(f"\n=== {args.ckpt.split('/')[-2]} iter={it}  layer {args.layer} attn branch ===")
    q, k, v = cap["qkv"].split(1280, dim=2)
    print(f"  branch_in   {s(cap['branch_in'])}")
    print(f"  ln_out      {s(cap['ln_out'])}")
    print(f"  q           {s(q)}")
    print(f"  k           {s(k)}")
    print(f"  v           {s(v)}")
    print(f"  y (attn)    {s(cap['y'])}")
    print(f"  h (c_proj)  {s(cap['h'])}")

    W = attn.c_attn.weight.detach().float()
    Wq, Wk, Wv = W.split(1280, dim=0)
    Wo = attn.c_proj.weight.detach().float()
    lnw = ln.weight.detach().float()
    print(f"  |W_q|F={Wq.norm():8.3f} sv1={torch.linalg.matrix_norm(Wq, 2):8.3f}")
    print(f"  |W_k|F={Wk.norm():8.3f} sv1={torch.linalg.matrix_norm(Wk, 2):8.3f}")
    print(f"  |W_v|F={Wv.norm():8.3f} sv1={torch.linalg.matrix_norm(Wv, 2):8.3f}")
    print(f"  |W_o|F={Wo.norm():8.3f} sv1={torch.linalg.matrix_norm(Wo, 2):8.3f}")
    print(f"  LN gain: mean={lnw.abs().mean():.4f} max={lnw.abs().max():.4f}")

    # is attention collapsing onto one token (sink)?  recompute the map for head 0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        qq = q.view(q.shape[0], q.shape[1], 20, 64).transpose(1, 2).float()
        kk = k.view(k.shape[0], k.shape[1], 20, 64).transpose(1, 2).float()
        att = (qq @ kk.transpose(-2, -1)) / 8.0
        T = att.shape[-1]
        att = att.masked_fill(torch.tril(torch.ones(T, T, device=att.device)) == 0, float("-inf"))
        p = att.softmax(-1)
        print(f"  attn mass on token 0 (mean over heads/queries): {float(p[..., 0].mean()):.4f}")
        print(f"  max single-token mass  (mean over heads/queries): {float(p.max(-1).values.mean()):.4f}")
        print(f"  ||v|| per token: max={float(v.float().norm(dim=-1).max()):.3f} "
              f"argmax_pos={int(v.float().norm(dim=-1).flatten().argmax()) % 1024}")


if __name__ == "__main__":
    main()
