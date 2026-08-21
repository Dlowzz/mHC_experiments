"""Forward/backward internals of the hyper-connection layers at a given checkpoint.

Answers: how big is each residual stream, how saturated are the alpha/beta gates,
and how large is the gradient arriving at each layer's stream input.  Run it on two
checkpoints (e.g. mhc vs mhc-group) and diff the profiles.
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
    ap.add_argument("--n_layer", type=int, default=28)
    ap.add_argument("--n_head", type=int, default=20)
    ap.add_argument("--n_embd", type=int, default=1280)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data", default="/home/work/data/guotianzizhe/data/openwebtext/train.bin")
    ap.add_argument("--data_seed", type=int, default=20260804)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    os.chdir(args.repo)
    from model import GPT, GPTConfig

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(1337)
    random.seed(1337)

    conf = GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                     block_size=args.block_size, bias=False, vocab_size=50304, dropout=0.0,
                     hyper_conn_n=4, hyper_conn_type=args.hyper_conn_type)
    model = GPT(conf)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
          for k, v in ck["model"].items()}
    model.load_state_dict(sd, strict=True)
    print(f"ckpt {args.ckpt} iter={ck.get('iter_num')} val={ck.get('val_loss')}")
    del ck, sd
    model.to(args.device).train()

    # PROBE
    hc_mods = [(n, m) for n, m in model.named_modules() if hasattr(m, "static_beta")]
    rec = {}

    def mk_hook(name):
        def hook(mod, inputs, output):
            x = inputs[0]
            d = rec.setdefault(name, {})
            # x is the stacked (b*s, T, C) residual stream
            f = x.detach().float()
            d["x_rms"] = float(f.pow(2).mean().sqrt())
            d["x_absmax"] = float(f.abs().max())
            x.retain_grad() if x.requires_grad else None
            d["_x"] = x
        return hook

    handles = [m.register_forward_hook(mk_hook(n)) for n, m in hc_mods]

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(args.data_seed)
    ix = rng.integers(0, len(data) - args.block_size - 1, size=args.batch_size)
    X = torch.from_numpy(np.stack([data[i:i + args.block_size].astype(np.int64) for i in ix]))
    Y = torch.from_numpy(np.stack([data[i + 1:i + 1 + args.block_size].astype(np.int64) for i in ix]))
    X, Y = X.to(args.device), Y.to(args.device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits, loss = model(X, Y)
    print(f"loss = {float(loss):.6f}")
    loss.backward()
    for h in handles:
        h.remove()

    # gate statistics, recomputed with the captured stream inputs (no_grad, cheap)
    print(f"\n{'module':22} {'x_rms':>10} {'x_absmax':>10} {'dL/dx_rms':>12} "
          f"{'beta_min':>9} {'beta_max':>9} {'a_diag':>8} {'a_off':>8} {'a_ent':>8}")
    rows = []
    with torch.no_grad():
        for n, m in hc_mods:
            d = rec[n]
            x = d.pop("_x")
            gx = x.grad
            g_rms = float(gx.detach().float().pow(2).mean().sqrt()) if gx is not None else float("nan")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                bi, res, kw = m.width_connection(x.detach())
            beta = kw["beta"].float()
            # recompute alpha the way width_connection does, to read its saturation
            from einops import rearrange, repeat
            from hyper_conn.mhc import sinkhorn_knopps
            streams = m.num_residual_streams
            r = rearrange(m.split_fracs(x.detach()), '(b s) ... d -> b ... s d', s=streams)
            normed = m.norm(rearrange(r, 'b ... s d -> b ... (s d)', s=streams).float())
            wc = rearrange(normed @ m.dynamic_alpha_fn.float(), '... (s t) -> ... s t', s=streams)
            scale = torch.cat((repeat(m.pre_branch_scale, '1 -> v', v=m.num_input_views * m.num_fracs),
                               repeat(m.residual_scale, '1 -> s', s=m.num_fracs * streams))).float()
            alpha = wc * scale + rearrange(m.static_alpha.float(), '(f s) t -> f s t', s=streams)
            alpha = m.split_fracs(alpha)[..., m.num_input_views:]
            a = sinkhorn_knopps(rearrange(alpha, '... f s g t -> ... f g s t'), m.sinkhorn_iters)
            a = a[..., 0, :, :]  # drop the frac dims (num_fracs == 1)
            diag = a.diagonal(dim1=-2, dim2=-1).mean()
            off = ((a.sum((-1, -2)) - a.diagonal(dim1=-2, dim2=-1).sum(-1)) / (streams * (streams - 1))).mean()
            ent = -(a.clamp_min(1e-9).log() * a).sum(-1).mean()
            row = (n, d["x_rms"], d["x_absmax"], g_rms, float(beta.min()), float(beta.max()),
                   float(diag), float(off), float(ent))
            rows.append(row)
            print(f"{n.replace('transformer.h.',''):22} {row[1]:>10.4f} {row[2]:>10.3f} {row[3]:>12.3e} "
                  f"{row[4]:>9.4f} {row[5]:>9.4f} {row[6]:>8.4f} {row[7]:>8.4f} {row[8]:>8.4f}")
            del x, r, normed, wc, alpha, a

    emb = dict(wte=model.transformer.wte.weight, wpe=model.transformer.wpe.weight)
    print()
    for k, v in emb.items():
        f = v.detach().float()
        print(f"{k}: rms={float(f.pow(2).mean().sqrt()):.6f} row_rms_mean="
              f"{float(f.pow(2).mean(-1).sqrt().mean()):.6f} "
              f"grad_rms={float(v.grad.detach().float().pow(2).mean().sqrt()):.3e} "
              f"grad_norm={float(v.grad.detach().float().norm()):.4f}")
    if args.out:
        torch.save(rows, args.out)


if __name__ == "__main__":
    main()
