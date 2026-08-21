"""Where does the layer-0 write magnitude actually come from?

Measures, per early layer: the stream coming in, the branch output h, the beta gate,
and the resulting write -- plus the per-channel structure (are a few channels carrying
everything?).  No inference, all hooks.
"""
import argparse
import os
import random
import sys

import numpy as np
import torch


def rms(t):
    return float(t.detach().float().pow(2).mean().sqrt())


def amax(t):
    return float(t.detach().float().abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/work/data/guotianzizhe/project/mhc-lite")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hyper_conn_type", default="mhc")
    ap.add_argument("--layers", type=int, default=4)
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
    it = ck.get("iter_num")
    del ck
    model.to(args.device).eval()

    rec = {}

    def hook(name):
        def f(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            rec[name] = dict(rms=rms(o), amax=amax(o))
            if name.endswith("stream_in"):
                x = o.detach().float()
                # per-channel rms over (batch*stream, token)
                ch = x.reshape(-1, x.shape[-1]).pow(2).mean(0).sqrt()
                rec[name]["ch_top"] = [(int(i), round(float(ch[i]), 3))
                                       for i in ch.topk(5).indices]
                rec[name]["ch_med"] = round(float(ch.median()), 4)
        return f

    hs = []
    for li in range(args.layers):
        blk = model.transformer.h[li]
        hs.append(blk.hc_attn.register_forward_pre_hook(
            lambda m, inp, n=f"{li}.attn.stream_in": hook(n)(m, inp, inp[0])))
        hs.append(blk.branch_attn.register_forward_hook(hook(f"{li}.attn.h")))
        hs.append(blk.branch_mlp.register_forward_hook(hook(f"{li}.mlp.h")))
        hs.append(blk.hc_mlp.register_forward_pre_hook(
            lambda m, inp, n=f"{li}.mlp.stream_in": hook(n)(m, inp, inp[0])))

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(20260804)
    ix = rng.integers(0, len(data) - 1025, size=args.batch_size)
    X = torch.from_numpy(np.stack([data[i:i + 1024].astype(np.int64) for i in ix])).to(args.device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model(X)
    for h in hs:
        h.remove()

    print(f"\n=== {args.ckpt.split('/')[-2]} ({args.hyper_conn_type}) iter={it} ===")
    print(f"  {'site':16} {'stream_in rms':>14} {'in amax':>9} {'branch h rms':>13} "
          f"{'h amax':>9} {'beta max':>9} {'write/in':>9}")
    for li in range(args.layers):
        for which in ("attn", "mlp"):
            sin = rec[f"{li}.{which}.stream_in"]
            h = rec[f"{li}.{which}.h"]
            mod = getattr(model.transformer.h[li], f"hc_{which}")
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                bmax = "n/a"
            print(f"  h.{li}.{which:12} {sin['rms']:>14.4f} {sin['amax']:>9.3f} "
                  f"{h['rms']:>13.4f} {h['amax']:>9.3f} {bmax:>9} "
                  f"{h['rms']/max(sin['rms'],1e-9):>9.1f}")
    print("\n  per-channel rms of the stream (top-5 channels, and median channel)")
    for li in range(args.layers):
        for which in ("attn", "mlp"):
            s = rec[f"{li}.{which}.stream_in"]
            print(f"  h.{li}.{which:5} median={s['ch_med']:<9} top={s['ch_top']}")

    # LayerNorm gains of the branches: mean vs max (outlier channels)
    print("\n  branch LayerNorm gain (mean / max / argmax) and hc RMSNorm gamma")
    for li in range(args.layers):
        blk = model.transformer.h[li]
        for which, m in (("attn", blk.branch_attn[0]), ("mlp", blk.branch_mlp[0])):
            w = m.weight.detach().float()
            g = getattr(blk, f"hc_{which}").norm.gamma.detach().float() + 1
            print(f"  h.{li}.{which:5} LN gain mean={float(w.abs().mean()):.4f} "
                  f"max={float(w.abs().max()):8.3f} @ch{int(w.abs().argmax())%1280:<5} | "
                  f"gamma mean={float(g.abs().mean()):.4f} max={float(g.abs().max()):8.3f}")


if __name__ == "__main__":
    main()
