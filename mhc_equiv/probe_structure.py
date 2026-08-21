"""Structural probe: group-wise read decoupling, and per-stream write differentiation.

Two questions, measured in the forward pass and reduced to scalars inside the hooks
(no tensors retained), so it runs on all 28 layers at once.

Q1 group decoupling -- for the group variants each channel group q gets its own read
   vector H_pre[q, :] over the 4 streams; in mhc/lora one read vector is shared by all
   channels (so the group rows are identical *by construction*, cos == 1 exactly).
     hpre_row_cos    mean pairwise cos between rows q != q' of H_pre  (1.0 = no decoupling)
     hpre_group_std  spread across q of each stream's read coefficient
     u_cos_vs_shared cos(actual group-wise branch input, branch input rebuilt with the
                     group-averaged read vector) -- 1.0 = grouping changes nothing

Q2 LoRA differentiation -- without LoRA the write to every stream is beta_s * h, i.e. all
   four writes are collinear; with LoRA it is beta_s(h + lambda*delta_s), so each stream
   gets its own direction.
     write_cos       mean pairwise cos between the writes to different streams
     write_orth_frac ||component of the write orthogonal to h|| / ||write||  (0 = no LoRA)
     resid_cos       mean pairwise cos between the residual streams after the write
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from einops import rearrange, einsum


def pair_cos(x):
    """x: [b, s, t, d] -> mean cos over stream pairs i<j and over (b, t)."""
    xn = torch.nn.functional.normalize(x.float(), dim=-1)
    s = xn.shape[1]
    vals = []
    for i in range(s):
        for j in range(i + 1, s):
            vals.append((xn[:, i] * xn[:, j]).sum(-1).mean())
    return float(torch.stack(vals).mean())


def orth_frac(write, h):
    """write: [b, s, t, d], h: [b, t, d] -> mean over s of ||write - proj_h(write)|| / ||write||."""
    w = write.float()
    hh = h.float().unsqueeze(1)
    denom = hh.pow(2).sum(-1, keepdim=True).clamp_min(1e-12)
    proj = (w * hh).sum(-1, keepdim=True) / denom * hh
    o = (w - proj).norm(dim=-1)
    n = w.norm(dim=-1).clamp_min(1e-12)
    return float((o / n).mean())


def install(model, acc):
    """Monkey-patch every hyper-conn instance; each hook reduces to scalars in `acc`."""
    mods = []
    for i, blk in enumerate(model.transformer.h):
        mods += [(f"{i}.attn", blk.hc_attn), (f"{i}.mlp", blk.hc_mlp)]

    for name, hc in mods:
        streams = hc.num_residual_streams
        is_group = hasattr(hc, "group_pre_weight")
        # the two group families spell the gate generator differently
        gate_attr = next((a for a in ("_compute_group_gates", "_compute_group_pre_gate")
                          if hasattr(hc, a)), None)
        is_group = is_group and gate_attr is not None

        if is_group:
            _gate = getattr(hc, gate_attr)

            def gate(normed, _o=_gate, _n=name, _hc=hc):
                H = _o(normed)                       # b t f q s
                with torch.no_grad():
                    h = H.float()[:, :, 0]           # b t q s   (num_fracs == 1)
                    hn = torch.nn.functional.normalize(h, dim=-1)
                    q = hn.shape[-2]
                    cs = [(hn[..., a, :] * hn[..., b, :]).sum(-1).mean()
                          for a in range(q) for b in range(a + 1, q)]
                    acc[_n]["hpre_row_cos"].append(float(torch.stack(cs).mean()))
                    acc[_n]["hpre_group_std"].append(float(h.std(dim=-2).mean()))
                    acc[_n]["hpre_rowsum"].append(float(h.sum(-1).mean()))
                    _hc._H = H
                return H
            setattr(hc, gate_attr, gate)

        _width = hc.width_connection

        def width(residuals, _o=_width, _n=name, _hc=hc, _s=streams, _g=is_group):
            out = _o(residuals)
            with torch.no_grad():
                if _g and getattr(_hc, "_H", None) is not None:
                    groups = _hc.group_embedding_groups
                    r = rearrange(_hc.split_fracs(residuals), '(b s) ... d -> b ... s d', s=_s)
                    rg = rearrange(r, 'b ... s (q d) -> b ... s q d', q=groups)
                    Hm = _hc._H.mean(dim=-2, keepdim=True).expand_as(_hc._H)
                    u_sh = einsum(Hm.float(), rg.float(),
                                  'b ... f q s, b ... f s q d -> b ... f q d')
                    u_sh = rearrange(u_sh, 'b ... f q d -> b ... f (q d)')[..., 0, :]
                    u = out[0].float()
                    acc[_n]["u_cos_vs_shared"].append(float(
                        (torch.nn.functional.normalize(u, dim=-1)
                         * torch.nn.functional.normalize(u_sh, dim=-1)).sum(-1).mean()))
                    _hc._H = None
            return out
        hc.width_connection = width

        _depth = hc.depth_connection

        def depth(branch_output, residuals, *, beta, _o=_depth, _n=name, _s=streams):
            ret = _o(branch_output, residuals, beta=beta)
            with torch.no_grad():
                w = rearrange((ret - residuals).detach(), '(b s) t d -> b s t d', s=_s)
                x = rearrange(ret.detach(), '(b s) t d -> b s t d', s=_s)
                acc[_n]["write_cos"].append(pair_cos(w))
                acc[_n]["resid_cos"].append(pair_cos(x))
                acc[_n]["write_orth_frac"].append(orth_frac(w, branch_output.detach()))
                acc[_n]["write_rms"].append(float(w.float().pow(2).mean().sqrt()))
                acc[_n]["beta_max"].append(float(beta.detach().float().max()))
            return ret
        hc.depth_connection = depth
    return [n for n, _ in mods]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/work/data/guotianzizhe/project/mhc-lite")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hyper_conn_type", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data", default="/home/work/data/guotianzizhe/data/openwebtext/val.bin")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    os.chdir(args.repo)
    from model import GPT, GPTConfig

    torch.manual_seed(1337)
    random.seed(1337)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ck["model_args"]))
    model.load_state_dict({k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
                           for k, v in ck["model"].items()})
    it = ck.get("iter_num")
    del ck
    model.to(args.device).eval()

    acc = defaultdict(lambda: defaultdict(list))
    names = install(model, acc)

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(20260806)
    with torch.no_grad():
        for _ in range(args.n_batches):
            ix = rng.integers(0, len(data) - 1025, size=args.batch_size)
            X = torch.from_numpy(np.stack([data[i:i + 1024].astype(np.int64) for i in ix]))
            model(X.to(args.device))

    keys = ["write_cos", "write_orth_frac", "resid_cos", "beta_max", "write_rms",
            "hpre_row_cos", "hpre_group_std", "u_cos_vs_shared", "hpre_rowsum"]
    rows = {n: {k: float(np.mean(acc[n][k])) for k in keys if acc[n][k]} for n in names}
    print(f"\n===== {args.tag}  ({args.hyper_conn_type}, iter={it}) =====")
    show = [k for k in keys if any(k in rows[n] for n in names)]
    print("  " + f"{'site':10}" + " ".join(f"{k[:15]:>16}" for k in show))
    for n in names:
        print("  " + f"{n:10}" + " ".join(f"{rows[n].get(k, float('nan')):>16.5f}" for k in show))
    print("\n  --- 全层平均 ---")
    for k in show:
        v = [rows[n][k] for n in names if k in rows[n]]
        print(f"  {k:18} mean={np.mean(v):.5f}  min={np.min(v):.5f}  max={np.max(v):.5f}")
    if args.out:
        json.dump({"tag": args.tag, "type": args.hyper_conn_type, "iter": it, "per_layer": rows},
                  open(args.out, "w"), indent=2)
        print(f"\n  -> {args.out}")


if __name__ == "__main__":
    main()
