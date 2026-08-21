"""Compare per-layer parameter scales across checkpoints (CPU, mmap).

No assumptions about which knob matters -- just dump every scale that can multiply
the branch output, plus the embeddings, and diff the checkpoints side by side.
"""
import argparse
import re

import torch

FIELDS = [
    ("ln_attn", "branch_attn.0.weight"),
    ("c_attn", "branch_attn.1.c_attn.weight"),
    ("c_proj_a", "branch_attn.1.c_proj.weight"),
    ("ln_mlp", "branch_mlp.0.weight"),
    ("c_fc", "branch_mlp.1.c_fc.weight"),
    ("c_proj_m", "branch_mlp.1.c_proj.weight"),
    ("hcA.gamma", "hc_attn.norm.gamma"),
    ("hcA.pre_s", "hc_attn.pre_branch_scale"),
    ("hcA.res_s", "hc_attn.residual_scale"),
    ("hcA.post_s", "hc_attn.h_post_scale"),
    ("hcA.sbeta_max", "hc_attn.static_beta"),
    ("hcA.salpha_off", "hc_attn.static_alpha"),
    ("hcA.dynB", "hc_attn.dynamic_beta_fn"),
]


def load(path):
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
          for k, v in ck["model"].items()}
    return sd, ck.get("iter_num")


def val(sd, layer, suffix):
    k = f"transformer.h.{layer}.{suffix}"
    if k not in sd:
        return float("nan")
    t = sd[k].float()
    if suffix.endswith("static_alpha"):
        # off-diagonal of the residual block: how much cross-stream leakage the
        # static bias still suppresses (init: -8 off-diagonal, 0 on the diagonal)
        blk = t[:, 1:]
        n = blk.shape[0]
        off = blk[~torch.eye(n, dtype=torch.bool)]
        return float(off.mean())
    if suffix.endswith("static_beta"):
        return float(t.max())
    if suffix.endswith("norm.gamma"):
        return float((1 + t).abs().mean())
    if t.numel() <= 1:
        return float(t.flatten()[0])
    if suffix.endswith(".0.weight"):        # LayerNorm gain
        return float(t.abs().mean())
    return float(t.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="tag=path")
    ap.add_argument("--layers", default="0,1,2,13,27")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    data = {}
    for spec in args.ckpts:
        tag, path = spec.split("=", 1)
        sd, it = load(path)
        data[tag] = (sd, it)
        e = {k: sd[k].float() for k in ("transformer.wte.weight", "transformer.wpe.weight")}
        print(f"{tag:12} iter={it}  wte_rms={float(e['transformer.wte.weight'].pow(2).mean().sqrt()):.6f}"
              f"  wpe_rms={float(e['transformer.wpe.weight'].pow(2).mean().sqrt()):.6f}"
              f"  ln_f={float(sd['transformer.ln_f.weight'].abs().mean()):.4f}")

    for li in layers:
        print(f"\n--- layer {li} ---")
        print(f"  {'field':16} " + " ".join(f"{t:>14}" for t in data))
        for name, suffix in FIELDS:
            row = [val(sd, li, suffix) for sd, _ in data.values()]
            print(f"  {name:16} " + " ".join(f"{v:>14.5f}" for v in row))


if __name__ == "__main__":
    main()
