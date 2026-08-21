"""Inspect the mHC gate parameters stored in a checkpoint (CPU only).

Prints, per layer, the scalars that control how far the alpha/beta logits can be
pushed away from their static init -- which is what decides whether Sinkhorn is
operating in a smooth or a near-permutation (ill-conditioned) regime.
"""
import argparse
import re

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--per_layer", type=int, default=0)
    args = ap.parse_args()

    try:
        ck = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=False)
    except Exception as e:  # noqa: BLE001
        print(f"mmap load failed ({e}); falling back to a full load")
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    sd = ck["model"]
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    print(f"ckpt {args.ckpt}")
    print(f"  iter_num={ck.get('iter_num')} best_val={ck.get('best_val_loss')} val={ck.get('val_loss')}")

    pat = re.compile(r"transformer\.h\.(\d+)\.(hc_attn|hc_mlp)\.(.*)")
    rows = {}
    for k, v in sd.items():
        m = pat.match(k)
        if m:
            rows.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = v

    def stat(t):
        f = t.float()
        return f

    keys = ["pre_branch_scale", "residual_scale", "h_post_scale"]
    agg = {k: [] for k in keys}
    agg["|dyn_alpha|"] = []
    agg["|dyn_beta|"] = []
    agg["gamma_absmax"] = []
    agg["alpha_logit_est"] = []
    agg["beta_logit_est"] = []
    agg["static_alpha_off_diag_max"] = []

    hdr = f"{'layer':>16} " + " ".join(f"{k:>13}" for k in
                                       ["pre_scale", "res_scale", "post_scale", "|dynA|", "|dynB|",
                                        "gamma_amax", "a_logit_est", "b_logit_est"])
    if args.per_layer:
        print(hdr)
    for (li, which), d in sorted(rows.items()):
        if not all(k in d for k in keys):
            continue
        pre = float(stat(d["pre_branch_scale"]).flatten()[0])
        res = float(stat(d["residual_scale"]).flatten()[0])
        post = float(stat(d["h_post_scale"]).flatten()[0])
        dA = stat(d["dynamic_alpha_fn"])
        dB = stat(d["dynamic_beta_fn"])
        gamma = stat(d["norm.gamma"])
        # `normed` has an exact L2 norm of sqrt(D)*|1+gamma| per token (RMSNorm), so
        # |logit| <= scale * |normed| * max_col_norm(dyn_fn) is a hard bound; use the
        # rms of the column norms for a typical-size estimate instead of the bound.
        D = gamma.numel()
        normed_rms = (D ** 0.5) * float((1 + gamma).pow(2).mean().sqrt())
        colA = dA.norm(dim=0)
        colB = dB.norm(dim=0)
        a_est = res * normed_rms * float(colA.mean()) / (D ** 0.5)
        b_est = post * normed_rms * float(colB.mean()) / (D ** 0.5)
        agg["pre_branch_scale"].append(pre)
        agg["residual_scale"].append(res)
        agg["h_post_scale"].append(post)
        agg["|dyn_alpha|"].append(float(dA.norm()))
        agg["|dyn_beta|"].append(float(dB.norm()))
        agg["gamma_absmax"].append(float(gamma.abs().max()))
        agg["alpha_logit_est"].append(a_est)
        agg["beta_logit_est"].append(b_est)
        sa = stat(d["static_alpha"])
        agg["static_alpha_off_diag_max"].append(float(sa[:, 1:].max()))
        if args.per_layer:
            print(f"{f'h.{li}.{which}':>16} " + " ".join(
                f"{v:>13.5f}" for v in [pre, res, post, float(dA.norm()), float(dB.norm()),
                                        float(gamma.abs().max()), a_est, b_est]))

    print(f"\nsummary over {len(agg['residual_scale'])} hyper-conn modules")
    for k, v in agg.items():
        t = torch.tensor(v)
        print(f"  {k:>26}  min={t.min():.5f}  med={t.median():.5f}  max={t.max():.5f}  mean={t.mean():.5f}")


if __name__ == "__main__":
    main()
