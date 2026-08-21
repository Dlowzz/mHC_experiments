"""Where does the total grad norm come from? Rank one run_one.py dump."""
import argparse
import re
from collections import defaultdict

import torch


def group_of(name):
    if ".hc_attn." in name or ".hc_mlp." in name:
        tail = name.split(".hc_attn.")[-1].split(".hc_mlp.")[-1]
        return f"hc/{tail}"
    if name.startswith("transformer.wte") or name.startswith("transformer.wpe"):
        return "embed"
    if ".branch_attn." in name:
        return "branch_attn/" + name.split(".branch_attn.")[-1].split(".", 1)[-1]
    if ".branch_mlp." in name:
        return "branch_mlp/" + name.split(".branch_mlp.")[-1].split(".", 1)[-1]
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    D = torch.load(args.dump, weights_only=False)
    g = D["grads"]
    total = D["total_grad_norm"]
    print(f"{args.dump}")
    print(f"  ckpt={D['args'].get('ckpt')}  total grad norm={total:.6f}")
    print(f"  losses: mean={sum(D['losses'])/len(D['losses']):.6f}")

    print(f"\ntop {args.top} individual params by grad norm")
    print(f"  {'param':62} {'norm':>12} {'%of total^2':>12} {'absmax':>12}")
    for n in sorted(g, key=lambda k: -g[k]["norm"])[:args.top]:
        sh = g[n]["norm"]
        print(f"  {n:62} {sh:>12.4e} {100*sh**2/total**2:>11.3f}% {g[n]['absmax']:>12.4e}")

    print("\nby parameter group (sum of squares)")
    agg = defaultdict(float)
    cnt = defaultdict(int)
    amax = defaultdict(float)
    for n, d in g.items():
        k = group_of(n)
        agg[k] += d["norm"] ** 2
        cnt[k] += 1
        amax[k] = max(amax[k], d["absmax"])
    print(f"  {'group':44} {'n':>4} {'norm':>12} {'%of total^2':>12} {'absmax':>12}")
    for k in sorted(agg, key=lambda k: -agg[k]):
        print(f"  {k:44} {cnt[k]:>4} {agg[k]**0.5:>12.4e} {100*agg[k]/total**2:>11.3f}% {amax[k]:>12.4e}")

    # per-layer profile of the hyper-connection gate grads
    print("\nper-layer hc gate grad norms (attn / mlp)")
    pat = re.compile(r"transformer\.h\.(\d+)\.(hc_attn|hc_mlp)\.(.*)")
    per = defaultdict(lambda: defaultdict(float))
    for n, d in g.items():
        m = pat.match(n)
        if m:
            per[int(m.group(1))][(m.group(2), m.group(3))] = d["norm"]
    fields = ["static_alpha", "dynamic_alpha_fn", "static_beta", "dynamic_beta_fn",
              "residual_scale", "h_post_scale", "pre_branch_scale", "norm.gamma"]
    print("  layer " + " ".join(f"{f[:13]:>13}" for f in fields))
    for li in sorted(per):
        for which in ("hc_attn", "hc_mlp"):
            vals = [per[li].get((which, f), float("nan")) for f in fields]
            print(f"  {li:>3}{which[3:6]:>4} " + " ".join(f"{v:>13.4e}" for v in vals))


if __name__ == "__main__":
    main()
