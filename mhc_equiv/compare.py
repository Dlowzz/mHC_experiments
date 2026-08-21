"""Diff two fingerprints produced by run_one.py."""
import argparse
import torch


def rel(a, b):
    d = abs(a - b)
    s = max(abs(a), abs(b), 1e-300)
    return d / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    A = torch.load(args.a, weights_only=False)
    B = torch.load(args.b, weights_only=False)

    print(f"A = {args.a}")
    print(f"B = {args.b}")
    print(f"torch: {A['torch']} vs {B['torch']}")
    for k in ("hyper_conn_type", "n_layer", "n_embd", "batch_size", "grad_accum",
              "dtype", "sdpa", "ckpt", "seed", "data_seed"):
        va, vb = A["args"].get(k), B["args"].get(k)
        flag = "" if va == vb else "   <-- DIFFERENT"
        print(f"  {k:16} {va} | {vb}{flag}")

    # --- init / weights ---
    ka, kb = set(A["params"]), set(B["params"])
    print(f"\n[params] {len(ka)} vs {len(kb)} tensors")
    if ka != kb:
        print(f"  only in A: {sorted(ka - kb)[:10]}")
        print(f"  only in B: {sorted(kb - ka)[:10]}")
    shared = sorted(ka & kb)
    bad = [n for n in shared if A["params"][n]["sha"] != B["params"][n]["sha"]]
    print(f"  identical weight bytes: {len(shared) - len(bad)}/{len(shared)}")
    for n in bad[:args.top]:
        pa, pb = A["params"][n], B["params"][n]
        print(f"    {n:60} norm {pa['norm']:.8e} vs {pb['norm']:.8e} "
              f"rel={rel(pa['norm'], pb['norm']):.2e}")

    ha = {n: h for n, h, _ in A["home"]}
    hb = {n: h for n, h, _ in B["home"]}
    same_home = sum(1 for n in ha if hb.get(n) == ha[n])
    print(f"  home stream (static_beta.argmax) identical: {same_home}/{len(ha)}")

    # --- forward ---
    print("\n[forward]")
    print(f"  {'micro':>5} {'loss A':>18} {'loss B':>18} {'rel':>10}  logits sha")
    for i, (la, lb) in enumerate(zip(A["losses"], B["losses"])):
        sa, sb = A["logits"][i]["sha"], B["logits"][i]["sha"]
        tag = "same" if sa == sb else f"DIFF absmax {A['logits'][i]['absmax']:.6f}/{B['logits'][i]['absmax']:.6f}"
        print(f"  {i:>5} {la:>18.12f} {lb:>18.12f} {rel(la, lb):>10.2e}  {tag}")

    # --- backward ---
    print("\n[backward]")
    ga, gb = set(A["grads"]), set(B["grads"])
    shared = sorted(ga & gb)
    if ga != gb:
        print(f"  grad key mismatch: only A {sorted(ga-gb)[:5]} only B {sorted(gb-ga)[:5]}")
    has_sha = all("sha" in A["grads"][n] for n in shared) and all("sha" in B["grads"][n] for n in shared)
    if has_sha:
        bad = [n for n in shared if A["grads"][n]["sha"] != B["grads"][n]["sha"]]
        print(f"  identical grad bytes: {len(shared) - len(bad)}/{len(shared)}")
    else:
        bad = [n for n in shared if A["grads"][n]["norm"] != B["grads"][n]["norm"]]
        print(f"  identical grad norms: {len(shared) - len(bad)}/{len(shared)}")
    worst = sorted(shared, key=lambda n: -rel(A["grads"][n]["norm"], B["grads"][n]["norm"]))
    print(f"  {'param':60} {'norm A':>14} {'norm B':>14} {'rel':>10}")
    for n in worst[:args.top]:
        na, nb = A["grads"][n]["norm"], B["grads"][n]["norm"]
        print(f"  {n:60} {na:>14.6e} {nb:>14.6e} {rel(na, nb):>10.2e}")
    ta, tb = A["total_grad_norm"], B["total_grad_norm"]
    print(f"\n  total grad norm: {ta:.10f} vs {tb:.10f}   rel={rel(ta, tb):.3e}")


if __name__ == "__main__":
    main()
