"""Cross-variant write-back gate comparison at equal budget (XL, bs6, 80000 steps).

Compares the four main variants -- mhc / group / lora / group-lora -- on the four
quantities the write-back analysis hinges on, per hyper-connection site:

    ||f||/||r||     branch-output norm over residual-stream norm (massive-activation gauge)
    dL/dbeta        autograd grad on the H_post gate
    z saturation    fraction of gate cells with beta>1.9 or beta<0.1 (both sigmoid tails)
    dbeta/dz        gate Jacobian beta(1-beta/2); 0.5 = wide open, ->0 = saturated

Read-only: reuses the Probe / loader from diagnose_beta_writeback (retain_grad only,
no model-logic change).  All four variants see the IDENTICAL token batches (generator
reseeded per variant) so the comparison is like-for-like.

Run from the repo root:  python -m diagnostics.diagnose_variant_compare
"""

import json
import os

import numpy as np
import torch

from diagnostics.diagnose_beta_writeback import Probe, get_batch, hc_sites, load_model

# ------------------------------------------------------------------- configuration
E = "/home/work/data/guotianzizhe/data/eval_round2"
CKPTS = [
    ("mhc",        f"{E}/out-owt-xl-mhc-bs6-80000step/ckpt_last.pt"),
    ("group",      f"{E}/out-owt-xl-mhc-group-embedding-bs6-80000step/ckpt_last.pt"),
    ("lora",       f"{E}/out-owt-xl-mhc-lora-residual-midnorm-bs6-80000step/ckpt_last.pt"),
    ("group_lora", f"{E}/out-owt-xl-mhc-group-lora-midnorm-bs6-80000step/ckpt_last.pt"),
]
VAL_BIN = "data/openwebtext/val.bin"
DEVICE = "cuda:0"
BATCH_SIZE = 2
BLOCK_SIZE = 1024
NUM_BATCHES = 4
SEED = 1337
OUT_DIR = "diagnostics/reports"
TAG = "variant_compare_xl"

# PLACEHOLDER_REST


def site_metrics(probe):
    """Per-batch scalar metrics for one hc site."""
    b, t, s = probe.beta.shape[0], probe.beta.shape[1], probe.streams
    beta = probe.beta.detach().float().reshape(b, t, s)
    gbeta = probe.beta.grad.detach().float().reshape(b, t, s)
    f = probe.f.detach().float()                                 # [b,t,d]
    d = f.shape[-1]
    r = probe.r.detach().float().view(b, s, t, d)                # [(b s),t,d] -> [b,s,t,d]

    fn = f.norm(dim=-1)                                          # ||f|| per token   [b,t]
    rn = r.norm(dim=-1)                                          # ||r_i|| per stream [b,s,t]
    fr = fn / rn.mean(dim=1).clamp_min(1e-30)                    # ||f|| / mean_i||r_i||

    write = beta.unsqueeze(-1) * f.unsqueeze(2)                  # beta_i f  [b,t,s,d]
    wr = write.norm(dim=-1) / rn.permute(0, 2, 1).clamp_min(1e-30)   # ||beta_i f|| / ||r_i||
    dbeta_dz = beta * (1 - beta / 2)

    return dict(
        fr_mean=fr.mean().item(), fr_max=fr.max().item(),
        write_to_resid_mean=wr.mean().item(), write_to_resid_max=wr.max().item(),
        dLdbeta_norm=gbeta.norm().item(),
        dLdbeta_max=gbeta.abs().max().item(),
        beta_mean=beta.mean().item(), beta_max=beta.max().item(),
        frac_beta_gt_1p9=(beta > 1.9).float().mean().item(),
        frac_beta_lt_0p1=(beta < 0.1).float().mean().item(),
        frac_saturated=((beta > 1.9) | (beta < 0.1)).float().mean().item(),
        dbeta_dz_mean=dbeta_dz.mean().item(),
        dbeta_dz_median=dbeta_dz.median().item(),
    )


def mean_over(dicts):
    return {k: sum(dd[k] for dd in dicts) / len(dicts) for k in dicts[0]}


def run_variant(path, data):
    model, meta = load_model(path, DEVICE)
    probes = [Probe(n, hc) for n, hc in hc_sites(model)]
    gen = torch.Generator().manual_seed(SEED)                   # identical batches per variant
    per_site = {p.name: [] for p in probes}
    for _ in range(NUM_BATCHES):
        x, y = get_batch(data, BATCH_SIZE, BLOCK_SIZE, DEVICE, gen)
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):      # training precision
            _, loss = model(x, y)
        loss.backward()
        for p in probes:
            per_site[p.name].append(site_metrics(p))
            p.clear()
        model.zero_grad(set_to_none=True)
    for p in probes:
        p.remove()
    sites = {name: mean_over(bs) for name, bs in per_site.items()}
    hc_type = meta["model_args"].get("hyper_conn_type")
    del model
    torch.cuda.empty_cache()
    return meta, hc_type, sites


# PLACEHOLDER_REST2


def band(sites, kind, lo, hi):
    return [v for name, v in sites.items()
            if name.endswith(kind) and lo <= int(name[1:3]) <= hi]


def agg(vals, key, fn=None):
    xs = [v[key] for v in vals]
    return (fn or (lambda a: sum(a) / len(a)))(xs)


def summarize(sites):
    """Overall / layer-0 / worst-site rollups for one variant."""
    allv = list(sites.values())
    out = dict(
        overall={k: sum(v[k] for v in allv) / len(allv) for k in allv[0]},
        L00_attn=sites.get("L00.attn"), L00_mlp=sites.get("L00.mlp"),
        worst_fr_site=max(sites, key=lambda n: sites[n]["fr_max"]),
        worst_fr=max(v["fr_max"] for v in allv),
        worst_dLdbeta_site=max(sites, key=lambda n: sites[n]["dLdbeta_norm"]),
        worst_dLdbeta=max(v["dLdbeta_norm"] for v in allv),
    )
    for lo, hi, tag in [(0, 6, "L00_06"), (7, 13, "L07_13"), (14, 20, "L14_20"), (21, 27, "L21_27")]:
        g = band(sites, ".attn", lo, hi) + band(sites, ".mlp", lo, hi)
        out[tag] = {k: sum(v[k] for v in g) / len(g) for k in g[0]}
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    data = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")

    report = {}
    for label, path in CKPTS:
        meta, hc_type, sites = run_variant(path, data)
        report[label] = dict(path=path, iter_num=meta["iter_num"], hyper_conn_type=hc_type,
                             summary=summarize(sites), sites=sites)
        print(f"{label:11s} iter {meta['iter_num']}  type={hc_type}  {len(sites)} sites")

    with open(os.path.join(OUT_DIR, f"{TAG}.json"), "w") as fh:
        json.dump(report, fh, indent=1)

    def row(label, m):
        return (f"{label:11s} | {m['fr_mean']:7.2f} {m['fr_max']:9.1f} | "
                f"{m['dLdbeta_norm']:8.3f} {m['dLdbeta_max']:8.3f} | "
                f"{m['frac_saturated']:.3f} {m['frac_beta_gt_1p9']:.3f} {m['frac_beta_lt_0p1']:.3f} | "
                f"{m['dbeta_dz_mean']:.4f} {m['beta_mean']:.3f}")

    hdr = ("variant     | fr_mean   fr_max | dLdb_norm dLdb_max | sat    b>1.9 b<0.1 | "
           "dbdz    b_mean")
    for scope, key in [("OVERALL (56 sites, mean)", "overall"),
                       ("LAYER 0 attention", "L00_attn"),
                       ("LAYER 0 mlp", "L00_mlp")]:
        print(f"\n=== {scope} ===\n{hdr}")
        for label, _ in CKPTS:
            print(row(label, report[label]["summary"][key]))

    print("\n=== worst site per variant ===")
    for label, _ in CKPTS:
        s = report[label]["summary"]
        print(f"{label:11s} | max ||f||/||r|| {s['worst_fr']:8.1f} @ {s['worst_fr_site']:9s} | "
              f"max ||dL/dbeta|| {s['worst_dLdbeta']:8.3f} @ {s['worst_dLdbeta_site']}")

    print("\n=== depth bands: fr_mean / dLdbeta_norm / frac_saturated / dbeta_dz_mean ===")
    for label, _ in CKPTS:
        s = report[label]["summary"]
        cells = " | ".join(
            f"{b}:{s[b]['fr_mean']:5.1f}/{s[b]['dLdbeta_norm']:6.2f}/{s[b]['frac_saturated']:.2f}/{s[b]['dbeta_dz_mean']:.3f}"
            for b in ("L00_06", "L07_13", "L14_20", "L21_27"))
        print(f"{label:11s} | {cells}")
    print(f"\nwrote {OUT_DIR}/{TAG}.json")


if __name__ == "__main__":
    main()
