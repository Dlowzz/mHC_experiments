"""g_i / v_i / cos(g_i, v_i) at layers L00-L06 for the four variants, and the identity

    |dL/dbeta_i| == ||g_i|| * ||v_i|| * |cos(g_i, v_i)|            (= |<g_i, v_i>|)

g_i = dL/dy_i is the grad on the returned residual stream (Probe retains it).  v_i is the
per-stream sensitivity of the write to its gate, v_i = d(write_i)/dbeta_i:

    mhc, group        : write_i = beta_i * f            -> v_i = f            (no LoRA branch)
    lora, group_lora  : write_i = beta_i * (f + lam d_i)-> v_i = f + lam d_i  (d_i = compute_lora(f)_i)

so dL/dbeta_i = <g_i, v_i> exactly (beta_i scales write_i linearly; d_i and r_i do not
depend on beta_i; num_fracs == 1).  The reconstructed <g_i, v_i> is compared cell-by-cell
against autograd's captured dL/dbeta_i -- this both verifies the identity and validates the
v_i reconstruction (a wrong v_i would break the match).

Read-only: Probe adds retain_grad only; delta is recomputed under no_grad from the captured
branch output.  fp32 so the identity holds to fp32 epsilon.

Run from the repo root:  python -m diagnostics.diagnose_gv_writeback
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
NUM_BATCHES = 2
SEED = 1337
LAYERS = range(0, 7)          # L00-L06
OUT_DIR = "diagnostics/reports"
TAG = "gv_writeback_L00_06_xl"

# PLACEHOLDER_REST


def site_gv(probe):
    """Per-(stream, token) g_i, v_i, cos and the identity check for one hc site.

    Returns a dict of already-reduced scalars plus per-stream vectors, all detached.
    """
    b, t, s = probe.beta.shape[0], probe.beta.shape[1], probe.streams
    f = probe.f.detach().float()                                    # [b,t,d]
    d = f.shape[-1]
    g = probe.y.grad.detach().float().view(b, s, t, d)              # g_i = dL/dy_i

    has_lora = hasattr(probe.hc, "compute_lora") and not getattr(
        probe.hc, "disable_lora_branch", False)
    if has_lora:
        with torch.no_grad():
            delta = probe.hc.compute_lora(probe.hc.split_fracs(f)).float()   # [b,t,1,s,d]
        delta = delta.reshape(b, t, s, d).permute(0, 2, 1, 3)       # [b,s,t,d]
        lam = float(getattr(probe.hc, "lora_lambda", 1.0))
        v = f.unsqueeze(1) + lam * delta                           # f + lam d_i, [b,s,t,d]
    else:
        v = f.unsqueeze(1).expand(b, s, t, d)                      # v_i = f

    gn = g.norm(dim=-1)                                            # ||g_i||   [b,s,t]
    vn = v.norm(dim=-1)                                            # ||v_i||   [b,s,t]
    gv = (g * v).sum(dim=-1)                                       # <g_i,v_i> [b,s,t]
    cos = gv / (gn * vn).clamp_min(1e-30)

    gbeta = probe.beta.grad.detach().float().reshape(b, t, s).permute(0, 2, 1)  # dL/dbeta [b,s,t]
    pred = gn * vn * cos.abs()                                     # = |<g_i,v_i>|
    abserr = (pred - gbeta.abs()).abs()
    rel = abserr / gbeta.abs().clamp_min(1e-30)
    # relative error is meaningless where |dL/dbeta| ~ 0; mask those out for a fair check
    mask = gbeta.abs() > 1e-5
    rel_masked = (abserr[mask] / gbeta.abs()[mask]) if mask.any() else torch.zeros(1)

    return dict(
        g_norm_mean=gn.mean().item(), g_norm_max=gn.max().item(),
        v_norm_mean=vn.mean().item(), v_norm_max=vn.max().item(),
        cos_mean=cos.mean().item(), abscos_mean=cos.abs().mean().item(),
        gv_absmean=gv.abs().mean().item(), dLdbeta_absmean=gbeta.abs().mean().item(),
        identity_max_rel=rel.max().item(), identity_mean_rel=rel.mean().item(),
        identity_max_abs=abserr.max().item(),
        identity_max_rel_masked=rel_masked.max().item(),
        identity_frac_cells_masked=mask.float().mean().item(),
        g_norm_per_stream=gn.mean(dim=(0, 2)).tolist(),
        v_norm_per_stream=vn.mean(dim=(0, 2)).tolist(),
        cos_per_stream=cos.mean(dim=(0, 2)).tolist(),
        abscos_per_stream=cos.abs().mean(dim=(0, 2)).tolist(),
        has_lora=has_lora,
    )


# PLACEHOLDER_REST2


def mean_over(dicts, key):
    return sum(dd[key] for dd in dicts) / len(dicts)


def run_variant(path, data, keep_sites):
    model, meta = load_model(path, DEVICE)
    probes = [Probe(n, hc) for n, hc in hc_sites(model) if n in keep_sites]
    gen = torch.Generator().manual_seed(SEED)
    per_site = {p.name: [] for p in probes}
    for _ in range(NUM_BATCHES):
        x, y = get_batch(data, BATCH_SIZE, BLOCK_SIZE, DEVICE, gen)
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=False):        # fp32: exact identity
            _, loss = model(x, y)
        loss.backward()
        for p in probes:
            per_site[p.name].append(site_gv(p))
            p.clear()
        model.zero_grad(set_to_none=True)
    for p in probes:
        p.remove()
    scalar_keys = [k for k, val in per_site[probes[0].name][0].items()
                   if isinstance(val, (int, float, bool)) and not isinstance(val, bool)]
    sites = {}
    for name, batches in per_site.items():
        agg = {k: mean_over(batches, k) for k in scalar_keys}
        for vk in ("g_norm_per_stream", "v_norm_per_stream", "cos_per_stream", "abscos_per_stream"):
            cols = list(zip(*[b[vk] for b in batches]))
            agg[vk] = [sum(c) / len(c) for c in cols]
        agg["has_lora"] = batches[0]["has_lora"]
        sites[name] = agg
    hc_type = meta["model_args"].get("hyper_conn_type")
    del model
    torch.cuda.empty_cache()
    return meta, hc_type, sites


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    data = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    keep = {f"L{ly:02d}.{k}" for ly in LAYERS for k in ("attn", "mlp")}

    report = {}
    for label, path in CKPTS:
        meta, hc_type, sites = run_variant(path, data, keep)
        allv = list(sites.values())
        band = {k: sum(v[k] for v in allv) / len(allv) for k in
                ("g_norm_mean", "v_norm_mean", "cos_mean", "abscos_mean",
                 "gv_absmean", "dLdbeta_absmean", "identity_max_rel", "identity_mean_rel",
                 "identity_max_abs", "identity_max_rel_masked")}
        band["identity_max_rel_masked"] = max(v["identity_max_rel_masked"] for v in allv)
        band["identity_max_abs"] = max(v["identity_max_abs"] for v in allv)
        report[label] = dict(hyper_conn_type=hc_type, has_lora=allv[0]["has_lora"],
                             L00_06=band, sites=sites)
        print(f"{label:11s} type={hc_type} has_lora={allv[0]['has_lora']} "
              f"({len(sites)} sites, L00-06)")

    with open(os.path.join(OUT_DIR, f"{TAG}.json"), "w") as fh:
        json.dump(report, fh, indent=1)

    print("\n=== identity  |dL/dbeta_i| == ||g_i|| ||v_i|| |cos(g_i,v_i)|  over L00-06 cells ===")
    for label, _ in CKPTS:
        b = report[label]["L00_06"]
        print(f"{label:11s} | mean rel {b['identity_mean_rel']:.2e}  max abs {b['identity_max_abs']:.2e}  "
              f"max rel (|dL/dbeta|>1e-5) {b['identity_max_rel_masked']:.2e}")

    print("\n=== L00-06 aggregate (mean over 14 sites, per-cell means) ===")
    print("variant     | ||g_i||    ||v_i||     cos      |cos|    |<g,v>|=|dL/dbeta|")
    for label, _ in CKPTS:
        b = report[label]["L00_06"]
        print(f"{label:11s} | {b['g_norm_mean']:.3e} {b['v_norm_mean']:8.2f} "
              f"{b['cos_mean']:+.4f}  {b['abscos_mean']:.4f}  {b['dLdbeta_absmean']:.4e}")

    print("\n=== L00.attn per-stream: ||g_i|| / ||v_i|| / cos(g_i,v_i) ===")
    for label, _ in CKPTS:
        st = report[label]["sites"]["L00.attn"]
        gs = " ".join(f"{x:.2e}" for x in st["g_norm_per_stream"])
        vs = " ".join(f"{x:8.1f}" for x in st["v_norm_per_stream"])
        cs = " ".join(f"{x:+.3f}" for x in st["cos_per_stream"])
        print(f"{label:11s} | g:[{gs}]  v:[{vs}]  cos:[{cs}]")
    print(f"\nwrote {OUT_DIR}/{TAG}.json")


if __name__ == "__main__":
    main()
