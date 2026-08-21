"""All-layer write-back gate / gradient dump for three variants (read-only, no training).

For every hyper-connection site (28 layers x {attn, mlp} = 56 sites) of each variant,
this dumps -- per residual stream i -- the quantities requested, into one Markdown file:

  (1) H_pre  (branch-input read gate)
        mhc         : 1 x n  alpha_pre  (per-source-stream read into the single branch input),
                      recomputed from the captured width_connection input, hyper_conn/mhc.py:337-369
        group /     : n x n  H_pre_grp [groups, streams], via hc.get_group_gates(x)
        group_lora
      H_post (branch-output write gate)
        1 x n  beta_i = 2*sigmoid(z_i)                       hyper_conn/mhc.py:389
  (2) ||g_i||   downstream gradient magnitude, g_i = dL/dy_i, and its normalised share
                ||g_i|| / sum_j ||g_j||
  (3) v_i       branch/write feature entering stream i's write, write_i = beta_i * v_i:
                mhc / group        v_i = f
                group_lora (LoRA)  v_i = f + lambda * delta_i,  delta_i = compute_lora(f)_i
                report ||v_i||  and  ||beta_i v_i|| / ||r_i||  (write magnitude vs the residual it lands on)
  (4) dL/dbeta_i   per stream, signed value and absolute value (= |<g_i, v_i>|)

Reuses Probe / load_model / get_batch / hc_sites from diagnose_beta_writeback (retain_grad
only; forward graph untouched).  A forward-pre-hook grabs the width_connection input x so
H_pre can be recomputed; recompute_beta(x) is asserted against the captured beta at every
site as a faithfulness anchor (same normed pathway feeds alpha_pre).  fp32 so gates and
gradients are exact.

Run from the repo root:  python -m diagnostics.diagnose_gates_dump
"""

import json
import os

import numpy as np
import torch
from torch import cat
from einops import rearrange, repeat

from diagnostics.diagnose_beta_writeback import Probe, get_batch, hc_sites, load_model

# ------------------------------------------------------------------- configuration
# Module-level constants only: this repo's diagnostics take no argparse flags.
E = "/home/work/data/guotianzizhe/data/eval_round2"
CKPTS = [
    ("mhc",        f"{E}/out-owt-xl-mhc-bs6-80000step/ckpt_last.pt"),
    ("group",      f"{E}/out-owt-xl-mhc-group-embedding-bs6-80000step/ckpt_last.pt"),
    ("group_lora", f"{E}/out-owt-xl-mhc-group-lora-midnorm-bs6-80000step/ckpt_last.pt"),
]
VAL_BIN = "data/openwebtext/val.bin"
DEVICE = "cuda:0"
BATCH_SIZE = 1              # fp32 XL + 56 retained-grad sites on a shared GPU: keep peak low
BLOCK_SIZE = 1024
NUM_BATCHES = 4            # 4 x 1 x 1024 = 4096 tokens (peak memory is per-batch, freed between)
SEED = 1337
OUT_DIR = "diagnostics/reports"
TAG = "gates_dump_all_layers_xl"

def _normed(hc, x):
    """RMSNorm(concat streams) from a width_connection input x, replicating
    hyper_conn/mhc.py:337-346 (and identical in the group subclasses)."""
    streams = hc.num_residual_streams
    res = hc.split_fracs(x)
    res = rearrange(res, "(b s) ... d -> b ... s d", s=streams)
    normed = rearrange(res, "b ... s d -> b ... (s d)")
    return hc.norm(normed)


def recompute_beta(hc, x):
    """beta = 2*sigmoid(z) recomputed from x; a faithfulness anchor for the x capture
    (the beta generator is identical across mhc / group / group_lora)."""
    streams = hc.num_residual_streams
    normed = _normed(hc, x)
    dc = rearrange(normed @ hc.dynamic_beta_fn, "... (s f) -> ... s f", s=streams)
    z = dc * hc.h_post_scale + rearrange(hc.static_beta, "... (s f) -> ... s f", s=streams)
    return (2 * torch.sigmoid(z)).reshape(x.shape[0] // streams, x.shape[1], streams)


def recompute_alpha_pre(hc, x):
    """mHC 1 x n H_pre read gate = sigmoid(alpha_pre) from x, replicating
    hyper_conn/mhc.py:349-369.  Returns [b, t, streams] (num_input_views == num_fracs == 1)."""
    streams = hc.num_residual_streams
    normed = _normed(hc, x)
    wc = rearrange(normed @ hc.dynamic_alpha_fn, "... (s t) -> ... s t", s=streams)
    pbs = repeat(hc.pre_branch_scale, "1 -> v", v=hc.num_input_views * hc.num_fracs)
    rs = repeat(hc.residual_scale, "1 -> s", s=hc.num_fracs * streams)
    alpha = wc * cat((pbs, rs)) + rearrange(hc.static_alpha, "(f s) t -> f s t", s=streams)
    alpha = hc.split_fracs(alpha)
    alpha_pre = alpha[..., : hc.num_input_views].sigmoid()      # [b, t, f1, s, f2, v]
    return alpha_pre.reshape(alpha_pre.shape[0], alpha_pre.shape[1], streams)


def variant_kind(hc):
    """('mhc' | 'group' | 'group_lora') from the module's capabilities."""
    is_group = hasattr(hc, "get_group_gates")
    has_lora = hasattr(hc, "compute_lora") and not getattr(hc, "disable_lora_branch", False)
    if is_group and has_lora:
        return "group_lora"
    return "group" if is_group else "mhc"


def new_acc(hc, groups):
    s = hc.num_residual_streams
    kind = variant_kind(hc)
    z = lambda *shp: torch.zeros(*shp, device=DEVICE, dtype=torch.float64)
    return dict(
        kind=kind, home=int(hc.init_residual_index), streams=s, groups=groups, n=0,
        beta_sum=z(s),
        hpre_sum=z(groups, s) if kind != "mhc" else z(s),   # 4x4 grp gate, or 1x4 alpha_pre
        g_sq=z(s), v_sq=z(s), write_sq=z(s), r_sq=z(s),
        gb_sum=z(s), gb_abs=z(s), gb_sq=z(s),
        beta_err=0.0,
    )


def accumulate(acc, probe, x):
    """Fold one batch of a single site into its accumulator (all per-stream, fp64)."""
    hc, s = probe.hc, probe.streams
    b, t = probe.beta.shape[0], probe.beta.shape[1]
    d = probe.f.shape[-1]

    beta = probe.beta.detach().double().reshape(b, t, s)                 # [b,t,s]
    gbeta = probe.beta.grad.detach().double().reshape(b, t, s)           # dL/dbeta
    f = probe.f.detach().double()                                        # [b,t,d]
    g = probe.y.grad.detach().double().view(b, s, t, d)                  # g_i = dL/dy_i
    r = probe.r.detach().double().view(b, s, t, d)                       # r_i (Sinkhorn-mixed)

    if acc["kind"] == "group_lora":
        with torch.no_grad():
            delta = hc.compute_lora(hc.split_fracs(probe.f)).double()    # [b,t,1,s,d]
        delta = delta.reshape(b, t, s, d).permute(0, 2, 1, 3)            # [b,s,t,d]
        lam = float(getattr(hc, "lora_lambda", 1.0))
        v = f.unsqueeze(1) + lam * delta                                 # f + lam d_i
    else:
        v = f.unsqueeze(1).expand(b, s, t, d)                            # v_i = f

    write = beta.permute(0, 2, 1).unsqueeze(-1) * v                      # beta_i v_i  [b,s,t,d]

    acc["beta_sum"] += beta.sum(dim=(0, 1))
    acc["gb_sum"] += gbeta.sum(dim=(0, 1))
    acc["gb_abs"] += gbeta.abs().sum(dim=(0, 1))
    acc["gb_sq"] += (gbeta ** 2).sum(dim=(0, 1))
    acc["g_sq"] += (g ** 2).sum(dim=(0, 2, 3))
    acc["v_sq"] += (v ** 2).sum(dim=(0, 2, 3))
    acc["write_sq"] += (write ** 2).sum(dim=(0, 2, 3))
    acc["r_sq"] += (r ** 2).sum(dim=(0, 2, 3))
    acc["n"] += b * t

    beta_rc = recompute_beta(hc, x).double()
    acc["beta_err"] = max(acc["beta_err"], (beta_rc - beta).abs().max().item())
    if acc["kind"] == "mhc":
        acc["hpre_sum"] += recompute_alpha_pre(hc, x).double().sum(dim=(0, 1))       # [s]
    else:
        hpg = hc.get_group_gates(x).double().reshape(b, t, acc["groups"], s)
        acc["hpre_sum"] += hpg.sum(dim=(0, 1))                                        # [groups,s]


def finalize(acc):
    """Accumulators -> reported per-stream vectors (pooled norms, token means)."""
    n = acc["n"]
    g_norm = acc["g_sq"].sqrt()
    out = dict(
        kind=acc["kind"], home=acc["home"], streams=acc["streams"], groups=acc["groups"],
        beta=(acc["beta_sum"] / n).tolist(),                                 # H_post 1xn
        hpre=(acc["hpre_sum"] / n).tolist(),                                 # 1xn or nxn
        g_norm=g_norm.tolist(),
        g_share=(g_norm / g_norm.sum().clamp_min(1e-30)).tolist(),
        v_norm=acc["v_sq"].sqrt().tolist(),
        bv_over_r=(acc["write_sq"].sqrt() / acc["r_sq"].sqrt().clamp_min(1e-30)).tolist(),
        dLdbeta_signed=(acc["gb_sum"] / n).tolist(),                         # token-mean, signed
        dLdbeta_abs=(acc["gb_abs"] / n).tolist(),                            # token-mean of |.|
        dLdbeta_norm=acc["gb_sq"].sqrt().tolist(),                           # pooled L2 over tokens
        beta_err=acc["beta_err"],
    )
    return out


def run_variant(path, data):
    """Install probes + width-input pre-hooks on all 56 sites, fold NUM_BATCHES fp32
    batches, and return {site: finalized per-stream dict}."""
    model, meta = load_model(path, DEVICE)
    sites = hc_sites(model)
    probes = [Probe(n, hc) for n, hc in sites]

    cache = {}
    handles = [hc.register_forward_pre_hook(
        lambda mod, args, nm=n: cache.__setitem__(nm, args[0].detach()))
        for n, hc in sites]

    accs = {p.name: new_acc(p.hc, int(getattr(p.hc, "group_embedding_groups", p.streams)))
            for p in probes}

    gen = torch.Generator().manual_seed(SEED)
    for _ in range(NUM_BATCHES):
        x, y = get_batch(data, BATCH_SIZE, BLOCK_SIZE, DEVICE, gen)
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=False):          # fp32: exact gates + grads
            _, loss = model(x, y)
        loss.backward()
        for p in probes:
            accumulate(accs[p.name], p, cache[p.name])
            p.clear()
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    for h in handles:
        h.remove()
    for p in probes:
        p.remove()

    out = {name: finalize(acc) for name, acc in accs.items()}
    hc_type = meta["model_args"].get("hyper_conn_type")
    del model
    torch.cuda.empty_cache()
    return meta, hc_type, out


# ------------------------------------------------------------------- markdown

def _fmt(v, p=4):
    return f"{v:.{p}f}"


def _vec(xs, p=4):
    return "[" + ", ".join(_fmt(x, p) for x in xs) + "]"


def site_block(name, st):
    """Markdown for one site: H_pre / H_post header + a per-stream table."""
    s, home, kind = st["streams"], st["home"], st["kind"]
    L = [f"#### {name}  (home stream {home})", ""]

    # ---- H_pre
    if kind == "mhc":
        L += [f"- **H_pre** (1x{s}, alpha_pre; source-stream read into the single branch input): "
              f"{_vec(st['hpre'])}"]
    else:
        L += [f"- **H_pre** ({st['groups']}x{s} H_pre_grp, rows = groups, cols = source streams):"]
        for q, row in enumerate(st["hpre"]):
            L.append(f"    - group {q}: {_vec(row)}")
    # ---- H_post
    L += [f"- **H_post** (1x{s}, beta_i = 2*sigmoid(z_i)): {_vec(st['beta'])}", ""]

    # ---- per-stream table
    L += ["| stream | H_post beta | \\|\\|g_i\\|\\| | g share | \\|\\|v_i\\|\\| | "
          "\\|\\|beta_i v_i\\|\\|/\\|\\|r_i\\|\\| | dL/dbeta (signed) | \\|dL/dbeta\\| |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for i in range(s):
        star = " (home)" if i == home else ""
        L.append(
            f"| {i}{star} | {_fmt(st['beta'][i])} | {st['g_norm'][i]:.3e} | "
            f"{_fmt(st['g_share'][i])} | {st['v_norm'][i]:.3e} | {_fmt(st['bv_over_r'][i])} | "
            f"{st['dLdbeta_signed'][i]:+.3e} | {st['dLdbeta_abs'][i]:.3e} |")
    L.append("")
    return "\n".join(L)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    data = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")

    report = {}
    for label, path in CKPTS:
        meta, hc_type, sites = run_variant(path, data)
        report[label] = dict(path=path, iter_num=meta["iter_num"], hyper_conn_type=hc_type,
                             sites=sites)
        max_err = max(s["beta_err"] for s in sites.values())
        print(f"{label:11s} iter {meta['iter_num']}  type={hc_type}  {len(sites)} sites  "
              f"| beta faithfulness max|rc-cap| = {max_err:.2e}")

    with open(os.path.join(OUT_DIR, f"{TAG}.json"), "w") as fh:
        json.dump(report, fh, indent=1)

    # ---------------------------------------------------------------- markdown
    kinds = {label: report[label]["sites"][next(iter(report[label]["sites"]))]["kind"]
             for label, _ in CKPTS}
    global_err = max(s["beta_err"] for label, _ in CKPTS for s in report[label]["sites"].values())

    md = [
        "# Write-back gates & gradients: all-layer per-stream dump",
        "",
        f"XL (28 layers, 4 residual streams, num_fracs=1), 56 hyper-connection sites per "
        f"variant. Batches: {NUM_BATCHES} x {BATCH_SIZE} x {BLOCK_SIZE} tokens, seed {SEED}, "
        "**fp32** (autocast off) so gates and gradients are exact. Read-only probe "
        "(`retain_grad` only; forward graph untouched).",
        "",
        "**Variants** (`ckpt_last.pt`, iter 80000):",
    ]
    for label, path in CKPTS:
        r = report[label]
        md.append(f"- `{label}` — type `{r['hyper_conn_type']}`, kind detected `{kinds[label]}`  \n"
                  f"  `{path}`")
    md += [
        "",
        "**Quantities (per residual stream i):**",
        "- **H_pre** — branch-input read gate. `mhc`: 1x4 `alpha_pre` (each source stream's "
        "read into the single branch input, `sigmoid`). `group`/`group_lora`: 4x4 `H_pre_grp` "
        "`[groups, streams]` via `hc.get_group_gates` (each channel-group's read of each source stream).",
        "- **H_post** — branch-output write gate `beta_i = 2*sigmoid(z_i)` (token mean).",
        "- **||g_i||** — pooled L2 norm of the downstream gradient `g_i = dL/dy_i`; **g share** "
        "= `||g_i|| / sum_j ||g_j||`.",
        "- **||v_i||** — pooled L2 norm of the write feature. `mhc`/`group`: `v_i = f`. "
        "`group_lora`: `v_i = f + lambda*delta_i` (`delta_i = compute_lora(f)_i`). "
        "(For `mhc`/`group`, `v_i = f` is stream-independent, so ||v_i|| is equal across streams.)",
        "- **||beta_i v_i||/||r_i||** — write magnitude vs the Sinkhorn-mixed residual it lands on.",
        "- **dL/dbeta (signed)** / **|dL/dbeta|** — token-mean of the H_post-gate gradient, "
        "signed and absolute (`dL/dbeta_i = <g_i, v_i>`).",
        "",
        f"**Faithfulness anchor:** for every site, `beta` recomputed from the captured "
        f"width-connection input matches the autograd-captured `beta` to "
        f"`max|recomputed - captured| = {global_err:.2e}`.",
        "",
    ]
    for label, _ in CKPTS:
        sites = report[label]["sites"]
        md += [f"## Variant: `{label}`  (kind `{kinds[label]}`)", ""]
        for name, st in sites.items():
            md.append(site_block(name, st))

    md_path = os.path.join(OUT_DIR, f"{TAG}.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"\nwrote {md_path}\nwrote {os.path.join(OUT_DIR, f'{TAG}.json')}")


if __name__ == "__main__":
    main()
