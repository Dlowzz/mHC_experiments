"""mHC branch write-back gradient diagnosis (read-only: no training, no model changes).

How the write-back algebra maps onto this repo (all line refs in hyper_conn/mhc.py):

    y_i    = r_i + beta_i * f                        depth_connection, L439-L452
    beta_i = 2 * sigmoid(z_i)                        L389
    z_i    = static_beta_i + h_post_scale * (normed @ dynamic_beta_fn)_i    L381-L388

  f              branch output entering ``ManifoldConstrainedHyperConnections.depth_connection``
                 (the attn or MLP output), shape [b, t, dim]
  r_i            the Sinkhorn-mixed residual returned by ``width_connection``
                 (``mix_h[..., 1:, :]``) -- NOT the raw incoming residual
  y_i            the tensor ``depth_connection`` returns, stacked as [(b s), t, dim]
  beta / H_post  the ``beta`` kwarg, shape [b, t, f1, s, f2]; num_fracs == 1 here -> [b, t, 1, s, 1]
  beta_static    ``static_beta``      init: -1 everywhere, +1 on the layer's home stream
  alpha          ``h_post_scale``     scalar, init 1e-2   (the scale on the dynamic term)
  W_post         ``dynamic_beta_fn``  [dim*streams, num_fracs*streams]

Note on the user-facing formula ``z_i = beta_static_i + alpha * x W_post_i``: the ``x`` that
feeds W_post is ``normed`` -- the RMSNorm of all streams concatenated (L344-L346) -- not the
raw block input, and ``alpha`` is ``h_post_scale`` (the H_post scale), distinct from
``pre_branch_scale`` / ``residual_scale`` which belong to the alpha / H_res side.

Instrumentation only wraps ``depth_connection`` to call ``retain_grad()`` on tensors that
are already in the graph.  No math is changed and no parameter is touched.

dL/dz is obtained exactly from beta without needing z: with sigma = beta/2,
    dbeta/dz = 2 * sigma * (1 - sigma) = beta * (2 - beta) / 2
and z itself is recovered as logit(beta/2).

Run from the repo root:  python -m diagnostics.diagnose_beta_writeback
"""

import json
import os

import numpy as np
import torch

from model import GPT, GPTConfig

# ----------------------------------------------------------------- configuration
# Deliberately module-level constants: this repo's diagnostics do not take argparse flags.

CKPT_PATH = "out-owt-xl-mhc-bs6-80000step/ckpt_last.pt"   # iter 80000, hyper_conn_type=mhc
VAL_BIN = "data/openwebtext/val.bin"
DEVICE = "cuda:0"
BATCH_SIZE = 2
BLOCK_SIZE = 1024
NUM_BATCHES = 4
SEED = 1337
FP32_VERIFY = True          # one extra fp32 batch, purely to check dL/df == sum_i beta_i g_i
OUT_DIR = "diagnostics/reports"
TAG = "beta_writeback_xl_mhc"

# static_beta / h_post_scale / ||W_post|| read straight out of these (mmap, no model build)
DRIFT_CKPTS = [
    ("bs4run_iter28500", "out-owt-xl-mhc-bs4-30000step/ckpt.pt"),
    ("bs6run_iter79000", "out-owt-xl-mhc-bs6-80000step/ckpt.pt"),
    ("bs6run_iter80000", "out-owt-xl-mhc-bs6-80000step/ckpt_last.pt"),
]
STATIC_BETA_INIT_HOME = 1.0
STATIC_BETA_INIT_OTHER = -1.0

# ----------------------------------------------------------------- data / model


def get_batch(data, batch_size, block_size, device, generator):
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model_args = dict(ckpt["model_args"])
    model = GPT(GPTConfig(**model_args))
    state_dict = ckpt["model"]
    for k in list(state_dict.keys()):                 # torch.compile prefix, as in train.py
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    meta = dict(iter_num=ckpt["iter_num"], best_val_loss=float(ckpt["best_val_loss"]),
                model_args=model_args)
    del ckpt, state_dict
    return model, meta


def hc_sites(model):
    """[(name, module)] for every hyper-connection module, tagged attn / mlp per layer."""
    sites = []
    for i, block in enumerate(model.transformer.h):
        for kind in ("attn", "mlp"):
            hc = getattr(block, f"hc_{kind}", None)
            if hc is not None and hasattr(hc, "depth_connection"):
                sites.append((f"L{i:02d}.{kind}", hc))
    return sites


# --------------------------------------------------------------- instrumentation


class Probe:
    """Stashes the three write-back tensors of one hc module and retains their grads.

    Wraps ``depth_connection`` only; the wrapper forwards to the original bound method
    unchanged, so the forward pass is bit-identical to an uninstrumented run.
    """

    def __init__(self, name, hc):
        self.name = name
        self.hc = hc
        self.streams = hc.num_residual_streams
        self._orig = hc.depth_connection
        self.f = self.beta = self.y = self.r = None

        def wrapped(branch_output, residuals, *, beta, **kw):
            if branch_output.requires_grad:
                branch_output.retain_grad()
            if beta.requires_grad:
                beta.retain_grad()
            y = self._orig(branch_output, residuals, beta=beta, **kw)
            if y.requires_grad:
                y.retain_grad()
            self.f, self.beta, self.y, self.r = branch_output, beta, y, residuals
            return y

        hc.depth_connection = wrapped

    def remove(self):
        self.hc.depth_connection = self._orig

    def clear(self):
        self.f = self.beta = self.y = self.r = None


def install_probes(model):
    return [Probe(name, hc) for name, hc in hc_sites(model)]


# ------------------------------------------------------------------ per-site stats


def _cos_matrix(g):
    """g: [s, N] -> (global cosine matrix [s, s], mean of the off-diagonal entries)."""
    gn = torch.nn.functional.normalize(g, dim=-1)
    c = gn @ gn.t()
    s = c.shape[0]
    off = c[~torch.eye(s, dtype=torch.bool, device=c.device)]
    return c, off.mean()


def _cos_per_token(gs):
    """gs: [s, b*t, d] -> (token-averaged cosine matrix, mean off-diagonal)."""
    gn = torch.nn.functional.normalize(gs, dim=-1)
    c = torch.einsum("i n d, j n d -> i j n", gn, gn).mean(dim=-1)
    s = c.shape[0]
    off = c[~torch.eye(s, dtype=torch.bool, device=c.device)]
    return c, off.mean()


def site_stats(probe):
    hc, s = probe.hc, probe.streams
    beta = probe.beta.detach().float().squeeze(4).squeeze(2)        # [b, t, s]
    b, t, _ = beta.shape
    d = probe.f.shape[-1]

    g = probe.y.grad.detach().float().view(b, s, t, d)              # g_i = dL/dy_i
    gf = probe.f.grad.detach().float()                              # dL/df, [b, t, d]

    g_flat = g.permute(1, 0, 2, 3).reshape(s, -1)                   # [s, b*t*d]
    g_tok = g.permute(1, 0, 2, 3).reshape(s, b * t, d)
    g_norm = g_flat.norm(dim=-1)

    cos_g, cos_g_off = _cos_matrix(g_flat)
    cos_tok, cos_tok_off = _cos_per_token(g_tok)

    # dL/df must equal sum_i beta_i g_i  (num_fracs == 1)
    terms = beta.permute(0, 2, 1).unsqueeze(-1) * g                 # [b, s, t, d]
    manual = terms.sum(dim=1)                                       # [b, t, d]
    term_norms = terms.permute(1, 0, 2, 3).reshape(s, -1).norm(dim=-1)
    gf_norm = gf.norm()
    ident_rel = (manual - gf).norm() / gf_norm.clamp_min(1e-30)

    # coherence: 1 if the beta_i g_i are orthogonal, sqrt(s) if all parallel, ->0 if cancelling
    coherence = gf_norm / term_norms.pow(2).sum().sqrt().clamp_min(1e-30)
    align = gf_norm / term_norms.sum().clamp_min(1e-30)

    # forward side: how big is the write beta_i*f next to the residual r_i it lands on
    r_i = probe.r.detach().float().view(b, s, t, d)
    write = beta.permute(0, 2, 1).unsqueeze(-1) * probe.f.detach().float().unsqueeze(1)
    write_norm = write.permute(1, 0, 2, 3).reshape(s, -1).norm(dim=-1)
    r_norm = r_i.permute(1, 0, 2, 3).reshape(s, -1).norm(dim=-1)

    # ---- one step further in: beta -> z -> (static_beta, h_post_scale, W_post)
    gbeta = probe.beta.grad.detach().float().squeeze(4).squeeze(2)   # dL/dbeta, [b, t, s]
    sig = (beta / 2).clamp(1e-7, 1 - 1e-7)
    z = torch.log(sig / (1 - sig))                                  # exact: z = logit(beta/2)
    dbeta_dz = 2 * sig * (1 - sig)                                  # = beta(2-beta)/2
    gz = gbeta * dbeta_dz

    out = dict(
        streams=s, home_stream=int(hc.init_residual_index), b=b, t=t, d=d,
        # ---- beta
        beta_mean=beta.mean().item(), beta_max=beta.max().item(), beta_min=beta.min().item(),
        beta_per_stream_mean=beta.mean(dim=(0, 1)).tolist(),
        beta_per_stream_max=beta.amax(dim=(0, 1)).tolist(),
        beta_norm_per_token=beta.norm(dim=-1).mean().item(),
        frac_beta_gt_1p9=(beta > 1.9).float().mean().item(),
        frac_beta_gt_1p99=(beta > 1.99).float().mean().item(),
        frac_beta_lt_0p1=(beta < 0.1).float().mean().item(),
        # ---- forward: write magnitude vs the residual it lands on
        write_norm_per_stream=write_norm.tolist(),
        resid_norm_per_stream=r_norm.tolist(),
        write_to_resid_per_stream=(write_norm / r_norm.clamp_min(1e-30)).tolist(),
        write_to_resid_mean=(write_norm / r_norm.clamp_min(1e-30)).mean().item(),
        write_to_resid_max=(write_norm / r_norm.clamp_min(1e-30)).max().item(),
        # ---- y_i -> f
        g_norm_per_stream=g_norm.tolist(),
        g_norm_max=g_norm.max().item(), g_norm_mean=g_norm.mean().item(),
        cos_g_global=cos_g.tolist(), cos_g_global_offdiag_mean=cos_g_off.item(),
        cos_g_pertoken=cos_tok.tolist(), cos_g_pertoken_offdiag_mean=cos_tok_off.item(),
        term_norm_per_stream=term_norms.tolist(),
        dLdf_norm=gf_norm.item(),
        identity_rel_err=ident_rel.item(),
        coherence=coherence.item(),                 # 1 = orthogonal, sqrt(s) = all parallel
        align_ratio=align.item(),                   # 1 = perfectly aligned
        amp_vs_mean_g=(gf_norm / g_norm.mean().clamp_min(1e-30)).item(),
        # ---- beta -> z
        dLdbeta_norm=gbeta.norm().item(),
        dLdbeta_per_stream=gbeta.permute(2, 0, 1).reshape(s, -1).norm(dim=-1).tolist(),
        dLdz_norm=gz.norm().item(),
        dLdz_per_stream=gz.permute(2, 0, 1).reshape(s, -1).norm(dim=-1).tolist(),
        sat_shrink=(gz.norm() / gbeta.norm().clamp_min(1e-30)).item(),
        dbeta_dz_mean=dbeta_dz.mean().item(), dbeta_dz_min=dbeta_dz.min().item(),
        z_mean=z.mean().item(), z_max=z.max().item(), z_min=z.min().item(),
        z_per_stream_mean=z.mean(dim=(0, 1)).tolist(),
    )

    # ---- leaf parameters of the write-back gate
    sb, hps, wp = hc.static_beta, hc.h_post_scale, hc.dynamic_beta_fn
    init = [STATIC_BETA_INIT_HOME if i == hc.init_residual_index else STATIC_BETA_INIT_OTHER
            for i in range(sb.numel())]
    out.update(
        static_beta=sb.detach().float().tolist(),
        static_beta_drift=[v - i for v, i in zip(sb.detach().float().tolist(), init)],
        h_post_scale=hps.detach().float().item(),
        W_post_fro=wp.detach().float().norm().item(),
        dL_dstatic_beta=None if sb.grad is None else sb.grad.detach().float().tolist(),
        dL_dh_post_scale=None if hps.grad is None else hps.grad.detach().float().item(),
        dL_dW_post_fro=None if wp.grad is None else wp.grad.detach().float().norm().item(),
    )
    # z splits as static_beta_i + h_post_scale * (normed @ W_post)_i, so the token-averaged
    # dynamic contribution is just the gap between mean z and the static term
    sb_list = sb.detach().float().tolist()
    z_dyn = [zm - v for zm, v in zip(out["z_per_stream_mean"], sb_list)]
    out["z_dynamic_per_stream_mean"] = z_dyn
    out["z_dynamic_mean"] = sum(z_dyn) / len(z_dyn)
    return out


# --------------------------------------------------------------------- averaging


def _avg(values):
    """Elementwise mean over a list of matching scalars / nested lists / None."""
    first = values[0]
    if first is None:
        return None
    if isinstance(first, bool) or isinstance(first, int):
        return first                            # shapes, indices: constant across batches
    if isinstance(first, float):
        return sum(values) / len(values)
    if isinstance(first, list):
        return [_avg([v[i] for v in values]) for i in range(len(first))]
    return first


def average_sites(per_batch):
    """per_batch: [ {site: stats} ] -> {site: averaged stats}."""
    out = {}
    for site in per_batch[0]:
        keys = per_batch[0][site]
        out[site] = {k: _avg([bat[site][k] for bat in per_batch]) for k in keys}
    return out


def read_gate_params(path):
    """static_beta / h_post_scale / ||W_post|| straight from a ckpt, no model build."""
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    sd, rows = ck["model"], {}
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    for k in sd:
        if k.endswith(".static_beta"):
            base = k[: -len(".static_beta")]
            # "transformer.h.7.hc_attn" -> "L07.attn", matching the live-probe site names
            parts = base.split(".")
            site = f"L{int(parts[2]):02d}.{parts[3][len('hc_'):]}"
            sb = sd[k].float()
            rows[site] = dict(
                static_beta=sb.tolist(),
                static_beta_mean=sb.mean().item(), static_beta_max=sb.max().item(),
                beta_from_static_only=(2 * torch.sigmoid(sb)).tolist(),
                h_post_scale=sd[base + ".h_post_scale"].float().item(),
                W_post_fro=sd[base + ".dynamic_beta_fn"].float().norm().item(),
            )
    meta = dict(iter_num=ck.get("iter_num"))
    del ck, sd
    return meta, rows


CSV_COLS = [
    "site", "layer", "kind", "home_stream",
    "beta_mean", "beta_max", "beta_norm_per_token", "frac_beta_gt_1p9", "frac_beta_lt_0p1",
    "write_to_resid_mean", "write_to_resid_max",
    "g_norm_mean", "g_norm_max", "cos_g_pertoken_offdiag_mean", "cos_g_global_offdiag_mean",
    "dLdf_norm", "coherence", "align_ratio", "amp_vs_mean_g", "identity_rel_err",
    "dLdbeta_norm", "dLdz_norm", "sat_shrink", "dbeta_dz_mean", "z_mean", "z_max",
    "z_dynamic_mean",
    "h_post_scale", "W_post_fro", "dL_dh_post_scale", "dL_dW_post_fro",
    "static_beta_mean", "static_beta_max", "static_beta_drift_mean", "dL_dstatic_beta_absmean",
]


def write_csv(path, sites):
    lines = [",".join(CSV_COLS)]
    for site, st in sites.items():
        layer, kind = site.split(".")
        sb, drift, gsb = st["static_beta"], st["static_beta_drift"], st["dL_dstatic_beta"]
        row = dict(st)
        row.update(
            site=site, layer=int(layer[1:]), kind=kind,
            static_beta_mean=sum(sb) / len(sb), static_beta_max=max(sb),
            static_beta_drift_mean=sum(drift) / len(drift),
            dL_dstatic_beta_absmean=None if gsb is None else sum(abs(v) for v in gsb) / len(gsb),
        )
        lines.append(",".join(
            "" if row.get(c) is None else
            (f"{row[c]:.6g}" if isinstance(row[c], float) else str(row[c]))
            for c in CSV_COLS
        ))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def run_batches(model, probes, data, device, n_batches, dtype, generator):
    per_batch = []
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, BATCH_SIZE, BLOCK_SIZE, device, generator)
        model.zero_grad(set_to_none=True)
        ctx = (torch.autocast(device_type="cuda", dtype=dtype)
               if dtype != torch.float32 else torch.autocast("cuda", enabled=False))
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        losses.append(loss.item())
        per_batch.append({p.name: site_stats(p) for p in probes})
        for p in probes:
            p.clear()
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    return per_batch, losses


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)

    data = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    model, meta = load_model(CKPT_PATH, DEVICE)
    probes = install_probes(model)
    print(f"{CKPT_PATH}: iter {meta['iter_num']}, {len(probes)} hyper-conn sites, "
          f"streams={probes[0].streams}, num_fracs={probes[0].hc.num_fracs}")
    assert probes[0].hc.num_fracs == 1, "the sum_i beta_i g_i identity below assumes num_fracs == 1"

    # main pass: bf16 autocast, matching the training precision
    per_batch, losses = run_batches(model, probes, data, DEVICE, NUM_BATCHES,
                                   torch.bfloat16, gen)
    sites = average_sites(per_batch)

    # one fp32 batch purely to show the identity holds exactly, free of bf16 rounding
    fp32 = None
    if FP32_VERIFY:
        f32_batch, _ = run_batches(model, probes, data, DEVICE, 1, torch.float32,
                                   torch.Generator().manual_seed(SEED + 1))
        errs = [s["identity_rel_err"] for s in f32_batch[0].values()]
        fp32 = dict(max_identity_rel_err=max(errs), mean_identity_rel_err=sum(errs) / len(errs))
        print(f"fp32 identity check  dL/df vs sum_i beta_i g_i : "
              f"max rel err {fp32['max_identity_rel_err']:.3e}")

    for p in probes:
        p.remove()

    drift = {}
    for label, path in DRIFT_CKPTS:
        if not os.path.exists(path):
            continue
        dmeta, rows = read_gate_params(path)
        drift[label] = dict(iter_num=dmeta["iter_num"], path=path, sites=rows)
        print(f"gate params read from {path} (iter {dmeta['iter_num']})")

    report = dict(
        ckpt=CKPT_PATH, ckpt_meta=meta, batches=NUM_BATCHES, batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE, seed=SEED, losses=losses,
        mean_loss=sum(losses) / len(losses),
        precision="bfloat16 autocast (training precision)",
        fp32_identity_check=fp32, sites=sites, gate_param_drift=drift,
    )
    json_path = os.path.join(OUT_DIR, f"{TAG}.json")
    csv_path = os.path.join(OUT_DIR, f"{TAG}.csv")
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=1)
    write_csv(csv_path, sites)
    print(f"wrote {json_path}\nwrote {csv_path}")


if __name__ == "__main__":
    main()

