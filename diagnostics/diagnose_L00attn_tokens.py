"""L00.attn token/stream-level write-back gate diagnosis (read-only, no training).

For the first attention layer's hyper-connection, this captures -- per token, per
stream, aligned -- the five quantities requested:

    beta_i          the write gate, beta_i = 2*sigmoid(z_i)          (hyper_conn/mhc.py:389)
    z_i             the pre-sigmoid logit, recovered exactly as z_i = logit(beta_i/2)
    dbeta_i/dz_i    the gate Jacobian, = beta_i * (1 - beta_i/2)
    dL/dbeta_i      autograd grad on the beta tensor fed to depth_connection
    dL/dz_i         autograd grad on z, obtained from an independent sigmoid-backward oracle

Elementwise identity being verified:

    dL/dz_i == dL/dbeta_i * beta_i * (1 - beta_i/2)

The RHS uses the analytic Jacobian; the LHS is produced by autograd's own backward
through the SAME op the model uses (b = 2*sigmoid(z)), fed the model's captured
dL/dbeta as the upstream gradient.  z is recovered as logit(beta/2) which is exact
because beta = 2*sigmoid(z) is bijective -- no model surgery, the forward graph is
untouched (Probe only calls retain_grad on tensors already in the graph).

Run in fp32 so the identity holds to fp32 epsilon rather than bf16 rounding.

Run from the repo root:  python -m diagnostics.diagnose_L00attn_tokens
"""

import json
import os

import numpy as np
import torch
from einops import rearrange

from diagnostics.diagnose_beta_writeback import Probe, get_batch, hc_sites, load_model

# ------------------------------------------------------------------- configuration
CKPT_PATH = "out-owt-xl-mhc-bs6-80000step/ckpt_last.pt"
VAL_BIN = "data/openwebtext/val.bin"
DEVICE = "cuda:0"
TARGET_SITE = "L00.attn"
BATCH_SIZE = 2
BLOCK_SIZE = 1024
NUM_BATCHES = 2
SEED = 1337
TOPK = 20
OUT_DIR = "diagnostics/reports"
TAG = "L00attn_tokens_xl_mhc"

# PLACEHOLDER_REST


def forward_z(hc, residuals):
    """Recompute the true pre-sigmoid gate z for one hc module, replicating the beta
    branch of width_connection (hyper_conn/mhc.py:337-388) exactly.  Values only, no
    grad -- z is bijective with beta (beta = 2 sigmoid(z)) so this just recovers the
    logit that the saturated tail loses when reconstructed from an underflowed beta.
    A caller-side assertion checks 2*sigmoid(z) matches the captured beta.
    """
    assert not hc.channel_first
    streams = hc.num_residual_streams
    res = hc.split_fracs(residuals)                                  # (b s) ... f d
    res = rearrange(res, "(b s) ... d -> b ... s d", s=streams)
    normed = rearrange(res, "b ... s d -> b ... (s d)")
    normed = hc.norm(normed)
    dc = normed @ hc.dynamic_beta_fn                                 # ... (s f)
    dc = rearrange(dc, "... (s f) -> ... s f", s=streams)
    z = dc * hc.h_post_scale + rearrange(hc.static_beta, "... (s f) -> ... s f", s=streams)
    return z                                                         # [b, ..., s, f]


def capture(model, probe, data, device, n_batches, gen):
    """Run n_batches in fp32 and collect per-(token, stream) rows for the target site.

    A forward-pre-hook grabs the width_connection input so the true forward z can be
    recomputed (recovering the logit on cells where beta underflowed the fp32 tail)."""
    beta_rows, gbeta_rows, z_rows, tokid_rows, meta_rows = [], [], [], [], []
    cache = {}
    handle = probe.hc.register_forward_pre_hook(
        lambda mod, args: cache.__setitem__("x", args[0].detach()))
    for bi in range(n_batches):
        x, y = get_batch(data, BATCH_SIZE, BLOCK_SIZE, device, gen)
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=False):        # fp32: exact identity
            _, loss = model(x, y)
        loss.backward()

        b, t, s = probe.beta.shape[0], probe.beta.shape[1], probe.streams
        beta = probe.beta.detach().float().reshape(b, t, s)          # [b,t,s]
        gbeta = probe.beta.grad.detach().float().reshape(b, t, s)    # dL/dbeta
        with torch.no_grad():
            z = forward_z(probe.hc, cache["x"]).float().reshape(b, t, s)   # true forward z
        assert torch.allclose(2 * torch.sigmoid(z), beta, atol=1e-4, rtol=1e-3), \
            "recomputed z does not reproduce the captured beta -- forward_z drifted from the model"

        beta_rows.append(beta.reshape(-1, s))
        gbeta_rows.append(gbeta.reshape(-1, s))
        z_rows.append(z.reshape(-1, s))
        tokid_rows.append(x.reshape(-1))                             # token id per position
        pos = torch.arange(t, device=device).expand(b, t).reshape(-1)
        bat = torch.full((b * t,), bi, device=device)
        meta_rows.append(torch.stack([bat, pos], dim=-1))
        probe.clear()
        model.zero_grad(set_to_none=True)

    handle.remove()
    return (torch.cat(beta_rows), torch.cat(gbeta_rows), torch.cat(z_rows),
            torch.cat(tokid_rows), torch.cat(meta_rows))


def verify_identity(z_true, gbeta):
    """Independent autograd oracle for dL/dz, compared against dL/dbeta * beta(1-beta/2).

    The TRUE forward z is rebuilt as a leaf and b = 2*sigmoid(z) reproduces beta; its
    backward is autograd's own d(2 sigmoid)/dz, fed the model's captured dL/dbeta.  No
    clamp: on saturated cells z is very negative, b -> 0 and both sides -> 0 honestly.
    """
    z = z_true.detach().clone().requires_grad_(True)
    b = 2 * torch.sigmoid(z)
    b.backward(gbeta)
    gz_auto = z.grad                                                 # autograd dL/dz
    beta = b.detach()
    dbeta_dz = beta * (1 - beta / 2)                                 # analytic Jacobian
    gz_analytic = gbeta * dbeta_dz
    denom = gz_auto.abs().clamp_min(1e-30)
    rel = (gz_auto - gz_analytic).abs() / denom
    return beta, dbeta_dz, gz_auto.detach(), gz_analytic, rel.detach()


# PLACEHOLDER_REST2


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    data = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")

    model, meta = load_model(CKPT_PATH, DEVICE)
    site = dict(hc_sites(model)).get(TARGET_SITE)
    assert site is not None, f"{TARGET_SITE} not found"
    assert site.num_fracs == 1, "num_fracs must be 1 for the [b,t,s] reshape"
    probe = Probe(TARGET_SITE, site)
    print(f"{CKPT_PATH}: iter {meta['iter_num']}, probing {TARGET_SITE} "
          f"(home stream {site.init_residual_index}), streams={probe.streams}")

    beta_cap, gbeta, z_true, tokid, rowmeta = capture(model, probe, data, DEVICE, NUM_BATCHES, gen)
    probe.remove()
    beta, dbeta_dz, gz_auto, gz_analytic, rel = verify_identity(z_true, gbeta)
    z = z_true

    s = probe.streams
    N = beta.shape[0]
    absg = gbeta.abs()
    flat_rank = torch.argsort(absg.reshape(-1), descending=True)[:TOPK]

    print(f"\nelementwise identity  dL/dz == dL/dbeta * beta(1-beta/2)  over {N*s} (token,stream) cells:")
    print(f"  max rel err {rel.max().item():.3e}   mean rel err {rel.mean().item():.3e}   "
          f"max abs err {(gz_auto - gz_analytic).abs().max().item():.3e}")

    print(f"\ntop {TOPK} (token, stream) by |dL/dbeta|:")
    hdr = " rank  batch  pos  stream  tok_id |   beta      z       dbeta/dz    dL/dbeta     dL/dz(auto)  dL/dz(analytic)  rel_err"
    print(hdr)
    top = []
    for r, idx in enumerate(flat_rank.tolist()):
        tok, st = divmod(idx, s)
        bat, pos = rowmeta[tok].tolist()
        rec = dict(
            rank=r, batch=bat, pos=pos, stream=st, tok_id=int(tokid[tok].item()),
            beta=beta[tok, st].item(), z=z[tok, st].item(), dbeta_dz=dbeta_dz[tok, st].item(),
            dL_dbeta=gbeta[tok, st].item(), dL_dz_auto=gz_auto[tok, st].item(),
            dL_dz_analytic=gz_analytic[tok, st].item(), rel_err=rel[tok, st].item(),
        )
        top.append(rec)
        print(f" {r:4d} {bat:5d} {pos:5d} {st:6d} {rec['tok_id']:7d} | "
              f"{rec['beta']:8.4f} {rec['z']:+8.3f} {rec['dbeta_dz']:10.3e} "
              f"{rec['dL_dbeta']:12.4e} {rec['dL_dz_auto']:12.4e} {rec['dL_dz_analytic']:13.4e} "
              f"{rec['rel_err']:.2e}")

    per_stream = dict(
        beta_mean=beta.mean(0).tolist(), beta_max=beta.amax(0).tolist(),
        z_mean=z.mean(0).tolist(),
        dLdbeta_absmean=gbeta.abs().mean(0).tolist(), dLdbeta_absmax=gbeta.abs().amax(0).tolist(),
        dLdz_absmean=gz_auto.abs().mean(0).tolist(), dLdz_absmax=gz_auto.abs().amax(0).tolist(),
        dbeta_dz_mean=dbeta_dz.mean(0).tolist(),
    )
    report = dict(
        ckpt=CKPT_PATH, ckpt_meta=meta, site=TARGET_SITE,
        home_stream=int(site.init_residual_index), streams=s,
        n_tokens=N, n_cells=N * s, precision="float32",
        identity=dict(max_rel_err=rel.max().item(), mean_rel_err=rel.mean().item(),
                      max_abs_err=(gz_auto - gz_analytic).abs().max().item()),
        per_stream=per_stream, topk=top,
    )
    with open(os.path.join(OUT_DIR, f"{TAG}.json"), "w") as fh:
        json.dump(report, fh, indent=1)

    cols = ["rank", "batch", "pos", "stream", "tok_id", "beta", "z", "dbeta_dz",
            "dL_dbeta", "dL_dz_auto", "dL_dz_analytic", "rel_err"]
    with open(os.path.join(OUT_DIR, f"{TAG}_top{TOPK}.csv"), "w") as fh:
        fh.write(",".join(cols) + "\n")
        for rec in top:
            fh.write(",".join(f"{rec[c]:.6g}" if isinstance(rec[c], float) else str(rec[c])
                              for c in cols) + "\n")
    print(f"\nwrote {OUT_DIR}/{TAG}.json and {OUT_DIR}/{TAG}_top{TOPK}.csv")


if __name__ == "__main__":
    main()
