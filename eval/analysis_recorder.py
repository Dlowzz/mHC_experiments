"""Paper-analysis evaluation recorder for mHC / mHC-Group-LoRA.

Minimally invasive: for every hyper-connection instance (block i, site attn|mlp) we
temporarily rebind its ``width_connection`` / ``depth_connection`` to capture wrappers
that reproduce the EXACT forward math (verbatim from hyper_conn/mhc.py and
hyper_conn/mhc_group_lora.py) while (a) stashing detached fp32 / CPU forward values and
(b) calling ``.retain_grad()`` on the real graph tensors (per-stream residual x_s, the
branch read input u_q, and beta) so a single real ``loss.backward()`` fills gradient.*.

The model math, parameters and checkpoint are never modified; ``test_no_side_effect``
asserts a bit-identical loss with the recorder on vs off.  ``num_fracs`` is assumed to be
1 (true for every S/M/L/XL config here); we assert it at install time.

Captured, per layer / site / sequence / token (see the 4 categories in the spec):
  routing.*      Hpre_raw, Hpost_raw(beta), Hres_raw, read_contrib_norm
  gradient.*     stream_grad_norm, beta_grad(signed), read_grad_norm
  stream_state.* rms_before_read, rms_after_hres, rms_after_write, same_group_stream_cos
  lora.*         delta/write/postwrite stream cos, lora_main_ratio, lora_perp_ratio
Only raw per-token values are stored; entropy / averages / max-share are left for offline.
"""
from __future__ import annotations

import types

import torch
from torch import cat
from einops import rearrange, repeat, einsum

from hyper_conn.mhc import ManifoldConstrainedHyperConnections, sinkhorn_knopps
from hyper_conn.mhc_group_lora import ManifoldConstrainedHyperConnectionsGroupLoRA

SITES = ("attn", "mlp")
_EPS = 1e-8

def _f32cpu(t):
    return t.detach().to(torch.float32).cpu()


def _rms_lastdim(x):
    """RMS over the last (feature) dim -> shape x.shape[:-1]."""
    return x.float().pow(2).mean(dim=-1).clamp_min(0).sqrt()


def _norm_lastdim(x):
    return x.float().pow(2).sum(dim=-1).clamp_min(0).sqrt()


def _pairwise_cos_streams(x):
    """Pairwise cosine between vectors along a `stream` axis.

    x: ``[..., s, d]`` -> ``[..., s, s]`` with entry [.., s1, s2] = cos(x[..,s1], x[..,s2]).
    Computed in fp32; a zero vector yields cos 0 (via eps in the norm).
    """
    xf = x.float()
    n = xf.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS * _EPS).sqrt()
    u = xf / n
    return einsum(u, u, "... a d, ... b d -> ... a b")


def _perp_ratio(delta, h):
    """||delta - proj_h(delta)|| / (||delta|| + eps), per stream.

    delta: ``[..., s, d]``   h: ``[..., d]`` (the shared main branch output).
    proj_h(delta) = (delta . h_hat) h_hat.  Returns ``[..., s]``.
    """
    delta = delta.float()
    h = h.float()
    hn = h.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS * _EPS).sqrt()
    h_hat = h / hn                                  # [..., d]
    coeff = einsum(delta, h_hat, "... s d, ... d -> ... s")      # delta . h_hat
    proj = coeff.unsqueeze(-1) * h_hat.unsqueeze(-2)             # [..., s, d]
    perp = _norm_lastdim(delta - proj)
    return perp / (_norm_lastdim(delta) + _EPS)

def _normed_per_stream(mod, residuals_in):
    """Reshape the (b s) ... d site input into per-stream b ... s d and its RMSNorm,
    exactly as width_connection does (num_fracs == 1)."""
    streams = mod.num_residual_streams
    r = residuals_in
    if mod.channel_first:
        r = rearrange(r, "b d ... -> b ... d")
    r = mod.split_fracs(r)                                   # b ... f d  (f=1)
    r = rearrange(r, "(b s) ... d -> b ... s d", s=streams)  # per-stream x_s
    normed = mod.norm(rearrange(r, "b ... s d -> b ... (s d)", s=streams))
    return r, normed


@torch.no_grad()
def _alpha_residual(mod, normed):
    """Hres (alpha_residual, post-sinkhorn) — shared by base & group (verbatim)."""
    streams = mod.num_residual_streams
    wc_weight = normed @ mod.dynamic_alpha_fn
    wc_weight = rearrange(wc_weight, "... (s t) -> ... s t", s=streams)
    pre_branch_scale = repeat(mod.pre_branch_scale, "1 -> v", v=mod.num_input_views * mod.num_fracs)
    residual_scale = repeat(mod.residual_scale, "1 -> s", s=mod.num_fracs * streams)
    alpha_scale = cat((pre_branch_scale, residual_scale))
    alpha = mod.split_fracs(wc_weight * alpha_scale + rearrange(mod.static_alpha, "(f s) t -> f s t", s=streams))
    res = alpha[..., mod.num_input_views:]
    res = rearrange(res, "... f s g t -> ... f g s t")
    res = sinkhorn_knopps(res, mod.sinkhorn_iters)
    res = rearrange(res, "... f g s t -> ... f s g t")
    return res, alpha


@torch.no_grad()
def _recompute_gates_base(mod, residuals_in):
    """Plain mHC: Hpre=[b..,1,n] (alpha_pre, post-sigmoid), Hres=[b..,n,n] (src,dst)."""
    x_stream, normed = _normed_per_stream(mod, residuals_in)
    res, alpha = _alpha_residual(mod, normed)
    # alpha / res carry the full [b .. f1 s f2 t] layout (num_fracs==1 -> f1=f2=1).
    alpha_pre = alpha[..., : mod.num_input_views].sigmoid()          # b .. f1 s f2 v
    Hpre = rearrange(alpha_pre, "b ... f1 s f2 v -> b ... (f1 f2 v) s")  # [b.. q=1 n]
    Hres = res.squeeze(-4).squeeze(-2)                               # [b.. n n] (src,dst)
    return {"Hpre": Hpre, "Hres": Hres, "x_stream": x_stream}


@torch.no_grad()
def _recompute_gates_group(mod, residuals_in):
    """Group-LoRA: Hpre=[b..,Q,n] (H_pre_grp, post-sigmoid via the module's own gate),
    Hres=[b..,n,n]."""
    x_stream, normed = _normed_per_stream(mod, residuals_in)
    Hpre = mod._compute_group_pre_gate(normed)                      # b ... f q s
    Hpre = Hpre.squeeze(-3) if Hpre.dim() >= 4 else Hpre            # drop frac -> [b.. q s]
    res, _ = _alpha_residual(mod, normed)
    Hres = res.squeeze(-4).squeeze(-2)
    return {"Hpre": Hpre, "Hres": Hres, "x_stream": x_stream}

def _to_streams(t, n):
    return rearrange(t, "(b s) ... d -> b ... s d", s=n)


def _cap_width(mod, residuals_in):
    """Wrapper for width_connection: real forward + record routing/stream_state read side
    and retain_grad on the real graph tensors (x_s, u, beta)."""
    n = mod.num_residual_streams
    residuals_in.retain_grad()                         # dL/dx_s
    branch_input, residuals_out, kw = mod._orig_width(residuals_in)
    beta = kw.get("beta")
    if beta is not None:
        beta.retain_grad()
    branch_input.retain_grad()                         # dL/du_q

    g = mod._rec_gate_fn(mod, residuals_in)            # no-grad recomputed gate values
    Hpre = g["Hpre"]                                   # [b.. q n]
    x_s = _to_streams(residuals_in.detach(), n)        # [b.. n d]  (real input, per stream)
    read_stream_cos = _pairwise_cos_streams(x_s)       # [b.. n n]  full-stream cos (both models)
    if mod._rec_is_group:
        Q = mod.group_embedding_groups
        xg = rearrange(x_s, "b ... s (q d) -> b ... s q d", q=Q)   # [b.. n Q dq]
        x_read_norm = _norm_lastdim(xg)                             # [b.. n Q]
        read_contrib = Hpre.abs() * rearrange(x_read_norm, "b ... s q -> b ... q s")
        same_cos = _pairwise_cos_streams(rearrange(xg, "b ... s q d -> b ... q s d"))  # [b.. Q n n]
    else:
        Q = 1
        x_read_norm = _norm_lastdim(x_s)                            # [b.. n]
        read_contrib = Hpre.abs() * x_read_norm.unsqueeze(-2)       # [b.. 1 n]
        same_cos = None

    mod._rec_pending = dict(
        layer=mod._rec_layer, site=mod._rec_site, is_group=mod._rec_is_group, Q=Q, n=n,
        fwd=dict(
            Hpre_raw=_f32cpu(Hpre),
            Hres_raw=_f32cpu(g["Hres"]),
            beta=_f32cpu(beta[..., 0, :, 0]) if beta is not None else None,   # [b.. n]
            read_contrib_norm=_f32cpu(read_contrib),
            rms_before_read=_f32cpu(_rms_lastdim(x_s)),                     # [b.. n]
            rms_after_hres=_f32cpu(_rms_lastdim(_to_streams(residuals_out.detach(), n))),
            read_stream_cos=_f32cpu(read_stream_cos),                       # [b.. n n]
        ),
        Hpre_abs=Hpre.abs().detach(),      # kept (cpu-ish) for read_grad after backward
        g_x=residuals_in, g_u=branch_input, g_beta=beta,
    )
    if same_cos is not None:
        mod._rec_pending["fwd"]["same_group_stream_cos"] = _f32cpu(same_cos)
    return branch_input, residuals_out, kw


def _cap_depth(mod, branch_output, residuals, *, beta):
    """Wrapper for depth_connection: run the real write-back, then record the per-stream
    write vectors.  ``write_stream_cos`` / ``postwrite_stream_cos`` are computed for BOTH
    models (mHC write = beta_s * h, collinear across streams by construction); the LoRA
    directional stats (delta cos, lora_main_ratio, lora_perp_ratio) are group-LoRA only.

    ``main_write_s`` and ``lora_lambda * lora_write_s`` are verified to sum to the true
    per-stream write-back (see tests / probe).
    """
    n = mod.num_residual_streams
    out = mod._orig_depth(branch_output, residuals, beta=beta)   # (b s) ... d  real residuals'
    pend = mod._rec_pending
    fwd = pend["fwd"]

    x_out = _to_streams(out.detach(), n)                          # [b.. n d] = x'_s
    fwd["rms_after_write"] = _f32cpu(_rms_lastdim(x_out))         # [b.. n]

    with torch.no_grad():
        bo = mod.split_fracs(branch_output)                      # [b.. f1 d]
        # main write per stream: verbatim depth_connection einsum (mHC: this IS the write)
        main_w = einsum(bo, beta, "b ... f1 d, b ... f1 s f2 -> b ... f2 s d")
        is_group = mod._rec_is_group and not getattr(mod, "disable_lora_branch", False)
        if is_group:
            lora_w = mod.lora_lambda * mod.lora_write(bo, beta)  # real final LoRA write
            w = (main_w + lora_w).squeeze(-3)                    # [b.. n d]  full write-back
            main_w = main_w.squeeze(-3)
            lora_w = lora_w.squeeze(-3)
            delta = mod.compute_lora(bo).squeeze(-3)             # [b.. n d]  directional update
            h = bo.squeeze(-2)                                   # [b.. d]    main branch output
            fwd["delta_stream_cos"] = _f32cpu(_pairwise_cos_streams(delta))     # [b.. n n]
            fwd["lora_main_ratio"] = _f32cpu(
                _norm_lastdim(lora_w) / (_norm_lastdim(main_w) + _EPS))         # [b.. n]
            fwd["lora_perp_ratio"] = _f32cpu(_perp_ratio(delta, h))            # [b.. n]
        else:
            w = main_w.squeeze(-3)                               # [b.. n d]  write = beta_s * h

        # per-stream write / post-write cosine -- recorded for BOTH models
        fwd["write_stream_cos"] = _f32cpu(_pairwise_cos_streams(w))         # [b.. n n]
        fwd["postwrite_stream_cos"] = _f32cpu(_pairwise_cos_streams(x_out)) # [b.. n n]
    return out


# category each recorded key belongs to (used only to organise the saved file)
_CATEGORY = {
    "Hpre_raw": "routing", "Hres_raw": "routing", "beta": "routing",
    "read_contrib_norm": "routing",
    "stream_grad_norm": "gradient", "beta_grad": "gradient", "read_grad_norm": "gradient",
    "rms_before_read": "stream_state", "rms_after_hres": "stream_state",
    "rms_after_write": "stream_state", "same_group_stream_cos": "stream_state",
    # inter-stream cosine of the full residual streams / write vectors -- both models
    "read_stream_cos": "stream_state", "write_stream_cos": "stream_state",
    "postwrite_stream_cos": "stream_state",
    # LoRA directional stats -- group-LoRA only
    "delta_stream_cos": "lora", "lora_main_ratio": "lora", "lora_perp_ratio": "lora",
}


class Recorder:
    """Install capture wrappers on every hyper-connection instance of ``model``.

    Usage (per mini-batch, model in eval() mode, NO torch.no_grad):
        rec = Recorder(model, "mhc"); rec.install()
        for xb, yb in batches:
            model.zero_grad(set_to_none=True)
            logits, loss = model(xb, yb)
            loss.backward()                 # real, unscaled gradients
            rec.collect()                   # read fwd + .grad, offload to CPU, free graph
        rec.save(path); rec.remove()
    """

    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.mods = []                       # [(layer, site, module)]
        self.records = {}                    # (layer, site) -> [per-minibatch dict]
        self._installed = False

    def _iter_hc(self):
        for i, block in enumerate(self.model.transformer.h):
            for site in SITES:
                mod = getattr(block, f"hc_{site}", None)
                if mod is not None and hasattr(mod, "width_connection"):
                    yield i, site, mod

    def install(self):
        assert not self._installed, "Recorder already installed"
        for layer, site, mod in self._iter_hc():
            assert getattr(mod, "num_fracs", 1) == 1, "recorder assumes num_fracs == 1"
            assert not getattr(mod, "channel_first", False), "recorder assumes channel_first == False"
            is_group = isinstance(mod, ManifoldConstrainedHyperConnectionsGroupLoRA)
            mod._orig_width = mod.width_connection
            mod._orig_depth = mod.depth_connection
            mod._rec_gate_fn = _recompute_gates_group if is_group else _recompute_gates_base
            mod._rec_is_group = is_group
            mod._rec_layer = layer
            mod._rec_site = site
            mod._rec_pending = None
            mod.width_connection = types.MethodType(_cap_width, mod)
            mod.depth_connection = types.MethodType(_cap_depth, mod)
            self.mods.append((layer, site, mod))
            self.records[(layer, site)] = []
        assert self.mods, "no hyper-connection instances found on model.transformer.h[*].hc_{attn,mlp}"
        self._installed = True
        return self

    def remove(self):
        for _, _, mod in self.mods:
            if hasattr(mod, "_orig_width"):
                mod.width_connection = mod._orig_width
            if hasattr(mod, "_orig_depth"):
                mod.depth_connection = mod._orig_depth
            for attr in ("_orig_width", "_orig_depth", "_rec_gate_fn", "_rec_is_group",
                         "_rec_layer", "_rec_site", "_rec_pending"):
                if hasattr(mod, attr):
                    delattr(mod, attr)
        self._installed = False

    @torch.no_grad()
    def collect(self):
        """Read forward values (already CPU) + real gradients off the retained graph
        tensors for the mini-batch just backward-ed, then drop all graph references."""
        for layer, site, mod in self.mods:
            pend = mod._rec_pending
            assert pend is not None, f"no pending capture for layer {layer} site {site}"
            n, Q = pend["n"], pend["Q"]
            rec = dict(pend["fwd"])           # copy of the fwd cpu tensors

            g_x, g_u, g_beta = pend["g_x"], pend["g_u"], pend["g_beta"]
            # stream_grad_norm[s] = ||dL/dx_s||
            rec["stream_grad_norm"] = _f32cpu(_norm_lastdim(_to_streams(g_x.grad, n)))
            # read_grad_norm[q,s] = |Hpre[q,s]| * ||dL/du_q||
            if pend["is_group"]:
                du = _norm_lastdim(rearrange(g_u.grad, "b ... (q d) -> b ... q d", q=Q))  # [b.. Q]
            else:
                du = _norm_lastdim(g_u.grad).unsqueeze(-1)                                # [b.. 1]
            rec["read_grad_norm"] = _f32cpu(pend["Hpre_abs"] * du.unsqueeze(-1))          # [b.. Q n]
            # beta_grad[s] (signed) = dL/dbeta_s
            if g_beta is not None and g_beta.grad is not None:
                rec["beta_grad"] = _f32cpu(g_beta.grad[..., 0, :, 0])                     # [b.. n]

            self.records[(layer, site)].append(rec)
            mod._rec_pending = None           # free retained graph tensors for this mini-batch

    def _stack(self):
        """Concatenate per-mini-batch records along the sequence (dim 0) axis and group
        the keys by category.  Returns {"L{layer}.{site}": {category: {key: tensor}}}."""
        out = {}
        for (layer, site), recs in self.records.items():
            if not recs:
                continue
            keys = [k for k in recs[0] if recs[0][k] is not None]
            cats = {}
            for k in keys:
                parts = [r[k] for r in recs if r.get(k) is not None]
                if not parts:
                    continue
                cats.setdefault(_CATEGORY[k], {})[k] = torch.cat(parts, dim=0)
            out[f"L{layer}.{site}"] = cats
        return out

    def save(self, path):
        assert self.mods
        n = self.mods[0][2].num_residual_streams
        sample = self.mods[0][2]
        meta = dict(
            model_name=self.model_name,
            hyper_conn_type=getattr(self.model.config, "hyper_conn_type", None),
            n_streams=n,
            n_layers=len(self.model.transformer.h),
            sites=list(SITES),
            is_group=bool(getattr(sample, "_rec_is_group",
                                  isinstance(sample, ManifoldConstrainedHyperConnectionsGroupLoRA))),
            dtype="float32",
            dims=dict(
                Hpre_raw="[sequence, token, group(Q; mHC Q=1), stream]",
                Hres_raw="[sequence, token, src_stream, dst_stream]",
                beta="[sequence, token, stream]",
                read_contrib_norm="[sequence, token, group, stream]",
                stream_grad_norm="[sequence, token, stream]",
                beta_grad="[sequence, token, stream]",
                read_grad_norm="[sequence, token, group, stream]",
                rms_before_read="[sequence, token, stream]",
                rms_after_hres="[sequence, token, stream]",
                rms_after_write="[sequence, token, stream]",
                same_group_stream_cos="[sequence, token, group, stream, stream]",
                read_stream_cos="[sequence, token, stream, stream]  (both models)",
                write_stream_cos="[sequence, token, stream, stream]  (both models)",
                postwrite_stream_cos="[sequence, token, stream, stream]  (both models)",
                delta_stream_cos="[sequence, token, stream, stream]",
                lora_main_ratio="[sequence, token, stream]",
                lora_perp_ratio="[sequence, token, stream]",
            ),
        )
        if getattr(sample, "group_embedding_groups", None) is not None:
            meta["n_groups"] = sample.group_embedding_groups
            meta["group_dim"] = sample.group_dim
            meta["lora_lambda"] = getattr(sample, "lora_lambda", None)
            meta["lora_rank"] = getattr(sample, "lora_rank", None)
        payload = dict(meta=meta, layers=self._stack())
        torch.save(payload, path)
        return path