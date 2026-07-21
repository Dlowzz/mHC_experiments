from __future__ import annotations

"""
mHC-group-LoRA
==============

Combination of two orthogonal (non-overlapping) mHC modifications, kept exactly
as in their standalone files, fused into one module:

  * READ side (from mhc_group_embedding.py):
        group-wise H^pre branch-input read.  The effective hidden dim is split
        into ``groups`` channel groups, each with its own 1 x n read coefficients
        H_pre_grp in R^{groups x streams}:
            u_{q,t} = sum_s H^pre_{q,s} X_grp_{s,q,t}

  * WRITE side (from mhc_lora_residual.py):
        original mHC beta write-back PLUS an independent per-stream LoRA (A_s, B_s),
        with the LoRA output passed through a parameter-free RMSNorm on the hidden
        feature dim (dim=-1) *after B_s* and *before* adding to the beta write-back:
            delta_s = (h @ A_s) @ B_s
            delta_s = rmsnorm(delta_s, dim=-1)            # no affine, eps=1e-6
            u_s     = beta_s * h + delta_s
        (A_s kaiming-uniform, B_s zero-init -> delta_s == 0 at init, and
         rmsnorm(0) == 0, so the write is identical to the original mHC at init.
         The RMSNorm removes the unbounded magnitude d.o.f. that blew up
         ||A_s B_s|| at L scale.  No lora_scale / alpha / (alpha/r) factor.)

Everything else is untouched and reused verbatim from the original mHC:
  * H_res / Sinkhorn / residual-stream mixing
  * the attention / FFN branch
  * the H^post (beta) generator (static_beta / dynamic_beta_fn / h_post_scale)

This file does NOT modify mhc.py, MHC-Lite, mhc_group_embedding.py or
mhc_lora_residual.py.  With both branches disabled it falls back to the exact
original mHC path.
"""

import math
from functools import partial

import torch
from torch import nn, cat
from einops import rearrange, repeat, einsum

from .mhc import (
    ManifoldConstrainedHyperConnections,
    Residual,
    sinkhorn_knopps,
    get_expand_reduce_stream_functions,
    default,
)

# fixed epsilon for the parameter-free LoRA RMSNorm
LORA_RMSNORM_EPS = 1e-6


def rmsnorm_lastdim(x, eps=LORA_RMSNORM_EPS):
    """Parameter-free RMSNorm over the last dim (no affine, fixed eps).

    rmsnorm(0) == 0, so a zero-initialised LoRA stays exactly zero at init.
    Computed in fp32 for stable norms, cast back to the input dtype.
    """
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x.to(dtype)


class ManifoldConstrainedHyperConnectionsGroupLoRA(ManifoldConstrainedHyperConnections):
    """Full mHC with group-wise H^pre read (mhc_group_embedding) AND per-stream
    LoRA write-back on top of the original beta (mhc_lora_residual)."""

    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        # --- group-wise read (mhc_group_embedding) ---
        use_group_embedding: bool = True,
        group_embedding_groups: int | None = None,
        disable_group_embedding: bool = False,
        # --- per-stream LoRA write (mhc_lora_residual) ---
        lora_rank: int = 8,
        disable_lora_branch: bool = False,
        **kwargs,
    ):
        super().__init__(num_residual_streams, dim=dim, **kwargs)

        streams = num_residual_streams
        num_fracs = kwargs.get("num_fracs", 1)
        effective_dim = dim // num_fracs
        assert effective_dim % 1 == 0

        # ---------- group-wise READ params (H_pre only) ----------
        groups = group_embedding_groups or streams
        assert effective_dim % groups == 0, (
            f"effective_dim ({effective_dim} = dim//num_fracs) must be divisible by "
            f"group_embedding_groups ({groups})"
        )
        group_dim = effective_dim // groups

        self.use_group_embedding = use_group_embedding
        self.disable_group_embedding = disable_group_embedding
        self.group_embedding_groups = groups
        self.group_dim = group_dim
        self.effective_dim = effective_dim

        in_feat = streams * group_dim
        # [groups, streams*group_dim, streams] -> numel == streams^2 * effective_dim (n^2 C)
        self.group_pre_weight = nn.Parameter(torch.zeros(groups, in_feat, streams))
        pre_bias = torch.full((groups, streams), -1.0)
        for q in range(groups):
            pre_bias[q, q % streams] = 1.0
        self.group_pre_bias = nn.Parameter(pre_bias)

        # ---------- per-stream LoRA WRITE params ----------
        self.lora_rank = lora_rank
        self.disable_lora_branch = disable_lora_branch
        self.stream_down_weight = nn.Parameter(torch.empty(streams, effective_dim, lora_rank))  # A_s
        self.stream_up_weight = nn.Parameter(torch.zeros(streams, lora_rank, effective_dim))     # B_s (zero)
        self._reset_lora_parameters()
        # NOTE: no lora_scale / alpha factor -- the LoRA output is instead passed
        # through a parameter-free RMSNorm (see compute_lora).

        # debug capture (does not change forward outputs)
        self._capture_gates = False
        self._last_H_pre_grp = None

        # NOTE: original mHC beta generators (static_beta / dynamic_beta_fn /
        # h_post_scale) are KEPT and used unchanged.

    def _reset_lora_parameters(self):
        # hook so subclasses (e.g. group-midnorm) can override the LoRA init
        # without double-initialising (parent init then re-init).
        nn.init.kaiming_uniform_(self.stream_down_weight, a=math.sqrt(5))
        nn.init.zeros_(self.stream_up_weight)

    @property
    def _group_read_enabled(self):
        return self.use_group_embedding and not self.disable_group_embedding

    @property
    def _lora_enabled(self):
        return not self.disable_lora_branch

    # ------------------------------------------------ group-wise read gate

    def _compute_group_pre_gate(self, normed):
        """H_pre_grp from normed residuals.  normed: ``b ... f (s d)`` ->
        returns ``b ... f groups streams``."""
        streams = self.num_residual_streams
        groups = self.group_embedding_groups

        normed_sd = rearrange(normed, "b ... (s d) -> b ... s d", s=streams)
        normed_grp = rearrange(normed_sd, "b ... s (q d) -> b ... s q d", q=groups)
        x_group_flat = rearrange(normed_grp, "b ... s q d -> b ... q (s d)")

        H_pre_dyn = einsum(x_group_flat, self.group_pre_weight, "b ... q i, q i s -> b ... q s")
        H_pre_raw = self.pre_branch_scale * H_pre_dyn + self.group_pre_bias
        return H_pre_raw.sigmoid()

    # ------------------------------------------------ per-stream LoRA

    def compute_lora(self, branch_output):
        """delta_s = rmsnorm_{-1}((h @ A_s) @ B_s) per stream (batched einsum, no loop).

        RMSNorm is applied on the hidden feature dim (dim=-1) AFTER B_s and before
        the beta write-back adds it.  branch_output : ``b ... f d`` -> ``b ... f s d``
        """
        down = einsum(branch_output, self.stream_down_weight, "b ... f d, s d r -> b ... f s r")
        delta = einsum(down, self.stream_up_weight, "b ... f s r, s r e -> b ... f s e")
        return rmsnorm_lastdim(delta)   # RMSNorm on hidden dim (=-1); rmsnorm(0)=0 at init

    # ------------------------------------------------------- width connection

    def width_connection(self, residuals):
        # original read (+ beta) when group read is off; LoRA write still applied in depth
        if not self._group_read_enabled:
            return super().width_connection(residuals)

        assert self.num_input_views == 1, (
            "group-wise read currently supports num_input_views == 1 only"
        )

        streams = self.num_residual_streams
        groups = self.group_embedding_groups

        if self.channel_first:
            residuals = rearrange(residuals, 'b d ... -> b ... d')

        residuals = self.split_fracs(residuals)
        residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s=streams)

        normed = rearrange(residuals, 'b ... s d -> b ... (s d)', s=streams)
        normed = self.norm(normed)

        # ---- H_res path: EXACTLY the original mHC (dynamic alpha + sinkhorn) ----
        wc_weight = normed @ self.dynamic_alpha_fn
        wc_weight = rearrange(wc_weight, '... (s t) -> ... s t', s=streams)

        pre_branch_scale = repeat(self.pre_branch_scale, '1 -> v', v=self.num_input_views * self.num_fracs)
        residual_scale = repeat(self.residual_scale, '1 -> s', s=self.num_fracs * streams)
        alpha_scale = cat((pre_branch_scale, residual_scale))

        dynamic_alpha = wc_weight * alpha_scale
        static_alpha = rearrange(self.static_alpha, '(f s) t -> f s t', s=streams)
        alpha = dynamic_alpha + static_alpha
        alpha = self.split_fracs(alpha)

        alpha_residual = alpha[..., self.num_input_views:]
        alpha_residual = rearrange(alpha_residual, '... f s g t -> ... f g s t')
        alpha_residual = sinkhorn_knopps(alpha_residual, self.sinkhorn_iters)
        alpha_residual = rearrange(alpha_residual, '... f g s t -> ... f s g t')
        residual_mix = einsum(alpha_residual, residuals,
                              '... f1 s f2 t, ... f1 s d -> ... f2 t d')

        # ---- group-wise READ (mhc_group_embedding) ----
        H_pre_grp = self._compute_group_pre_gate(normed)     # b ... f q s
        if self._capture_gates:
            self._last_H_pre_grp = H_pre_grp

        residuals_grp = rearrange(residuals, 'b ... s (q d) -> b ... s q d', q=groups)
        u_grp = einsum(H_pre_grp, residuals_grp, 'b ... f q s, b ... f s q d -> b ... f q d')
        branch_input = rearrange(u_grp, 'b ... f q d -> b ... f (q d)')

        # ---- H^post (beta): ORIGINAL mHC write-back generator, unchanged ----
        beta = None
        if self.add_branch_out_to_residual:
            dc_weight = normed @ self.dynamic_beta_fn
            dc_weight = rearrange(dc_weight, '... (s f) -> ... s f', s=streams)
            dynamic_beta = dc_weight * self.h_post_scale
            static_beta = rearrange(self.static_beta, '... (s f) -> ... s f', s=streams)
            beta = dynamic_beta + static_beta
            beta = beta.sigmoid() * 2

        # ---- tail: identical shape handling to the original mHC ----
        if self.channel_first:
            branch_input = rearrange(branch_input, 'b ... d -> b d ...')
        branch_input = self.merge_fracs(branch_input)

        residuals_out = rearrange(residual_mix, 'b ... s d -> (b s) ... d')
        if self.channel_first:
            residuals_out = rearrange(residuals_out, 'b ... d -> b d ...')
        residuals_out = self.merge_fracs(residuals_out)

        return branch_input, residuals_out, dict(beta=beta)

    # ------------------------------------------------------- depth connection

    def depth_connection(self, branch_output, residuals, *, beta):
        # original beta write-back + per-stream LoRA (mhc_lora_residual)
        assert self.add_branch_out_to_residual

        branch_output = self.split_fracs(branch_output)

        if self.channel_first:
            branch_output = rearrange(branch_output, 'b d ... -> b ... d')

        output = einsum(branch_output, beta, 'b ... f1 d, b ... f1 s f2 -> b ... f2 s d')

        if self._lora_enabled:
            delta = self.compute_lora(branch_output)     # b ... f s d
            output = output + delta

        output = rearrange(output, 'b ... s d -> (b s) ... d')
        output = self.merge_fracs(output)

        if self.channel_first:
            output = rearrange(output, 'b ... d -> b d ...')

        residuals = self.depth_residual_fn(output, residuals)
        return self.dropout(residuals)

    # ---------------------------------------------------- debug interface

    @torch.no_grad()
    def get_group_gates(self, residuals):
        """Return H_pre_grp actually generated for ``residuals`` (shape
        ``[batch, ..., num_fracs, groups, streams]``)."""
        prev = self._capture_gates
        self._capture_gates = True
        try:
            self.width_connection(residuals)
        finally:
            self._capture_gates = prev
        return self._last_H_pre_grp


# convenience factory

def get_init_and_expand_reduce_stream_functions(
    num_streams,
    num_fracs=1,
    dim=None,
    add_stream_embed=False,
    disable=None,
    sinkhorn_iters=20,
    **kwargs,
):
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = (
        ManifoldConstrainedHyperConnectionsGroupLoRA if not disable else Residual
    )

    init_hyper_conn_fn = partial(
        hyper_conn_klass, num_streams, num_fracs=num_fracs, sinkhorn_iters=sinkhorn_iters, **kwargs
    )
    expand_reduce_fns = get_expand_reduce_stream_functions(
        num_streams, add_stream_embed=add_stream_embed, dim=dim, disable=disable
    )

    if dim is not None:
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim=dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)


MHCGroupLoRA = ManifoldConstrainedHyperConnectionsGroupLoRA
