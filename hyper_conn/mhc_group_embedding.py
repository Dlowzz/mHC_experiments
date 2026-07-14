from __future__ import annotations

"""
mHC-group-embedding
===================

Group-wise feature read/write extension of the *full* Manifold-Constrained
Hyper-Connections (mHC).

In the original mHC every hidden channel shares a single set of stream
read/write coefficients: H^pre in R^{1 x n} and H^post in R^{1 x n}.  Here the
effective hidden dimension is split into ``groups`` channel groups::

    X_grp in R^{streams x groups x group_dim}          (group_dim = eff_dim / groups)

and each channel group gets its own 1 x n read (H^pre) and write (H^post)
coefficients::

    H_pre_grp, H_post_grp in R^{groups x streams}

Group-wise read (per group q):
    u_{q,t} = sum_s H^pre_{q,s} X_grp_{s,q,t}
Group-wise write (per group q):
    dX_{s,q,t} = H^post_{q,s} h_{q,t}

Only the H^pre branch-input read and the H^post branch-output write-back change.
``H_res`` / Sinkhorn / residual-stream mixing, the attention/FFN branch, and the
whole width/depth plumbing are kept identical to the original mHC.

The per-group logit generators are **group-local** (each group only reads its own
slice ``X^(q) in R^{streams x group_dim}``), so the parameter cost is n^2 C, NOT
the n^3 C of a dense ``Linear(streams*eff_dim, groups*streams)``.

This file does NOT modify mhc.py or MHC-Lite.  ``disable_group_embedding=True``
falls back to the exact original mHC path.
"""

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


class ManifoldConstrainedHyperConnectionsGroupEmbedding(ManifoldConstrainedHyperConnections):
    """Full mHC with group-wise (per channel-group) H^pre / H^post."""

    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        use_group_embedding: bool = True,
        group_embedding_groups: int | None = None,
        disable_group_embedding: bool = False,
        **kwargs,
    ):
        super().__init__(num_residual_streams, dim=dim, **kwargs)

        streams = num_residual_streams
        num_fracs = kwargs.get("num_fracs", 1)
        effective_dim = dim // num_fracs                       # matches parent's `dim //= num_fracs`

        groups = group_embedding_groups or streams             # default: groups == streams
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

        # group-local generators.  input slice per group is [streams, group_dim] -> flatten
        in_feat = streams * group_dim                          # == effective_dim only when groups==1
        # weight shape [groups, streams*group_dim, streams] -> numel == streams^2 * effective_dim
        self.group_pre_weight = nn.Parameter(torch.zeros(groups, in_feat, streams))
        self.group_post_weight = nn.Parameter(torch.zeros(groups, in_feat, streams))

        # static biases (analogue of static_alpha / static_beta): "home" stream per group
        pre_bias = torch.full((groups, streams), -1.0)
        post_bias = torch.full((groups, streams), -1.0)
        for q in range(groups):
            pre_bias[q, q % streams] = 1.0
            post_bias[q, q % streams] = 1.0
        self.group_pre_bias = nn.Parameter(pre_bias)
        self.group_post_bias = nn.Parameter(post_bias)

        # optional gate capture for debugging (does not change forward outputs)
        self._capture_gates = False
        self._last_H_pre_grp = None
        self._last_H_post_grp = None

        # When the group path is active it fully replaces the original H^post
        # (beta) generators, leaving `static_beta` / `dynamic_beta_fn` unused.
        # Remove them so DDP does not complain about parameters with no gradient
        # (the original H_res generators alpha are still used and kept intact).
        if self._group_enabled and self.add_branch_out_to_residual:
            for dead in ("static_beta", "dynamic_beta_fn"):
                if hasattr(self, dead):
                    delattr(self, dead)

    @property
    def _group_enabled(self):
        return self.use_group_embedding and not self.disable_group_embedding

    # ---------------------------------------------------------------- gates

    def _compute_group_gates(self, normed):
        """Generate per-group H_pre_grp / H_post_grp from normed residuals.

        normed : ``b ... f (s d)``  (RMSNorm output, same as parent)
        returns H_pre_grp, H_post_grp : ``b ... f groups streams``
        """
        streams = self.num_residual_streams
        groups = self.group_embedding_groups

        # reshape normed -> per-group flattened local slice  [b ... f q (s d)]
        normed_sd = rearrange(normed, "b ... (s d) -> b ... s d", s=streams)
        normed_grp = rearrange(normed_sd, "b ... s (q d) -> b ... s q d", q=groups)
        x_group_flat = rearrange(normed_grp, "b ... s q d -> b ... q (s d)")

        # group-local dynamic logits (preserve the parent's scale style)
        H_pre_dyn = einsum(x_group_flat, self.group_pre_weight, "b ... q i, q i s -> b ... q s")
        H_post_dyn = einsum(x_group_flat, self.group_post_weight, "b ... q i, q i s -> b ... q s")

        H_pre_raw = self.pre_branch_scale * H_pre_dyn + self.group_pre_bias
        H_post_raw = self.h_post_scale * H_post_dyn + self.group_post_bias

        H_pre_grp = H_pre_raw.sigmoid()          # H^pre in [0, 1]
        H_post_grp = H_post_raw.sigmoid() * 2     # H^post in [0, 2]  (same as original beta)
        return H_pre_grp, H_post_grp

    # ------------------------------------------------------- width connection

    def width_connection(self, residuals):
        if not self._group_enabled:
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
        # residual-stream mixing (H_res) -- identical to original
        residual_mix = einsum(alpha_residual, residuals,
                              '... f1 s f2 t, ... f1 s d -> ... f2 t d')

        # ---- group-wise gates + read (the new logic) ----
        H_pre_grp, H_post_grp = self._compute_group_gates(normed)  # b ... f q s
        if self._capture_gates:
            self._last_H_pre_grp = H_pre_grp
            self._last_H_post_grp = H_post_grp

        residuals_grp = rearrange(residuals, 'b ... s (q d) -> b ... s q d', q=groups)
        u_grp = einsum(H_pre_grp, residuals_grp,
                       'b ... f q s, b ... f s q d -> b ... f q d')
        branch_input = rearrange(u_grp, 'b ... f q d -> b ... f (q d)')

        # ---- tail: identical shape handling to the original mHC ----
        if self.channel_first:
            branch_input = rearrange(branch_input, 'b ... d -> b d ...')
        branch_input = self.merge_fracs(branch_input)

        residuals_out = rearrange(residual_mix, 'b ... s d -> (b s) ... d')
        if self.channel_first:
            residuals_out = rearrange(residuals_out, 'b ... d -> b d ...')
        residuals_out = self.merge_fracs(residuals_out)

        return branch_input, residuals_out, dict(H_post_grp=H_post_grp, group_embedding=True)

    # ------------------------------------------------------- depth connection

    def depth_connection(self, branch_output, residuals, *, beta=None,
                         H_post_grp=None, group_embedding=False):
        if not group_embedding:
            return super().depth_connection(branch_output, residuals, beta=beta)

        assert self.add_branch_out_to_residual
        groups = self.group_embedding_groups

        branch_output = self.split_fracs(branch_output)          # b ... f d

        if self.channel_first:
            branch_output = rearrange(branch_output, 'b d ... -> b ... d')

        # group-wise write-back: dX_{s,q,t} = H^post_{q,s} h_{q,t}
        branch_output_grp = rearrange(branch_output, 'b ... f (q d) -> b ... f q d', q=groups)
        output_grp = einsum(H_post_grp, branch_output_grp,
                            'b ... f1 q s, b ... f1 q d -> b ... f1 s q d')
        output = rearrange(output_grp, 'b ... f s q d -> b ... f s (q d)')

        # identical tail to original mHC depth connection
        output = rearrange(output, 'b ... s d -> (b s) ... d')
        output = self.merge_fracs(output)

        if self.channel_first:
            output = rearrange(output, 'b ... d -> b d ...')

        residuals = self.depth_residual_fn(output, residuals)
        return self.dropout(residuals)

    # ---------------------------------------------------- debug interface

    @torch.no_grad()
    def get_group_gates(self, residuals):
        """Return (H_pre_grp, H_post_grp) actually generated for ``residuals``
        without changing the default forward return values.

        Each has shape ``[batch, ..., num_fracs, groups, streams]``.
        """
        prev = self._capture_gates
        self._capture_gates = True
        try:
            self.width_connection(residuals)
        finally:
            self._capture_gates = prev
        return self._last_H_pre_grp, self._last_H_post_grp


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
        ManifoldConstrainedHyperConnectionsGroupEmbedding if not disable else Residual
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


MHCGroupEmbedding = ManifoldConstrainedHyperConnectionsGroupEmbedding
