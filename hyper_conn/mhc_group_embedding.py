from __future__ import annotations

"""
mHC-group-embedding
===================

Group-wise feature read extension of the *full* Manifold-Constrained
Hyper-Connections (mHC).

Only the H^pre branch-input read is group-wise.
The H^post branch-output write-back remains the original mHC beta path.

In the original mHC every hidden channel shares a single set of stream read
coefficients H^pre in R^{1 x n}.  Here the effective hidden dimension is split
into ``groups`` channel groups::

    X_grp in R^{streams x groups x group_dim}          (group_dim = eff_dim / groups)

and each channel group gets its own 1 x n read coefficients::

    H_pre_grp in R^{groups x streams}

Group-wise read (per group q):
    u_{q,t} = sum_s H^pre_{q,s} X_grp_{s,q,t}

The write-back is unchanged from the original mHC:
    dX_s = beta_s h            (beta = H^post via static_beta / dynamic_beta_fn)

``H_res`` / Sinkhorn / residual-stream mixing, the attention/FFN branch, and the
whole width/depth plumbing are kept identical to the original mHC.

The per-group read generator is **group-local** (each group only reads its own
slice ``X^(q) in R^{streams x group_dim}``), so the added parameter cost is
n^2 C (from ``group_pre_weight`` alone), NOT the n^3 C of a dense
``Linear(streams*eff_dim, groups*streams)``.

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
    """Full mHC with a group-wise (per channel-group) H^pre read.

    The H^post write-back stays the original mHC beta path (static_beta /
    dynamic_beta_fn / h_post_scale are kept and used unchanged).
    """

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

        # group-local READ generator only.  input slice per group is
        # [streams, group_dim] -> flatten.  weight shape [groups, streams*group_dim, streams]
        # -> numel == streams^2 * effective_dim (== n^2 C when groups == streams).
        in_feat = streams * group_dim
        self.group_pre_weight = nn.Parameter(torch.zeros(groups, in_feat, streams))

        # static read bias (analogue of static_alpha): "home" stream per group
        pre_bias = torch.full((groups, streams), -1.0)
        for q in range(groups):
            pre_bias[q, q % streams] = 1.0
        self.group_pre_bias = nn.Parameter(pre_bias)

        # optional gate capture for debugging (does not change forward outputs)
        self._capture_gates = False
        self._last_H_pre_grp = None

        # NOTE: the original mHC H^post generators (static_beta / dynamic_beta_fn /
        # h_post_scale) are intentionally KEPT and used unchanged for write-back.

    @property
    def _group_enabled(self):
        return self.use_group_embedding and not self.disable_group_embedding

    # ---------------------------------------------------------------- gates

    def _compute_group_gates(self, normed):
        """Generate per-group H_pre_grp from normed residuals.

        normed : ``b ... f (s d)``  (RMSNorm output, same as parent)
        returns H_pre_grp : ``b ... f groups streams``

        The read is group-local: group q only sees its own ``[streams, group_dim]``
        slice.  Gathering those slices used to need a ``b s q d -> b q (s d)`` rearrange,
        i.e. a full permute copy of ``normed`` (21M elements at XL) on every call.
        Instead the compact weight is expanded into the equivalent block-diagonal dense
        matrix ``[(s q dl), (q e)]`` -- built from the parameter on every forward, so
        gradients only reach the compact parameter and the off-block zeros stay zero --
        and applied as one GEMM on the untouched ``normed``.  The dense form does
        ``streams`` x redundant FLOPs into a tiny (groups*streams) output, which
        measured ~0.8% faster end-to-end at XL than either the rearrange or a
        strided-view einsum.
        """
        streams = self.num_residual_streams
        groups = self.group_embedding_groups

        w = self.group_pre_weight.unflatten(1, (streams, self.group_dim))   # q s dl e (view)
        block_eye = torch.eye(groups, device=w.device, dtype=w.dtype)
        dense = einsum(w, block_eye, "q s l e, q p -> s q l p e").reshape(
            streams * groups * self.group_dim, groups * streams
        )

        # group-local dynamic logits (preserve the parent's scale style)
        H_pre_dyn = (normed @ dense).unflatten(-1, (groups, streams))
        H_pre_raw = self.pre_branch_scale * H_pre_dyn + self.group_pre_bias
        H_pre_grp = H_pre_raw.sigmoid()          # H^pre in [0, 1]
        return H_pre_grp

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
        # alpha and beta gate logits come from one packed GEMM (see parent gate_logits)
        wc_weight, dc_weight = self.gate_logits(normed)
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

        # ---- group-wise READ (the only new logic) ----
        H_pre_grp = self._compute_group_gates(normed)     # b ... f q s
        if self._capture_gates:
            self._last_H_pre_grp = H_pre_grp

        residuals_grp = rearrange(residuals, 'b ... s (q d) -> b ... s q d', q=groups)
        u_grp = einsum(H_pre_grp, residuals_grp,
                       'b ... f q s, b ... f s q d -> b ... f q d')
        branch_input = rearrange(u_grp, 'b ... f q d -> b ... f (q d)')

        # ---- H^post (beta): ORIGINAL mHC write-back generator, unchanged ----
        beta = None
        if self.add_branch_out_to_residual:
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

    # depth_connection is inherited unchanged from ManifoldConstrainedHyperConnections
    # (original beta write-back: dX_s = beta_s h).

    # ---------------------------------------------------- debug interface

    @torch.no_grad()
    def get_group_gates(self, residuals):
        """Return H_pre_grp actually generated for ``residuals`` without changing
        the default forward return values.

        Shape: ``[batch, ..., num_fracs, groups, streams]``.
        """
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
