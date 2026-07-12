from __future__ import annotations

"""
mHC-embedding
=============

An extension of the *full* Manifold-Constrained Hyper-Connections (mHC) that adds
an extra low-rank "embedding" flow direction to the depth connection, independent
of the beta gate.

Original mHC depth connection writes the branch output ``h`` back to residual
stream ``s`` as::

    u_s = beta_s * h

This module changes it to::

    u_s = beta_s * h + lambda * (h @ A_s) @ B

where
  * each residual stream owns an independent  A_s in R^{C x r}   (``stream_down_weight``)
  * all streams share a single             B    in R^{r x C}   (``shared_up_weight``)
  * ``lambda`` is the learnable ``embedding_scale``
  * the number of A_s is decided by ``num_residual_streams`` (never hard-coded)
  * the new term lives *outside* beta (it is NOT ``beta * (h + gamma)``)
  * everything is computed with batched einsum, no per-stream python loop

``shared_up_weight`` (B) is zero-initialised so that at init ``gamma == 0`` and the
module is numerically identical to the original mHC. ``stream_down_weight`` (A_s)
uses the standard LoRA init (kaiming-uniform).

This file does NOT modify the original mHC (``mhc.py``); it only subclasses it and
overrides ``depth_connection``.
"""

import math
from functools import partial

import torch
from torch import nn
from einops import einsum, rearrange

from .mhc import (
    ManifoldConstrainedHyperConnections,
    Residual,
    get_expand_reduce_stream_functions,
    default,
)


class ManifoldConstrainedHyperConnectionsWithEmbedding(ManifoldConstrainedHyperConnections):
    """Full mHC + independent low-rank embedding branch in the depth connection."""

    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        embedding_rank: int = 8,        # rank r of the low-rank branch (ablation knob)
        embedding_scale: float = 1.0,   # lambda, learnable
        disable_embedding_branch: bool = False,
        **kwargs,
    ):
        super().__init__(num_residual_streams, dim=dim, **kwargs)

        # effective per-fraction feature dim (mirrors the parent's internal `dim //= num_fracs`)
        num_fracs = kwargs.get("num_fracs", 1)
        assert dim % num_fracs == 0, f"dim ({dim}) must be divisible by num_fracs ({num_fracs})"
        eff_dim = dim // num_fracs

        self.embedding_rank = embedding_rank
        self.disable_embedding_branch = disable_embedding_branch

        # independent down-projection A_s per residual stream: [num_streams, dim, rank]
        self.stream_down_weight = nn.Parameter(
            torch.empty(num_residual_streams, eff_dim, embedding_rank)
        )
        # shared up-projection B: [rank, dim]
        self.shared_up_weight = nn.Parameter(torch.zeros(embedding_rank, eff_dim))

        # learnable scale lambda
        self.embedding_scale = nn.Parameter(torch.tensor(float(embedding_scale)))

        self.reset_embedding_parameters()

    def reset_embedding_parameters(self):
        # standard LoRA init: A ~ kaiming-uniform (non-zero), B = 0 (so init == original mHC)
        nn.init.kaiming_uniform_(self.stream_down_weight, a=math.sqrt(5))
        nn.init.zeros_(self.shared_up_weight)

    def compute_gamma(self, branch_output):
        """gamma_s = (h @ A_s) @ B, computed for all streams at once (no loop).

        branch_output : ``b ... f d``  (already split into fractions)
        returns gamma : ``b ... f s d``  (per stream)
        """
        # h @ A_s  ->  per-stream low-rank code
        down = einsum(
            branch_output, self.stream_down_weight,
            "b ... f d, s d r -> b ... f s r",
        )
        # (h @ A_s) @ B  (B shared across streams)
        gamma = einsum(
            down, self.shared_up_weight,
            "b ... f s r, r e -> b ... f s e",
        )
        return gamma

    def depth_connection(
        self,
        branch_output,
        residuals,
        *,
        beta,
    ):
        assert self.add_branch_out_to_residual

        # maybe split fractions
        branch_output = self.split_fracs(branch_output)

        # channel first handling (same as parent)
        if self.channel_first:
            branch_output = rearrange(branch_output, "b d ... -> b ... d")

        # --- original beta path:  u_s = beta_s * h  -> (b ... f2 s d) ---
        output = einsum(branch_output, beta, "b ... f1 d, b ... f1 s f2 -> b ... f2 s d")

        # --- new low-rank embedding branch, OUTSIDE beta:  + lambda * (h A_s) B ---
        if not self.disable_embedding_branch:
            gamma = self.compute_gamma(branch_output)          # b ... f s d
            output = output + self.embedding_scale * gamma     # beta_output + scale * gamma

        output = rearrange(output, "b ... s d -> (b s) ... d")

        # merge fractions back
        output = self.merge_fracs(output)

        # channel first
        if self.channel_first:
            output = rearrange(output, "b ... d -> b d ...")

        residuals = self.depth_residual_fn(output, residuals)

        return self.dropout(residuals)


# convenience factory mirroring mhc.get_init_and_expand_reduce_stream_functions


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
        ManifoldConstrainedHyperConnectionsWithEmbedding if not disable else Residual
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


# keep a short alias
MHCEmbedding = ManifoldConstrainedHyperConnectionsWithEmbedding
