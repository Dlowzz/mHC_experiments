from __future__ import annotations

"""
mHC-LoRA-Residual
=================

Variant of mHC-embedding that gives **each residual stream its own independent
LoRA pair (A_s, B_s)** in the depth connection, instead of sharing a single B
across all streams.  The learnable scalar ``embedding_scale`` is also removed;
the effective scaling is absorbed into the per-stream B_s initialisation.

Depth connection formula (per residual stream s):

    u_s = beta_s * h  +  (h @ A_s) @ B_s

where
  * A_s in R^{C x r}  –  independent per stream, kaiming-uniform init
  * B_s in R^{r x C}  –  independent per stream, **zero init**
                          (so at init the branch is off -> equals original mHC)
  * the LoRA term lives *outside* beta
  * batched einsum, no per-stream Python loop

Parameter shapes::

    stream_down_weight : [num_streams, dim, rank]   (A_s, independent)
    stream_up_weight   : [num_streams, rank, dim]   (B_s, independent, zero-init)

Both A_s and B_s count are derived from ``num_residual_streams`` – never hard-coded.

This file does NOT touch mhc.py / mhc_lite.py / mhc_embedding.py /
mhc_orthogonal_diff.py.  Setting ``disable_lora_branch=True`` falls back
exactly to the original mHC depth connection.
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


class ManifoldConstrainedHyperConnectionsLoRAResidual(ManifoldConstrainedHyperConnections):
    """Full mHC + per-stream independent LoRA (A_s, B_s) in the depth connection."""

    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        lora_rank: int = 8,               # rank r  (ablation knob)
        disable_lora_branch: bool = False,
        **kwargs,
    ):
        super().__init__(num_residual_streams, dim=dim, **kwargs)

        num_fracs = kwargs.get("num_fracs", 1)
        assert dim % num_fracs == 0
        eff_dim = dim // num_fracs

        self.lora_rank = lora_rank
        self.disable_lora_branch = disable_lora_branch

        # independent A_s : [s, C, r]  –  kaiming-uniform
        self.stream_down_weight = nn.Parameter(
            torch.empty(num_residual_streams, eff_dim, lora_rank)
        )
        # independent B_s : [s, r, C]  –  zero-init (branch off at init)
        self.stream_up_weight = nn.Parameter(
            torch.zeros(num_residual_streams, lora_rank, eff_dim)
        )

        self._reset_lora_parameters()

    def _reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.stream_down_weight, a=math.sqrt(5))
        nn.init.zeros_(self.stream_up_weight)

    def compute_lora(self, branch_output):
        """delta_s = (h @ A_s) @ B_s for every stream at once (batched einsum).

        branch_output : ``b ... f d``  (already split into fractions)
        returns delta : ``b ... f s d``
        """
        # down: [b ... f s r]
        down = einsum(
            branch_output, self.stream_down_weight,
            "b ... f d, s d r -> b ... f s r",
        )
        # up:   [b ... f s d]
        delta = einsum(
            down, self.stream_up_weight,
            "b ... f s r, s r e -> b ... f s e",
        )
        return delta

    def depth_connection(
        self,
        branch_output,
        residuals,
        *,
        beta,
    ):
        assert self.add_branch_out_to_residual

        branch_output = self.split_fracs(branch_output)

        if self.channel_first:
            branch_output = rearrange(branch_output, "b d ... -> b ... d")

        # --- original beta path ---
        output = einsum(branch_output, beta, "b ... f1 d, b ... f1 s f2 -> b ... f2 s d")

        # --- per-stream LoRA branch, OUTSIDE beta ---
        if not self.disable_lora_branch:
            delta = self.compute_lora(branch_output)   # b ... f s d
            output = output + delta

        output = rearrange(output, "b ... s d -> (b s) ... d")
        output = self.merge_fracs(output)

        if self.channel_first:
            output = rearrange(output, "b ... d -> b d ...")

        residuals = self.depth_residual_fn(output, residuals)
        return self.dropout(residuals)


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
        ManifoldConstrainedHyperConnectionsLoRAResidual if not disable else Residual
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


MHCLoRAResidual = ManifoldConstrainedHyperConnectionsLoRAResidual
