from __future__ import annotations

"""
mHC-LoRA-Residual
=================

Variant of mHC-embedding that gives **each residual stream its own independent
LoRA pair (A_s, B_s)** in the depth connection, instead of sharing a single B
across all streams.  The learnable scalar ``embedding_scale`` is also removed;
the effective scaling is absorbed into the per-stream B_s initialisation.

Depth connection formula (per residual stream s):

    delta_s = (h @ A_s) @ B_s
    delta_s = rmsnorm(delta_s, dim=-1)      # parameter-free, no affine, eps=1e-6
    u_s     = beta_s * h + delta_s

RMSNorm is applied on the hidden feature dim (dim=-1), after B_s and before the
beta write-back adds delta_s.  Since B_s is zero-initialised, delta_s == 0 at init
and rmsnorm(0) == 0, so the module is identical to the original mHC at init.

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


class ManifoldConstrainedHyperConnectionsLoRAResidual(ManifoldConstrainedHyperConnections):
    """Full mHC + per-stream independent LoRA (A_s, B_s) in the depth connection."""

    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        lora_rank: int = 8,               # rank r  (ablation knob)
        lora_lambda: float = 1.0,         # lambda in u_s = beta_s (h + lambda * delta_s)
        disable_lora_branch: bool = False,
        **kwargs,
    ):
        super().__init__(num_residual_streams, dim=dim, **kwargs)

        num_fracs = kwargs.get("num_fracs", 1)
        assert dim % num_fracs == 0
        eff_dim = dim // num_fracs

        self.lora_rank = lora_rank
        self.lora_lambda = lora_lambda
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
        """delta_s = rmsnorm_{-1}((h @ A_s) @ B_s) for every stream (batched einsum).

        RMSNorm is applied on the hidden feature dim (dim=-1) AFTER B_s and before
        the beta write-back adds it.  branch_output : ``b ... f d`` -> ``b ... f s d``
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
        return rmsnorm_lastdim(delta)   # RMSNorm on hidden dim (=-1); rmsnorm(0)=0 at init

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

        # --- per-stream LoRA branch, INSIDE beta:  u_s = beta_s (h + lambda * delta_s) ---
        # equivalently  output += lambda * beta_s * delta_s  (gate delta by the same
        # per-stream beta write gate, summing over input fractions f1).
        if not self.disable_lora_branch:
            delta = self.compute_lora(branch_output)   # b ... f1 s d
            delta_write = einsum(delta, beta, "b ... f1 s d, b ... f1 s f2 -> b ... f2 s d")
            output = output + self.lora_lambda * delta_write

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
