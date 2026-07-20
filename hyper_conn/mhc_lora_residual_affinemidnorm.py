from __future__ import annotations

"""
mHC-LoRA-Residual-affine-midnorm  (ablation)
============================================

Identical to ``mhc_lora_residual_midnorm`` (per-stream LoRA A_s, B_s with a
RMSNorm on the LoRA **rank** dim between A_s and B_s) EXCEPT the RMSNorm is a
normal *affine* RMSNorm with a learnable per-channel scale and bias on the rank
dim (the parameter-free version has neither):

    down_s = h @ A_s
    down_s = rmsnorm(down_s, dim=-1)                 # normalise rank dim r, fp32, eps=1e-6
    down_s = down_s * gamma_s + beta_s               # affine: per-(stream, rank) scale & bias
    delta_s = down_s @ B_s
    u_s     = beta_write_s * h + delta_s

Init (mirrors the parameter-free version at init):
  * gamma (``lora_rmsnorm_weight``) : ones  [num_streams, rank]
  * beta  (``lora_rmsnorm_bias``)   : zeros [num_streams, rank]
  => at init the affine RMSNorm == the parameter-free RMSNorm, and since B_s is
     zero-initialised, delta_s == 0 => identical to mhc_lora_residual_midnorm /
     original mHC at init.

The affine parameters are named ``lora_rmsnorm_*`` so that
``GPT.configure_optimizers`` places them in a dedicated optimizer group with
weight_decay=0 and ``no_lr_decay=True`` (constant LR, no cosine decay / warmup).

Only ``__init__`` and ``compute_lora`` differ from
``mhc_lora_residual_midnorm``; everything else (depth_connection, beta generator,
H_res/Sinkhorn, attn/FFN) is inherited unchanged.  ``disable_lora_branch=True``
still falls back to the exact original mHC depth connection.
"""

from functools import partial

import torch
from torch import nn
from einops import einsum

from .mhc import Residual, get_expand_reduce_stream_functions, default
from .mhc_lora_residual import LORA_RMSNORM_EPS
from .mhc_lora_residual_midnorm import ManifoldConstrainedHyperConnectionsLoRAResidualMidNorm


class ManifoldConstrainedHyperConnectionsLoRAResidualAffineMidNorm(
    ManifoldConstrainedHyperConnectionsLoRAResidualMidNorm
):
    """per-stream LoRA-residual with an *affine* RMSNorm (learnable per-(stream,rank)
    scale + bias) on the LoRA rank dim, between A_s and B_s."""

    def __init__(self, num_residual_streams, *, dim, lora_rank: int = 8, **kwargs):
        super().__init__(num_residual_streams, dim=dim, lora_rank=lora_rank, **kwargs)
        # affine params on the rank dim, per stream; init scale=1, bias=0
        # (=> affine RMSNorm == parameter-free RMSNorm at init).
        self.lora_rmsnorm_weight = nn.Parameter(torch.ones(num_residual_streams, lora_rank))
        self.lora_rmsnorm_bias = nn.Parameter(torch.zeros(num_residual_streams, lora_rank))

    def compute_lora(self, branch_output):
        """delta_s = (affine_rmsnorm_{-1}(h @ A_s)) @ B_s per stream (batched einsum).

        Affine RMSNorm on the rank dim r (dim=-1 of ``down``), computed in fp32.
        branch_output : ``b ... f d`` -> ``b ... f s d``
        """
        down = einsum(
            branch_output, self.stream_down_weight,
            "b ... f d, s d r -> b ... f s r",
        )
        dtype = down.dtype
        d = down.float()
        d = d * torch.rsqrt(d.pow(2).mean(dim=-1, keepdim=True) + LORA_RMSNORM_EPS)
        # affine: gamma/beta broadcast over [b ... f] against trailing [s, r]
        d = d * self.lora_rmsnorm_weight.float() + self.lora_rmsnorm_bias.float()
        down = d.to(dtype)
        delta = einsum(
            down, self.stream_up_weight,
            "b ... f s r, s r e -> b ... f s e",
        )
        return delta   # B_s zero-init -> delta == 0 at init


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
        ManifoldConstrainedHyperConnectionsLoRAResidualAffineMidNorm if not disable else Residual
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


MHCLoRAResidualAffineMidNorm = ManifoldConstrainedHyperConnectionsLoRAResidualAffineMidNorm
