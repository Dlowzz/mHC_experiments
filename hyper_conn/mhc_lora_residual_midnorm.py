from __future__ import annotations

"""
mHC-LoRA-Residual-midnorm  (ablation)
=====================================

Identical to ``mhc_lora_residual`` (per-stream independent LoRA A_s, B_s in the
depth connection) EXCEPT the RMSNorm is placed on the LoRA **rank** dimension,
between A_s and B_s (mid-norm), instead of on the hidden feature dim after B_s.

    down_s  = h @ A_s
    down_s  = rmsnorm(down_s, dim=-1)       # normalise the rank dim r (no affine, eps=1e-6)
    delta_s = down_s @ B_s
    u_s     = beta_s * h + delta_s

Since B_s is zero-initialised, ``delta_s == 0`` at init regardless of ``down_s``,
so the module is identical to the original mHC at init.

Only ``compute_lora`` differs from ``mhc_lora_residual``; everything else
(depth_connection, beta generator, H_res/Sinkhorn, attn/FFN) is inherited
unchanged.  No lora_scale / alpha factor.
"""

from functools import partial

from einops import einsum

from .mhc import Residual, get_expand_reduce_stream_functions, default
from .mhc_lora_residual import ManifoldConstrainedHyperConnectionsLoRAResidual, rmsnorm_lastdim


class ManifoldConstrainedHyperConnectionsLoRAResidualMidNorm(ManifoldConstrainedHyperConnectionsLoRAResidual):
    """per-stream LoRA-residual with RMSNorm on the LoRA rank dim (between A_s and B_s)."""

    def compute_lora(self, branch_output):
        """delta_s = (rmsnorm_{-1}(h @ A_s)) @ B_s for every stream (batched einsum).

        RMSNorm is applied on the rank dim r (dim=-1 of ``down``), between A_s and B_s.
        branch_output : ``b ... f d`` -> ``b ... f s d``
        """
        down = einsum(
            branch_output, self.stream_down_weight,
            "b ... f d, s d r -> b ... f s r",
        )
        down = rmsnorm_lastdim(down)   # RMSNorm on rank dim (=-1 = r)
        delta = einsum(
            down, self.stream_up_weight,
            "b ... f s r, s r e -> b ... f s e",
        )
        return delta                    # B_s zero-init -> delta == 0 at init


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
        ManifoldConstrainedHyperConnectionsLoRAResidualMidNorm if not disable else Residual
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


MHCLoRAResidualMidNorm = ManifoldConstrainedHyperConnectionsLoRAResidualMidNorm
