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

Only ``compute_lora`` and the A_s init differ from ``mhc_lora_residual``;
everything else (depth_connection, beta generator, H_res/Sinkhorn, attn/FFN) is
inherited unchanged.  No lora_scale / alpha factor.

A_s init fix: ``nn.init.kaiming_uniform_`` on the full 3D ``[s, d, r]`` tensor
treats it as a conv weight (fan_in = d * r), shrinking the init by sqrt(r) vs
the intended per-matrix value.  The matmul is ``h @ A_s`` with A_s in R^{d x r},
so the correct fan_in is d; we init each stream's matrix with bound 1/sqrt(d)
(the a=sqrt(5) kaiming bound with the correct fan_in).

Weight-decay note: with the rank-dim RMSNorm right after ``h @ A_s``, A_s is
scale-invariant -- decaying it only shrinks ||A_s|| without changing the forward
pass (silently inflating its effective LR).  ``GPT.configure_optimizers``
therefore excludes ``stream_down_weight`` from weight decay for midnorm variants.
"""

import math
from functools import partial

from torch import nn
from einops import einsum

from .mhc import Residual, get_expand_reduce_stream_functions, default
from .mhc_lora_residual import ManifoldConstrainedHyperConnectionsLoRAResidual, rmsnorm_lastdim


def init_lora_A_per_stream_(stream_down_weight):
    """Kaiming-uniform init of A_s ([s, d, r]) with the correct per-matrix fan_in (= d).

    Equivalent to ``nn.init.kaiming_uniform_(A_s.T, a=sqrt(5))`` per stream: A_s is
    ``[d, r]`` and the matmul is ``h @ A_s`` (fan_in = d), but kaiming on a 2D tensor
    takes fan_in from dim 1 (= r here), so it must be applied to the transpose
    ``[r, d]`` to use fan_in = d; the resulting bound is 1/sqrt(d).
    """
    fan_in = stream_down_weight.shape[1]  # effective hidden dim d
    bound = 1.0 / math.sqrt(fan_in)
    nn.init.uniform_(stream_down_weight, -bound, bound)


class ManifoldConstrainedHyperConnectionsLoRAResidualMidNorm(ManifoldConstrainedHyperConnectionsLoRAResidual):
    """per-stream LoRA-residual with RMSNorm on the LoRA rank dim (between A_s and B_s)."""

    def _reset_lora_parameters(self):
        # A_s: kaiming-uniform with the correct per-stream fan_in (= d, not d*r)
        init_lora_A_per_stream_(self.stream_down_weight)
        nn.init.zeros_(self.stream_up_weight)
        # A_s is scale-invariant under the rank-dim RMSNorm -> exclude from weight
        # decay (picked up by GPT.configure_optimizers via this flag).
        self.stream_down_weight._no_weight_decay = True

    def lora_down(self, branch_output):
        """``rmsnorm_r(h @ A_s)`` : ``b ... f d`` -> ``b ... f s r`` (norm on the rank dim)."""
        down = einsum(
            branch_output, self.stream_down_weight,
            "b ... f d, s d r -> b ... f s r",
        )
        return rmsnorm_lastdim(down)   # RMSNorm on rank dim (=-1 = r)

    def lora_up(self, down):
        """``down @ B_s`` : ``b ... f s r`` -> ``b ... f s d``."""
        return einsum(
            down, self.stream_up_weight,
            "b ... f s r, s r e -> b ... f s e",
        )

    def compute_lora(self, branch_output):
        """delta_s = (rmsnorm_{-1}(h @ A_s)) @ B_s for every stream (batched einsum).

        RMSNorm is applied on the rank dim r (dim=-1 of ``down``), between A_s and B_s.
        branch_output : ``b ... f d`` -> ``b ... f s d``
        """
        return self.lora_up(self.lora_down(branch_output))   # B_s zero-init -> delta == 0 at init

    def lora_write(self, branch_output, beta):
        """beta-gated LoRA delta with the gate applied in the LoRA rank space.

        beta_s is a per-stream scalar and B_s maps r -> d inside one stream and is
        shared across fractions, so

            (sum_f1 down[f1,s] beta[f1,s,f2]) B_s == sum_f1 (down[f1,s] B_s) beta[f1,s,f2]

        which is exactly the parent's result -- but the gate multiplies a
        ``[.. s r]`` tensor instead of a ``[.. s d]`` one and the intermediate
        ``[b ... f s d]`` delta is never materialised (r=8 vs d=1280 at XL).
        Valid only because the RMSNorm sits *before* B_s in this variant.
        """
        return self.lora_up(self.beta_write_per_stream(self.lora_down(branch_output), beta))


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
