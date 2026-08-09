from __future__ import annotations

"""
mHC-LoRA-Residual-postnorm  (ablation)
======================================

Norm-position ablation: the parameter-free RMSNorm sits **after** the LoRA up
projection, i.e. it normalises the delta itself on the hidden dim:

    down_s  = h @ A_s
    delta_s = rmsnorm(down_s @ B_s, dim=-1)   # parameter-free, no affine, eps=1e-6
    u_s     = beta_s (h + lambda * delta_s)   # in-beta write, as in every current variant

The three positions under ablation:
  * ``mhc_lora_residual_prenorm``  : rmsnorm on the **hidden** dim of h, before A_s
  * ``mhc_lora_residual_midnorm``  : rmsnorm on the **rank** dim, between A_s and B_s
  * ``mhc_lora_residual_postnorm`` : rmsnorm on the **hidden** dim, after B_s   <-- this file

Relation to ``mhc_lora_residual``: that module already computes this exact forward
pass, so this class only subclasses it and re-does the initialisation / weight-decay
bookkeeping, keeping the three ablation arms comparable:

  1. A_s init uses the corrected per-matrix fan_in (= d) via ``init_lora_A_per_stream_``.
     ``mhc_lora_residual`` calls ``kaiming_uniform_`` on the 3D ``[s, d, r]`` tensor,
     which treats it as a conv weight (fan_in = d*r) and shrinks the init by sqrt(r).
  2. Both A_s and B_s are flagged ``_no_weight_decay``.  With the norm downstream of
     *both* matrices, delta depends only on their directions -- scaling either one is
     cancelled by the norm -- so decay would shrink ||A_s||, ||B_s|| without changing
     the forward pass, silently inflating their effective LR.  (midnorm flags A_s only,
     since there B_s is downstream of the norm and its scale is real.)

Rank-space beta gating is **invalid** here: RMSNorm is scale-invariant, so moving the
per-stream scalar beta_s in front of the norm cancels the gate entirely.  ``lora_write``
therefore stays the parent's naive "gate after the norm" form; a unit test pins this
asymmetry against midnorm.
"""

from functools import partial

from torch import nn

from .mhc import Residual, get_expand_reduce_stream_functions, default
from .mhc_lora_residual import ManifoldConstrainedHyperConnectionsLoRAResidual
from .mhc_lora_residual_midnorm import init_lora_A_per_stream_


class ManifoldConstrainedHyperConnectionsLoRAResidualPostNorm(
    ManifoldConstrainedHyperConnectionsLoRAResidual
):
    """per-stream LoRA whose parameter-free RMSNorm is applied to delta after B_s."""

    def _reset_lora_parameters(self):
        # correct per-matrix fan_in (= d) for A_s, zero B_s -- same as the other two arms.
        init_lora_A_per_stream_(self.stream_down_weight)
        nn.init.zeros_(self.stream_up_weight)
        # the norm is downstream of both matrices -> both are scale-invariant.
        self.stream_down_weight._no_weight_decay = True
        self.stream_up_weight._no_weight_decay = True

    # compute_lora (norm after B_s) and lora_write (gate after the norm) are inherited
    # from ManifoldConstrainedHyperConnectionsLoRAResidual unchanged -- see module docstring.


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
        ManifoldConstrainedHyperConnectionsLoRAResidualPostNorm if not disable else Residual
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


MHCLoRAResidualPostNorm = ManifoldConstrainedHyperConnectionsLoRAResidualPostNorm
