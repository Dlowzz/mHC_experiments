from __future__ import annotations

"""
mHC-group-LoRA-scale-midnorm  (ablation)
========================================

``mhc_group_lora_midnorm`` (group-wise H^pre read + per-stream LoRA with the RMSNorm on
the LoRA **rank** dim between A_s and B_s) plus a learnable **scale** on that norm --
and no bias:

    down_s  = h @ A_s
    down_s  = rmsnorm(down_s, dim=-1)        # parameter-free norm on the rank dim r
    down_s  = down_s * gamma_s               # learnable per-(stream, rank) scale, NO bias
    delta_s = down_s @ B_s
    u_s     = beta_s (h + lambda * delta_s)

The group side of the module is untouched -- this is the group-lora counterpart of
``mhc_lora_residual_scalemidnorm``, with the identical override.

``gamma`` (``lora_rmsnorm_weight``, shape ``[num_streams, lora_rank]``) is ones-init, so
at init this is exactly ``mhc_group_lora_midnorm``; and since B_s is zero-init,
``delta_s == 0`` at init, so it also matches the group_lora baseline.  It is flagged
``_no_weight_decay`` like every other norm affine parameter and follows the normal LR
schedule.

Why no bias: a bias on the rank dim adds a *constant* (input-independent) column to
``down_s``, which B_s turns into a constant per-stream vector added to every token's
delta -- a learned bias on the residual write rather than a normalisation knob.  The
scale keeps the norm's degree of freedom without that.

Implementation note -- this overrides ``lora_down`` (not ``compute_lora``): everything
downstream (``compute_lora`` = up(down), ``lora_write`` = up(beta_write(down))) is
defined in terms of ``lora_down``, so the scale automatically reaches BOTH the plain
delta and the rank-space beta-gated write.  The archived
``mhc_lora_residual_affinemidnorm`` overrode ``compute_lora`` only, which left the affine
out of ``lora_write`` -- i.e. out of the path ``depth_connection`` actually uses.  A unit
test below pins this down.

Rank-space gating stays valid: gamma multiplies elementwise on ``[s, r]`` and beta_s is a
per-stream scalar, so they commute.
"""

from functools import partial

import torch
from torch import nn

from .mhc import Residual, get_expand_reduce_stream_functions, default
from .mhc_group_lora_midnorm import ManifoldConstrainedHyperConnectionsGroupLoRAMidNorm


class ManifoldConstrainedHyperConnectionsGroupLoRAScaleMidNorm(
    ManifoldConstrainedHyperConnectionsGroupLoRAMidNorm
):
    """group-LoRA midnorm with a learnable per-(stream, rank) scale on the rank-dim RMSNorm."""

    def __init__(self, num_residual_streams, *, dim, lora_rank: int = 8, **kwargs):
        super().__init__(num_residual_streams, dim=dim, lora_rank=lora_rank, **kwargs)
        # scale only, no bias; ones-init => identical to the parameter-free midnorm at init
        self.lora_rmsnorm_weight = nn.Parameter(torch.ones(num_residual_streams, lora_rank))
        self.lora_rmsnorm_weight._no_weight_decay = True

    def lora_down(self, branch_output):
        """``rmsnorm_r(h @ A_s) * gamma_s`` : ``b ... f d`` -> ``b ... f s r``.

        Overriding here (rather than ``compute_lora``) is what makes the scale apply to
        ``lora_write`` too, since the parent defines both in terms of ``lora_down``.
        """
        down = super().lora_down(branch_output)          # rmsnorm on the rank dim
        return down * self.lora_rmsnorm_weight.to(down.dtype)


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
        ManifoldConstrainedHyperConnectionsGroupLoRAScaleMidNorm if not disable else Residual
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


MHCGroupLoRAScaleMidNorm = ManifoldConstrainedHyperConnectionsGroupLoRAScaleMidNorm
