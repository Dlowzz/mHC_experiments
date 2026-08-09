from __future__ import annotations

"""
mHC-group-LoRA-prenorm  (ablation)
==================================

Norm-position ablation on ``mhc_group_lora_midnorm``: the parameter-free RMSNorm sits
**before** the LoRA down projection, i.e. it normalises the branch output h itself
(hidden dim), not the rank activations and not the delta:

    h_n     = rmsnorm(h, dim=-1)             # parameter-free, no affine, eps=1e-6
    down_s  = h_n @ A_s
    delta_s = down_s @ B_s
    u_s     = beta_s (h + lambda * delta_s)  # in-beta write, as in every current variant

Contrast with the other two positions on the group-LoRA line:
  * ``mhc_group_lora_midnorm``  : rmsnorm on the **rank** dim, between A_s and B_s
  * ``mhc_group_lora_postnorm`` : rmsnorm on the **hidden** dim, after B_s

The group-wise H^pre read is untouched -- this is the group-LoRA counterpart of
``mhc_lora_residual_prenorm``, with the identical override.

Only the *main* write ``beta_s * h`` uses the raw h; the LoRA branch sees the normalised
copy.  B_s is zero-initialised, so ``delta_s == 0`` at init and the module is identical
to the group_lora baseline there.

Weight decay: unlike midnorm, A_s is **not** scale-invariant here (the norm is upstream
of A_s, so scaling A_s does change delta), so A_s stays in the normal weight-decay group.
Only the A_s fan_in fix from ``mhc_lora_residual_midnorm`` is reused.

Rank-space beta gating stays valid: the norm does not involve beta and sits before A_s,
so ``(h_n A_s beta_s) B_s == (h_n A_s B_s) beta_s``.  ``lora_write`` therefore keeps the
cheap rank-space form (a unit test pins it against the naive hidden-dim gating).
"""

from functools import partial

from torch import nn
from einops import einsum

from .mhc import Residual, get_expand_reduce_stream_functions, default
from .mhc_group_lora import rmsnorm_lastdim
from .mhc_group_lora_midnorm import ManifoldConstrainedHyperConnectionsGroupLoRAMidNorm
from .mhc_lora_residual_midnorm import init_lora_A_per_stream_


class ManifoldConstrainedHyperConnectionsGroupLoRAPreNorm(
    ManifoldConstrainedHyperConnectionsGroupLoRAMidNorm
):
    """group-LoRA whose parameter-free RMSNorm is applied to h before A_s."""

    def _reset_lora_parameters(self):
        # correct per-matrix fan_in (= d) for A_s, zero B_s -- same as midnorm.
        # A_s is NOT flagged _no_weight_decay: with the norm upstream of A_s it is not
        # scale-invariant, so the usual decay applies.
        init_lora_A_per_stream_(self.stream_down_weight)
        nn.init.zeros_(self.stream_up_weight)

    def lora_down(self, branch_output):
        """``rmsnorm_d(h) @ A_s`` : ``b ... f d`` -> ``b ... f s r`` (norm on the hidden dim)."""
        h = rmsnorm_lastdim(branch_output)          # normalise h, then project down
        return einsum(h, self.stream_down_weight, "b ... f d, s d r -> b ... f s r")

    # lora_up / compute_lora / lora_write are inherited from the midnorm class: they are
    # all expressed through lora_down + lora_up, so moving the norm here is enough.


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
        ManifoldConstrainedHyperConnectionsGroupLoRAPreNorm if not disable else Residual
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


MHCGroupLoRAPreNorm = ManifoldConstrainedHyperConnectionsGroupLoRAPreNorm
