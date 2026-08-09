from __future__ import annotations

"""
mHC-group-LoRA-dense-midnorm  (ablation)
========================================

``mhc_group_lora_midnorm`` (group-wise H^pre read + per-stream LoRA write-back with the
RMSNorm on the LoRA rank dim) with the H^pre generator swapped for the **dense n^3 C**
form -- the same change ``mhc_group_dense_embedding`` makes to ``mhc_group_embedding``.

    group_pre_weight_dense : [streams * effective_dim, groups * streams]

so every group's ``n`` read coefficients are computed from all ``n x n`` group streams
(the whole ``n C`` normed vector) instead of only that group's own
``[streams, group_dim]`` slice.  Zero-init, so H_pre == sigmoid(group_pre_bias) at init,
identical to the group-local parent.

The LoRA side is untouched: midnorm (RMSNorm on the rank dim between A_s and B_s),
in-beta write ``u_s = beta_s (h + lambda * delta_s)``, gate applied in the rank space,
A_s excluded from weight decay.  H_res / Sinkhorn and the beta generator are also
unchanged.
"""

from functools import partial

import torch
from torch import nn

from .mhc import Residual, get_expand_reduce_stream_functions, default
from .mhc_group_dense_embedding import dense_from_group_local
from .mhc_group_lora_midnorm import ManifoldConstrainedHyperConnectionsGroupLoRAMidNorm


class ManifoldConstrainedHyperConnectionsGroupLoRADenseMidNorm(
    ManifoldConstrainedHyperConnectionsGroupLoRAMidNorm
):
    """group-LoRA-midnorm whose per-group read coefficients see every group stream."""

    def __init__(self, num_residual_streams, *, dim, **kwargs):
        super().__init__(num_residual_streams, dim=dim, **kwargs)

        streams = self.num_residual_streams
        groups = self.group_embedding_groups
        del self.group_pre_weight            # replaced by the dense read
        self.group_pre_weight_dense = nn.Parameter(
            torch.zeros(streams * self.effective_dim, groups * streams)
        )

    def load_group_local_(self, group_local_module):
        """Copy a group-local module's read weight into the dense layout, in place."""
        with torch.no_grad():
            self.group_pre_weight_dense.copy_(dense_from_group_local(
                group_local_module.group_pre_weight, self.num_residual_streams,
                self.group_embedding_groups, self.group_dim,
            ))
            self.group_pre_bias.copy_(group_local_module.group_pre_bias)
        return self

    def _compute_group_pre_gate(self, normed):
        """Dense per-group read gates: ``b ... f (s d)`` -> ``b ... f groups streams``."""
        streams = self.num_residual_streams
        groups = self.group_embedding_groups

        H_pre_dyn = (normed @ self.group_pre_weight_dense).unflatten(-1, (groups, streams))
        H_pre_raw = self.pre_branch_scale * H_pre_dyn + self.group_pre_bias
        return H_pre_raw.sigmoid()


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
        ManifoldConstrainedHyperConnectionsGroupLoRADenseMidNorm if not disable else Residual
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


MHCGroupLoRADenseMidNorm = ManifoldConstrainedHyperConnectionsGroupLoRADenseMidNorm
