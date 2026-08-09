from __future__ import annotations

"""
mHC-group-dense-embedding  (ablation)
=====================================

Same group-wise H^pre read as ``mhc_group_embedding`` EXCEPT the per-group read
coefficients are generated from the **whole** normed residual vector instead of only
the group's own slice.

``mhc_group_embedding`` is *group-local*: group q's coefficients H_pre[q, :] see only
``X^(q) in R^{streams x group_dim}`` -- the q-th channel slice of every stream.  The
compact parameter is ``[groups, streams*group_dim, streams]``, i.e. n^2 C values, and
the forward expands it into a block-diagonal dense matrix so that the off-block
entries stay exactly zero.

Here that block-diagonal constraint is dropped: one dense

    group_pre_weight_dense : [streams * effective_dim, groups * streams]

maps all ``n x n`` group streams (the full ``n C`` normed vector) to every group's
``n`` coefficients, i.e. **n^3 C** parameters (n=4, C=1280 at XL: 20,480 -> 81,920).
Group q may now read channel group q' != q of any stream when deciding how much of
each stream to pull into its own slice.

    H_pre_raw = pre_branch_scale * (normed @ W_dense) + group_pre_bias
    H_pre     = sigmoid(H_pre_raw)                      # b ... f groups streams
    u_{q}     = sum_s H_pre[q, s] * X[s, q, :]          # the read itself is unchanged

``W_dense`` is zero-initialised (as the compact weight is), so at init
``H_pre == sigmoid(group_pre_bias)`` exactly as in ``mhc_group_embedding`` -- the two
variants are identical at init, and the dense one can reproduce the group-local one
exactly by writing the compact weight into the matching block-diagonal positions
(``dense_from_group_local`` below; the unit test uses it).

Everything else -- H_res / Sinkhorn, the beta write-back, the attn/FFN branch, the
whole width/depth plumbing -- is inherited from ``mhc_group_embedding`` unchanged.
"""

from functools import partial

import torch
from torch import nn
from einops import einsum

from .mhc import Residual, get_expand_reduce_stream_functions, default
from .mhc_group_embedding import ManifoldConstrainedHyperConnectionsGroupEmbedding


def dense_from_group_local(group_pre_weight, streams, groups, group_dim):
    """Embed a compact ``[groups, streams*group_dim, streams]`` weight into the dense
    ``[streams*groups*group_dim, groups*streams]`` layout (block-diagonal over groups).

    Mirrors the expansion inside ``mhc_group_embedding._compute_group_gates``, so
    ``dense @ normed`` reproduces the group-local read exactly.  Used by the tests and
    by ``load_group_local_`` to initialise a dense module from a group-local one.
    """
    w = group_pre_weight.unflatten(1, (streams, group_dim))      # q s dl e
    block_eye = torch.eye(groups, device=w.device, dtype=w.dtype)
    return einsum(w, block_eye, "q s l e, q p -> s q l p e").reshape(
        streams * groups * group_dim, groups * streams
    )


class ManifoldConstrainedHyperConnectionsGroupDenseEmbedding(
    ManifoldConstrainedHyperConnectionsGroupEmbedding
):
    """Group-wise H^pre read whose per-group coefficients see every group stream."""

    def __init__(self, num_residual_streams, *, dim, **kwargs):
        super().__init__(num_residual_streams, dim=dim, **kwargs)

        streams = self.num_residual_streams
        groups = self.group_embedding_groups
        # drop the parent's compact group-local weight; the dense one replaces it
        # (keeping both would silently double the read and break state_dict shapes)
        del self.group_pre_weight
        # [streams*effective_dim, groups*streams] == n^3 C values when groups == streams.
        # zero-init, so H_pre == sigmoid(group_pre_bias) at init -- same as the parent.
        self.group_pre_weight_dense = nn.Parameter(
            torch.zeros(streams * self.effective_dim, groups * streams)
        )
        # group_pre_bias is inherited unchanged (+1 on the layer's home column, -1 else)

    def load_group_local_(self, group_local_module):
        """Copy a group-local module's read weight into the dense layout, in place.

        Afterwards both modules produce bit-identical H_pre, which is what the
        equivalence test asserts.  Nothing else needs copying: every other parameter
        has the same name and shape in both classes.
        """
        with torch.no_grad():
            self.group_pre_weight_dense.copy_(dense_from_group_local(
                group_local_module.group_pre_weight, self.num_residual_streams,
                self.group_embedding_groups, self.group_dim,
            ))
            self.group_pre_bias.copy_(group_local_module.group_pre_bias)
        return self

    def _compute_group_gates(self, normed):
        """Dense per-group read gates.

        normed : ``b ... f (s d)``  ->  ``b ... f groups streams``

        One plain GEMM on the full ``n C`` vector, no block-diagonal masking, so group q
        can use any stream's any channel group.  Same scale/bias convention as the parent.
        """
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
        ManifoldConstrainedHyperConnectionsGroupDenseEmbedding if not disable else Residual
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


MHCGroupDenseEmbedding = ManifoldConstrainedHyperConnectionsGroupDenseEmbedding
