from __future__ import annotations

"""
mHC-group-LoRA-capped
=====================

Identical to ``mhc_group_lora`` (group-wise H^pre read + per-stream LoRA
write-back) EXCEPT the LoRA write is bounded **relative to the main mHC beta
write-back**, to fix the late-training grad-norm blow-up observed at L scale.

Motivation
----------
In ``mhc_group_lora`` the depth connection is

    u_s = beta_s * h  +  delta_s ,      delta_s = (h @ A_s) @ B_s

and ``delta_s`` is added *raw and unbounded* to the residual stream.  Because
``B_s`` is zero-initialised but otherwise unconstrained, ``||delta_s||`` grows
during training; over a deep (24-layer) stack this creates a positive-feedback
loop that eventually explodes the gradient norm (seen ~1.2B tokens on L).

Fix (relative cap, applied in ``depth_connection`` where beta is available)
---------------------------------------------------------------------------
Let the main write be ``m_s = beta_s * h`` and pick a fraction ``rho``:

    tau_s      = rho * ||m_s||_2                       (detached; a reference)
    ratio      = ||delta_s||_2 / tau_s
    delta_s   <- delta_s * tanh(ratio) / ratio         (smooth soft-clip)

so that

    ||delta_s||_2  =  rho * ||m_s||_2 * tanh(ratio)  <  rho * ||m_s||_2 .

Properties:
  * ``tanh(r)/r -> 1`` as ``r -> 0``  => small deltas pass through unchanged
    (identical to uncapped group-LoRA early on; delta == 0 at init since B==0).
  * hard upper bound ``||delta_s|| < rho * ||beta_s h||`` => no explosion.
  * scale is direction-preserving and self-adapting (tracks the main write),
    no absolute threshold to tune.
  * effective scale ``~ 1/ratio`` when saturated => contractive gradient on
    ``delta`` (hence on A_s, B_s) => the growth feedback is damped.

Only ``depth_connection`` differs from ``mhc_group_lora``; everything else
(group read, ``compute_lora``, initialisation, H_res/Sinkhorn, beta generator)
is inherited unchanged.  ``disable_lora_branch=True`` still falls back to the
exact original mHC depth connection.
"""

from functools import partial

import torch
from einops import rearrange, einsum

from .mhc import Residual, get_expand_reduce_stream_functions, default
from .mhc_group_lora import ManifoldConstrainedHyperConnectionsGroupLoRA


class ManifoldConstrainedHyperConnectionsGroupLoRACapped(ManifoldConstrainedHyperConnectionsGroupLoRA):
    """group-LoRA with the per-stream LoRA write soft-capped to ``rho`` times the
    main mHC beta write-back norm."""

    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        lora_relative_cap: float = 0.25,   # rho: ||delta_s|| < rho * ||beta_s h||
        **kwargs,
    ):
        super().__init__(num_residual_streams, dim=dim, **kwargs)
        self.lora_relative_cap = float(lora_relative_cap)

    def depth_connection(self, branch_output, residuals, *, beta):
        # original beta write-back + per-stream LoRA, LoRA soft-capped relative
        # to the main beta write norm.
        assert self.add_branch_out_to_residual

        branch_output = self.split_fracs(branch_output)

        if self.channel_first:
            branch_output = rearrange(branch_output, 'b d ... -> b ... d')

        # main mHC write-back  m_s = beta_s * h
        output = einsum(branch_output, beta, 'b ... f1 d, b ... f1 s f2 -> b ... f2 s d')

        if self._lora_enabled:
            delta = self.compute_lora(branch_output)                # b ... f s d

            # ---- relative soft-cap: ||delta_s|| < rho * ||m_s|| ----
            # computed in fp32 for stable norms; main norm detached (reference only)
            main_norm = output.detach().float().norm(dim=-1, keepdim=True)     # b ... s 1
            delta_f = delta.float()
            delta_norm = delta_f.norm(dim=-1, keepdim=True)                     # b ... s 1

            tau = self.lora_relative_cap * main_norm.clamp_min(1e-6)
            ratio = delta_norm / tau.clamp_min(1e-8)
            delta_scale = torch.where(
                delta_norm > 0,
                torch.tanh(ratio) / ratio.clamp_min(1e-8),
                torch.ones_like(delta_norm),
            )
            delta = (delta_f * delta_scale).to(delta.dtype)

            # move the (capped) LoRA write inside beta: u_s = beta_s (h + lambda * delta_s)
            delta_write = einsum(delta, beta, 'b ... f1 s d, b ... f1 s f2 -> b ... f2 s d')
            output = output + self.lora_lambda * delta_write

        output = rearrange(output, 'b ... s d -> (b s) ... d')
        output = self.merge_fracs(output)

        if self.channel_first:
            output = rearrange(output, 'b ... d -> b d ...')

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
        ManifoldConstrainedHyperConnectionsGroupLoRACapped if not disable else Residual
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


MHCGroupLoRACapped = ManifoldConstrainedHyperConnectionsGroupLoRACapped
