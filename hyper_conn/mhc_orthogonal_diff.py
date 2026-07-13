from __future__ import annotations

"""
mHC-orthogonal-diff
===================

A variant of the *full* Manifold-Constrained Hyper-Connections (mHC) that replaces
**only** the parameterization of the residual mixing matrix ``H_res``.

Everything else of the original mHC is kept byte-for-byte identical:
  * ``H_pre``  (the ``alpha_pre.sigmoid()`` branch-input gate)
  * ``H_post`` (the ``beta`` write-back gate)
  * the dynamic conditioning network that produces the residual logits ``Z``
  * the branch computation and the depth write-back

Original mHC turns the dynamic residual logits ``Z`` into a non-negative doubly
stochastic matrix via Sinkhorn.  Here we instead map ``Z`` to a matrix that is
"identity on the common mode, orthogonal on the difference modes":

    H_res = q0 q0^T + Q_perp R Q_perp^T

with
    q0 = (1/sqrt(n)) * 1,      Q_perp^T Q_perp = I,     Q_perp^T q0 = 0

Construction from the (per batch/token) residual logits ``Z`` (shape [.., n, n]):
    S = 0.5 * (Z - Z^T)                      # antisymmetric
    A = cayley_scale * (Q_perp^T S Q_perp)   # [.., n-1, n-1], still antisymmetric
    R = (I - A)(I + A)^{-1}                   # Cayley transform -> orthogonal
      = torch.linalg.solve(I + A, I - A)      # (I+-A) are polynomials in A -> commute
    H_res = P0 + Q_perp R Q_perp^T            # P0 = q0 q0^T (all entries 1/n)

No Sinkhorn / softmax / clamp / non-negativity is applied to the new ``H_res``;
negative entries are allowed but the matrix satisfies (up to numerical error)::

    H_res^T @ H_res ~= I
    H_res @ ones    ~= ones
    ones^T @ H_res  ~= ones^T

``q0``, ``P0`` and ``Q_perp`` are non-trainable buffers whose sizes are derived
from ``num_residual_streams`` (never hard-coded).  ``Q_perp`` is a deterministic
Helmert orthonormal basis of the mean-zero (difference) subspace.

The small (n-1)x(n-1) Cayley solve is done in float32 and cast back to the input
dtype to avoid fp16/bf16 numerical issues.

This file does NOT modify mhc.py / mhc_lite.py / mhc_embedding.py, and does not
change their default behaviour.  Setting ``disable_orthogonal_residual=True``
falls back to the exact original mHC Sinkhorn path.
"""

import math

import torch
from torch import nn, cat
from einops import rearrange, repeat, einsum

from .mhc import (
    ManifoldConstrainedHyperConnections,
    Residual,
    sinkhorn_knopps,
    get_expand_reduce_stream_functions,
    default,
)


# ------------------------------------------------------------------ helpers


def make_helmert_qperp(num_streams: int, dtype=torch.float32) -> torch.Tensor:
    """Deterministic Helmert orthonormal basis of the mean-zero subspace.

    Returns Q_perp of shape [num_streams, num_streams - 1] with
        Q_perp^T @ Q_perp == I_{n-1}   and   Q_perp^T @ ones == 0.

    Column k-1 (k = 1..n-1):  entries 1/sqrt(k(k+1)) on the first k positions,
    -k/sqrt(k(k+1)) on position k, 0 afterwards.
    """
    n = num_streams
    qp = torch.zeros(n, n - 1, dtype=dtype)
    for k in range(1, n):
        val = 1.0 / math.sqrt(k * (k + 1))
        qp[:k, k - 1] = val
        qp[k, k - 1] = -k * val
    return qp


def cayley_orthogonal(A: torch.Tensor) -> torch.Tensor:
    """R = (I - A)(I + A)^{-1} via a float32 linear solve (no explicit inverse).

    ``A`` is expected to be antisymmetric of shape [.., m, m]. Computation is done
    in float32 and cast back to ``A.dtype``.
    """
    orig_dtype = A.dtype
    Af = A.float()
    m = Af.shape[-1]
    eye = torch.eye(m, device=Af.device, dtype=Af.dtype)
    # (I+A) and (I-A) are polynomials in A -> commute, so
    # (I-A)(I+A)^{-1} == (I+A)^{-1}(I-A) == solve(I+A, I-A)
    R = torch.linalg.solve(eye + Af, eye - Af)
    return R.to(orig_dtype)


def orthogonal_residual_matrix(
    Z: torch.Tensor,
    q_perp: torch.Tensor,
    P0: torch.Tensor,
    cayley_scale=1.0,
) -> torch.Tensor:
    """Build H_res = P0 + Q_perp R Q_perp^T from residual logits Z.

    Z      : [.., n, n]   (dynamic residual logits)
    q_perp : [n, n-1]
    P0     : [n, n]       (== q0 q0^T, all entries 1/n)
    returns: [.., n, n]   in Z.dtype
    """
    orig_dtype = Z.dtype
    Zf = Z.float()
    qp = q_perp.float()
    p0 = P0.float()

    S = 0.5 * (Zf - Zf.transpose(-1, -2))                    # antisymmetric [.., n, n]
    A = qp.transpose(-1, -2) @ S @ qp                        # [.., n-1, n-1]
    if not isinstance(cayley_scale, (int, float)):
        cayley_scale = cayley_scale.float()
    A = cayley_scale * A

    R = cayley_orthogonal(A)                                 # [.., n-1, n-1] orthogonal
    H = p0 + qp @ R @ qp.transpose(-1, -2)                   # [.., n, n]
    return H.to(orig_dtype)


# --------------------------------------------- debug / analysis helpers


def _eye_like(H):
    return torch.eye(H.shape[-1], device=H.device, dtype=H.dtype)


def distance_to_identity(H):
    n = H.shape[-1]
    return (torch.linalg.matrix_norm(H - _eye_like(H)) / math.sqrt(n)).mean()


def orthogonal_error(H):
    return torch.linalg.matrix_norm(H.transpose(-1, -2) @ H - _eye_like(H)).mean()


def row_sum_error(H):
    ones = torch.ones(H.shape[-1], device=H.device, dtype=H.dtype)
    return (H @ ones - ones).abs().mean()


def column_sum_error(H):
    ones = torch.ones(H.shape[-1], device=H.device, dtype=H.dtype)
    return (ones @ H - ones).abs().mean()


# ------------------------------------------------------------------ main class


class ManifoldConstrainedHyperConnectionsOrthogonalDiff(ManifoldConstrainedHyperConnections):
    """Full mHC with an orthogonal (common-mode identity / difference-mode
    orthogonal) parameterization of ``H_res`` in place of Sinkhorn."""

    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        cayley_scale: float = 1.0,
        disable_orthogonal_residual: bool = False,
        **kwargs,
    ):
        super().__init__(num_residual_streams, dim=dim, **kwargs)

        n = num_residual_streams
        self.disable_orthogonal_residual = disable_orthogonal_residual

        # learnable scale on the antisymmetric generator (must receive gradients)
        self.cayley_scale = nn.Parameter(torch.tensor(float(cayley_scale)))

        # fixed (non-trainable) geometry buffers, sizes derived from num_streams
        q0 = torch.ones(n, dtype=torch.float32) / math.sqrt(n)          # [n]
        P0 = q0[:, None] @ q0[None, :]                                  # [n, n] == 1/n
        Q_perp = make_helmert_qperp(n, dtype=torch.float32)             # [n, n-1]
        self.register_buffer("q0", q0, persistent=False)
        self.register_buffer("P0", P0, persistent=False)
        self.register_buffer("Q_perp", Q_perp, persistent=False)

        # optional H_res capture (does not affect forward return values)
        self._capture_H_res = False
        self._last_H_res = None

    # ----- the only changed piece of the mHC width connection ------------

    def _make_residual_matrix(self, alpha_residual):
        """alpha_residual are the residual logits Z in layout '... f g s t'
        with the last two dims being the (n x n) stream matrix.

        Returns the mixing matrix in the same layout.
        """
        if self.disable_orthogonal_residual:
            # exact original mHC path
            return sinkhorn_knopps(alpha_residual, self.sinkhorn_iters)
        return orthogonal_residual_matrix(
            alpha_residual, self.Q_perp, self.P0, self.cayley_scale
        )

    def width_connection(self, residuals):
        # NOTE: copied verbatim from ManifoldConstrainedHyperConnections.width_connection,
        # with ONLY the Sinkhorn block swapped for self._make_residual_matrix(...).
        streams = self.num_residual_streams

        maybe_transformed_residuals = self.residual_transform(residuals)  # noqa: F841

        if self.channel_first:
            residuals = rearrange(residuals, 'b d ... -> b ... d')

        residuals = self.split_fracs(residuals)
        residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s=streams)

        normed = rearrange(residuals, 'b ... s d -> b ... (s d)', s=streams)
        normed = self.norm(normed)

        wc_weight = normed @ self.dynamic_alpha_fn
        wc_weight = rearrange(wc_weight, '... (s t) -> ... s t', s=streams)

        pre_branch_scale = repeat(self.pre_branch_scale, '1 -> v', v=self.num_input_views * self.num_fracs)
        residual_scale = repeat(self.residual_scale, '1 -> s', s=self.num_fracs * streams)
        alpha_scale = cat((pre_branch_scale, residual_scale))

        dynamic_alpha = wc_weight * alpha_scale
        static_alpha = rearrange(self.static_alpha, '(f s) t -> f s t', s=streams)
        alpha = dynamic_alpha + static_alpha

        alpha = self.split_fracs(alpha)

        alpha_pre, alpha_residual = alpha[..., :self.num_input_views], alpha[..., self.num_input_views:]

        alpha_pre = alpha_pre.sigmoid()

        # ---- H_res parameterization (the ONLY change vs original mHC) ----
        alpha_residual = rearrange(alpha_residual, '... f s g t -> ... f g s t')
        alpha_residual = self._make_residual_matrix(alpha_residual)
        if self._capture_H_res:
            self._last_H_res = alpha_residual
        alpha_residual = rearrange(alpha_residual, '... f g s t -> ... f s g t')
        # ------------------------------------------------------------------

        alpha = cat((alpha_pre, alpha_residual), dim=-1)

        beta = None
        if self.add_branch_out_to_residual:
            dc_weight = normed @ self.dynamic_beta_fn
            dc_weight = rearrange(dc_weight, '... (s f) -> ... s f', s=streams)
            dynamic_beta = dc_weight * self.h_post_scale
            static_beta = rearrange(self.static_beta, '... (s f) -> ... s f', s=streams)
            beta = dynamic_beta + static_beta
            beta = beta.sigmoid() * 2

        mix_h = einsum(alpha, residuals, '... f1 s f2 t, ... f1 s d -> ... f2 t d')

        if self.num_input_views == 1:
            branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]
        else:
            branch_input, residuals = mix_h[..., :self.num_input_views, :], mix_h[..., self.num_input_views:, :]
            branch_input = rearrange(branch_input, 'b ... v d -> v b ... d')

        if self.channel_first:
            branch_input = rearrange(branch_input, 'b ... d -> b d ...')

        branch_input = self.merge_fracs(branch_input)

        residuals = rearrange(residuals, 'b ... s d -> (b s) ... d')
        if self.channel_first:
            residuals = rearrange(residuals, 'b ... d -> b d ...')
        residuals = self.merge_fracs(residuals)
        return branch_input, residuals, dict(beta=beta)

    # ----- debug interface: fetch the actually-generated H_res -----------

    @torch.no_grad()
    def get_H_res(self, residuals):
        """Return the H_res matrix actually generated for ``residuals`` without
        changing the default forward return values.

        For num_fracs == 1 the returned shape is ``[batch, ..., n, n]``.
        """
        prev = self._capture_H_res
        self._capture_H_res = True
        try:
            self.width_connection(residuals)
        finally:
            self._capture_H_res = prev
        H = self._last_H_res  # layout '... f g s t'
        if self.num_fracs == 1:
            H = H[..., 0, 0, :, :]
        return H


# convenience factory mirroring mhc.get_init_and_expand_reduce_stream_functions


def get_init_and_expand_reduce_stream_functions(
    num_streams,
    num_fracs=1,
    dim=None,
    add_stream_embed=False,
    disable=None,
    sinkhorn_iters=20,
    **kwargs,
):
    from functools import partial

    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = (
        ManifoldConstrainedHyperConnectionsOrthogonalDiff if not disable else Residual
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


# short alias
MHCOrthogonalDiff = ManifoldConstrainedHyperConnectionsOrthogonalDiff
