"""Minimal tests for the mHC orthogonal-difference H_res parameterization.

Run from the repo root (mhc-lite) with:
    python -m hyper_conn.test_mhc_orthogonal_diff
"""

import torch
from torch import nn
from einops import repeat

from .mhc import ManifoldConstrainedHyperConnections
from .mhc_lite import MHCLite
from .mhc_orthogonal_diff import (
    ManifoldConstrainedHyperConnectionsOrthogonalDiff as MHCOrtho,
    make_helmert_qperp,
    orthogonal_residual_matrix,
    distance_to_identity,
    orthogonal_error,
    row_sum_error,
    column_sum_error,
)

ATOL = 1e-4


def _expand(x, s):
    return repeat(x, "b n d -> (b s) n d", s=s)


def test_runs_and_shape_for_various_streams():
    b, seq, d = 2, 4, 16
    for s in (4, 8):
        torch.manual_seed(0)
        m = MHCOrtho(s, dim=d, branch=nn.Linear(d, d))
        torch.manual_seed(0)
        ref = ManifoldConstrainedHyperConnections(s, dim=d, branch=nn.Linear(d, d))
        x = _expand(torch.randn(b, seq, d), s)
        out = m(x)
        out_ref = ref(x)
        assert out.shape == x.shape
        assert out.shape == out_ref.shape
        # buffer sizes are derived from num_streams, not hard-coded
        assert m.Q_perp.shape == (s, s - 1)
        assert m.P0.shape == (s, s)
        assert m.q0.shape == (s,)
    print("[ok] runs for num_streams in {4, 8}; output shape matches original mHC")


def test_qperp_properties():
    for s in (4, 8):
        qp = make_helmert_qperp(s)
        eye = torch.eye(s - 1)
        assert torch.allclose(qp.T @ qp, eye, atol=1e-6), "Q_perp not orthonormal"
        assert torch.allclose(qp.T @ torch.ones(s), torch.zeros(s - 1), atol=1e-6), \
            "Q_perp not orthogonal to ones"
    print("[ok] Q_perp^T Q_perp == I and Q_perp^T ones == 0")


def test_hres_row_col_sums_and_orthogonality():
    b, seq, d, s = 2, 4, 16, 4
    torch.manual_seed(1)
    m = MHCOrtho(s, dim=d, branch=nn.Linear(d, d))
    x = _expand(torch.randn(b, seq, d), s)
    H = m.get_H_res(x)  # [b, seq, s, s]
    assert H.shape == (b, seq, s, s)
    assert row_sum_error(H) < ATOL, f"row sum error {float(row_sum_error(H))}"
    assert column_sum_error(H) < ATOL, f"col sum error {float(column_sum_error(H))}"
    assert orthogonal_error(H) < ATOL, f"orthogonal error {float(orthogonal_error(H))}"
    print("[ok] H_res row/col sums ~= 1 and H_res^T H_res ~= I")


def test_norm_preservation_on_mean_zero_input():
    s = 8
    torch.manual_seed(2)
    qp = make_helmert_qperp(s)
    q0 = torch.ones(s) / (s ** 0.5)
    P0 = q0[:, None] @ q0[None, :]
    Z = torch.randn(s, s)  # arbitrary residual logits
    H = orthogonal_residual_matrix(Z, qp, P0, cayley_scale=1.0)  # [s, s]

    for _ in range(5):
        v = torch.randn(s)
        v_perp = v - v.mean()  # mean-zero (difference) mode, orthogonal to ones
        out = H @ v_perp
        assert torch.allclose(out.norm(), v_perp.norm(), atol=1e-4), \
            f"||H x_perp|| {float(out.norm())} != ||x_perp|| {float(v_perp.norm())}"
    print("[ok] ||H_res @ x_perp|| == ||x_perp|| for mean-zero inputs")


def test_zero_logits_give_identity():
    for s in (4, 8):
        qp = make_helmert_qperp(s)
        q0 = torch.ones(s) / (s ** 0.5)
        P0 = q0[:, None] @ q0[None, :]
        Z = torch.zeros(3, s, s)
        H = orthogonal_residual_matrix(Z, qp, P0, cayley_scale=1.0)
        eye = torch.eye(s).expand_as(H)
        assert torch.allclose(H, eye, atol=1e-6), "zero logits did not give identity"
    print("[ok] zero residual logits -> H_res == I")


def test_gradients_flow_to_logits_generator_and_scale():
    b, seq, d, s = 2, 4, 16, 4
    torch.manual_seed(3)
    m = MHCOrtho(s, dim=d, branch=nn.Linear(d, d))
    # activate the dynamic residual-logits path so the antisymmetric part (and thus
    # cayley_scale) is non-trivial (dynamic_alpha_fn is zero-initialised by design,
    # which would make the antisymmetric generator exactly zero at init).
    m.dynamic_alpha_fn.data.normal_(std=0.02)
    x = _expand(torch.randn(b, seq, d), s)
    out = m(x)
    out.sum().backward()

    assert m.dynamic_alpha_fn.grad is not None and m.dynamic_alpha_fn.grad.abs().sum() > 0, \
        "residual logits generator (dynamic_alpha_fn) got no gradient"
    assert m.cayley_scale.grad is not None and m.cayley_scale.grad.abs().item() > 0, \
        "cayley_scale got no gradient"
    print("[ok] residual-logits generator and cayley_scale receive non-zero gradients")


def test_disable_matches_original_mhc():
    b, seq, d, s = 2, 4, 16, 4
    torch.manual_seed(4)
    ref = ManifoldConstrainedHyperConnections(s, dim=d, branch=nn.Linear(d, d)).eval()
    var = MHCOrtho(s, dim=d, branch=nn.Linear(d, d),
                   disable_orthogonal_residual=True).eval()
    missing = var.load_state_dict(ref.state_dict(), strict=False)
    assert set(missing.missing_keys) <= {"cayley_scale", "q0", "P0", "Q_perp"}, \
        f"unexpected missing keys: {missing.missing_keys}"
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        assert torch.allclose(var(x), ref(x), atol=1e-6), \
            "disable_orthogonal_residual=True does not match original mHC"
    print("[ok] disable_orthogonal_residual=True reproduces original mHC exactly")


def test_original_mhc_and_mhc_lite_unaffected():
    b, seq, d, s = 2, 4, 16, 4
    torch.manual_seed(5)
    mhc = ManifoldConstrainedHyperConnections(s, dim=d, branch=nn.Linear(d, d))
    lite = MHCLite(s, dim=d, branch=nn.Linear(d, d))
    x = _expand(torch.randn(b, seq, d), s)
    out_mhc = mhc(x)
    out_lite = lite(x)
    assert out_mhc.shape == x.shape and out_lite.shape == x.shape
    print("[ok] original mHC and MHC-Lite still run and are unaffected")


def main():
    test_runs_and_shape_for_various_streams()
    test_qperp_properties()
    test_hres_row_col_sums_and_orthogonality()
    test_norm_preservation_on_mean_zero_input()
    test_zero_logits_give_identity()
    test_gradients_flow_to_logits_generator_and_scale()
    test_disable_matches_original_mhc()
    test_original_mhc_and_mhc_lite_unaffected()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
