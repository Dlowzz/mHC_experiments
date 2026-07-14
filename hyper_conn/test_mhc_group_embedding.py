"""Minimal tests for mHC group-wise H_pre read (H_post stays original mHC beta).

Run from the repo root (mhc-lite) with:
    python -m hyper_conn.test_mhc_group_embedding
"""

import torch
from torch import nn
from einops import repeat, einsum

from .mhc import ManifoldConstrainedHyperConnections
from .mhc_group_embedding import (
    ManifoldConstrainedHyperConnectionsGroupEmbedding as MHCGroup,
)


def _expand(x, s):
    return repeat(x, "b n d -> (b s) n d", s=s)


def test_runs_streams4_dim64():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(0)
    m = MHCGroup(s, dim=d, branch=nn.Linear(d, d))
    x = _expand(torch.randn(b, seq, d), s)
    assert m(x).shape == x.shape
    print("[ok] num_streams=4, dim=64 runs")


def test_runs_streams8_dim128():
    b, seq, d, s = 2, 4, 128, 8
    torch.manual_seed(0)
    m = MHCGroup(s, dim=d, branch=nn.Linear(d, d))
    x = _expand(torch.randn(b, seq, d), s)
    assert m(x).shape == x.shape
    print("[ok] num_streams=8, dim=128 runs")


def test_output_shape_matches_original_mhc():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(0)
    m = MHCGroup(s, dim=d, branch=nn.Linear(d, d))
    torch.manual_seed(0)
    ref = ManifoldConstrainedHyperConnections(s, dim=d, branch=nn.Linear(d, d))
    x = _expand(torch.randn(b, seq, d), s)
    assert m(x).shape == ref(x).shape
    print("[ok] output shape (incl. depth_connection) matches original mHC")


def test_disable_matches_original_mhc():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(1)
    ref = ManifoldConstrainedHyperConnections(s, dim=d, branch=nn.Linear(d, d)).eval()
    var = MHCGroup(s, dim=d, branch=nn.Linear(d, d), disable_group_embedding=True).eval()
    missing = var.load_state_dict(ref.state_dict(), strict=False)
    assert set(missing.missing_keys) <= {"group_pre_weight", "group_pre_bias"}, \
        f"unexpected missing keys: {missing.missing_keys}"
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        assert torch.allclose(var(x), ref(x), atol=1e-6), \
            "disable_group_embedding=True does not match original mHC"
    print("[ok] disable_group_embedding=True reproduces original mHC exactly")


def test_hpre_gate_shape_groups_equal_streams():
    b, seq, d, s = 2, 4, 64, 4  # groups defaults to streams
    torch.manual_seed(2)
    m = MHCGroup(s, dim=d, branch=nn.Linear(d, d))
    x = _expand(torch.randn(b, seq, d), s)
    H_pre_grp = m.get_group_gates(x)
    # [batch, ..., num_fracs, groups, streams]
    assert H_pre_grp.shape == (b, seq, 1, s, s), H_pre_grp.shape
    print("[ok] H_pre_grp shape == [b, ..., num_fracs, groups, streams]")


def test_no_group_post_and_beta_kwargs():
    b, seq, d, s = 2, 4, 64, 4
    m = MHCGroup(s, dim=d, branch=nn.Linear(d, d))
    # group-wise H_post must be gone
    assert not hasattr(m, "group_post_weight"), "group_post_weight should not exist"
    assert not hasattr(m, "group_post_bias"), "group_post_bias should not exist"
    assert not hasattr(m, "_last_H_post_grp"), "_last_H_post_grp should not exist"
    # original mHC beta generators must be kept
    assert hasattr(m, "static_beta") and hasattr(m, "dynamic_beta_fn") and hasattr(m, "h_post_scale")
    # width_connection returns beta, not H_post_grp
    x = _expand(torch.randn(b, seq, d), s)
    _, _, kwargs = m.width_connection(x)
    assert "beta" in kwargs and "H_post_grp" not in kwargs and "group_embedding" not in kwargs, kwargs.keys()
    assert kwargs["beta"] is not None
    print("[ok] no group-wise H_post; width_connection returns dict(beta=...) only")


def test_parameter_count_is_n2C():
    for d, s in [(64, 4), (128, 8)]:
        m = MHCGroup(s, dim=d, branch=nn.Linear(d, d))
        eff = m.effective_dim
        expected = s * s * eff  # n^2 * C   (single generator, NOT 2 n^2 C, NOT n^3 C)
        assert m.group_pre_weight.numel() == expected, \
            f"group_pre_weight {m.group_pre_weight.numel()} != {expected}"
        assert not hasattr(m, "group_post_weight")
    print("[ok] added read-generator param count == n^2 C (group_pre_weight only)")


def test_group_read_einsum_matches_explicit_sum():
    b, f, q, s, gd = 2, 1, 4, 4, 16
    H_pre = torch.randn(b, f, q, s)
    X = torch.randn(b, f, s, q, gd)
    u = einsum(H_pre, X, "b f q s, b f s q d -> b f q d")
    Xr = X.permute(0, 1, 3, 2, 4)                     # [b, f, q, s, gd]
    manual = (H_pre.unsqueeze(-1) * Xr).sum(dim=3)    # sum over s
    assert torch.allclose(u, manual, atol=1e-5)
    print("[ok] group-wise read einsum matches explicit sum")


def test_gradients_flow_to_read_gen_and_original_beta():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(3)
    m = MHCGroup(s, dim=d, branch=nn.Linear(d, d))
    # activate dynamic paths so gradients are clearly non-zero
    m.group_pre_weight.data.normal_(std=0.02)
    m.dynamic_beta_fn.data.normal_(std=0.02)
    x = _expand(torch.randn(b, seq, d), s)
    m(x).sum().backward()

    # group-wise read generator
    for name, p in [("group_pre_weight", m.group_pre_weight),
                    ("group_pre_bias", m.group_pre_bias)]:
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} no grad"
    # original mHC beta write-back generators
    for name, p in [("dynamic_beta_fn", m.dynamic_beta_fn),
                    ("static_beta", m.static_beta),
                    ("h_post_scale", m.h_post_scale)]:
        assert p.grad is not None, f"{name} grad is None"
        assert p.grad.abs().sum() > 0, f"{name} grad is all zero"
    print("[ok] read generator AND original beta (static_beta/dynamic_beta_fn/h_post_scale) get gradients")


def test_indivisible_raises():
    try:
        MHCGroup(4, dim=64, branch=nn.Linear(64, 64), group_embedding_groups=3)
    except AssertionError as e:
        assert "divisible" in str(e)
        print("[ok] effective_dim % groups != 0 raises a clear assertion error")
        return
    raise AssertionError("expected AssertionError for indivisible groups")


def main():
    test_runs_streams4_dim64()
    test_runs_streams8_dim128()
    test_output_shape_matches_original_mhc()
    test_disable_matches_original_mhc()
    test_hpre_gate_shape_groups_equal_streams()
    test_no_group_post_and_beta_kwargs()
    test_parameter_count_is_n2C()
    test_group_read_einsum_matches_explicit_sum()
    test_gradients_flow_to_read_gen_and_original_beta()
    test_indivisible_raises()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
