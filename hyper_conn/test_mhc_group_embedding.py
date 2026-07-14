"""Minimal tests for mHC group-wise (per channel-group) H_pre / H_post.

Run from the repo root (mhc-lite) with:
    python -m hyper_conn.test_mhc_group_embedding
"""

import torch
from torch import nn
from einops import repeat, rearrange, einsum

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
    out = m(x)
    assert out.shape == x.shape
    print("[ok] num_streams=4, dim=64 runs")


def test_runs_streams8_dim128():
    b, seq, d, s = 2, 4, 128, 8
    torch.manual_seed(0)
    m = MHCGroup(s, dim=d, branch=nn.Linear(d, d))
    x = _expand(torch.randn(b, seq, d), s)
    out = m(x)
    assert out.shape == x.shape
    print("[ok] num_streams=8, dim=128 runs")


def test_output_shape_matches_original_mhc():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(0)
    m = MHCGroup(s, dim=d, branch=nn.Linear(d, d))
    torch.manual_seed(0)
    ref = ManifoldConstrainedHyperConnections(s, dim=d, branch=nn.Linear(d, d))
    x = _expand(torch.randn(b, seq, d), s)
    assert m(x).shape == ref(x).shape
    print("[ok] output shape matches original mHC")


def test_disable_matches_original_mhc():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(1)
    ref = ManifoldConstrainedHyperConnections(s, dim=d, branch=nn.Linear(d, d)).eval()
    var = MHCGroup(s, dim=d, branch=nn.Linear(d, d), disable_group_embedding=True).eval()
    missing = var.load_state_dict(ref.state_dict(), strict=False)
    assert set(missing.missing_keys) <= {
        "group_pre_weight", "group_post_weight", "group_pre_bias", "group_post_bias"
    }, f"unexpected missing keys: {missing.missing_keys}"
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        assert torch.allclose(var(x), ref(x), atol=1e-6), \
            "disable_group_embedding=True does not match original mHC"
    print("[ok] disable_group_embedding=True reproduces original mHC exactly")


def test_gate_shapes_groups_equal_streams():
    b, seq, d, s = 2, 4, 64, 4  # groups defaults to streams
    torch.manual_seed(2)
    m = MHCGroup(s, dim=d, branch=nn.Linear(d, d))
    x = _expand(torch.randn(b, seq, d), s)
    H_pre_grp, H_post_grp = m.get_group_gates(x)
    # [batch, ..., num_fracs, groups, streams]
    assert H_pre_grp.shape == (b, seq, 1, s, s), H_pre_grp.shape
    assert H_post_grp.shape == (b, seq, 1, s, s), H_post_grp.shape
    print("[ok] H_pre_grp / H_post_grp shape == [b, ..., num_fracs, groups, streams]")


def test_parameter_count_is_n2C_not_n3C():
    for d, s in [(64, 4), (128, 8)]:
        m = MHCGroup(s, dim=d, branch=nn.Linear(d, d))
        eff = m.effective_dim
        expected = s * s * eff  # n^2 * C  (NOT n^3 * C)
        assert m.group_pre_weight.numel() == expected, \
            f"group_pre_weight {m.group_pre_weight.numel()} != {expected}"
        assert m.group_post_weight.numel() == expected, \
            f"group_post_weight {m.group_post_weight.numel()} != {expected}"
        # sanity: dense version would be s*eff*(s*s) = n^3 C, which is s times bigger
        assert expected * s == s * eff * (s * s)
    print("[ok] group generator param count == n^2 C (not n^3 C)")


def test_group_read_einsum_matches_explicit_sum():
    b, f, q, s, gd = 2, 1, 4, 4, 16
    H_pre = torch.randn(b, f, q, s)
    X = torch.randn(b, f, s, q, gd)
    u = einsum(H_pre, X, "b f q s, b f s q d -> b f q d")
    # explicit: u[b,f,q,d] = sum_s H_pre[b,f,q,s] * X[b,f,s,q,d]
    Xr = X.permute(0, 1, 3, 2, 4)                     # [b, f, q, s, gd]
    manual = (H_pre.unsqueeze(-1) * Xr).sum(dim=3)    # sum over s
    assert torch.allclose(u, manual, atol=1e-5)
    print("[ok] group-wise read einsum matches explicit sum")


def test_group_write_einsum_matches_explicit():
    b, f, q, s, gd = 2, 1, 4, 4, 16
    H_post = torch.randn(b, f, q, s)
    h = torch.randn(b, f, q, gd)
    out = einsum(H_post, h, "b f q s, b f q d -> b f s q d")
    # explicit: out[b,f,s,q,d] = H_post[b,f,q,s] * h[b,f,q,d]
    manual = H_post.permute(0, 1, 3, 2).unsqueeze(-1) * h.unsqueeze(2)  # [b,f,s,q,gd]
    assert torch.allclose(out, manual, atol=1e-5)
    print("[ok] group-wise write einsum matches explicit product")


def test_gradients_flow_to_group_generators():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(3)
    m = MHCGroup(s, dim=d, branch=nn.Linear(d, d))
    m.group_pre_weight.data.normal_(std=0.02)   # activate dynamic path
    m.group_post_weight.data.normal_(std=0.02)
    x = _expand(torch.randn(b, seq, d), s)
    m(x).sum().backward()
    for name, p in [
        ("group_pre_weight", m.group_pre_weight),
        ("group_post_weight", m.group_post_weight),
        ("group_pre_bias", m.group_pre_bias),
        ("group_post_bias", m.group_post_bias),
    ]:
        assert p.grad is not None, f"{name} has no gradient"
        assert p.grad.abs().sum() > 0, f"{name} grad is all zero"
    print("[ok] group_{pre,post}_{weight,bias} all receive non-zero gradients")


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
    test_gate_shapes_groups_equal_streams()
    test_parameter_count_is_n2C_not_n3C()
    test_group_read_einsum_matches_explicit_sum()
    test_group_write_einsum_matches_explicit()
    test_gradients_flow_to_group_generators()
    test_indivisible_raises()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
