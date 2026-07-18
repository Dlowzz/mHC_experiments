"""Minimal tests for mHC group-wise H_pre read + per-stream LoRA write-back.

Run from the repo root (mhc-lite) with:
    python -m hyper_conn.test_mhc_group_lora
"""

import torch
from torch import nn
from einops import repeat, einsum

from .mhc import ManifoldConstrainedHyperConnections
from .mhc_group_embedding import ManifoldConstrainedHyperConnectionsGroupEmbedding as MHCGroup
from .mhc_group_lora import ManifoldConstrainedHyperConnectionsGroupLoRA as MHCGroupLoRA


def _expand(x, s):
    return repeat(x, "b n d -> (b s) n d", s=s)


def test_runs_streams4_dim64():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(0)
    m = MHCGroupLoRA(s, dim=d, branch=nn.Linear(d, d))
    x = _expand(torch.randn(b, seq, d), s)
    assert m(x).shape == x.shape
    print("[ok] num_streams=4, dim=64 runs")


def test_runs_streams8_dim128():
    b, seq, d, s = 2, 4, 128, 8
    torch.manual_seed(0)
    m = MHCGroupLoRA(s, dim=d, branch=nn.Linear(d, d), lora_rank=8)
    x = _expand(torch.randn(b, seq, d), s)
    assert m(x).shape == x.shape
    print("[ok] num_streams=8, dim=128 runs")


def test_output_shape_matches_original_mhc():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(0)
    m = MHCGroupLoRA(s, dim=d, branch=nn.Linear(d, d))
    torch.manual_seed(0)
    ref = ManifoldConstrainedHyperConnections(s, dim=d, branch=nn.Linear(d, d))
    x = _expand(torch.randn(b, seq, d), s)
    assert m(x).shape == ref(x).shape
    print("[ok] output shape matches original mHC")


def test_disable_both_matches_original_mhc():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(1)
    ref = ManifoldConstrainedHyperConnections(s, dim=d, branch=nn.Linear(d, d)).eval()
    var = MHCGroupLoRA(s, dim=d, branch=nn.Linear(d, d),
                       disable_group_embedding=True, disable_lora_branch=True).eval()
    missing = var.load_state_dict(ref.state_dict(), strict=False)
    assert set(missing.missing_keys) <= {
        "group_pre_weight", "group_pre_bias", "stream_down_weight", "stream_up_weight", "lora_scale"
    }, f"unexpected missing keys: {missing.missing_keys}"
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        assert torch.allclose(var(x), ref(x), atol=1e-6), \
            "disabling both branches does not match original mHC"
    print("[ok] disable_group_embedding + disable_lora_branch reproduces original mHC")


def test_hpre_gate_shape():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(2)
    m = MHCGroupLoRA(s, dim=d, branch=nn.Linear(d, d))
    x = _expand(torch.randn(b, seq, d), s)
    H_pre_grp = m.get_group_gates(x)
    assert H_pre_grp.shape == (b, seq, 1, s, s), H_pre_grp.shape
    print("[ok] H_pre_grp shape == [b, ..., num_fracs, groups, streams]")


def test_param_shapes():
    for d, s, r in [(64, 4, 8), (128, 8, 8)]:
        m = MHCGroupLoRA(s, dim=d, branch=nn.Linear(d, d), lora_rank=r)
        eff = m.effective_dim
        assert m.group_pre_weight.numel() == s * s * eff, "group_pre_weight != n^2 C"
        assert m.stream_down_weight.shape == (s, eff, r), m.stream_down_weight.shape
        assert m.stream_up_weight.shape == (s, r, eff), m.stream_up_weight.shape
        assert not hasattr(m, "group_post_weight")
    print("[ok] group_pre_weight == n^2 C; A_s [s,C,r]; B_s [s,r,C]")


def test_group_read_einsum_matches_explicit():
    b, f, q, s, gd = 2, 1, 4, 4, 16
    H_pre = torch.randn(b, f, q, s)
    X = torch.randn(b, f, s, q, gd)
    u = einsum(H_pre, X, "b f q s, b f s q d -> b f q d")
    Xr = X.permute(0, 1, 3, 2, 4)
    manual = (H_pre.unsqueeze(-1) * Xr).sum(dim=3)
    assert torch.allclose(u, manual, atol=1e-5)
    print("[ok] group-wise read einsum matches explicit sum")


def test_lora_is_additive_and_off_at_init():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(4)
    m = MHCGroupLoRA(s, dim=d, branch=nn.Linear(d, d)).eval()
    m_lora_off = MHCGroupLoRA(s, dim=d, branch=nn.Linear(d, d), disable_lora_branch=True).eval()
    m_lora_off.load_state_dict(m.state_dict())
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        # B_s is zero-initialised -> LoRA delta == 0 -> identical to lora-off
        assert torch.allclose(m(x), m_lora_off(x), atol=1e-6), "LoRA not zero at init"
        # activate B_s -> outputs must now differ (LoRA is a real additive term)
        m.stream_up_weight.data.normal_()
        assert not torch.allclose(m(x), m_lora_off(x), atol=1e-5), "LoRA had no effect"
    print("[ok] LoRA write-back is additive and zero at init (B_s=0)")


def test_equivalent_to_group_embedding_when_lora_off():
    # group read + beta (LoRA disabled) must equal the standalone Hpre-only group module
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(5)
    grp = MHCGroup(s, dim=d, branch=nn.Linear(d, d)).eval()
    combo = MHCGroupLoRA(s, dim=d, branch=nn.Linear(d, d), disable_lora_branch=True).eval()
    combo.load_state_dict(grp.state_dict(), strict=False)  # copies shared mHC + group_pre; B_s stays 0
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        assert torch.allclose(combo(x), grp(x), atol=1e-6), \
            "combo(LoRA off) != group-embedding (Hpre-only)"
    print("[ok] LoRA-off combo == standalone group-embedding (Hpre-only)")


def test_gradients_flow_everywhere():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(6)
    m = MHCGroupLoRA(s, dim=d, branch=nn.Linear(d, d))
    # activate dynamic paths for clearly non-zero grads
    m.group_pre_weight.data.normal_(std=0.02)
    m.dynamic_beta_fn.data.normal_(std=0.02)
    m.stream_up_weight.data.normal_(std=0.02)
    x = _expand(torch.randn(b, seq, d), s)
    m(x).sum().backward()
    for name, p in [
        ("group_pre_weight", m.group_pre_weight),
        ("group_pre_bias", m.group_pre_bias),
        ("stream_down_weight (A_s)", m.stream_down_weight),
        ("stream_up_weight (B_s)", m.stream_up_weight),
        ("dynamic_beta_fn", m.dynamic_beta_fn),
        ("static_beta", m.static_beta),
        ("h_post_scale", m.h_post_scale),
    ]:
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} no/zero grad"
    print("[ok] group read gen + LoRA (A_s,B_s) + original beta all get non-zero gradients")


def test_indivisible_raises():
    try:
        MHCGroupLoRA(4, dim=64, branch=nn.Linear(64, 64), group_embedding_groups=3)
    except AssertionError as e:
        assert "divisible" in str(e)
        print("[ok] effective_dim % groups != 0 raises a clear assertion error")
        return
    raise AssertionError("expected AssertionError for indivisible groups")


def main():
    test_runs_streams4_dim64()
    test_runs_streams8_dim128()
    test_output_shape_matches_original_mhc()
    test_disable_both_matches_original_mhc()
    test_hpre_gate_shape()
    test_param_shapes()
    test_group_read_einsum_matches_explicit()
    test_lora_is_additive_and_off_at_init()
    test_equivalent_to_group_embedding_when_lora_off()
    test_gradients_flow_everywhere()
    test_indivisible_raises()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
