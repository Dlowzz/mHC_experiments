"""Minimal tests for mHC-group-LoRA-midnorm (group H_pre read + LoRA rank-dim RMSNorm).

Run from the repo root (mhc-lite) with:
    python -m hyper_conn.test_mhc_group_lora_midnorm
"""

import torch
from torch import nn
from einops import repeat, einsum

from .mhc import ManifoldConstrainedHyperConnections
from .mhc_group_embedding import ManifoldConstrainedHyperConnectionsGroupEmbedding as MHCGroup
from .mhc_group_lora import rmsnorm_lastdim
from .mhc_group_lora_midnorm import ManifoldConstrainedHyperConnectionsGroupLoRAMidNorm as MHCMid


def _expand(x, s):
    return repeat(x, "b n d -> (b s) n d", s=s)


def test_runs_and_shape_matches_mhc():
    for d, s in [(64, 4), (128, 8)]:
        torch.manual_seed(0)
        m = MHCMid(s, dim=d, branch=nn.Linear(d, d))
        torch.manual_seed(0)
        ref = ManifoldConstrainedHyperConnections(s, dim=d, branch=nn.Linear(d, d))
        x = _expand(torch.randn(2, 4, d), s)
        assert m(x).shape == x.shape == ref(x).shape
    print("[ok] runs for streams {4,8}; shape matches original mHC")


def test_disable_both_equals_original_mhc():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(1)
    ref = ManifoldConstrainedHyperConnections(s, dim=d, branch=nn.Linear(d, d)).eval()
    m = MHCMid(s, dim=d, branch=nn.Linear(d, d),
              disable_group_embedding=True, disable_lora_branch=True).eval()
    missing = m.load_state_dict(ref.state_dict(), strict=False)
    assert set(missing.missing_keys) <= {
        "group_pre_weight", "group_pre_bias", "stream_down_weight", "stream_up_weight"
    }, missing.missing_keys
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        err = (m(x) - ref(x)).abs().max().item()
    assert err <= 1e-6, f"disable-both not equal to original mHC (max abs err {err})"
    print(f"[ok] disable group+LoRA == original mHC (max abs err {err:.2e})")


def test_init_equals_group_baseline():
    # at init (B=0) delta=0 -> midnorm == the group_embedding (Hpre-only) baseline
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(2)
    grp = MHCGroup(s, dim=d, branch=nn.Linear(d, d)).eval()
    m = MHCMid(s, dim=d, branch=nn.Linear(d, d)).eval()
    m.load_state_dict(grp.state_dict(), strict=False)  # shared mHC + group_pre; B_s stays 0
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        err = (m(x) - grp(x)).abs().max().item()
    assert err <= 1e-6, f"init != group baseline (max abs err {err})"
    print(f"[ok] init == group_lora baseline (max abs err {err:.2e})")


def test_hpre_gate_shape():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(3)
    m = MHCMid(s, dim=d, branch=nn.Linear(d, d))
    x = _expand(torch.randn(b, seq, d), s)
    H = m.get_group_gates(x)
    assert H.shape == (b, seq, 1, s, s), H.shape
    print("[ok] H_pre_grp shape == [b, ..., num_fracs, groups, streams]")


def test_midnorm_dim_is_rank():
    b, seq, d, s, r = 2, 4, 64, 4, 8
    torch.manual_seed(4)
    m = MHCMid(s, dim=d, branch=nn.Linear(d, d), lora_rank=r)
    h = m.split_fracs(torch.randn(b, seq, d))
    down = einsum(h, m.stream_down_weight, "b ... f d, s d r -> b ... f s r")
    rms = rmsnorm_lastdim(down).pow(2).mean(dim=-1).sqrt()  # over rank dim
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3), "mid RMSNorm not on rank dim"
    # exact: module computes delta = (rmsnorm_{-1}(h@A_s)) @ B_s (norm on rank, before B)
    m.stream_up_weight.data.normal_()
    with torch.no_grad():
        delta_mod = m.compute_lora(h)
        down2 = einsum(h, m.stream_down_weight, "b ... f d, s d r -> b ... f s r")
        delta_manual = einsum(rmsnorm_lastdim(down2), m.stream_up_weight,
                              "b ... f s r, s r e -> b ... f s e")
    assert torch.allclose(delta_mod, delta_manual, atol=1e-6), "mid RMSNorm not applied between A and B"
    print("[ok] mid RMSNorm normalises the rank dim (=r); applied between A_s and B_s")


def test_per_stream_independent_AB():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(5)
    m = MHCMid(s, dim=d, branch=nn.Linear(d, d))
    m.stream_up_weight.data.normal_()
    with torch.no_grad():
        delta = m.compute_lora(m.split_fracs(torch.randn(b, seq, d)))
    diffs = [not torch.allclose(delta[..., i, :], delta[..., j, :])
             for i in range(s) for j in range(i + 1, s)]
    assert all(diffs), "streams share the same LoRA output"
    print("[ok] each stream uses independent A_s/B_s")


def test_lora_off_and_group_off_reverts():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(6)
    # LoRA off -> equals group_embedding (Hpre-only)
    grp = MHCGroup(s, dim=d, branch=nn.Linear(d, d)).eval()
    m = MHCMid(s, dim=d, branch=nn.Linear(d, d), disable_lora_branch=True).eval()
    m.load_state_dict(grp.state_dict(), strict=False)
    m.stream_up_weight.data.normal_(std=5.0)  # no effect when disabled
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        assert torch.allclose(m(x), grp(x), atol=1e-6), "LoRA-off != group baseline"
    print("[ok] LoRA-off reverts to group baseline; group-off path covered by disable-both test")


def test_no_nan_inf_and_grads():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(7)
    m = MHCMid(s, dim=d, branch=nn.Linear(d, d))
    m.group_pre_weight.data.normal_(std=0.02)
    m.dynamic_beta_fn.data.normal_(std=0.02)
    m.stream_up_weight.data.normal_(std=0.5)
    x = _expand(torch.randn(b, seq, d), s)
    out = m(x)
    assert torch.isfinite(out).all()
    out.sum().backward()
    for name, p in [("group_pre_weight", m.group_pre_weight), ("A_s", m.stream_down_weight),
                    ("B_s", m.stream_up_weight), ("dynamic_beta_fn", m.dynamic_beta_fn)]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, f"{name} grad bad"
    print("[ok] forward/backward finite; group/LoRA/beta all get gradients")


def test_state_dict_roundtrip():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(8)
    m = MHCMid(s, dim=d, branch=nn.Linear(d, d)).eval()
    m.stream_up_weight.data.normal_(std=0.3)
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        o1 = m(x)
    m2 = MHCMid(s, dim=d, branch=nn.Linear(d, d)).eval()
    m2.load_state_dict(m.state_dict())
    with torch.no_grad():
        o2 = m2(x)
    assert torch.allclose(o1, o2, atol=1e-6)
    print("[ok] state_dict save/load reproduces output")


def main():
    test_runs_and_shape_matches_mhc()
    test_disable_both_equals_original_mhc()
    test_init_equals_group_baseline()
    test_hpre_gate_shape()
    test_midnorm_dim_is_rank()
    test_per_stream_independent_AB()
    test_lora_off_and_group_off_reverts()
    test_no_nan_inf_and_grads()
    test_state_dict_roundtrip()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
