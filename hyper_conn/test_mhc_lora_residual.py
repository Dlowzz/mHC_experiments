"""Minimal tests for mHC-LoRA-Residual (per-stream LoRA + output RMSNorm).

Run from the repo root (mhc-lite) with:
    python -m hyper_conn.test_mhc_lora_residual
"""

import torch
from torch import nn
from einops import repeat

from .mhc import ManifoldConstrainedHyperConnections
from .mhc_lora_residual import ManifoldConstrainedHyperConnectionsLoRAResidual as MHCLoRA


def _expand(x, s):
    return repeat(x, "b n d -> (b s) n d", s=s)


def test_runs_and_shape():
    for d, s in [(64, 4), (128, 8)]:
        b, seq = 2, 4
        torch.manual_seed(0)
        m = MHCLoRA(s, dim=d, branch=nn.Linear(d, d))
        x = _expand(torch.randn(b, seq, d), s)
        out = m(x)
        assert out.shape == x.shape
    print("[ok] runs for streams {4,8}; output shape preserved")


def test_init_equals_original_mhc():
    # B_s zero-init -> delta = rmsnorm(0) = 0 -> identical to original mHC at init
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(1)
    ref = ManifoldConstrainedHyperConnections(s, dim=d, branch=nn.Linear(d, d)).eval()
    m = MHCLoRA(s, dim=d, branch=nn.Linear(d, d)).eval()
    missing = m.load_state_dict(ref.state_dict(), strict=False)
    assert set(missing.missing_keys) <= {"stream_down_weight", "stream_up_weight"}, missing.missing_keys
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        err = (m(x) - ref(x)).abs().max().item()
    assert err <= 1e-6, f"init not equal to original mHC (max abs err {err})"
    print(f"[ok] init == original mHC (max abs err {err:.2e})")


def test_disable_lora_equals_original_mhc():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(2)
    ref = ManifoldConstrainedHyperConnections(s, dim=d, branch=nn.Linear(d, d)).eval()
    m = MHCLoRA(s, dim=d, branch=nn.Linear(d, d), disable_lora_branch=True).eval()
    m.load_state_dict(ref.state_dict(), strict=False)
    m.stream_up_weight.data.normal_(std=5.0)  # must have NO effect when disabled
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        assert torch.allclose(m(x), ref(x), atol=1e-6), "disable_lora_branch != original mHC"
    print("[ok] disable_lora_branch=True fully reverts to original mHC")


def test_output_rmsnorm_dim_is_hidden():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(3)
    m = MHCLoRA(s, dim=d, branch=nn.Linear(d, d))
    m.stream_up_weight.data.normal_()
    with torch.no_grad():
        delta = m.compute_lora(m.split_fracs(torch.randn(b, seq, d)))  # b seq f s d
        rms = delta.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3), "output RMSNorm not on hidden dim (-1)"
    print("[ok] output RMSNorm normalises the hidden dim (=-1) to unit RMS")


def test_per_stream_independent_AB():
    b, seq, d, s, r = 2, 4, 64, 4, 8
    torch.manual_seed(4)
    m = MHCLoRA(s, dim=d, branch=nn.Linear(d, d), lora_rank=r)
    assert m.stream_down_weight.shape == (s, d, r) and m.stream_up_weight.shape == (s, r, d)
    m.stream_up_weight.data.normal_()  # activate; A_s differ per stream (kaiming)
    with torch.no_grad():
        delta = m.compute_lora(m.split_fracs(torch.randn(b, seq, d)))  # b seq f s d
    diffs = [not torch.allclose(delta[..., i, :], delta[..., j, :])
             for i in range(s) for j in range(i + 1, s)]
    assert all(diffs), "streams share the same LoRA output (A_s/B_s not independent)"
    print("[ok] each stream uses independent A_s/B_s (distinct LoRA outputs)")


def test_no_nan_inf_fwd_bwd():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(5)
    m = MHCLoRA(s, dim=d, branch=nn.Linear(d, d))
    m.stream_up_weight.data.normal_(std=0.5)
    x = _expand(torch.randn(b, seq, d), s)
    out = m(x)
    assert torch.isfinite(out).all()
    out.sum().backward()
    for name, p in [("A_s", m.stream_down_weight), ("B_s", m.stream_up_weight),
                    ("static_beta", m.static_beta), ("dynamic_beta_fn", m.dynamic_beta_fn)]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, f"{name} grad bad"
    print("[ok] forward/backward finite; A_s/B_s/beta all get gradients")


def test_state_dict_roundtrip():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(6)
    m = MHCLoRA(s, dim=d, branch=nn.Linear(d, d)).eval()
    m.stream_up_weight.data.normal_(std=0.3)
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        o1 = m(x)
    m2 = MHCLoRA(s, dim=d, branch=nn.Linear(d, d)).eval()
    m2.load_state_dict(m.state_dict())
    with torch.no_grad():
        o2 = m2(x)
    assert torch.allclose(o1, o2, atol=1e-6)
    print("[ok] state_dict save/load reproduces output")


def main():
    test_runs_and_shape()
    test_init_equals_original_mhc()
    test_disable_lora_equals_original_mhc()
    test_output_rmsnorm_dim_is_hidden()
    test_per_stream_independent_AB()
    test_no_nan_inf_fwd_bwd()
    test_state_dict_roundtrip()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
