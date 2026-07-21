"""Minimal tests for mHC-LoRA-Residual-midnorm (RMSNorm on the LoRA rank dim).

Run from the repo root (mhc-lite) with:
    python -m hyper_conn.test_mhc_lora_residual_midnorm
"""

import torch
from torch import nn
from einops import repeat, einsum, rearrange

from .mhc import ManifoldConstrainedHyperConnections
from .mhc_lora_residual import (
    ManifoldConstrainedHyperConnectionsLoRAResidual as MHCLoRA,
    rmsnorm_lastdim,
)
from .mhc_lora_residual_midnorm import ManifoldConstrainedHyperConnectionsLoRAResidualMidNorm as MHCMid


def _expand(x, s):
    return repeat(x, "b n d -> (b s) n d", s=s)


def test_runs_and_shape():
    for d, s in [(64, 4), (128, 8)]:
        torch.manual_seed(0)
        m = MHCMid(s, dim=d, branch=nn.Linear(d, d))
        x = _expand(torch.randn(2, 4, d), s)
        assert m(x).shape == x.shape
    print("[ok] runs for streams {4,8}; output shape preserved")


def test_init_equals_original_mhc():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(1)
    ref = ManifoldConstrainedHyperConnections(s, dim=d, branch=nn.Linear(d, d)).eval()
    m = MHCMid(s, dim=d, branch=nn.Linear(d, d)).eval()
    missing = m.load_state_dict(ref.state_dict(), strict=False)
    assert set(missing.missing_keys) <= {"stream_down_weight", "stream_up_weight"}, missing.missing_keys
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        err = (m(x) - ref(x)).abs().max().item()
    assert err <= 1e-6, f"init not equal to original mHC (max abs err {err})"
    print(f"[ok] init == original mHC (max abs err {err:.2e})")


def test_midnorm_dim_is_rank():
    # the norm is applied to `down = h@A_s` on its last dim (=rank r)
    b, seq, d, s, r = 2, 4, 64, 4, 8
    torch.manual_seed(2)
    m = MHCMid(s, dim=d, branch=nn.Linear(d, d), lora_rank=r)
    h = m.split_fracs(torch.randn(b, seq, d))
    down = einsum(h, m.stream_down_weight, "b ... f d, s d r -> b ... f s r")  # last dim = r
    downn = rmsnorm_lastdim(down)
    rms = downn.pow(2).mean(dim=-1).sqrt()  # RMS over rank dim
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3), "mid RMSNorm not on rank dim"
    # exact: module computes delta = (rmsnorm_{-1}(h@A_s)) @ B_s  (norm on rank, before B)
    m.stream_up_weight.data.normal_()
    with torch.no_grad():
        xx = m.split_fracs(torch.randn(b, seq, d))
        delta_mod = m.compute_lora(xx)
        down2 = einsum(xx, m.stream_down_weight, "b ... f d, s d r -> b ... f s r")
        delta_manual = einsum(rmsnorm_lastdim(down2), m.stream_up_weight,
                              "b ... f s r, s r e -> b ... f s e")
    assert torch.allclose(delta_mod, delta_manual, atol=1e-6), "mid RMSNorm not applied between A and B"
    print("[ok] mid RMSNorm normalises the rank dim (=r); applied between A_s and B_s")


def test_per_stream_independent_AB():
    b, seq, d, s, r = 2, 4, 64, 4, 8
    torch.manual_seed(3)
    m = MHCMid(s, dim=d, branch=nn.Linear(d, d), lora_rank=r)
    m.stream_up_weight.data.normal_()
    with torch.no_grad():
        delta = m.compute_lora(m.split_fracs(torch.randn(b, seq, d)))
    diffs = [not torch.allclose(delta[..., i, :], delta[..., j, :])
             for i in range(s) for j in range(i + 1, s)]
    assert all(diffs), "streams share the same LoRA output"
    print("[ok] each stream uses independent A_s/B_s")


def test_disable_lora_reverts():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(4)
    ref = ManifoldConstrainedHyperConnections(s, dim=d, branch=nn.Linear(d, d)).eval()
    m = MHCMid(s, dim=d, branch=nn.Linear(d, d), disable_lora_branch=True).eval()
    m.load_state_dict(ref.state_dict(), strict=False)
    m.stream_up_weight.data.normal_(std=5.0)
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        assert torch.allclose(m(x), ref(x), atol=1e-6)
    print("[ok] disable_lora_branch=True fully reverts to original mHC")


def test_no_nan_inf_and_grads():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(5)
    m = MHCMid(s, dim=d, branch=nn.Linear(d, d))
    m.stream_up_weight.data.normal_(std=0.5)
    x = _expand(torch.randn(b, seq, d), s)
    out = m(x)
    assert torch.isfinite(out).all()
    out.sum().backward()
    for name, p in [("A_s", m.stream_down_weight), ("B_s", m.stream_up_weight),
                    ("dynamic_beta_fn", m.dynamic_beta_fn)]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, f"{name} grad bad"
    print("[ok] forward/backward finite; A_s/B_s/beta get gradients")


def test_state_dict_roundtrip():
    b, seq, d, s = 2, 4, 64, 4
    torch.manual_seed(6)
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


def test_inside_beta_formula_consistency():
    # depth write must equal  einsum(h, beta) + lora_lambda * einsum(delta, beta)
    # (LoRA moved INSIDE beta).  depth_connection is exercised directly with
    # fabricated inputs, which also covers num_fracs=2 (base-mHC width_connection
    # RMSNorm is separately mis-sized for num_fracs>1 -- out of scope here).
    b, seq, d, s = 2, 4, 64, 4
    for nf in (1, 2):
        torch.manual_seed(10 + nf)
        m = MHCMid(s, dim=d, num_fracs=nf, branch=nn.Linear(d, d)).eval()
        with torch.no_grad():
            m.stream_up_weight.normal_(std=0.5)   # non-zero delta so the LoRA term matters
            bo = torch.randn(b, seq, d)                     # branch output (pre-split)
            res = torch.randn(b * s, seq, d)                # residual streams
            beta = torch.rand(b, seq, nf, s, nf) * 2.0      # [b, seq, f1, s, f2] in (0,2)
            actual = m.depth_connection(bo, res, beta=beta)
            # manual reconstruction of u_s = beta_s (h + lambda * delta_s)
            h = m.split_fracs(bo)
            main = einsum(h, beta, 'b ... f1 d, b ... f1 s f2 -> b ... f2 s d')
            delta = m.compute_lora(h)
            dwrite = einsum(delta, beta, 'b ... f1 s d, b ... f1 s f2 -> b ... f2 s d')
            write = main + m.lora_lambda * dwrite
            write = m.merge_fracs(rearrange(write, 'b ... s d -> (b s) ... d'))
            expected = write + res
        err = (actual - expected).abs().max().item()
        assert err <= 1e-5, f"num_fracs={nf}: depth write != einsum(h,beta)+lambda*einsum(delta,beta) (err {err:.2e})"
    print("[ok] inside-beta formula: actual == einsum(h,beta) + lambda*einsum(delta,beta) (num_fracs 1,2)")


def test_beta_mask_zeros_stream_write():
    # zeroing beta for one stream must zero BOTH its main write and its LoRA write.
    b, seq, d, s = 2, 4, 64, 4
    for nf in (1, 2):
        torch.manual_seed(20 + nf)
        m = MHCMid(s, dim=d, num_fracs=nf, branch=nn.Linear(d, d)).eval()
        with torch.no_grad():
            m.stream_up_weight.normal_(std=0.5)   # LoRA write would be non-zero if not masked
            bo = torch.randn(b, seq, d)
            res = torch.randn(b * s, seq, d)
            beta = torch.rand(b, seq, nf, s, nf) * 2.0
            s0 = 1
            beta[..., s0, :] = 0.0                 # beta[.., f1, s0, f2] = 0  (stream axis = -2)
            out = m.depth_connection(bo, res, beta=beta)
            write = rearrange(out, '(b s) ... -> b s ...', s=s) - rearrange(res, '(b s) ... -> b s ...', s=s)
        err = write[:, s0].abs().max().item()
        assert err < 1e-6, f"num_fracs={nf}: masked stream write not zero (max {err:.2e})"
    print("[ok] beta=0 for a stream -> that stream's main AND LoRA write are exactly 0 (num_fracs 1,2)")


def main():
    test_runs_and_shape()
    test_init_equals_original_mhc()
    test_midnorm_dim_is_rank()
    test_per_stream_independent_AB()
    test_disable_lora_reverts()
    test_no_nan_inf_and_grads()
    test_state_dict_roundtrip()
    test_inside_beta_formula_consistency()
    test_beta_mask_zeros_stream_write()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
