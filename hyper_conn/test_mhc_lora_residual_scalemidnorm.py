"""Tests for mHC-LoRA-Residual-scale-midnorm (learnable scale, no bias, on the rank norm).

Run from the repo root (mhc-lite) with:
    python -m hyper_conn.test_mhc_lora_residual_scalemidnorm
"""

import random

import torch
from torch import nn
from einops import repeat

from .mhc_lora_residual_midnorm import ManifoldConstrainedHyperConnectionsLoRAResidualMidNorm as MHCMid
from .mhc_lora_residual_scalemidnorm import (
    ManifoldConstrainedHyperConnectionsLoRAResidualScaleMidNorm as MHCScale,
)


def _expand(x, s):
    return repeat(x, "b n d -> (b s) n d", s=s)


def _model(d=64, s=4, seed=0, cls=MHCScale):
    random.seed(seed)
    torch.manual_seed(seed)
    return cls(s, dim=d, branch=nn.Linear(d, d)).eval()


def test_runs_and_shape():
    for d, s in [(64, 4), (128, 8)]:
        m = _model(d, s)
        x = _expand(torch.randn(2, 4, d), s)
        assert m(x).shape == x.shape
    print("[ok] runs for streams {4,8}; output shape preserved")


def test_scale_only_no_bias():
    d, s = 64, 4
    m = _model(d, s)
    assert m.lora_rmsnorm_weight.shape == (s, m.lora_rank)
    assert torch.equal(m.lora_rmsnorm_weight, torch.ones(s, m.lora_rank))
    assert getattr(m.lora_rmsnorm_weight, "_no_weight_decay", False)
    assert not hasattr(m, "lora_rmsnorm_bias"), "this variant must have no bias"
    names = dict(m.named_parameters())
    assert "lora_rmsnorm_weight" in names and "lora_rmsnorm_bias" not in names
    print(f"[ok] gamma is [{s}, {m.lora_rank}] ones-init, flagged no-weight-decay; no bias exists")


def test_init_equals_midnorm():
    b, seq, d, s = 2, 5, 64, 4
    ref, m = _model(d, s, cls=MHCMid), _model(d, s, cls=MHCScale)
    incompat = m.load_state_dict(ref.state_dict(), strict=False)
    assert incompat.missing_keys == ["lora_rmsnorm_weight"], incompat.missing_keys
    assert incompat.unexpected_keys == [], incompat.unexpected_keys
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        assert torch.equal(ref(x), m(x))
    print("[ok] gamma == 1 makes it bit-identical to mhc_lora_residual_midnorm")


def test_gamma_reaches_lora_write():
    """Regression guard: the scale must apply to `lora_write`, the path depth_connection
    uses -- not only to `compute_lora`.  The archived affinemidnorm variant overrode
    compute_lora alone and silently dropped the affine from training."""
    b, seq, d, s = 2, 5, 64, 4
    m = _model(d, s)
    with torch.no_grad():
        m.stream_up_weight.normal_(0, 0.2)
        m.lora_rmsnorm_weight.normal_(1.0, 0.5)          # non-trivial gamma
    h = torch.randn(b, seq, 1, d)                        # b t f d (num_fracs == 1)
    beta = torch.rand(b, seq, 1, s, 1) * 2
    with torch.no_grad():
        fast = m.lora_write(h, beta)
        naive = m.beta_write_per_stream(m.compute_lora(h), beta)
    assert (fast - naive).abs().max() < 1e-5, (fast - naive).abs().max()
    # and it is genuinely gamma-dependent
    with torch.no_grad():
        before = m.lora_write(h, beta).clone()
        m.lora_rmsnorm_weight.mul_(3.0)
        after = m.lora_write(h, beta)
    assert (after - 3.0 * before).abs().max() < 1e-5, (after - 3.0 * before).abs().max()
    print("[ok] gamma reaches lora_write (== naive gated compute_lora) and scales it exactly")


def test_gamma_is_a_pure_scale():
    """delta with gamma = c must be exactly c * delta with gamma = 1."""
    b, seq, d, s = 2, 5, 64, 4
    m = _model(d, s)
    with torch.no_grad():
        m.stream_up_weight.normal_(0, 0.2)
    h = torch.randn(b, seq, 1, d)
    with torch.no_grad():
        base = m.compute_lora(h).clone()
        m.lora_rmsnorm_weight.fill_(2.5)
        scaled = m.compute_lora(h)
    assert (scaled - 2.5 * base).abs().max() < 1e-5, (scaled - 2.5 * base).abs().max()
    print("[ok] gamma acts as an exact multiplicative scale on the rank dim")


def test_gamma_changes_module_output():
    """End-to-end: gamma must move the module's forward output, not just the delta."""
    b, seq, d, s = 2, 5, 64, 4
    m = _model(d, s)
    with torch.no_grad():
        m.stream_up_weight.normal_(0, 0.2)
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        before = m(x).clone()
        m.lora_rmsnorm_weight.mul_(4.0)
        after = m(x)
    assert (after - before).abs().max() > 1e-4, (after - before).abs().max()
    print(f"[ok] scaling gamma moves the module output by {float((after-before).abs().max()):.3e}")


def test_grads_and_finite():
    b, seq, d, s = 2, 5, 64, 4
    random.seed(0)
    torch.manual_seed(0)
    m = MHCScale(s, dim=d, branch=nn.Linear(d, d))
    with torch.no_grad():
        m.stream_up_weight.normal_(0, 0.2)      # B_s != 0 so gamma is on the graph
    x = _expand(torch.randn(b, seq, d), s)
    out = m(x)
    assert torch.isfinite(out).all()
    out.sum().backward()
    g = m.lora_rmsnorm_weight.grad
    assert g is not None and torch.isfinite(g).all()
    assert g.abs().sum() > 0, "gamma got no gradient"
    for n, p in m.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), n
    print(f"[ok] forward/backward finite; gamma gets gradient (|g|={float(g.abs().sum()):.4e})")


def test_zero_B_keeps_init_identity():
    """With B_s == 0 (init) any gamma must leave the module equal to the original mHC."""
    b, seq, d, s = 2, 5, 64, 4
    m = _model(d, s)
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        base = m(x).clone()
        m.lora_rmsnorm_weight.normal_(0, 5.0)   # wild gamma, but B_s is still zero
        assert torch.equal(m(x), base)
    print("[ok] B_s == 0 keeps the module at the mHC baseline for any gamma")


def test_state_dict_roundtrip():
    d, s = 64, 4
    a = _model(d, s, seed=0)
    with torch.no_grad():
        a.lora_rmsnorm_weight.normal_(1.0, 0.3)
        a.stream_up_weight.normal_(0, 0.2)
    b_ = _model(d, s, seed=0)
    b_.load_state_dict(a.state_dict())
    x = _expand(torch.randn(2, 5, d), s)
    with torch.no_grad():
        assert torch.equal(a(x), b_(x))
    assert "lora_rmsnorm_weight" in a.state_dict()
    print("[ok] state_dict roundtrip is exact and carries lora_rmsnorm_weight")


def main():
    test_runs_and_shape()
    test_scale_only_no_bias()
    test_init_equals_midnorm()
    test_gamma_reaches_lora_write()
    test_gamma_is_a_pure_scale()
    test_gamma_changes_module_output()
    test_grads_and_finite()
    test_zero_B_keeps_init_identity()
    test_state_dict_roundtrip()
    print("PASS: mhc_lora_residual_scalemidnorm")


if __name__ == "__main__":
    main()
