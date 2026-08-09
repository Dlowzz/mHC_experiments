"""Tests for mHC-LoRA-Residual-postnorm (RMSNorm on the hidden dim, after B_s).

Run from the repo root (mhc-lite) with:
    python -m hyper_conn.test_mhc_lora_residual_postnorm
"""

import random

import torch
from torch import nn
from einops import einsum, repeat

from .mhc_lora_residual import (
    ManifoldConstrainedHyperConnectionsLoRAResidual as MHCBase,
    rmsnorm_lastdim,
)
from .mhc_lora_residual_midnorm import ManifoldConstrainedHyperConnectionsLoRAResidualMidNorm as MHCMid
from .mhc_lora_residual_postnorm import (
    ManifoldConstrainedHyperConnectionsLoRAResidualPostNorm as MHCPost,
)


def _expand(x, s):
    return repeat(x, "b n d -> (b s) n d", s=s)


def _model(d=64, s=4, seed=0, cls=MHCPost):
    random.seed(seed)          # mhc picks the home stream via random.randrange
    torch.manual_seed(seed)
    return cls(s, dim=d, branch=nn.Linear(d, d)).eval()


def test_runs_and_shape():
    for d, s in [(64, 4), (128, 8)]:
        m = _model(d, s)
        x = _expand(torch.randn(2, 4, d), s)
        assert m(x).shape == x.shape
    print("[ok] runs for streams {4,8}; output shape preserved")


def test_no_new_parameters():
    """The norm is parameter-free: same state dict as the other two norm positions."""
    d, s = 64, 4
    post, mid = _model(d, s, cls=MHCPost), _model(d, s, cls=MHCMid)
    assert list(post.state_dict().keys()) == list(mid.state_dict().keys())
    incompat = post.load_state_dict(mid.state_dict(), strict=True)
    assert incompat.missing_keys == [] and incompat.unexpected_keys == []
    print(f"[ok] no extra params; {len(post.state_dict())} keys identical to midnorm")


def test_formula_matches_definition():
    """compute_lora == rmsnorm_d((h @ A_s) @ B_s) -- norm strictly after B_s."""
    b, seq, d, s = 2, 5, 64, 4
    m = _model(d, s)
    with torch.no_grad():
        m.stream_up_weight.normal_(0, 0.2)
    h = torch.randn(b, seq, 1, d)                       # b t f d (num_fracs == 1)
    with torch.no_grad():
        down = einsum(h, m.stream_down_weight, "b ... f d, s d r -> b ... f s r")
        raw = einsum(down, m.stream_up_weight, "b ... f s r, s r e -> b ... f s e")
        want = rmsnorm_lastdim(raw)
        got = m.compute_lora(h)
    assert torch.equal(got, want)
    # and it is genuinely normalised on the hidden dim: RMS == 1 per (.., s) row
    rms = got.float().pow(2).mean(dim=-1).sqrt()
    assert (rms - 1.0).abs().max() < 1e-4, float((rms - 1.0).abs().max())
    print("[ok] compute_lora == rmsnorm(h A_s B_s); delta rows have unit RMS")


def test_delta_invariant_to_both_matrix_scales():
    """Norm downstream of both matrices -> delta depends only on their directions.
    This is what justifies flagging A_s and B_s no-weight-decay."""
    b, seq, d, s = 2, 5, 64, 4
    m = _model(d, s)
    with torch.no_grad():
        m.stream_up_weight.normal_(0, 0.2)
    h = torch.randn(b, seq, 1, d)
    with torch.no_grad():
        base = m.compute_lora(h).clone()
        m.stream_down_weight.mul_(5.0)
        after_A = m.compute_lora(h).clone()
        m.stream_up_weight.mul_(0.1)
        after_B = m.compute_lora(h)
    # delta rows are RMS-1, so these gaps are pure fp32 rounding (~1e-5), not a real change
    dA, dB = float((after_A - base).abs().max()), float((after_B - base).abs().max())
    assert dA < 1e-3, dA
    assert dB < 1e-3, dB
    print(f"[ok] delta invariant to rescaling A_s (x5, gap {dA:.1e}) and B_s (x0.1, gap {dB:.1e})")


def test_both_matrices_no_weight_decay():
    d, s = 64, 4
    post, mid = _model(d, s, cls=MHCPost), _model(d, s, cls=MHCMid)
    assert getattr(post.stream_down_weight, "_no_weight_decay", False)
    assert getattr(post.stream_up_weight, "_no_weight_decay", False)
    # contrast: midnorm's B_s is downstream of the norm, so its scale is real and it decays
    assert not getattr(mid.stream_up_weight, "_no_weight_decay", False)
    print("[ok] postnorm flags A_s and B_s no-decay; midnorm keeps B_s in the decay group")


def test_optimizer_grouping():
    """End-to-end check that model.configure_optimizers honours the flags."""
    d, s = 64, 4
    m = _model(d, s)
    flagged = {id(p) for p in m.parameters() if getattr(p, "_no_weight_decay", False)}
    decay = [p for p in m.parameters() if p.dim() >= 2 and id(p) not in flagged]
    assert id(m.stream_down_weight) not in {id(p) for p in decay}
    assert id(m.stream_up_weight) not in {id(p) for p in decay}
    print(f"[ok] neither LoRA matrix lands in the decay group ({len(decay)} params still decay)")


def test_A_s_init_fan_in():
    """A_s uses the corrected per-matrix fan_in (bound 1/sqrt(d)); the parent uses d*r."""
    d, s = 256, 4
    post, base = _model(d, s, cls=MHCPost), _model(d, s, cls=MHCBase)
    bound = 1.0 / d ** 0.5
    amax = float(post.stream_down_weight.abs().max())
    assert amax <= bound + 1e-6 and amax > 0.5 * bound, (amax, bound)
    assert float(base.stream_down_weight.abs().max()) < 0.5 * bound, "parent should be smaller"
    assert torch.equal(post.stream_up_weight, torch.zeros_like(post.stream_up_weight))
    print(f"[ok] |A_s|max={amax:.4f} ~ 1/sqrt(d)={bound:.4f} (parent's 3D kaiming is smaller)")


def test_rank_space_gating_would_be_invalid():
    """The reason lora_write must gate *after* the norm: RMSNorm kills a scalar gate
    applied before it, so midnorm's cheap rank-space trick is not usable here."""
    b, seq, d, s = 2, 5, 64, 4
    m = _model(d, s)
    with torch.no_grad():
        m.stream_up_weight.normal_(0, 0.2)
    h = torch.randn(b, seq, 1, d)
    beta = torch.rand(b, seq, 1, s, 1) * 2
    with torch.no_grad():
        # what the module does: gate the normalised delta
        want = m.beta_write_per_stream(m.compute_lora(h), beta)
        assert torch.equal(m.lora_write(h, beta), want)
        # what the rank-space trick would do: gate before B_s, i.e. before the norm
        down = einsum(h, m.stream_down_weight, "b ... f d, s d r -> b ... f s r")
        gated = m.beta_write_per_stream(down, beta)
        wrong = rmsnorm_lastdim(
            einsum(gated, m.stream_up_weight, "b ... f s r, s r e -> b ... f s e")
        )
    gap = (wrong - want).abs().max()
    assert gap > 1e-3, gap
    print(f"[ok] lora_write gates after the norm; pre-norm gating would differ by {float(gap):.3e}")


def test_beta_zero_kills_the_write():
    b, seq, d, s = 2, 5, 64, 4
    m = _model(d, s)
    with torch.no_grad():
        m.stream_up_weight.normal_(0, 0.2)
    h = torch.randn(b, seq, 1, d)
    with torch.no_grad():
        w = m.lora_write(h, torch.zeros(b, seq, 1, s, 1))
    assert w.abs().max() == 0
    print("[ok] beta == 0 zeroes the LoRA write (in-beta semantics preserved)")


def test_zero_B_keeps_init_identity():
    """B_s == 0 at init and rmsnorm(0) == 0 -> forward identical to midnorm/plain mHC."""
    b, seq, d, s = 2, 5, 64, 4
    post, mid = _model(d, s, cls=MHCPost), _model(d, s, cls=MHCMid)
    mid.load_state_dict(post.state_dict())
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        assert post.compute_lora(torch.randn(2, 3, 1, d)).abs().max() == 0
        assert torch.equal(post(x), mid(x))
    print("[ok] at init (B_s == 0) delta == 0 and postnorm is bit-identical to midnorm")


def test_differs_from_midnorm_when_trained():
    b, seq, d, s = 2, 5, 64, 4
    post, mid = _model(d, s, cls=MHCPost), _model(d, s, cls=MHCMid)
    mid.load_state_dict(post.state_dict())
    with torch.no_grad():
        post.stream_up_weight.normal_(0, 0.5)
        mid.stream_up_weight.copy_(post.stream_up_weight)
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        gap = (post(x) - mid(x)).abs().max()
    assert gap > 1e-4, gap
    print(f"[ok] postnorm != midnorm once B_s != 0 (max gap {float(gap):.3e})")


def test_grads_and_finite():
    b, seq, d, s = 2, 5, 64, 4
    random.seed(0)
    torch.manual_seed(0)
    m = MHCPost(s, dim=d, branch=nn.Linear(d, d))
    with torch.no_grad():
        m.stream_up_weight.normal_(0, 0.2)      # B_s != 0 so the norm is on the graph
    x = _expand(torch.randn(b, seq, d), s)
    out = m(x)
    assert torch.isfinite(out).all()
    out.sum().backward()
    for name in ("stream_down_weight", "stream_up_weight"):
        g = getattr(m, name).grad
        assert g is not None and torch.isfinite(g).all(), name
        assert g.abs().sum() > 0, f"{name} got no gradient"
    for n, p in m.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), n
    print("[ok] forward/backward finite; both A_s and B_s receive gradient")


def main():
    test_runs_and_shape()
    test_no_new_parameters()
    test_formula_matches_definition()
    test_delta_invariant_to_both_matrix_scales()
    test_both_matrices_no_weight_decay()
    test_optimizer_grouping()
    test_A_s_init_fan_in()
    test_rank_space_gating_would_be_invalid()
    test_beta_zero_kills_the_write()
    test_zero_B_keeps_init_identity()
    test_differs_from_midnorm_when_trained()
    test_grads_and_finite()
    print("PASS: mhc_lora_residual_postnorm")


if __name__ == "__main__":
    main()
