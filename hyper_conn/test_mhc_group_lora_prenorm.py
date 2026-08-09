"""Tests for mHC-group-LoRA-prenorm (RMSNorm on h, before the LoRA down projection).

Run from the repo root (mhc-lite) with:
    python -m hyper_conn.test_mhc_group_lora_prenorm
"""

import random

import torch
from torch import nn
from einops import einsum, repeat

from .mhc_group_lora import rmsnorm_lastdim
from .mhc_group_lora_midnorm import ManifoldConstrainedHyperConnectionsGroupLoRAMidNorm as MHCMid
from .mhc_group_lora_prenorm import (
    ManifoldConstrainedHyperConnectionsGroupLoRAPreNorm as MHCPre,
)


def _expand(x, s):
    return repeat(x, "b n d -> (b s) n d", s=s)


def _model(d=64, s=4, seed=0, cls=MHCPre, **kw):
    random.seed(seed)          # mhc picks the home stream via random.randrange
    torch.manual_seed(seed)
    return cls(s, dim=d, branch=nn.Linear(d, d), **kw).eval()


def test_runs_and_shape():
    for d, s in [(64, 4), (128, 8)]:
        m = _model(d, s)
        x = _expand(torch.randn(2, 4, d), s)
        assert m(x).shape == x.shape
    print("[ok] runs for streams {4,8}; output shape preserved")


def test_no_new_parameters():
    """The norm is parameter-free: the state dict must match midnorm exactly."""
    d, s = 64, 4
    pre, mid = _model(d, s, cls=MHCPre), _model(d, s, cls=MHCMid)
    assert list(pre.state_dict().keys()) == list(mid.state_dict().keys())
    incompat = pre.load_state_dict(mid.state_dict(), strict=True)
    assert incompat.missing_keys == [] and incompat.unexpected_keys == []
    print(f"[ok] no extra params; {len(pre.state_dict())} keys identical to midnorm")


def test_group_read_untouched():
    """The ablation only touches the LoRA norm: the group H_pre read must be identical."""
    d, s = 64, 4
    pre, mid = _model(d, s, cls=MHCPre), _model(d, s, cls=MHCMid)
    assert torch.equal(pre.group_pre_weight, mid.group_pre_weight)
    assert torch.equal(pre.group_pre_bias, mid.group_pre_bias)
    x = _expand(torch.randn(2, 5, d), s)
    with torch.no_grad():
        gp, gm = pre.get_group_gates(x), mid.get_group_gates(x)
    assert torch.equal(gp, gm)
    assert gp.shape[-2:] == (pre.group_embedding_groups, s)
    print(f"[ok] group H_pre gate identical to mhc_group_lora_midnorm, shape {tuple(gp.shape)}")


def test_formula_matches_definition():
    """compute_lora == (rmsnorm_d(h) @ A_s) @ B_s, and lora_down == rmsnorm_d(h) @ A_s."""
    b, seq, d, s = 2, 5, 64, 4
    m = _model(d, s)
    with torch.no_grad():
        m.stream_up_weight.normal_(0, 0.2)
    h = torch.randn(b, seq, 1, d)                       # b t f d (num_fracs == 1)
    with torch.no_grad():
        hn = rmsnorm_lastdim(h)
        want_down = einsum(hn, m.stream_down_weight, "b ... f d, s d r -> b ... f s r")
        want = einsum(want_down, m.stream_up_weight, "b ... f s r, s r e -> b ... f s e")
        got_down, got = m.lora_down(h), m.compute_lora(h)
    assert torch.equal(got_down, want_down)
    assert torch.equal(got, want)
    print("[ok] lora_down/compute_lora match rmsnorm(h)->A_s->B_s exactly")


def test_norm_is_before_A_not_after():
    """Discriminator vs midnorm: here delta scales linearly with A_s; under midnorm the
    rank-dim norm downstream of A_s makes delta invariant to that same rescale."""
    b, seq, d, s = 2, 5, 64, 4
    pre, mid = _model(d, s, cls=MHCPre), _model(d, s, cls=MHCMid)
    mid.load_state_dict(pre.state_dict())
    with torch.no_grad():
        pre.stream_up_weight.normal_(0, 0.2)
        mid.stream_up_weight.copy_(pre.stream_up_weight)
    h = torch.randn(b, seq, 1, d)
    with torch.no_grad():
        pre_base, mid_base = pre.compute_lora(h).clone(), mid.compute_lora(h).clone()
        pre.stream_down_weight.mul_(3.0)
        mid.stream_down_weight.mul_(3.0)
        pre_after, mid_after = pre.compute_lora(h), mid.compute_lora(h)
    assert (pre_after - 3.0 * pre_base).abs().max() < 1e-5, (pre_after - 3.0 * pre_base).abs().max()
    assert (mid_after - mid_base).abs().max() < 1e-5, (mid_after - mid_base).abs().max()
    print("[ok] delta is A_s-homogeneous here (x3 -> x3) while midnorm is A_s-invariant")


def test_delta_invariant_to_h_scale():
    """The norm still removes the branch-output magnitude, just one stage earlier."""
    b, seq, d, s = 2, 5, 64, 4
    m = _model(d, s)
    with torch.no_grad():
        m.stream_up_weight.normal_(0, 0.2)
    h = torch.randn(b, seq, 1, d)
    with torch.no_grad():
        a, c = m.compute_lora(h), m.compute_lora(h * 7.0)
    assert (a - c).abs().max() < 1e-5, (a - c).abs().max()
    print("[ok] delta is invariant to ||h|| (rmsnorm sits on h)")


def test_A_s_decays():
    """A_s is NOT scale-invariant here, so it must stay in the weight-decay group
    (midnorm flags it _no_weight_decay; prenorm must not)."""
    d, s = 64, 4
    pre, mid = _model(d, s, cls=MHCPre), _model(d, s, cls=MHCMid)
    assert getattr(mid.stream_down_weight, "_no_weight_decay", False), "midnorm should flag A_s"
    assert not getattr(pre.stream_down_weight, "_no_weight_decay", False)
    assert not getattr(pre.stream_up_weight, "_no_weight_decay", False)
    print("[ok] prenorm leaves A_s in the decay group (no _no_weight_decay flag)")


def test_A_s_init_fan_in():
    """A_s keeps the corrected per-matrix fan_in (bound 1/sqrt(d)), not kaiming's d*r."""
    d, s = 256, 4
    m = _model(d, s)
    bound = 1.0 / d ** 0.5
    amax = float(m.stream_down_weight.abs().max())
    assert amax <= bound + 1e-6, (amax, bound)
    assert amax > 0.5 * bound, (amax, bound)
    assert torch.equal(m.stream_up_weight, torch.zeros_like(m.stream_up_weight))
    print(f"[ok] |A_s|max={amax:.4f} <= 1/sqrt(d)={bound:.4f}; B_s zero-init")


def test_rank_space_gating_is_valid():
    """The norm does not involve beta and sits before A_s, so the cheap rank-space gate
    must equal the naive hidden-dim gate."""
    b, seq, d, s = 2, 5, 64, 4
    m = _model(d, s)
    with torch.no_grad():
        m.stream_up_weight.normal_(0, 0.2)
    h = torch.randn(b, seq, 1, d)
    beta = torch.rand(b, seq, 1, s, 1) * 2
    with torch.no_grad():
        fast = m.lora_write(h, beta)
        naive = m.beta_write_per_stream(m.compute_lora(h), beta)
    assert (fast - naive).abs().max() < 1e-5, (fast - naive).abs().max()
    with torch.no_grad():
        assert m.lora_write(h, torch.zeros_like(beta)).abs().max() == 0
    print("[ok] rank-space lora_write == naive gated compute_lora; beta=0 kills the write")


def test_zero_B_keeps_init_identity():
    """B_s == 0 at init -> forward identical to midnorm (and to the group_lora baseline)."""
    b, seq, d, s = 2, 5, 64, 4
    pre, mid = _model(d, s, cls=MHCPre), _model(d, s, cls=MHCMid)
    mid.load_state_dict(pre.state_dict())
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        assert torch.equal(pre(x), mid(x))
    print("[ok] at init (B_s == 0) prenorm is bit-identical to midnorm")


def test_differs_from_midnorm_when_trained():
    """Once B_s != 0 the two norm positions must give different outputs."""
    b, seq, d, s = 2, 5, 64, 4
    pre, mid = _model(d, s, cls=MHCPre), _model(d, s, cls=MHCMid)
    mid.load_state_dict(pre.state_dict())
    with torch.no_grad():
        pre.stream_up_weight.normal_(0, 0.5)
        mid.stream_up_weight.copy_(pre.stream_up_weight)
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        gap = (pre(x) - mid(x)).abs().max()
    assert gap > 1e-4, gap
    print(f"[ok] prenorm != midnorm once B_s != 0 (max gap {float(gap):.3e})")


def test_grads_and_finite():
    b, seq, d, s = 2, 5, 64, 4
    random.seed(0)
    torch.manual_seed(0)
    m = MHCPre(s, dim=d, branch=nn.Linear(d, d))
    with torch.no_grad():
        m.stream_up_weight.normal_(0, 0.2)      # B_s != 0 so A_s is on the graph
    x = _expand(torch.randn(b, seq, d), s)
    out = m(x)
    assert torch.isfinite(out).all()
    out.sum().backward()
    for name in ("stream_down_weight", "stream_up_weight", "group_pre_weight"):
        g = getattr(m, name).grad
        assert g is not None and torch.isfinite(g).all(), name
        assert g.abs().sum() > 0, f"{name} got no gradient"
    for n, p in m.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), n
    print("[ok] forward/backward finite; A_s, B_s and the group read all get gradients")


def test_groups_kwarg_and_state_dict():
    d, s, groups = 64, 4, 2
    a = _model(d, s, seed=0, group_embedding_groups=groups)
    assert a.group_embedding_groups == groups
    with torch.no_grad():
        a.stream_up_weight.normal_(0, 0.2)
    b_ = _model(d, s, seed=0, group_embedding_groups=groups)
    b_.load_state_dict(a.state_dict())
    x = _expand(torch.randn(2, 5, d), s)
    with torch.no_grad():
        assert torch.equal(a(x), b_(x))
    print(f"[ok] group_embedding_groups={groups} honoured; state_dict roundtrip exact")


def main():
    test_runs_and_shape()
    test_no_new_parameters()
    test_group_read_untouched()
    test_formula_matches_definition()
    test_norm_is_before_A_not_after()
    test_delta_invariant_to_h_scale()
    test_A_s_decays()
    test_A_s_init_fan_in()
    test_rank_space_gating_is_valid()
    test_zero_B_keeps_init_identity()
    test_differs_from_midnorm_when_trained()
    test_grads_and_finite()
    test_groups_kwarg_and_state_dict()
    print("PASS: mhc_group_lora_prenorm")


if __name__ == "__main__":
    main()
