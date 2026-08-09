"""Tests for mHC-group-LoRA-scale-midnorm (learnable scale, no bias, on the rank norm).

Run from the repo root (mhc-lite) with:
    python -m hyper_conn.test_mhc_group_lora_scalemidnorm
"""

import random

import torch
from torch import nn
from einops import repeat

from .mhc_group_lora_midnorm import ManifoldConstrainedHyperConnectionsGroupLoRAMidNorm as MHCMid
from .mhc_group_lora_scalemidnorm import (
    ManifoldConstrainedHyperConnectionsGroupLoRAScaleMidNorm as MHCScale,
)


def _expand(x, s):
    return repeat(x, "b n d -> (b s) n d", s=s)


def _model(d=64, s=4, seed=0, cls=MHCScale, **kw):
    random.seed(seed)          # mhc picks the home stream via random.randrange
    torch.manual_seed(seed)
    return cls(s, dim=d, branch=nn.Linear(d, d), **kw).eval()


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


def test_group_read_untouched():
    """The ablation only touches the LoRA norm: the group H_pre read must be identical."""
    d, s = 64, 4
    ref, m = _model(d, s, cls=MHCMid), _model(d, s, cls=MHCScale)
    assert m.group_pre_weight.shape == ref.group_pre_weight.shape
    assert torch.equal(m.group_pre_bias, ref.group_pre_bias)
    x = _expand(torch.randn(2, 5, d), s)
    with torch.no_grad():
        gm, gr = m.get_group_gates(x), ref.get_group_gates(x)
    assert torch.equal(gm, gr), (gm - gr).abs().max()
    assert gm.shape[-2:] == (m.group_embedding_groups, s)
    print(f"[ok] group H_pre gate identical to mhc_group_lora_midnorm, shape {tuple(gm.shape)}")


def test_init_equals_midnorm():
    b, seq, d, s = 2, 5, 64, 4
    ref, m = _model(d, s, cls=MHCMid), _model(d, s, cls=MHCScale)
    incompat = m.load_state_dict(ref.state_dict(), strict=False)
    assert incompat.missing_keys == ["lora_rmsnorm_weight"], incompat.missing_keys
    assert incompat.unexpected_keys == [], incompat.unexpected_keys
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        assert torch.equal(ref(x), m(x))
    print("[ok] gamma == 1 makes it bit-identical to mhc_group_lora_midnorm")


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
    for name in ("lora_rmsnorm_weight", "stream_down_weight", "stream_up_weight",
                 "group_pre_weight"):
        g = getattr(m, name).grad
        assert g is not None and torch.isfinite(g).all(), name
        assert g.abs().sum() > 0, f"{name} got no gradient"
    for n, p in m.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), n
    print("[ok] forward/backward finite; gamma, A_s, B_s and the group read all get gradients")


def test_zero_B_keeps_init_identity():
    """With B_s == 0 (init) any gamma must leave the module at the group_lora baseline."""
    b, seq, d, s = 2, 5, 64, 4
    m = _model(d, s)
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        base = m(x).clone()
        m.lora_rmsnorm_weight.normal_(0, 5.0)   # wild gamma, but B_s is still zero
        assert torch.equal(m(x), base)
    print("[ok] B_s == 0 keeps the module at the group_lora baseline for any gamma")


def test_groups_kwarg_and_state_dict():
    d, s, groups = 64, 4, 2
    a = _model(d, s, seed=0, group_embedding_groups=groups)
    assert a.group_embedding_groups == groups
    with torch.no_grad():
        a.lora_rmsnorm_weight.normal_(1.0, 0.3)
        a.stream_up_weight.normal_(0, 0.2)
    b_ = _model(d, s, seed=0, group_embedding_groups=groups)
    b_.load_state_dict(a.state_dict())
    x = _expand(torch.randn(2, 5, d), s)
    with torch.no_grad():
        assert torch.equal(a(x), b_(x))
    assert "lora_rmsnorm_weight" in a.state_dict()
    print(f"[ok] group_embedding_groups={groups} honoured; state_dict roundtrip exact")


def main():
    test_runs_and_shape()
    test_scale_only_no_bias()
    test_group_read_untouched()
    test_init_equals_midnorm()
    test_gamma_reaches_lora_write()
    test_gamma_is_a_pure_scale()
    test_gamma_changes_module_output()
    test_grads_and_finite()
    test_zero_B_keeps_init_identity()
    test_groups_kwarg_and_state_dict()
    print("PASS: mhc_group_lora_scalemidnorm")


if __name__ == "__main__":
    main()
