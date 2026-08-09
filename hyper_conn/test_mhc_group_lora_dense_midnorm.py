"""Tests for mHC-group-LoRA-dense-midnorm (dense n^3 C H_pre + midnorm in-beta LoRA).

Run from the repo root (mhc-lite) with:
    python -m hyper_conn.test_mhc_group_lora_dense_midnorm
"""

import random

import torch
from torch import nn
from einops import repeat, einsum

from .mhc_group_dense_embedding import dense_from_group_local
from .mhc_group_lora_midnorm import ManifoldConstrainedHyperConnectionsGroupLoRAMidNorm as MHCLocal
from .mhc_group_lora_dense_midnorm import (
    ManifoldConstrainedHyperConnectionsGroupLoRADenseMidNorm as MHCDense,
)


def _expand(x, s):
    return repeat(x, "b n d -> (b s) n d", s=s)


def _pair(d=64, s=4, seed=0):
    random.seed(seed); torch.manual_seed(seed)
    g = MHCLocal(s, dim=d, branch=nn.Linear(d, d)).eval()
    random.seed(seed); torch.manual_seed(seed)
    dn = MHCDense(s, dim=d, branch=nn.Linear(d, d)).eval()
    incompat = dn.load_state_dict(g.state_dict(), strict=False)
    assert incompat.unexpected_keys == ["group_pre_weight"], incompat.unexpected_keys
    assert incompat.missing_keys == ["group_pre_weight_dense"], incompat.missing_keys
    return g, dn


def test_runs_and_shape():
    for d, s in [(64, 4), (128, 8)]:
        random.seed(0); torch.manual_seed(0)
        m = MHCDense(s, dim=d, branch=nn.Linear(d, d))
        x = _expand(torch.randn(2, 4, d), s)
        assert m(x).shape == x.shape
    print("[ok] runs for streams {4,8}; output shape preserved")


def test_param_shape_and_lora_intact():
    d, s = 64, 4
    g, dn = _pair(d, s)
    eff, groups = dn.effective_dim, dn.group_embedding_groups
    assert dn.group_pre_weight_dense.shape == (s * eff, groups * s)
    assert dn.group_pre_weight_dense.numel() == s ** 3 * eff
    assert dn.group_pre_weight_dense.numel() == s * g.group_pre_weight.numel()
    assert not hasattr(dn, "group_pre_weight")
    # LoRA side untouched: same shapes, B_s zero-init, A_s excluded from weight decay
    assert dn.stream_down_weight.shape == (s, eff, dn.lora_rank)
    assert dn.stream_up_weight.shape == (s, dn.lora_rank, eff)
    assert getattr(dn.stream_down_weight, "_no_weight_decay", False)
    print(f"[ok] dense read n^3 C = {dn.group_pre_weight_dense.numel()} "
          f"(local n^2 C = {g.group_pre_weight.numel()}); LoRA A_s/B_s shapes and "
          f"no-weight-decay flag unchanged")


def test_zero_init_matches_group_local():
    b, seq, d, s = 2, 5, 64, 4
    g, dn = _pair(d, s)
    assert dn.group_pre_weight_dense.abs().sum() == 0
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        assert torch.equal(g(x), dn(x))
    print("[ok] at init the dense variant is bit-identical to mhc_group_lora_midnorm")


def test_dense_reproduces_group_local():
    b, seq, d, s = 2, 5, 64, 4
    g, dn = _pair(d, s)
    with torch.no_grad():
        g.group_pre_weight.normal_(0, 0.5)
        g.pre_branch_scale.fill_(1.0); dn.pre_branch_scale.fill_(1.0)
        # switch the LoRA branch on too, so the equivalence covers the LoRA path
        g.stream_up_weight.normal_(0, 0.2)
        dn.stream_up_weight.copy_(g.stream_up_weight)
    dn.load_group_local_(g)
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        og, od = g(x), dn(x)
    assert (og - od).abs().max() < 1e-6, (og - od).abs().max()
    print(f"[ok] dense reproduces a non-zero group-local read exactly with the LoRA branch "
          f"live (max |d out|={float((og-od).abs().max()):.2e})")


def test_dense_reads_across_groups():
    b, seq, d, s = 2, 5, 64, 4
    g, dn = _pair(d, s)
    with torch.no_grad():
        dn.pre_branch_scale.fill_(1.0)
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        before = dn(x).clone()
    mask = dense_from_group_local(torch.ones_like(g.group_pre_weight), s,
                                  dn.group_embedding_groups, dn.group_dim)
    rows, cols = (mask == 0).nonzero(as_tuple=True)
    assert rows.numel() > 0
    with torch.no_grad():
        dn.group_pre_weight_dense[rows[0], cols[0]] = 5.0
        after = dn(x).clone()
    assert (after - before).abs().max() > 1e-5
    print(f"[ok] off-block weight moves the output by {float((after-before).abs().max()):.3e} "
          f"-> cross-group reading is live")


def test_rank_space_gate_equals_naive():
    """The inherited rank-space `lora_write` must equal gating the materialised delta."""
    b, seq, d, s = 2, 5, 64, 4
    random.seed(0); torch.manual_seed(0)
    m = MHCDense(s, dim=d, branch=nn.Linear(d, d)).eval()
    with torch.no_grad():
        m.stream_up_weight.normal_(0, 0.2)
    h = torch.randn(b, seq, 1, d)                       # b t f d (num_fracs == 1)
    beta = torch.rand(b, seq, 1, s, 1) * 2              # b t f1 s f2
    with torch.no_grad():
        fast = m.lora_write(h, beta)
        naive = m.beta_write_per_stream(m.compute_lora(h), beta)
    assert (fast - naive).abs().max() < 1e-5, (fast - naive).abs().max()
    print(f"[ok] rank-space beta gate == naive hidden-dim gate "
          f"(max diff {float((fast-naive).abs().max()):.2e})")


def test_beta_zero_kills_both_writes():
    """in-beta semantics: zeroing a stream's beta must zero BOTH its main and LoRA write."""
    b, seq, d, s = 2, 5, 64, 4
    random.seed(0); torch.manual_seed(0)
    m = MHCDense(s, dim=d, branch=nn.Linear(d, d)).eval()
    with torch.no_grad():
        m.stream_up_weight.normal_(0, 0.2)
    h = torch.randn(b, seq, d)
    beta = torch.rand(b, seq, 1, s, 1) * 2
    beta[..., 0, :] = 0.0                               # kill stream 0
    resid = torch.zeros(b * s, seq, d)
    with torch.no_grad():
        out = m.depth_connection(h, resid, beta=beta)
    per_stream = out.unflatten(0, (b, s))
    assert per_stream[:, 0].abs().max() == 0, per_stream[:, 0].abs().max()
    assert per_stream[:, 1:].abs().max() > 0
    print("[ok] beta_s = 0 zeroes that stream's main *and* LoRA write (in-beta)")


def test_grads_and_finite():
    b, seq, d, s = 2, 5, 64, 4
    random.seed(0); torch.manual_seed(0)
    m = MHCDense(s, dim=d, branch=nn.Linear(d, d))
    x = _expand(torch.randn(b, seq, d), s)
    out = m(x)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert m.group_pre_weight_dense.grad is not None
    assert m.group_pre_weight_dense.grad.abs().sum() > 0
    assert m.stream_down_weight.grad is not None and m.stream_up_weight.grad is not None
    for n, p in m.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), n
    print("[ok] forward/backward finite; dense read and both LoRA factors get gradients")


def test_state_dict_roundtrip():
    d, s = 64, 4
    random.seed(0); torch.manual_seed(0)
    a = MHCDense(s, dim=d, branch=nn.Linear(d, d)).eval()
    with torch.no_grad():
        a.group_pre_weight_dense.normal_(0, 0.3)
        a.stream_up_weight.normal_(0, 0.2)
    random.seed(0); torch.manual_seed(1)
    b_ = MHCDense(s, dim=d, branch=nn.Linear(d, d)).eval()
    b_.load_state_dict(a.state_dict())
    x = _expand(torch.randn(2, 5, d), s)
    with torch.no_grad():
        assert torch.equal(a(x), b_(x))
    print("[ok] state_dict roundtrip is exact")


def main():
    test_runs_and_shape()
    test_param_shape_and_lora_intact()
    test_zero_init_matches_group_local()
    test_dense_reproduces_group_local()
    test_dense_reads_across_groups()
    test_rank_space_gate_equals_naive()
    test_beta_zero_kills_both_writes()
    test_grads_and_finite()
    test_state_dict_roundtrip()
    print("PASS: mhc_group_lora_dense_midnorm")


if __name__ == "__main__":
    main()
