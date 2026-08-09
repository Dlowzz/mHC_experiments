"""Tests for mHC-group-dense-embedding (dense n^3 C per-group H_pre read).

Run from the repo root (mhc-lite) with:
    python -m hyper_conn.test_mhc_group_dense_embedding
"""

import random

import torch
from torch import nn
from einops import repeat

from .mhc_group_embedding import ManifoldConstrainedHyperConnectionsGroupEmbedding as MHCGroup
from .mhc_group_dense_embedding import (
    ManifoldConstrainedHyperConnectionsGroupDenseEmbedding as MHCDense,
    dense_from_group_local,
)


def _expand(x, s):
    return repeat(x, "b n d -> (b s) n d", s=s)


def _pair(d=64, s=4, seed=0):
    """A group-local and a dense module that agree on everything except the read weight.

    `random.seed` is reset before each construction because the base mHC picks its
    per-layer home stream with `random.randrange` -- without this the two modules would
    get different static_alpha / static_beta / group_pre_bias and nothing would match.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    g = MHCGroup(s, dim=d, branch=nn.Linear(d, d)).eval()
    random.seed(seed)
    torch.manual_seed(seed)
    dn = MHCDense(s, dim=d, branch=nn.Linear(d, d)).eval()
    # every shared parameter by name, then the read weight through the dense embedding
    incompat = dn.load_state_dict(g.state_dict(), strict=False)
    assert incompat.unexpected_keys == ["group_pre_weight"], incompat.unexpected_keys
    assert incompat.missing_keys == ["group_pre_weight_dense"], incompat.missing_keys
    return g, dn


def test_runs_and_shape():
    for d, s in [(64, 4), (128, 8)]:
        random.seed(0)
        torch.manual_seed(0)
        m = MHCDense(s, dim=d, branch=nn.Linear(d, d))
        x = _expand(torch.randn(2, 4, d), s)
        assert m(x).shape == x.shape
    print("[ok] runs for streams {4,8}; output shape preserved")


def test_param_shape_is_n3C():
    d, s = 64, 4
    random.seed(0)
    torch.manual_seed(0)
    g = MHCGroup(s, dim=d)
    random.seed(0)
    torch.manual_seed(0)
    dn = MHCDense(s, dim=d)
    eff, groups = dn.effective_dim, dn.group_embedding_groups
    assert dn.group_pre_weight_dense.shape == (s * eff, groups * s)
    assert dn.group_pre_weight_dense.numel() == s ** 3 * eff        # n^3 C
    assert g.group_pre_weight.numel() == s ** 2 * eff               # n^2 C
    assert dn.group_pre_weight_dense.numel() == s * g.group_pre_weight.numel()
    # the compact weight must be gone, not shadowed
    assert not hasattr(dn, "group_pre_weight")
    assert "group_pre_weight" not in dict(dn.named_parameters())
    print(f"[ok] dense read is [{s*eff}, {groups*s}] = n^3 C = {dn.group_pre_weight_dense.numel()}"
          f" (group-local n^2 C = {g.group_pre_weight.numel()}); compact weight removed")


def test_zero_init_matches_group_local():
    """Both read weights are zero-init -> H_pre == sigmoid(group_pre_bias) for both."""
    b, seq, d, s = 2, 5, 64, 4
    g, dn = _pair(d, s)
    assert dn.group_pre_weight_dense.abs().sum() == 0
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        og, od = g(x), dn(x)
        hg, hd = g.get_group_gates(x), dn.get_group_gates(x)
    assert torch.allclose(hg, hd, atol=0, rtol=0), (hg - hd).abs().max()
    assert torch.allclose(og, od, atol=0, rtol=0), (og - od).abs().max()
    print("[ok] at init the dense variant is bit-identical to mhc_group_embedding")


def test_dense_reproduces_group_local():
    """The key test: a *non-zero* group-local read embedded into the dense layout must
    give the same H_pre and the same output, so the dense variant is a strict superset."""
    b, seq, d, s = 2, 5, 64, 4
    g, dn = _pair(d, s)
    with torch.no_grad():
        g.group_pre_weight.normal_(0, 0.5)          # a read that actually moves the sigmoid
        g.pre_branch_scale.fill_(1.0)
        dn.pre_branch_scale.fill_(1.0)
    dn.load_group_local_(g)

    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        hg, hd = g.get_group_gates(x), dn.get_group_gates(x)
        og, od = g(x), dn(x)
    assert (hg - hd).abs().max() < 1e-6, (hg - hd).abs().max()
    assert (og - od).abs().max() < 1e-6, (og - od).abs().max()
    assert hd.std() > 1e-3, hd.std()                # gates are non-trivial, not stuck at sigmoid(bias)
    print(f"[ok] dense reproduces a non-zero group-local read exactly "
          f"(max |dH|={float((hg-hd).abs().max()):.2e}, gate std={float(hd.std()):.4f})")


def test_dense_reads_across_groups():
    """An off-block entry must change the output -- group q really can read the channel
    groups of other groups, which the block-diagonal parent cannot."""
    b, seq, d, s = 2, 5, 64, 4
    g, dn = _pair(d, s)
    groups, gdim = dn.group_embedding_groups, dn.group_dim
    with torch.no_grad():
        dn.pre_branch_scale.fill_(1.0)
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        before = dn(x).clone()

    # positions the block-diagonal layout forces to zero
    dense_mask = dense_from_group_local(torch.ones_like(g.group_pre_weight), s, groups, gdim)
    zero_rows, zero_cols = (dense_mask == 0).nonzero(as_tuple=True)
    assert zero_rows.numel() > 0, "no off-block position -- layout assumption broken"
    with torch.no_grad():
        dn.group_pre_weight_dense[zero_rows[0], zero_cols[0]] = 5.0
        after = dn(x).clone()
    assert (after - before).abs().max() > 1e-5, (after - before).abs().max()
    print(f"[ok] an off-block weight moves the output by {float((after-before).abs().max()):.3e} "
          f"-> cross-group reading is live ({zero_rows.numel()} off-block positions)")


def test_grads_and_finite():
    b, seq, d, s = 2, 5, 64, 4
    random.seed(0)
    torch.manual_seed(0)
    m = MHCDense(s, dim=d, branch=nn.Linear(d, d))
    x = _expand(torch.randn(b, seq, d), s)
    out = m(x)
    assert torch.isfinite(out).all()
    out.sum().backward()
    gw = m.group_pre_weight_dense.grad
    assert gw is not None and torch.isfinite(gw).all()
    assert gw.abs().sum() > 0, "dense read weight got no gradient"
    for n, p in m.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), n
    print(f"[ok] forward/backward finite; dense read weight gets gradient "
          f"(|g|={float(gw.abs().sum()):.4e})")


def test_state_dict_roundtrip():
    d, s = 64, 4
    random.seed(0)
    torch.manual_seed(0)
    a = MHCDense(s, dim=d, branch=nn.Linear(d, d)).eval()
    with torch.no_grad():
        a.group_pre_weight_dense.normal_(0, 0.3)
    random.seed(0)
    torch.manual_seed(1)
    b_ = MHCDense(s, dim=d, branch=nn.Linear(d, d)).eval()
    b_.load_state_dict(a.state_dict())
    x = _expand(torch.randn(2, 5, d), s)
    with torch.no_grad():
        assert torch.equal(a(x), b_(x))
    assert "group_pre_weight_dense" in a.state_dict()
    print("[ok] state_dict roundtrip is exact and carries group_pre_weight_dense")


def main():
    test_runs_and_shape()
    test_param_shape_is_n3C()
    test_zero_init_matches_group_local()
    test_dense_reproduces_group_local()
    test_dense_reads_across_groups()
    test_grads_and_finite()
    test_state_dict_roundtrip()
    print("PASS: mhc_group_dense_embedding")


if __name__ == "__main__":
    main()
