"""Minimal tests for the mHC-embedding low-rank branch.

Run from the repo root (mhc-lite) with:
    python -m hyper_conn.test_mhc_embedding
"""

import torch
from torch import nn
from einops import repeat

from .mhc import ManifoldConstrainedHyperConnections
from .mhc_embedding import ManifoldConstrainedHyperConnectionsWithEmbedding as MHCEmbedding


def _make_input(b, n, d, s):
    x = torch.randn(b, n, d)
    return repeat(x, "b n d -> (b s) n d", s=s)  # expand to residual streams


def test_runs_and_shape_for_various_streams():
    b, n, d, rank = 2, 5, 16, 8
    for s in (4, 8):
        torch.manual_seed(0)
        emb = MHCEmbedding(s, dim=d, branch=nn.Linear(d, d), embedding_rank=rank)
        # reference original mHC with identical config
        torch.manual_seed(0)
        ref = ManifoldConstrainedHyperConnections(s, dim=d, branch=nn.Linear(d, d))

        x = _make_input(b, n, d, s)
        out = emb(x)
        out_ref = ref(x)

        assert out.shape == x.shape, f"emb out shape {out.shape} != input {x.shape}"
        assert out.shape == out_ref.shape, (
            f"emb out shape {out.shape} != original mHC out shape {out_ref.shape}"
        )
        # A_s count is driven by num_streams, never hard-coded
        assert emb.stream_down_weight.shape == (s, d, rank)
        assert emb.shared_up_weight.shape == (rank, d)
    print("[ok] runs for num_streams in {4, 8}; output shape matches original mHC")


def test_gamma_differs_across_streams():
    b, n, d, s, rank = 2, 5, 16, 4, 8
    torch.manual_seed(1)
    emb = MHCEmbedding(s, dim=d, branch=nn.Linear(d, d), embedding_rank=rank)
    # B is zero-initialised -> make it non-zero so gamma is active
    emb.shared_up_weight.data.normal_()

    h = torch.randn(b, n, d)
    gamma = emb.compute_gamma(emb.split_fracs(h))  # b n f s d, f == num_fracs == 1
    gamma = gamma[..., 0, :, :]  # drop frac dim -> b n s d

    # different streams must not produce identical gamma (A_s are independent)
    diffs = [
        not torch.allclose(gamma[..., i, :], gamma[..., j, :])
        for i in range(s) for j in range(i + 1, s)
    ]
    assert all(diffs), "gamma_s are not distinct across residual streams"
    print("[ok] gamma_s differ across residual streams")


def test_gradients_flow_to_A_B_scale():
    b, n, d, s, rank = 2, 5, 16, 4, 8
    torch.manual_seed(2)
    emb = MHCEmbedding(s, dim=d, branch=nn.Linear(d, d), embedding_rank=rank)
    emb.shared_up_weight.data.normal_()  # activate branch so grads are non-trivial

    x = _make_input(b, n, d, s).requires_grad_(False)
    out = emb(x)
    out.sum().backward()

    for name, p in [
        ("stream_down_weight (A_s)", emb.stream_down_weight),
        ("shared_up_weight (B)", emb.shared_up_weight),
        ("embedding_scale (lambda)", emb.embedding_scale),
    ]:
        assert p.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} grad not finite"
        assert p.grad.abs().sum() > 0, f"{name} grad is all zero"
    print("[ok] A_s, B and scale all receive non-zero finite gradients")


def test_equivalent_to_original_when_branch_off_or_B_zero():
    b, n, d, s = 2, 5, 16, 4
    torch.manual_seed(3)
    ref = ManifoldConstrainedHyperConnections(s, dim=d, branch=nn.Linear(d, d)).eval()
    x = _make_input(b, n, d, s)

    # case 1: explicitly disabled embedding branch
    emb_off = MHCEmbedding(s, dim=d, branch=nn.Linear(d, d),
                           disable_embedding_branch=True).eval()
    missing = emb_off.load_state_dict(ref.state_dict(), strict=False)
    # only the embedding-specific params should be "missing" from the ref dict
    assert set(missing.missing_keys) <= {
        "stream_down_weight", "shared_up_weight", "embedding_scale"
    }, f"unexpected missing keys: {missing.missing_keys}"

    # case 2: branch enabled but B == 0 (default zero-init)
    emb_bzero = MHCEmbedding(s, dim=d, branch=nn.Linear(d, d),
                             disable_embedding_branch=False).eval()
    emb_bzero.load_state_dict(ref.state_dict(), strict=False)
    assert torch.allclose(emb_bzero.shared_up_weight, torch.zeros_like(emb_bzero.shared_up_weight))

    with torch.no_grad():
        out_ref = ref(x)
        out_off = emb_off(x)
        out_bzero = emb_bzero(x)

    assert torch.allclose(out_off, out_ref, atol=1e-6), "disabled branch != original mHC"
    assert torch.allclose(out_bzero, out_ref, atol=1e-6), "B==0 != original mHC"
    print("[ok] disabling branch or B==0 reproduces original mHC exactly")


def main():
    test_runs_and_shape_for_various_streams()
    test_gamma_differs_across_streams()
    test_gradients_flow_to_A_B_scale()
    test_equivalent_to_original_when_branch_off_or_B_zero()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
