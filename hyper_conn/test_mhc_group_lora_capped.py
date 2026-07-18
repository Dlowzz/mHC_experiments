"""Minimal tests for mHC-group-LoRA-capped (relative-cap LoRA write).

Run from the repo root (mhc-lite) with:
    python -m hyper_conn.test_mhc_group_lora_capped
"""

import torch
from torch import nn
from einops import repeat

from .mhc import ManifoldConstrainedHyperConnections
from .mhc_group_lora import ManifoldConstrainedHyperConnectionsGroupLoRA as MHCGroupLoRA
from .mhc_group_lora_capped import ManifoldConstrainedHyperConnectionsGroupLoRACapped as MHCCap


def _expand(x, s):
    return repeat(x, "b n d -> (b s) n d", s=s)


def _relcap(delta, main, rho):
    """Standalone replica of the module's relative soft-cap (for the math test)."""
    main_norm = main.float().norm(dim=-1, keepdim=True)
    delta_f = delta.float()
    delta_norm = delta_f.norm(dim=-1, keepdim=True)
    tau = rho * main_norm.clamp_min(1e-6)
    ratio = delta_norm / tau.clamp_min(1e-8)
    scale = torch.where(
        delta_norm > 0,
        torch.tanh(ratio) / ratio.clamp_min(1e-8),
        torch.ones_like(delta_norm),
    )
    return delta_f * scale


def test_runs_and_shape_matches_mhc():
    b, seq, d = 2, 4, 16
    for s in (4, 8):
        torch.manual_seed(0)
        m = MHCCap(s, dim=d, branch=nn.Linear(d, d))
        torch.manual_seed(0)
        ref = ManifoldConstrainedHyperConnections(s, dim=d, branch=nn.Linear(d, d))
        x = _expand(torch.randn(b, seq, d), s)
        assert m(x).shape == x.shape == ref(x).shape
    print("[ok] runs for streams {4,8}; output shape matches original mHC")


def test_relcap_math_bound_and_linearity():
    torch.manual_seed(1)
    rho = 0.25
    main = torch.randn(3, 5, 8) * 2.0
    main_norm = main.float().norm(dim=-1, keepdim=True)
    # large delta -> bounded by rho*||main||
    big = torch.randn(3, 5, 8) * 100.0
    capped = _relcap(big, main, rho)
    assert torch.all(capped.norm(dim=-1, keepdim=True) < rho * main_norm + 1e-4), "cap bound violated"
    # small delta -> passes through ~unchanged (linear regime)
    small = torch.randn(3, 5, 8) * 1e-3 * main_norm.mean().item()
    assert torch.allclose(_relcap(small, main, rho), small.float(), rtol=1e-2, atol=1e-4), "small delta distorted"
    # zero delta -> stays zero (no NaN)
    z = torch.zeros(3, 5, 8)
    out = _relcap(z, main, rho)
    assert torch.all(out == 0) and torch.isfinite(out).all()
    print("[ok] relcap: ||delta||<rho||main|| for large, ~identity for small, safe at 0")


def test_init_matches_uncapped_group_lora():
    # at init B_s=0 -> delta=0 -> cap is a no-op -> identical to uncapped group_lora
    b, seq, d, s = 2, 4, 16, 4
    torch.manual_seed(2)
    cap = MHCCap(s, dim=d, branch=nn.Linear(d, d), lora_relative_cap=0.25).eval()
    base = MHCGroupLoRA(s, dim=d, branch=nn.Linear(d, d)).eval()
    base.load_state_dict(cap.state_dict())  # identical params (cap adds no Parameter)
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        assert torch.allclose(cap(x), base(x), atol=1e-6), "capped != uncapped at init"
    print("[ok] at init (B=0, delta=0) capped == uncapped group_lora")


def test_cap_shrinks_large_lora():
    # with large B_s, capped LoRA contribution must be much smaller than uncapped
    b, seq, d, s = 2, 4, 16, 4
    torch.manual_seed(3)
    cap = MHCCap(s, dim=d, branch=nn.Linear(d, d), lora_relative_cap=0.25).eval()
    base = MHCGroupLoRA(s, dim=d, branch=nn.Linear(d, d)).eval()
    base.load_state_dict(cap.state_dict())
    # blow up B_s (stream_up_weight) in BOTH identically
    with torch.no_grad():
        cap.stream_up_weight.normal_(std=5.0)
        base.stream_up_weight.copy_(cap.stream_up_weight)
    x = _expand(torch.randn(b, seq, d), s)
    with torch.no_grad():
        out_cap = cap(x)
        out_base = base(x)
        # LoRA-off reference (same on both, since weights identical)
        cap.disable_lora_branch = True
        out_nolora = cap(x)
        cap.disable_lora_branch = False
    lora_effect_cap = (out_cap - out_nolora).norm()
    lora_effect_base = (out_base - out_nolora).norm()
    assert torch.isfinite(out_cap).all(), "capped output not finite"
    assert lora_effect_cap < lora_effect_base, "cap did not shrink the large LoRA write"
    assert lora_effect_cap < 0.9 * lora_effect_base, "cap effect too weak"
    print(f"[ok] cap shrinks large LoRA write: ||Δ_cap||={float(lora_effect_cap):.3f} "
          f"< ||Δ_base||={float(lora_effect_base):.3f}")


def test_disable_lora_equals_no_delta():
    b, seq, d, s = 2, 4, 16, 4
    torch.manual_seed(4)
    cap = MHCCap(s, dim=d, branch=nn.Linear(d, d), disable_lora_branch=True).eval()
    with torch.no_grad():
        cap.stream_up_weight.normal_(std=5.0)  # should have NO effect when disabled
        x = _expand(torch.randn(b, seq, d), s)
        cap.disable_lora_branch = False
        out_on = cap(x)
        cap.disable_lora_branch = True
        out_off = cap(x)
    assert not torch.allclose(out_on, out_off, atol=1e-5), "disable flag had no effect"
    print("[ok] disable_lora_branch=True removes the LoRA write")


def test_grad_flows_through_cap():
    b, seq, d, s = 2, 4, 16, 4
    torch.manual_seed(5)
    cap = MHCCap(s, dim=d, branch=nn.Linear(d, d), lora_relative_cap=0.25)
    with torch.no_grad():
        cap.stream_up_weight.normal_(std=0.5)  # non-zero so LoRA path is active
    x = _expand(torch.randn(b, seq, d), s)
    cap(x).sum().backward()
    for name in ("stream_up_weight", "stream_down_weight", "group_pre_weight"):
        p = getattr(cap, name)
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} got no gradient"
    print("[ok] gradients flow to stream_up/down_weight and group_pre_weight through the cap")


def main():
    test_runs_and_shape_matches_mhc()
    test_relcap_math_bound_and_linearity()
    test_init_matches_uncapped_group_lora()
    test_cap_shrinks_large_lora()
    test_disable_lora_equals_no_delta()
    test_grad_flows_through_cap()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
