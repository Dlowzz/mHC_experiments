"""Minimal tests for mHC-LoRA-Residual-affine-midnorm (affine rank-dim RMSNorm)
and the no-cosine-LR-decay optimizer grouping.

Run from the repo root (mhc-lite) with:
    python -m hyper_conn.test_mhc_lora_residual_affinemidnorm
"""

import torch
from torch import nn
from einops import repeat

from .mhc import ManifoldConstrainedHyperConnections
from .mhc_lora_residual_midnorm import ManifoldConstrainedHyperConnectionsLoRAResidualMidNorm as MidNorm
from .mhc_lora_residual_affinemidnorm import ManifoldConstrainedHyperConnectionsLoRAResidualAffineMidNorm as Affine


def _expand(x, s):
    return repeat(x, "b n d -> (b s) n d", s=s)


def test_runs_and_shape_matches_mhc():
    b, seq, d = 2, 4, 16
    for s in (4, 8):
        torch.manual_seed(0)
        m = Affine(s, dim=d, branch=nn.Linear(d, d))
        torch.manual_seed(0)
        ref = ManifoldConstrainedHyperConnections(s, dim=d, branch=nn.Linear(d, d))
        x = _expand(torch.randn(b, seq, d), s)
        assert m(x).shape == x.shape == ref(x).shape
        assert m.lora_rmsnorm_weight.shape == (s, m.lora_rank)
        assert m.lora_rmsnorm_bias.shape == (s, m.lora_rank)
    print("[ok] runs for streams {4,8}; shape matches mHC; affine params [streams,rank]")


def test_init_weight1_bias0():
    m = Affine(4, dim=16, branch=nn.Linear(16, 16))
    assert torch.all(m.lora_rmsnorm_weight == 1.0) and torch.all(m.lora_rmsnorm_bias == 0.0)
    print("[ok] affine init: scale=1, bias=0")


def test_affine_reduces_to_paramfree_at_init_scale():
    # with gamma=1, beta=0 the affine RMSNorm must equal the parameter-free midnorm,
    # even for a NON-zero B_s (isolates the affine layer, not just the B=0 case).
    b, seq, d, s = 2, 4, 16, 4
    torch.manual_seed(1)
    aff = Affine(s, dim=d, branch=nn.Linear(d, d)).eval()
    mid = MidNorm(s, dim=d, branch=nn.Linear(d, d)).eval()
    mid.load_state_dict(aff.state_dict(), strict=False)  # share base + A_s + B_s
    with torch.no_grad():
        aff.stream_up_weight.normal_(std=0.5)
        mid.stream_up_weight.copy_(aff.stream_up_weight)
        # aff keeps gamma=1, beta=0
        x = _expand(torch.randn(b, seq, d), s)
        assert torch.allclose(aff(x), mid(x), atol=1e-5), "affine(gamma=1,beta=0) != param-free midnorm"
    print("[ok] affine RMSNorm with gamma=1,beta=0 == parameter-free midnorm")


def test_affine_is_active_when_params_move():
    b, seq, d, s = 2, 4, 16, 4
    torch.manual_seed(2)
    aff = Affine(s, dim=d, branch=nn.Linear(d, d)).eval()
    mid = MidNorm(s, dim=d, branch=nn.Linear(d, d)).eval()
    mid.load_state_dict(aff.state_dict(), strict=False)
    with torch.no_grad():
        aff.stream_up_weight.normal_(std=0.5)
        mid.stream_up_weight.copy_(aff.stream_up_weight)
        aff.lora_rmsnorm_weight.normal_(mean=1.0, std=0.3)
        aff.lora_rmsnorm_bias.normal_(std=0.3)
        x = _expand(torch.randn(b, seq, d), s)
        assert not torch.allclose(aff(x), mid(x), atol=1e-4), "affine params had no effect"
    print("[ok] moving gamma/beta changes the output (affine is active)")


def test_gradients_flow_to_affine_params():
    b, seq, d, s = 2, 4, 16, 4
    torch.manual_seed(3)
    aff = Affine(s, dim=d, branch=nn.Linear(d, d))
    with torch.no_grad():
        aff.stream_up_weight.normal_(std=0.3)  # activate LoRA path
    x = _expand(torch.randn(b, seq, d), s)
    aff(x).sum().backward()
    for name in ("lora_rmsnorm_weight", "lora_rmsnorm_bias", "stream_up_weight", "stream_down_weight"):
        p = getattr(aff, name)
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} got no gradient"
    print("[ok] gradients flow to lora_rmsnorm_weight/bias and stream_up/down_weight")


def test_optimizer_group_no_lr_decay():
    # affine RMSNorm params must land in a dedicated group: weight_decay=0 + no_lr_decay=True
    from model import GPTConfig, GPT
    cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=4, n_embd=64,
                    hyper_conn_type='mhc_lora_residual_affinemidnorm', hyper_conn_n=4)
    m = GPT(cfg)
    opt = m.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95), device_type='cpu')
    nd_groups = [g for g in opt.param_groups if g.get('no_lr_decay', False)]
    assert len(nd_groups) == 1, f"expected exactly 1 no_lr_decay group, got {len(nd_groups)}"
    g = nd_groups[0]
    assert g['weight_decay'] == 0.0, "no_lr_decay group must have weight_decay=0"
    n_rms = sum(1 for name, _ in m.named_parameters() if 'lora_rmsnorm' in name)
    assert len(g['params']) == n_rms and n_rms > 0, "not all lora_rmsnorm params in the group"
    # and they must NOT be in the (decayed) first group
    decay_ids = {id(p) for p in opt.param_groups[0]['params']}
    assert all(id(p) not in decay_ids for p in g['params']), "rmsnorm params leaked into decay group"
    print(f"[ok] configure_optimizers: {n_rms} lora_rmsnorm tensors in a no_lr_decay/wd=0 group")


def test_disable_lora_equals_mhc_write():
    b, seq, d, s = 2, 4, 16, 4
    torch.manual_seed(4)
    aff = Affine(s, dim=d, branch=nn.Linear(d, d), disable_lora_branch=True).eval()
    with torch.no_grad():
        aff.stream_up_weight.normal_(std=0.5)   # must have no effect when disabled
        aff.lora_rmsnorm_bias.normal_(std=0.5)
        x = _expand(torch.randn(b, seq, d), s)
        out_off = aff(x)
        aff.disable_lora_branch = False
        out_on = aff(x)
    assert not torch.allclose(out_on, out_off, atol=1e-5), "disable flag had no effect"
    print("[ok] disable_lora_branch=True removes the LoRA write")


def main():
    test_runs_and_shape_matches_mhc()
    test_init_weight1_bias0()
    test_affine_reduces_to_paramfree_at_init_scale()
    test_affine_is_active_when_params_move()
    test_gradients_flow_to_affine_params()
    test_optimizer_group_no_lr_decay()
    test_disable_lora_equals_mhc_write()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
