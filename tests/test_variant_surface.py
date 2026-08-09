"""Surface test for the pruned hyper_conn package.

After dropping every non-main variant, this pins down what is left: the selectable
types, that removed types fail loudly, that no leftover file references a removed
module, and that every shipped `config/with_*.py` still resolves.

Run from the repo root::

    python -m tests.test_variant_surface
"""
import re
from pathlib import Path

import torch
from torch import nn

import hyper_conn
from model import GPT, GPTConfig

REPO = Path(__file__).resolve().parents[1]

EXPECTED_TYPES = (
    "none",
    "mhc",
    "mhc_lite",
    "mhc_group_embedding",
    "mhc_lora_residual_midnorm",
    "mhc_group_lora_midnorm",
    # ablations: dense (n^3*C) H_pre read for the group variants
    "mhc_group_dense_embedding",
    "mhc_group_lora_dense_midnorm",
    # ablations: LoRA norm affine / position
    "mhc_lora_residual_scalemidnorm",
    "mhc_lora_residual_prenorm",
    "mhc_lora_residual_postnorm",
    # the same three norm ablations on the group-LoRA line
    "mhc_group_lora_scalemidnorm",
    "mhc_group_lora_prenorm",
    "mhc_group_lora_postnorm",
)
REMOVED_TYPES = (
    "hc", "shc", "mhc_embedding", "mhc_orthogonal_diff", "mhc_group_lora_capped",
    "mhc_lora_residual_affinemidnorm", "mhc_lora_residual", "mhc_group_lora", "analysis",
)
REMOVED_MODULES = (
    "hyper_connections", "mhc_analysis", "mhc_embedding", "mhc_group_lora_capped",
    "mhc_lora_residual_affinemidnorm", "mhc_orthogonal_diff", "residuals",
)
EXPECTED_FILES = {
    "__init__.py", "mhc.py", "mhc_lite.py", "mhc_group_embedding.py",
    "mhc_lora_residual.py", "mhc_lora_residual_midnorm.py",
    "mhc_group_lora.py", "mhc_group_lora_midnorm.py",
    "test_mhc_group_embedding.py", "test_mhc_group_lora.py",
    "test_mhc_group_lora_midnorm.py", "test_mhc_lora_residual.py",
    "test_mhc_lora_residual_midnorm.py",
    # ablation variants + their tests
    "mhc_group_dense_embedding.py", "test_mhc_group_dense_embedding.py",
    "mhc_group_lora_dense_midnorm.py", "test_mhc_group_lora_dense_midnorm.py",
    "mhc_lora_residual_scalemidnorm.py", "test_mhc_lora_residual_scalemidnorm.py",
    "mhc_lora_residual_prenorm.py", "test_mhc_lora_residual_prenorm.py",
    "mhc_lora_residual_postnorm.py", "test_mhc_lora_residual_postnorm.py",
    "mhc_group_lora_scalemidnorm.py", "test_mhc_group_lora_scalemidnorm.py",
    "mhc_group_lora_prenorm.py", "test_mhc_group_lora_prenorm.py",
    "mhc_group_lora_postnorm.py", "test_mhc_group_lora_postnorm.py",
}


def test_file_set():
    present = {p.name for p in (REPO / "hyper_conn").glob("*.py")}
    assert present == EXPECTED_FILES, (
        f"unexpected: {sorted(present - EXPECTED_FILES)}, missing: {sorted(EXPECTED_FILES - present)}"
    )
    print(f"  hyper_conn/ holds exactly the {len(EXPECTED_FILES)} expected files")


def test_no_dangling_imports():
    for path in (REPO / "hyper_conn").glob("*.py"):
        src = path.read_text()
        for mod in REMOVED_MODULES:
            assert not re.search(rf"(from|import)\s+\.?{mod}\b", src), f"{path.name} still imports {mod}"
    print(f"  no surviving file imports any of: {', '.join(REMOVED_MODULES)}")


def test_supported_types():
    assert hyper_conn.SUPPORTED_HYPER_CONN_TYPES == EXPECTED_TYPES
    for hc_type in EXPECTED_TYPES:
        streams = 1 if hc_type == "none" else 4
        cfg = GPTConfig(block_size=16, vocab_size=64, n_layer=2, n_head=4, n_embd=32,
                        dropout=0.0, bias=False, hyper_conn_n=streams, hyper_conn_type=hc_type)
        m = GPT(cfg)
        x = torch.randint(64, (2, 16))
        logits, loss = m(x, x)
        loss.backward()
        assert torch.isfinite(logits).all() and torch.isfinite(loss), hc_type
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters()), hc_type
    print(f"  all {len(EXPECTED_TYPES)} supported types build, forward and backward")


def test_removed_types_raise():
    for hc_type in REMOVED_TYPES:
        try:
            hyper_conn.hyper_conn_init_func(hc_type, 4)
        except ValueError as e:
            assert "final" in str(e), f"{hc_type}: error should point at the final branch, got: {e}"
        else:
            raise AssertionError(f"{hc_type} should no longer be selectable")
    print(f"  removed types raise ValueError pointing at `final`: {', '.join(REMOVED_TYPES)}")


def test_none_is_plain_residual():
    init_hc, expand, reduce = hyper_conn.hyper_conn_init_func("none", 4)
    assert isinstance(expand, nn.Identity) and isinstance(reduce, nn.Identity)
    hc = init_hc(dim=32)
    assert isinstance(hc, hyper_conn.Residual), type(hc)
    assert not list(hc.parameters()), "plain residual must not add parameters"
    x = torch.randn(2, 16, 32)
    branch_in, residuals, kw = hc.width_connection(x)
    assert kw == {} and torch.equal(branch_in, x) and torch.equal(residuals, x)
    out = torch.randn_like(x)
    assert torch.equal(hc.depth_connection(out, residuals), out + x)
    print("  `none` is a parameter-free passthrough residual (as hyper_connections.py was)")


def test_shipped_configs_resolve():
    for path in sorted((REPO / "config").glob("with_*.py")):
        m = re.search(r'hyper_conn_type\s*=\s*"([^"]+)"', path.read_text())
        assert m, f"{path.name}: no hyper_conn_type"
        assert m.group(1) in EXPECTED_TYPES, f"{path.name} -> unsupported {m.group(1)}"
        print(f"    {path.name:44s} -> {m.group(1)}")
    print("  every shipped config/with_*.py resolves to a supported type")


def main():
    test_file_set()
    test_no_dangling_imports()
    test_supported_types()
    test_removed_types_raise()
    test_none_is_plain_residual()
    test_shipped_configs_resolve()
    print("PASS: pruned hyper_conn exposes exactly the five main variants plus `none` and the ablations")


if __name__ == "__main__":
    main()
