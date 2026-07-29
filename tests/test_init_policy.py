"""Initialisation policy tests: the seeded home-stream choice and the group read bias.

Two things are checked here that the equivalence tests deliberately do not cover:

  * the beta "home" (write-in) stream per layer comes from ``random.randrange``
    (``hyper_conn/mhc.py``), so it is only reproducible if ``random`` is seeded -- which
    ``train.py`` now does;
  * ``group_pre_bias`` starts every channel group on that same home stream (one +1
    column, -1 elsewhere) instead of on the diagonal, matching the original mHC
    ``static_alpha`` / ``static_beta`` convention.

Run from the repo root::

    python -m tests.test_init_policy
"""
import random
from pathlib import Path

import torch

import model as model_mod
from model import GPTConfig
from hyper_conn.mhc_group_embedding import ManifoldConstrainedHyperConnectionsGroupEmbedding as Group
from hyper_conn.mhc_group_lora_midnorm import ManifoldConstrainedHyperConnectionsGroupLoRAMidNorm as GroupLoRA

REPO = Path(__file__).resolve().parents[1]
GROUP_TYPES = ("mhc_group_embedding", "mhc_group_lora_midnorm")


def build_gpt(hc_type, random_seed, n_layer=6):
    """`random` seed varies, torch seed fixed -- only the home-stream layout changes."""
    cfg = GPTConfig(block_size=16, vocab_size=64, n_layer=n_layer, n_head=4, n_embd=32,
                    dropout=0.0, bias=False, hyper_conn_n=4, hyper_conn_type=hc_type)
    random.seed(random_seed)
    torch.manual_seed(1337)
    return model_mod.GPT(cfg)


def hyper_conns(gpt):
    for i, blk in enumerate(gpt.transformer.h):
        yield f"layer{i}.attn", blk.hc_attn
        yield f"layer{i}.mlp", blk.hc_mlp


def test_seed_controls_home_stream():
    a, b, c = build_gpt("mhc", 1337), build_gpt("mhc", 1337), build_gpt("mhc", 999)
    idx = lambda g: [m.init_residual_index for _, m in hyper_conns(g)]
    ia, ib, ic = idx(a), idx(b), idx(c)

    assert ia == ib, f"same seed gave different home streams:\n{ia}\n{ib}"
    # 12 modules x 4 streams: a collision would need 4^-12 luck
    assert ia != ic, f"different seeds gave the same home streams: {ia}"

    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb and torch.equal(pa, pb), f"same seed but {na} differs"

    print(f"  home streams reproducible under a fixed random.seed: {ia}")
    print(f"  and change with the seed:                            {ic}")


def test_group_bias_pattern():
    for klass in (Group, GroupLoRA):
        for streams, dim, groups in ((4, 64, 4), (4, 64, 2), (8, 64, 8), (2, 48, 2)):
            random.seed(0)
            torch.manual_seed(0)
            m = klass(streams, dim=dim, group_embedding_groups=groups)

            home = m.init_residual_index
            expected = torch.full((groups, streams), -1.0)
            expected[:, home] = 1.0
            bias = m.group_pre_bias.detach()
            assert torch.equal(bias, expected), (
                f"{klass.__name__} s={streams} g={groups}: group_pre_bias is\n{bias}\nexpected\n{expected}"
            )

            # the three "home" definitions must agree
            assert int(m.static_beta.argmax()) == home, f"{klass.__name__}: static_beta home != {home}"
            assert int(m.static_alpha[:, 0].argmax()) == home, f"{klass.__name__}: static_alpha home != {home}"
    print("  group_pre_bias == one +1 column at the layer's home stream, -1 elsewhere")


def test_group_bias_per_layer():
    for hc_type in GROUP_TYPES:
        gpt = build_gpt(hc_type, 1337)
        for name, m in hyper_conns(gpt):
            home = m.init_residual_index
            bias = m.group_pre_bias.detach()
            assert bias.argmax(dim=-1).eq(home).all(), f"{hc_type} {name}: bias home != {home}"
            assert bias.eq(-1.0).sum().item() == bias.numel() - bias.shape[0], (
                f"{hc_type} {name}: expected exactly one +1 per group row"
            )
            assert int(m.static_beta.argmax()) == home, f"{hc_type} {name}: static_beta home != {home}"
    print(f"  per-layer bias follows each layer's own home stream ({', '.join(GROUP_TYPES)})")


def test_trainers_seed_random():
    for fname, call in (("train.py", "random.seed(seed)"), ("train_analysis.py", "random.seed(1337)")):
        src = (REPO / fname).read_text()
        assert call in src, f"{fname} must seed the `random` module with {call}"
        assert src.index(call) > src.index("torch.manual_seed"), (
            f"{fname}: seed `random` next to torch.manual_seed, after wandb.init, so the "
            f"wandb run-name suffix stays random"
        )
    print("  train.py / train_analysis.py seed the `random` module")


def main():
    test_seed_controls_home_stream()
    test_group_bias_pattern()
    test_group_bias_per_layer()
    test_trainers_seed_random()
    print("PASS: home-stream choice is seed-controlled and the group read bias matches "
          "the original mHC convention")


if __name__ == "__main__":
    main()
