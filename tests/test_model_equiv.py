"""GPT-level equivalence + short optimisation trajectory vs the frozen reference.

Note: ``train.py`` cannot be used for this A/B. ``init_residual_index`` comes from
Python's ``random.randrange`` (``hyper_conn/mhc.py:236``) and train.py never seeds the
``random`` module, so two runs of *identical* code already pick different home streams
per layer and diverge on their own. Here both RNGs are seeded before every build, so
any remaining difference is attributable to the code change.

Run from the repo root::

    python -m tests.test_model_equiv
"""
import random

import torch

import model as model_mod
from model import GPTConfig

from tests.reference.mhc import get_init_and_expand_reduce_stream_functions as ref_mhc
from tests.reference.mhc_lite import get_init_and_expand_reduce_stream_functions as ref_lite
from tests.reference.mhc_group_embedding import get_init_and_expand_reduce_stream_functions as ref_group
from tests.reference.mhc_lora_residual_midnorm import get_init_and_expand_reduce_stream_functions as ref_lora
from tests.reference.mhc_group_lora_midnorm import get_init_and_expand_reduce_stream_functions as ref_group_lora

TOL = 1e-6
STEPS = 5

REF_FACTORY = {
    "mhc": ref_mhc,
    "mhc_lite": ref_lite,
    "mhc_group_embedding": ref_group,
    "mhc_lora_residual_midnorm": ref_lora,
    "mhc_group_lora_midnorm": ref_group_lora,
}


def ref_init_func(hyper_conn_type, hyper_conn_n):
    return REF_FACTORY[hyper_conn_type](hyper_conn_n)


def scaled_diff(a, b):
    diff = (a.detach().float() - b.detach().float()).abs().max().item()
    scale = max(1.0, b.detach().float().abs().max().item())
    return diff / scale


def build_gpt(hc_type, use_reference, device, seed=1337):
    cfg = GPTConfig(
        block_size=16, vocab_size=64, n_layer=2, n_head=4, n_embd=32,
        dropout=0.0, bias=False, hyper_conn_n=4, hyper_conn_type=hc_type,
    )
    original = model_mod.hyper_conn_init_func
    if use_reference:
        model_mod.hyper_conn_init_func = ref_init_func
    try:
        random.seed(seed)
        torch.manual_seed(seed)
        gpt = model_mod.GPT(cfg)
    finally:
        model_mod.hyper_conn_init_func = original
    return gpt.to(device)


def run_variant(hc_type, device):
    new = build_gpt(hc_type, False, device)
    ref = build_gpt(hc_type, True, device)

    # ---- identical init (also proves RNG consumption order is unchanged) ----
    # `group_pre_bias` is intentionally initialised differently from the reference now
    # (see tests/test_init_policy.py); everything else must match, and the reference then
    # takes this model's weights so the comparisons below are weight-for-weight.
    sd_new, sd_ref = new.state_dict(), ref.state_dict()
    assert list(sd_new.keys()) == list(sd_ref.keys()), f"{hc_type}: state_dict keys differ"
    compared = [k for k in sd_new if not k.endswith("group_pre_bias")]
    worst_init = max(scaled_diff(sd_new[k], sd_ref[k]) for k in compared)
    assert worst_init <= TOL, f"{hc_type}: init differs, scaled diff {worst_init:.3e}"
    ref.load_state_dict(sd_new)

    # ---- fixed batches ----
    torch.manual_seed(7)
    batches = [
        (torch.randint(64, (2, 16), device=device), torch.randint(64, (2, 16), device=device))
        for _ in range(STEPS)
    ]

    # ---- forward / backward on the first batch ----
    x, y = batches[0]
    logits_new, loss_new = new(x, y)
    logits_ref, loss_ref = ref(x, y)
    d_logits = scaled_diff(logits_new, logits_ref)
    d_loss = scaled_diff(loss_new, loss_ref)
    assert d_logits <= TOL, f"{hc_type}: logits differ {d_logits:.3e}"
    assert d_loss <= TOL, f"{hc_type}: loss differs {d_loss:.3e}"

    loss_new.backward()
    loss_ref.backward()
    grads_new = {n: p.grad for n, p in new.named_parameters() if p.grad is not None}
    grads_ref = {n: p.grad for n, p in ref.named_parameters() if p.grad is not None}
    assert grads_new.keys() == grads_ref.keys(), f"{hc_type}: grad key sets differ"
    worst_grad_name, worst_grad = max(
        ((n, scaled_diff(grads_new[n], grads_ref[n])) for n in grads_new), key=lambda kv: kv[1]
    )
    assert worst_grad <= TOL, f"{hc_type}: grad {worst_grad_name} differs {worst_grad:.3e}"

    # ---- short optimisation trajectory on the same batch sequence ----
    opt_new = new.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    opt_ref = ref.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    traj = []
    for x, y in batches:
        for m, opt, store in ((new, opt_new, "new"), (ref, opt_ref, "ref")):
            opt.zero_grad(set_to_none=True)
            _, loss = m(x, y)
            loss.backward()
            opt.step()
            if store == "new":
                ln = loss.item()
            else:
                lr_ = loss.item()
        traj.append((ln, lr_))
    worst_traj = max(abs(a - b) for a, b in traj)
    traj_scale = max(1.0, max(abs(b) for _, b in traj))
    assert worst_traj / traj_scale <= TOL, (
        f"{hc_type}: loss trajectory diverges by {worst_traj:.3e} (scaled {worst_traj / traj_scale:.3e})\n{traj}"
    )

    print(f"  [{device}] {hc_type}: init {worst_init:.2e}, logits {d_logits:.2e}, "
          f"loss {d_loss:.2e}, worst grad ({worst_grad_name}) {worst_grad:.2e}, "
          f"{STEPS}-step traj {worst_traj:.2e} abs / {worst_traj / traj_scale:.2e} scaled")


def main():
    devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
    for device in devices:
        for hc_type in REF_FACTORY:
            run_variant(hc_type, device)
    print(f"PASS: GPT-level forward/backward/{STEPS}-step training identical to the "
          f"reference (scaled diffs <= {TOL:.0e}, fp32)")


if __name__ == "__main__":
    main()
