"""Equivalence test for the packed alpha/beta gate projection.

Every kept variant is compared against the frozen pre-optimization snapshot in
``tests/reference/`` (branch ``final``, 7e4c78a):

  * parameter init, including RNG consumption order (same seed -> same weights)
  * ``state_dict`` keys / shapes (checkpoint compatibility)
  * ``width_connection`` outputs (branch_input, residuals, beta)
  * ``depth_connection`` output
  * every parameter gradient and the input gradient

Run from the repo root::

    python -m tests.test_gate_packing
"""
import random

import torch

from hyper_conn.mhc import ManifoldConstrainedHyperConnections
from hyper_conn.mhc_lite import MHCLite
from hyper_conn.mhc_group_embedding import ManifoldConstrainedHyperConnectionsGroupEmbedding
from hyper_conn.mhc_lora_residual_midnorm import ManifoldConstrainedHyperConnectionsLoRAResidualMidNorm
from hyper_conn.mhc_group_lora_midnorm import ManifoldConstrainedHyperConnectionsGroupLoRAMidNorm

from tests.reference.mhc import ManifoldConstrainedHyperConnections as RefMHC
from tests.reference.mhc_lite import MHCLite as RefMHCLite
from tests.reference.mhc_group_embedding import ManifoldConstrainedHyperConnectionsGroupEmbedding as RefGroup
from tests.reference.mhc_lora_residual_midnorm import ManifoldConstrainedHyperConnectionsLoRAResidualMidNorm as RefLoRA
from tests.reference.mhc_group_lora_midnorm import ManifoldConstrainedHyperConnectionsGroupLoRAMidNorm as RefGroupLoRA

TOL = 1e-6

VARIANTS = [
    ("mhc", ManifoldConstrainedHyperConnections, RefMHC),
    ("mhc-lite", MHCLite, RefMHCLite),
    ("mhc-group", ManifoldConstrainedHyperConnectionsGroupEmbedding, RefGroup),
    ("mhc-lora", ManifoldConstrainedHyperConnectionsLoRAResidualMidNorm, RefLoRA),
    ("mhc-group-lora", ManifoldConstrainedHyperConnectionsGroupLoRAMidNorm, RefGroupLoRA),
]

CASES = [
    dict(streams=4, dim=64, num_fracs=1),
    dict(streams=2, dim=48, num_fracs=1),
    dict(streams=8, dim=64, num_fracs=1),
    dict(streams=4, dim=64, num_fracs=1, add_branch_out_to_residual=False),
]
# note: num_fracs > 1 is not covered because it is already broken upstream --
# `self.norm` is sized dim*streams*num_fracs while `normed` is only streams*(dim//num_fracs),
# so the reference implementation raises before reaching the gate projection.
# every experiment runs num_fracs=1.


def scaled_diff(a, b):
    """max |a - b| normalised by the reference magnitude (>= 1 to stay absolute for small values)."""
    diff = (a.detach().float() - b.detach().float()).abs().max().item()
    scale = max(1.0, b.detach().float().abs().max().item())
    return diff / scale


def check(name, a, b, report):
    d = scaled_diff(a, b)
    report.append((name, d))
    assert d <= TOL, f"{name}: scaled diff {d:.3e} > {TOL:.0e}"


def build(klass, seed, case):
    kw = dict(case)
    streams = kw.pop("streams")
    dim = kw.pop("dim")
    # seed both python random (layer_index = randrange) and torch (param init)
    random.seed(seed)
    torch.manual_seed(seed)
    return klass(streams, dim=dim, **kw)


def run_case(label, klass, ref_klass, case, device, seed=1234):
    new = build(klass, seed, case).to(device).eval()
    ref = build(ref_klass, seed, case).to(device).eval()

    report = []

    # ---- structural: same keys, same shapes, same init values ----
    sd_new, sd_ref = new.state_dict(), ref.state_dict()
    assert list(sd_new.keys()) == list(sd_ref.keys()), (
        f"{label}: state_dict keys differ\n{list(sd_new.keys())}\n{list(sd_ref.keys())}"
    )
    for k in sd_new:
        assert sd_new[k].shape == sd_ref[k].shape, f"{label}: shape mismatch for {k}"
        check(f"init:{k}", sd_new[k], sd_ref[k], report)

    # ---- forward: width_connection ----
    streams, dim = case["streams"], case["dim"]
    batch, seq = 3, 7
    torch.manual_seed(0)
    x = torch.randn(batch * streams, seq, dim, device=device)
    x_new = x.clone().requires_grad_(True)
    x_ref = x.clone().requires_grad_(True)

    bi_new, res_new, kw_new = new.width_connection(x_new)
    bi_ref, res_ref, kw_ref = ref.width_connection(x_ref)

    check("branch_input", bi_new, bi_ref, report)
    check("residuals", res_new, res_ref, report)

    beta_new, beta_ref = kw_new["beta"], kw_ref["beta"]
    assert (beta_new is None) == (beta_ref is None), f"{label}: beta presence differs"

    if beta_new is not None:
        check("beta", beta_new, beta_ref, report)

        torch.manual_seed(1)
        branch_out = torch.randn_like(bi_new)
        out_new = new.depth_connection(branch_out, res_new, beta=beta_new)
        out_ref = ref.depth_connection(branch_out, res_ref, beta=beta_ref)
        check("depth_out", out_new, out_ref, report)
        loss_new, loss_ref = out_new.square().mean(), out_ref.square().mean()
    else:
        loss_new = bi_new.square().mean() + res_new.square().mean()
        loss_ref = bi_ref.square().mean() + res_ref.square().mean()

    # ---- backward: every parameter grad + input grad ----
    loss_new.backward()
    loss_ref.backward()

    grads_new = {n: p.grad for n, p in new.named_parameters()}
    grads_ref = {n: p.grad for n, p in ref.named_parameters()}
    assert grads_new.keys() == grads_ref.keys()
    for n in grads_new:
        gn, gr = grads_new[n], grads_ref[n]
        assert (gn is None) == (gr is None), f"{label}: grad presence differs for {n}"
        if gn is not None:
            check(f"grad:{n}", gn, gr, report)
    check("grad:input", x_new.grad, x_ref.grad, report)

    worst = max(report, key=lambda kv: kv[1])
    print(f"  [{device}] {label} {case}: {len(report)} checks, worst {worst[0]} = {worst[1]:.3e}")


def main():
    devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
    for device in devices:
        for label, klass, ref_klass in VARIANTS:
            for case in CASES:
                run_case(label, klass, ref_klass, case, device)
    print("PASS: packed gate projection is equivalent to the reference "
          f"(all scaled diffs <= {TOL:.0e}, fp32)")


if __name__ == "__main__":
    main()
