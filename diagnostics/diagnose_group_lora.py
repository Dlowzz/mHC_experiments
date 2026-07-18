"""
Diagnose the grad-norm spike in the L + mhc_group_lora run (Experiments A + B).

A. Checkpoint weight forensics (no data):
   per layer x {attn,mlp}: ||A_s||_F, ||B_s||_F, ||A_s B_s||_2 (composed LoRA gain),
   ||group_pre_weight||_F, |group_pre_bias|max, pre_branch_scale, h_post_scale, NaN/Inf.

B. Forward+backward probe over N val batches (single GPU):
   per hyper-conn module: max_q Sum_s Hpre, %gate>0.95, RMS(residual_in / branch_input /
   LoRA delta / beta*h), r_LoRA = RMS(delta)/RMS(beta*h), r_res = RMS(delta)/RMS(X);
   per batch: 6-way pre-clip grad-norm decomposition + total grad-norm distribution;
   dumps the worst-k batches. Also scans train.bin near the ~0.53B-token spike offset.

Read-only: does NOT modify repo files (all instrumentation is runtime monkey-patching
on the loaded model instance). Run from the mhc-lite repo root, e.g.:

    CUDA_VISIBLE_DEVICES=1 venv/bin/python diagnostics/diagnose_group_lora.py \
        --ckpt out-owt-large-mhc-group-lora/ckpt.pt --n_batches 64
"""

import os
import sys
import json
import math
import argparse
from contextlib import nullcontext

import numpy as np
import torch
from einops import einsum

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from model import GPT, GPTConfig  # noqa: E402


def rms(t):
    return t.detach().float().pow(2).mean().sqrt().item()


def fnum(x):
    return float(x.detach().float().reshape(-1)[0]) if x.numel() == 1 else float(x.detach().float().mean())


# ----------------------------------------------------------------------------- load

def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    conf = GPTConfig(**ck["model_args"])
    model = GPT(conf)
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in ck["model"].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    # tied lm_head.weight may be reported missing (shares wte); tolerate that only
    assert all("lm_head" in m for m in missing), f"missing keys: {missing[:5]}"
    model.eval().to(device)
    return model, ck, conf


def hc_modules(model):
    """Return list of (layer_index, kind, module) for every group-lora hyper-conn."""
    mods = []
    for i, blk in enumerate(model.transformer.h):
        mods.append((i, "attn", blk.hc_attn))
        mods.append((i, "mlp", blk.hc_mlp))
    return mods


# ----------------------------------------------------------------------------- Part A

def part_A(model):
    rows = []
    bad = []
    for i, kind, hc in hc_modules(model):
        A = hc.stream_down_weight.detach().float()   # [s, C, r]
        B = hc.stream_up_weight.detach().float()      # [s, r, C]
        AB = torch.bmm(A, B)                          # [s, C, C]
        specs = torch.linalg.matrix_norm(AB, ord=2)   # [s]
        gpw = hc.group_pre_weight.detach().float()
        gpb = hc.group_pre_bias.detach().float()
        row = dict(
            layer=i, kind=kind,
            A_fro=A.norm().item(), B_fro=B.norm().item(),
            AB_spec_max=specs.max().item(), AB_spec_mean=specs.mean().item(),
            gpw_fro=gpw.norm().item(), gpb_absmax=gpb.abs().max().item(),
            pre_scale=fnum(hc.pre_branch_scale), h_post_scale=fnum(hc.h_post_scale),
        )
        rows.append(row)
        for name, p in [("A", A), ("B", B), ("gpw", gpw)]:
            if not torch.isfinite(p).all():
                bad.append(f"L{i}.{kind}.{name} has NaN/Inf")
    return rows, bad


# ----------------------------------------------------------------------------- Part B instrumentation

def install_probes(model):
    for _, _, hc in hc_modules(model):
        hc._diag = {}

        orig_gate = hc._compute_group_pre_gate

        def gate(normed, _o=orig_gate, _hc=hc):
            H = _o(normed)  # b ... f q s
            with torch.no_grad():
                rowsum = H.sum(dim=-1)                       # Sum_s Hpre_{q,s}
                _hc._diag["hpre_rowsum_max"] = rowsum.max().item()
                _hc._diag["hpre_rowsum_mean"] = rowsum.mean().item()
                _hc._diag["gate_max"] = H.max().item()
                _hc._diag["gate_gt095"] = (H > 0.95).float().mean().item()
            return H
        hc._compute_group_pre_gate = gate

        orig_width = hc.width_connection

        def width(residuals, _o=orig_width, _hc=hc):
            with torch.no_grad():
                _hc._diag["resid_in_rms"] = rms(residuals)
            out = _o(residuals)
            with torch.no_grad():
                _hc._diag["branch_input_rms"] = rms(out[0])
            return out
        hc.width_connection = width

        # capture the REAL delta (grad-safe): wrapping compute_lora, NOT recomputing it.
        # (recomputing compute_lora under no_grad inside autocast poisons the bf16 weight
        #  cache and silently zeroes LoRA grads.)
        orig_lora = hc.compute_lora

        def lora(bo, _o=orig_lora, _hc=hc):
            delta = _o(bo)
            with torch.no_grad():
                _hc._diag["delta_rms"] = rms(delta)
            return delta
        hc.compute_lora = lora

        orig_depth = hc.depth_connection

        def depth(branch_output, residuals, *, beta, _o=orig_depth, _hc=hc):
            # beta*h uses NO parameters (only activations) -> safe to recompute under no_grad
            with torch.no_grad():
                h = _hc.split_fracs(branch_output.detach())
                betah = einsum(h, beta.detach(), "b ... f1 d, b ... f1 s f2 -> b ... f2 s d")
                _hc._diag["betah_rms"] = rms(betah)
            out = _o(branch_output, residuals, beta=beta)  # real path sets delta_rms via wrapped compute_lora
            br = _hc._diag.get("betah_rms", 0.0)
            dr = _hc._diag.get("delta_rms", 0.0)
            _hc._diag["r_lora"] = dr / (br + 1e-8)
            _hc._diag["r_res"] = dr / (_hc._diag.get("resid_in_rms", 0.0) + 1e-8)
            return out
        hc.depth_connection = depth


def bucket(name):
    if ".stream_down_weight" in name:
        return "lora_A(down)"
    if ".stream_up_weight" in name:
        return "lora_B(up)"
    if ".group_pre_weight" in name:
        return "group_pre_weight"
    if ".group_pre_bias" in name:
        return "group_pre_bias"
    if "branch_attn." in name or "branch_mlp." in name:
        return "attn_ffn"
    if name.startswith(("transformer.wte", "transformer.wpe", "lm_head", "transformer.ln_f")):
        return "embed_head"
    return "mhc_other"


def grad_decomp(model):
    buckets = {}
    total_sq = 0.0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g2 = p.grad.detach().float().pow(2).sum().item()
        total_sq += g2
        buckets[bucket(n)] = buckets.get(bucket(n), 0.0) + g2
    out = {k: math.sqrt(v) for k, v in buckets.items()}
    out["TOTAL"] = math.sqrt(total_sq)
    return out


def get_batch(data, bs, block, device, start=None):
    if start is None:
        ix = torch.randint(len(data) - block - 1, (bs,))
    else:
        ix = torch.tensor([start + j * block for j in range(bs)])
    x = torch.stack([torch.from_numpy(data[i:i + block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def part_B(model, conf, device, ctx, n_batches, bs, spike_token):
    block = conf.block_size
    train = np.memmap(os.path.join(REPO, "data/openwebtext/train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(REPO, "data/openwebtext/val.bin"), dtype=np.uint16, mode="r")

    install_probes(model)
    mods = hc_modules(model)

    per_layer = {(i, k): {} for i, k, _ in mods}
    per_batch = []          # list of dict(total_gn, decomp, snapshot)

    def run_one(x, y, tag):
        model.zero_grad(set_to_none=True)
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        dec = grad_decomp(model)
        snap = {f"L{i}.{k}": dict(hc._diag) for i, k, hc in mods}
        per_batch.append(dict(tag=tag, loss=loss.item(), total_gn=dec["TOTAL"], decomp=dec, snap=snap))
        for i, k, hc in mods:
            for kk, vv in hc._diag.items():
                per_layer[(i, k)].setdefault(kk, []).append(vv)

    # random val batches
    for b in range(n_batches):
        x, y = get_batch(val, bs, block, device)
        run_one(x, y, "val")

    # batches drawn from the spike token region of train.bin
    spike_stats = {}
    if spike_token is not None and spike_token < len(train):
        seg = np.asarray(train[spike_token: spike_token + bs * block])
        spike_stats = dict(
            max_token_id=int(seg.max()), min_token_id=int(seg.min()),
            unique_ratio=float(len(np.unique(seg)) / seg.size),
            top1_token_frac=float(np.bincount(seg).max() / seg.size),
        )
        for b in range(8):
            x, y = get_batch(train, bs, block, device, start=spike_token + b * bs * block)
            run_one(x, y, "spike_region")

    # aggregate per-layer means
    layer_summary = []
    for (i, k), d in per_layer.items():
        layer_summary.append(dict(
            layer=i, kind=k,
            hpre_rowsum_max=float(np.mean(d.get("hpre_rowsum_max", [0]))),
            gate_gt095=float(np.mean(d.get("gate_gt095", [0]))),
            resid_in_rms=float(np.mean(d.get("resid_in_rms", [0]))),
            branch_input_rms=float(np.mean(d.get("branch_input_rms", [0]))),
            betah_rms=float(np.mean(d.get("betah_rms", [0]))),
            delta_rms=float(np.mean(d.get("delta_rms", [0]))),
            r_lora=float(np.mean(d.get("r_lora", [0]))),
            r_res=float(np.mean(d.get("r_res", [0]))),
        ))

    gns = np.array([pb["total_gn"] for pb in per_batch if pb["tag"] == "val"])
    worst = sorted(per_batch, key=lambda pb: pb["total_gn"], reverse=True)[:5]
    return layer_summary, per_batch, gns, worst, spike_stats


# ----------------------------------------------------------------------------- report

def fmt_top(rows, key, n=5, rev=True):
    s = sorted(rows, key=lambda r: r[key], reverse=rev)[:n]
    return " | ".join(f"L{r['layer']}.{r['kind']}={r[key]:.3g}" for r in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out-owt-large-mhc-group-lora/ckpt.pt")
    ap.add_argument("--n_batches", type=int, default=64)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--spike_token", type=int, default=530_000_000)  # ~iter 8090 tokens-seen
    ap.add_argument("--out", default="diagnostics/reports/group_lora_diag.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ptdtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    model, ck, conf = load_model(os.path.join(REPO, args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt, device)
    print(f"loaded ckpt iter_num={ck.get('iter_num')} best_val_loss={float(ck.get('best_val_loss',-1)):.4f} "
          f"| type={conf.hyper_conn_type} n_layer={conf.n_layer} n_embd={conf.n_embd} device={device} dtype={ptdtype}")

    # ---- Part A ----
    A_rows, bad = part_A(model)
    print("\n===== PART A: checkpoint weight forensics =====")
    print("NaN/Inf:", bad if bad else "none")
    print("top ||A_sB_s||_2 (composed LoRA gain):     ", fmt_top(A_rows, "AB_spec_max"))
    print("top ||group_pre_weight||_F:                ", fmt_top(A_rows, "gpw_fro"))
    print("top pre_branch_scale:                      ", fmt_top(A_rows, "pre_scale"))
    print("top |group_pre_bias|max:                   ", fmt_top(A_rows, "gpb_absmax"))
    print("top ||B_s||_F (LoRA up, was zero-init):    ", fmt_top(A_rows, "B_fro"))

    # ---- Part B ----
    print("\n===== PART B: forward/backward probe =====")
    layer_sum, per_batch, gns, worst, spike_stats = part_B(
        model, conf, device, ctx, args.n_batches, args.bs, args.spike_token)
    print(f"val grad-norm (single micro-batch, pre-clip): mean={gns.mean():.3g} "
          f"p50={np.percentile(gns,50):.3g} p95={np.percentile(gns,95):.3g} max={gns.max():.3g}")
    print("top r_LoRA = RMS(delta)/RMS(beta*h):        ", fmt_top(layer_sum, "r_lora"))
    print("top max_q Sum_s Hpre (read gain):           ", fmt_top(layer_sum, "hpre_rowsum_max"))
    print("top branch_input RMS:                       ", fmt_top(layer_sum, "branch_input_rms"))
    print("top delta(LoRA) RMS:                        ", fmt_top(layer_sum, "delta_rms"))
    print("top gate>0.95 fraction:                     ", fmt_top(layer_sum, "gate_gt095"))
    print("\nmean 6-way grad decomposition over val batches:")
    agg = {}
    for pb in per_batch:
        if pb["tag"] != "val":
            continue
        for k, v in pb["decomp"].items():
            agg.setdefault(k, []).append(v)
    for k in sorted(agg, key=lambda k: -np.mean(agg[k])):
        print(f"    {k:18s} {np.mean(agg[k]):.4g}")
    print("\nworst-5 val batches (total grad norm):")
    for pb in worst:
        rl = max((d.get("r_lora", 0) for d in pb["snap"].values()))
        bi = max((d.get("branch_input_rms", 0) for d in pb["snap"].values()))
        print(f"    gn={pb['total_gn']:.3g} loss={pb['loss']:.3f} maxR_lora={rl:.3g} maxBranchInputRMS={bi:.3g}")
    print("\nspike-region train.bin token stats:", spike_stats)

    os.makedirs(os.path.dirname(os.path.join(REPO, args.out)), exist_ok=True)
    with open(os.path.join(REPO, args.out), "w") as f:
        json.dump(dict(ckpt_iter=ck.get("iter_num"), part_A=A_rows, bad=bad,
                       layer_summary=layer_sum,
                       grad_decomp_mean={k: float(np.mean(v)) for k, v in agg.items()},
                       val_gradnorm=dict(mean=float(gns.mean()), p95=float(np.percentile(gns, 95)),
                                         max=float(gns.max())),
                       worst=[{k: pb[k] for k in ("tag", "loss", "total_gn", "decomp")} for pb in worst],
                       spike_stats=spike_stats), f, indent=2)
    print(f"\nsaved full report -> {args.out}")


if __name__ == "__main__":
    main()
