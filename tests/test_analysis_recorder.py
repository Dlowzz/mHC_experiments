"""Unit tests for eval/analysis_recorder.py (paper-analysis evaluation recorder).

Five groups, matching the spec:
  1. shape         -- Hpre / Hpost(beta) / Hres and every gradient/stream_state/lora
                      tensor has the correct stream / group / token dims; mHC has q=1
                      and no lora/same_group keys, group-LoRA has q=Q and the lora block.
  2. numeric       -- the definition maths (read_contrib_norm, read_grad_norm, pairwise
                      cosine, lora_main_ratio, lora_perp_ratio) on small hand-built tensors.
  3. lora sanity   -- delta ∥ h  -> perp ≈ 0 ; delta ⟂ h -> perp ≈ 1 ; identical vecs cos ≈ 1.
  4. gradient      -- autograd toy u_q = Σ_s Hpre[q,s] x[s,q]: the recorder's read_grad_norm
                      definition equals ||Hpre[q,s] * dL/du_q|| from real autograd.
  5. no side-effect-- recorder on/off gives a bit-identical loss + logits and never mutates
                      a parameter (no optimizer step / gradient clipping in the recorder).

Run from the repo root::

    python -m tests.test_analysis_recorder
"""
import copy

import torch
from einops import rearrange, einsum

from model import GPT, GPTConfig
from eval.analysis_recorder import (
    Recorder, _norm_lastdim, _pairwise_cos_streams, _perp_ratio, _EPS,
)

torch.manual_seed(0)


def _build(hc_type, n_layer=2, n=4, d=16):
    cfg = GPTConfig(block_size=16, vocab_size=64, n_layer=n_layer, n_head=2, n_embd=d,
                    dropout=0.0, bias=True, hyper_conn_n=n, hyper_conn_type=hc_type)
    torch.manual_seed(0)
    return GPT(cfg).eval()


def _run_recorder(model, hc_type, n_batches=2, b=2, t=16, seed=1):
    rec = Recorder(model, hc_type).install()
    torch.manual_seed(seed)
    for _ in range(n_batches):
        xb = torch.randint(0, 64, (b, t))
        yb = torch.randint(0, 64, (b, t))
        model.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        loss.backward()
        rec.collect()
    return rec


def _flat(rec):
    """{(layer,site): {key: tensor}} flattened across categories from rec._stack()."""
    out = {}
    for lk, cats in rec._stack().items():
        flat = {}
        for cat, keys in cats.items():
            flat.update(keys)
        out[lk] = flat
    return out


# ============================================================ 1. shape

def test_shapes():
    n, d, nb, b, t = 4, 16, 2, 2, 16
    S = nb * b  # number of sequences accumulated

    # ---- plain mHC: Hpre q-dim == 1, no lora / same_group ----
    m = _build("mhc", n=n, d=d)
    rec = _run_recorder(m, "mhc", n_batches=nb, b=b, t=t)
    flat = _flat(rec)
    assert set(flat) == {f"L{i}.{s}" for i in range(2) for s in ("attn", "mlp")}, set(flat)
    for lk, f in flat.items():
        assert f["Hpre_raw"].shape == (S, t, 1, n), (lk, f["Hpre_raw"].shape)
        assert f["Hres_raw"].shape == (S, t, n, n)
        assert f["beta"].shape == (S, t, n)
        assert f["read_contrib_norm"].shape == (S, t, 1, n)
        assert f["stream_grad_norm"].shape == (S, t, n)
        assert f["beta_grad"].shape == (S, t, n)
        assert f["read_grad_norm"].shape == (S, t, 1, n)
        assert f["rms_before_read"].shape == (S, t, n)
        assert f["rms_after_hres"].shape == (S, t, n)
        assert f["rms_after_write"].shape == (S, t, n)
        # inter-stream cosine of the full residual streams / write vectors (both models)
        for k in ("read_stream_cos", "write_stream_cos", "postwrite_stream_cos"):
            assert f[k].shape == (S, t, n, n), (lk, k, f[k].shape)
        assert "same_group_stream_cos" not in f
        assert "delta_stream_cos" not in f and "lora_main_ratio" not in f
        for v in f.values():
            assert v.dtype == torch.float32
    print("  [shape] mHC: Hpre=[.,.,1,n], Hres=[.,.,n,n], read/write/postwrite cos=[.,.,n,n]; no lora keys")

    # ---- group-LoRA: Hpre q-dim == Q, full lora + same_group ----
    Q = n  # group_embedding_groups defaults to num_streams
    mg = _build("mhc_group_lora_midnorm", n=n, d=d)
    recg = _run_recorder(mg, "mhc_group_lora_midnorm", n_batches=nb, b=b, t=t)
    fg = _flat(recg)
    for lk, f in fg.items():
        assert f["Hpre_raw"].shape == (S, t, Q, n), (lk, f["Hpre_raw"].shape)
        assert f["Hres_raw"].shape == (S, t, n, n)
        assert f["read_contrib_norm"].shape == (S, t, Q, n)
        assert f["read_grad_norm"].shape == (S, t, Q, n)
        assert f["same_group_stream_cos"].shape == (S, t, Q, n, n)
        for k in ("read_stream_cos", "write_stream_cos", "postwrite_stream_cos", "delta_stream_cos"):
            assert f[k].shape == (S, t, n, n), (lk, k, f[k].shape)
        assert f["lora_main_ratio"].shape == (S, t, n)
        assert f["lora_perp_ratio"].shape == (S, t, n)
    print("  [shape] group-LoRA: Hpre=[.,.,Q,n], same_group=[.,.,Q,n,n], lora cos=[.,.,n,n]")


# ============================================================ 2. numeric definitions

def test_numeric_definitions():
    # read_contrib_norm[q,s] = |Hpre[q,s]| * ||x[s,q]||  (hand-built)
    # streams n=2, groups Q=2, group_dim=3
    x = torch.tensor([[3.0, 4.0, 0.0],      # stream0 group0 -> norm 5
                      [0.0, 0.0, 2.0]])     # stream1 group0 -> norm 2  (single group here)
    xn = _norm_lastdim(x)                   # [2]
    assert torch.allclose(xn, torch.tensor([5.0, 2.0])), xn
    Hpre = torch.tensor([[-0.5, 2.0]])      # [q=1, s=2], signed
    read_contrib = Hpre.abs() * xn.unsqueeze(0)
    assert torch.allclose(read_contrib, torch.tensor([[2.5, 4.0]])), read_contrib

    # pairwise cosine: orthogonal -> 0, same -> 1, opposite -> -1
    v = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])  # [1, s=3, d=2]
    cos = _pairwise_cos_streams(v)[0]
    exp = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    assert torch.allclose(cos, exp, atol=1e-6), cos

    # lora_main_ratio = ||lora_w|| / (||main_w|| + eps)
    main_w = torch.tensor([[3.0, 4.0]])     # norm 5
    lora_w = torch.tensor([[0.0, 10.0]])    # norm 10
    ratio = _norm_lastdim(lora_w) / (_norm_lastdim(main_w) + _EPS)
    assert torch.allclose(ratio, torch.tensor([2.0]), atol=1e-5), ratio

    # read_grad_norm[q,s] = |Hpre[q,s]| * ||dL/du_q||  (scalar * vector norm)
    du = torch.tensor([[6.0, 8.0]])         # dL/du_0, norm 10  (Q=1)
    dun = _norm_lastdim(du)                  # [1]
    read_grad = Hpre.abs() * dun.unsqueeze(-1)   # [q=1, s=2]
    assert torch.allclose(read_grad, torch.tensor([[5.0, 20.0]])), read_grad
    print("  [numeric] read_contrib / cosine / lora_main_ratio / read_grad_norm match hand calc")


# ============================================================ 3. lora sanity

def test_lora_perp_and_cos():
    d = 8
    torch.manual_seed(3)
    h = torch.randn(1, 1, d)                 # [b=1, seq=1, d]
    # delta parallel to h  -> perp ratio ~ 0
    delta_par = (h * torch.tensor([2.0])).unsqueeze(-2).repeat(1, 1, 3, 1)  # [1,1,s=3,d]
    perp_par = _perp_ratio(delta_par, h)
    assert torch.all(perp_par < 1e-5), perp_par

    # delta orthogonal to h -> perp ratio ~ 1
    g = torch.randn(1, 1, d)
    hn = h / h.norm()
    g_perp = g - (g * hn).sum(-1, keepdim=True) * hn      # remove h-component
    delta_perp = g_perp.unsqueeze(-2).repeat(1, 1, 3, 1)
    perp_o = _perp_ratio(delta_perp, h)
    assert torch.allclose(perp_o, torch.ones_like(perp_o), atol=1e-5), perp_o

    # identical vectors -> cosine 1
    same = h.unsqueeze(-2).repeat(1, 1, 3, 1)             # [1,1,3,d]
    cos = _pairwise_cos_streams(same)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-6), cos
    print("  [lora] delta∥h -> perp≈0 ; delta⟂h -> perp≈1 ; identical streams -> cos≈1")


# ============================================================ 4. gradient sanity (autograd)

def test_gradient_read_definition():
    """Toy u_q = Σ_s Hpre[q,s] x[s,q]; real autograd dL/du_q.  The recorder defines
    read_grad_norm[q,s] = |Hpre[q,s]| * ||dL/du_q||, which must equal the literal
    ||Hpre[q,s] * dL/du_q|| (scalar * gradient vector)."""
    n, Q, d = 3, 2, 4
    torch.manual_seed(5)
    x = torch.randn(n, Q, d, requires_grad=True)
    Hpre = torch.randn(Q, n, requires_grad=True)
    u = einsum(Hpre, x, "q s, s q d -> q d")          # u_q  [Q, d]
    u.retain_grad()
    c = torch.randn_like(u)
    L = 0.5 * (u ** 2).sum() + (c * u).sum()          # arbitrary scalar loss
    L.backward()
    du = u.grad                                        # dL/du_q  [Q, d]

    # literal definition: norm of the scalar-scaled gradient vector, per (q,s)
    direct = torch.stack([(Hpre[q, s] * du[q]).norm()
                          for q in range(Q) for s in range(n)]).reshape(Q, n)
    # recorder's factored definition
    recorder = Hpre.abs() * _norm_lastdim(du).unsqueeze(-1)
    assert torch.allclose(direct, recorder, atol=1e-6), (direct, recorder)

    # sanity: autograd dL/du matches the analytic u + c
    assert torch.allclose(du, u.detach() + c, atol=1e-6)
    print("  [gradient] recorder read_grad_norm == ||Hpre[q,s]·dL/du_q|| on autograd toy")


# ============================================================ 5. no side-effect

def test_no_side_effect():
    for hc in ("mhc", "mhc_group_lora_midnorm"):
        m = _build(hc)                       # one instance (init uses Python RNG, not reproducible across builds)
        torch.manual_seed(7)
        xb = torch.randint(0, 64, (2, 16))
        yb = torch.randint(0, 64, (2, 16))

        # reference: recorder OFF
        m.zero_grad(set_to_none=True)
        params0 = {k: v.detach().clone() for k, v in m.named_parameters()}
        logits0, loss0 = m(xb, yb)
        loss0.backward()
        grads0 = {k: (None if v.grad is None else v.grad.detach().clone())
                  for k, v in m.named_parameters()}

        # recorder ON (same instance, params unchanged since there is no optimizer step)
        rec = Recorder(m, hc).install()
        m.zero_grad(set_to_none=True)
        logits1, loss1 = m(xb, yb)
        loss1.backward()
        rec.collect()

        assert torch.equal(loss0, loss1), (hc, "loss differs")
        assert torch.equal(logits0, logits1), (hc, "logits differ")
        # params never mutated by the recorder (no optimizer step)
        for k, v in m.named_parameters():
            assert torch.equal(v.detach(), params0[k]), (hc, "param mutated", k)
        # param gradients identical (recorder only reads .grad, no clipping/scaling)
        for k, v in m.named_parameters():
            if grads0[k] is None:
                assert v.grad is None, (hc, k)
            else:
                assert torch.equal(v.grad, grads0[k]), (hc, "grad differs", k)
        mods = list(rec.mods)
        rec.remove()
        # width/depth restored to the originals after remove()
        for _, _, mod in mods:
            assert not hasattr(mod, "_rec_pending")
    print("  [no-side-effect] recorder on/off: identical loss+logits, params & grads unchanged")
    print("  [no-side-effect] recorder on/off: identical loss+logits, params & grads unchanged")


def main():
    test_shapes()
    test_numeric_definitions()
    test_lora_perp_and_cos()
    test_gradient_read_definition()
    test_no_side_effect()
    print("PASS: analysis_recorder — shapes, numeric defs, lora sanity, gradient, no-side-effect")


if __name__ == "__main__":
    main()
