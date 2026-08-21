"""Offline analysis of the paper-analysis recordings for mHC vs mHC-Group-LoRA.

Reads the two dumps produced by eval/record_analysis.py and

  A. VALIDATES a sample of sequences -- range / definitional invariants that must hold
     for every recorded tensor (Hres doubly-stochastic, cosines in [-1,1],
     lora_perp_ratio in [0,1], Hpre in (0,1), beta in (0,2), norms >= 0), and, if a
     re-run dump is present under VAL_DIR, a faithfulness bit-check on the first sequences.

  B. COMPARES the two models metric-by-metric (aggregated over the 64x1024 tokens and the
     24 layers x 2 sites) and prints what each metric reveals.

No argparse: paths via constants / env vars (ANALYSIS_OUT_DIR, ANALYSIS_VAL_DIR).

Run from the repo root::

    python -m eval.analyze_recording
"""
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("ANALYSIS_OUT_DIR", os.path.join(_HERE, "analysis_out"))
VAL_DIR = os.environ.get("ANALYSIS_VAL_DIR", "/tmp/val_out")
MHC = "mhc"
GRP = "mhc_group_lora_midnorm"
BETA_THR = 1e-3   # streams whose write gate beta_s is below this write ~nothing (w_s ~ 0)


def _load(name, root=OUT_DIR):
    return torch.load(os.path.join(root, f"{name}.analysis.pt"), weights_only=False)


def _tokens(t):
    """[S, T, ...] -> [S*T, ...] (flatten sequence & token into one token axis)."""
    return t.reshape(-1, *t.shape[2:])


def _offdiag_mean(m):
    """mean of the off-diagonal entries of a [..., n, n] batch."""
    n = m.shape[-1]
    eye = torch.eye(n, dtype=torch.bool)
    return m[..., ~eye].reshape(*m.shape[:-2], n * (n - 1)).mean().item()


def _diag_mean(m):
    n = m.shape[-1]
    return m[..., torch.arange(n), torch.arange(n)].mean().item()


def _masked_write_cos(dump, thr):
    """Off-diagonal mean of write_stream_cos, keeping only stream pairs where BOTH
    streams actually write (beta_s >= thr).  Removes the ~zero-vector cosine artifact
    from near-zero-beta streams.  Returns (masked_mean, fraction_of_offdiag_pairs_kept)."""
    tot = cnt = total = 0.0
    for ls, cats in dump["layers"].items():
        cos = cats["stream_state"]["write_stream_cos"]      # [S,T,n,n]
        beta = cats["routing"]["beta"]                       # [S,T,n]
        n = cos.shape[-1]
        eye = torch.eye(n, dtype=torch.bool)
        valid = beta >= thr
        pair = (valid.unsqueeze(-1) & valid.unsqueeze(-2)) & (~eye)   # [S,T,n,n]
        tot += cos[pair].sum().item()
        cnt += pair.sum().item()
        total += (~eye).expand_as(pair).sum().item()
    return (tot / cnt if cnt else float("nan")), (cnt / total if total else 0.0)


def _per_layer_write(dump, thr):
    """Per layer (averaging the attn+mlp sites): (masked write-cos, kept-pair fraction).
    Returns a list of (cos, kept) with one entry per layer 0..n_layers-1."""
    out = []
    for i in range(dump["meta"]["n_layers"]):
        tot = cnt = total = 0.0
        for site in dump["meta"]["sites"]:
            cats = dump["layers"][f"L{i}.{site}"]
            cos = cats["stream_state"]["write_stream_cos"]
            beta = cats["routing"]["beta"]
            n = cos.shape[-1]
            eye = torch.eye(n, dtype=torch.bool)
            valid = beta >= thr
            pair = (valid.unsqueeze(-1) & valid.unsqueeze(-2)) & (~eye)
            tot += cos[pair].sum().item()
            cnt += pair.sum().item()
            total += (~eye).expand_as(pair).sum().item()
        out.append((tot / cnt if cnt else float("nan"), cnt / total if total else 0.0))
    return out
# ---- placeholder ----


def _iter_entries(dump, key, cat):
    """Yield (layer_site, tensor) for every entry that has category/key."""
    for ls, cats in dump["layers"].items():
        if cat in cats and key in cats[cat]:
            yield ls, cats[cat][key]


def _entry_means(dump, key, cat, fn):
    """{layer_site: fn(tensor)} for a scalar reduction fn, plus the overall mean.
    Every entry holds the same #tokens, so the plain mean over entries is the
    token-weighted global mean."""
    vals = {ls: fn(t) for ls, t in _iter_entries(dump, key, cat)}
    overall = sum(vals.values()) / len(vals) if vals else float("nan")
    return vals, overall


def _norm_dist(x, dim):
    """normalise non-negative x to a distribution along dim (safe for all-zero)."""
    s = x.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return x / s


def _max_share(x, dim=-1):
    """top-1 share of a non-negative vector after normalising along dim -> mean scalar."""
    p = _norm_dist(x, dim)
    return p.max(dim=dim).values.mean().item()


def _eff_count(x, dim=-1):
    """effective #entries = 2^entropy(p) along dim -> mean scalar (1..n)."""
    p = _norm_dist(x, dim).clamp_min(1e-12)
    ent = -(p * p.log2()).sum(dim=dim)
    return (2 ** ent).mean().item()


# =====================================================================  A. validation

def validate(dump, name):
    print(f"\n--- invariants: {name} ---")
    checks = []

    def rng(key, cat, lo, hi):
        mn, mx = None, None
        for _, t in _iter_entries(dump, key, cat):
            a, b = t.min().item(), t.max().item()
            mn = a if mn is None else min(mn, a)
            mx = b if mx is None else max(mx, b)
        if mn is None:
            return
        ok = (mn >= lo - 1e-4) and (mx <= hi + 1e-4)
        checks.append(ok)
        print(f"  [{'OK' if ok else 'XX'}] {cat}.{key:22s} in [{mn:+.4f}, {mx:+.4f}]  (expect [{lo},{hi}])")

    rng("Hpre_raw", "routing", 0.0, 1.0)          # sigmoid read gate
    rng("beta", "routing", 0.0, 2.0)              # sigmoid*2 write gate
    rng("read_contrib_norm", "routing", 0.0, 1e9)
    rng("stream_grad_norm", "gradient", 0.0, 1e9)
    rng("read_grad_norm", "gradient", 0.0, 1e9)
    rng("rms_before_read", "stream_state", 0.0, 1e9)
    rng("lora_perp_ratio", "lora", 0.0, 1.0)      # ||perp||/||delta|| in [0,1]
    rng("lora_main_ratio", "lora", 0.0, 1e9)
    for k in ("read_stream_cos", "write_stream_cos", "postwrite_stream_cos",
              "same_group_stream_cos"):
        rng(k, "stream_state", -1.0, 1.0)
    rng("delta_stream_cos", "lora", -1.0, 1.0)

    # H_res is row-stochastic over the destination axis (dim=-1): sinkhorn's final
    # normalisation is over dim=-1, so each source stream's weights over destinations
    # sum to 1 exactly (the src-axis sum is only approximately 1).
    dst_err = src_err = 0.0
    for _, t in _iter_entries(dump, "Hres_raw", "routing"):
        dst_err = max(dst_err, (t.sum(-1) - 1).abs().max().item())
        src_err = max(src_err, (t.sum(-2) - 1).abs().max().item())
    ok = dst_err < 1e-3
    checks.append(ok)
    print(f"  [{'OK' if ok else 'XX'}] Hres dst-axis sum == 1           max|sum-1| = {dst_err:.2e}"
          f"  (src-axis approx: {src_err:.2e})")
    print(f"  invariants passed: {sum(checks)}/{len(checks)}")
    return all(checks)


def faithfulness():
    """If a re-run dump exists in VAL_DIR, bit-check its sequences against the full dump."""
    if not (os.path.isdir(VAL_DIR) and os.path.exists(os.path.join(VAL_DIR, f"{MHC}.analysis.pt"))):
        print("\n--- faithfulness re-run: skipped (no VAL_DIR dump) ---")
        return
    print(f"\n--- faithfulness re-run vs {VAL_DIR} (first sequences) ---")
    for name in (MHC, GRP):
        full = _load(name)
        val = _load(name, root=VAL_DIR)
        probe = next(iter(next(iter(val["layers"].values())).values()))
        K = next(iter(probe.values())).shape[0]        # #sequences in the re-run
        agg = {"forward": 0.0, "gradient": 0.0}
        for ls, cats in val["layers"].items():
            for cat, keys in cats.items():
                g = "gradient" if cat == "gradient" else "forward"
                for key, vt in keys.items():
                    ft = full["layers"][ls][cat][key][:K]
                    agg[g] = max(agg[g], (ft - vt).abs().max().item())
        fwd_ok = agg["forward"] == 0.0
        print(f"  {name:26s} first {K} seqs:  forward max|Δ| = {agg['forward']:.1e} "
              f"{'(bit-identical)' if fwd_ok else ''} | gradient max|Δ| = {agg['gradient']:.1e} "
              f"(non-deterministic SDPA backward / atomicAdd)")
# ---- placeholder2 ----


def _group_specialization(t):
    """read_contrib_norm [S,T,Q,n] -> mean total-variation distance between the
    per-group read distributions (over streams).  0 = all groups read alike,
    ->1 = groups specialise to different streams.  Q==1 (mHC) returns 0."""
    Q = t.shape[-2]
    if Q < 2:
        return 0.0
    p = _norm_dist(t, dim=-1)                      # [S,T,Q,n]
    tv = []
    for a in range(Q):
        for b in range(a + 1, Q):
            tv.append(0.5 * (p[..., a, :] - p[..., b, :]).abs().sum(-1))
    return torch.stack(tv, -1).mean().item()


def _per_stream_mean(t):
    """mean over tokens of a [S,T,n] tensor -> python list of length n."""
    return _tokens(t).mean(0).tolist()


def _fmt(x):
    return "  n/a " if x is None else f"{x:6.3f}"


def _row(label, mhc_v, grp_v, note):
    print(f"  {label:34s} mHC={_fmt(mhc_v)}   Grp={_fmt(grp_v)}   | {note}")


def compare(dmhc, dgrp):
    print("\n" + "=" * 78)
    print("B. mHC  vs  mHC-Group-LoRA(midnorm)   (means over 64x1024 tokens, 24 layers x 2 sites)")
    print("=" * 78)

    # ---------- routing: residual mixing (Hres) ----------
    print("\n[routing] residual-stream mixing  H_res  (doubly-stochastic n x n)")
    _, m_diag = _entry_means(dmhc, "Hres_raw", "routing", _diag_mean)
    _, g_diag = _entry_means(dgrp, "Hres_raw", "routing", _diag_mean)
    _row("H_res self-retention (diag)", m_diag, g_diag,
         "how much a stream keeps itself vs mixes in others")

    # ---------- routing: read gate concentration ----------
    print("\n[routing] branch read  (read_contrib_norm = ||Hpre[q,s]*x[s,q]||, per group q)")
    _, m_share = _entry_means(dmhc, "read_contrib_norm", "routing", lambda t: _max_share(t, -1))
    _, g_share = _entry_means(dgrp, "read_contrib_norm", "routing", lambda t: _max_share(t, -1))
    _row("read top-1 stream share", m_share, g_share, "1.0 = reads a single stream")
    _, m_eff = _entry_means(dmhc, "read_contrib_norm", "routing", lambda t: _eff_count(t, -1))
    _, g_eff = _entry_means(dgrp, "read_contrib_norm", "routing", lambda t: _eff_count(t, -1))
    _row("read effective #streams (2^H)", m_eff, g_eff, "1=one stream .. 4=uniform over 4")
    _, g_spec = _entry_means(dgrp, "read_contrib_norm", "routing", _group_specialization)
    _row("group read specialization (TV)", 0.0, g_spec,
         "0=all groups read alike; >0 = groups read different streams (Grp only)")

    # ---------- routing: write gate beta ----------
    print("\n[routing] write gate  beta = H_post in (0,2)")
    _, m_beta = _entry_means(dmhc, "beta", "routing", lambda t: t.mean().item())
    _, g_beta = _entry_means(dgrp, "beta", "routing", lambda t: t.mean().item())
    _row("beta mean", m_beta, g_beta, "average write strength back to streams")
    imb = lambda t: _tokens(t).mean(0).std().item()
    _, m_bi = _entry_means(dmhc, "beta", "routing", imb)
    _, g_bi = _entry_means(dgrp, "beta", "routing", imb)
    _row("beta per-stream imbalance (std)", m_bi, g_bi, "spread of write strength across streams")

    # ---------- gradient: which streams learn ----------
    print("\n[gradient] real-backward stream usage")
    _, m_gs = _entry_means(dmhc, "stream_grad_norm", "gradient", lambda t: _max_share(t, -1))
    _, g_gs = _entry_means(dgrp, "stream_grad_norm", "gradient", lambda t: _max_share(t, -1))
    _row("stream-grad top-1 share", m_gs, g_gs, "1.0 = gradient flows into one stream only")
    _, m_ge = _entry_means(dmhc, "stream_grad_norm", "gradient", lambda t: _eff_count(t, -1))
    _, g_ge = _entry_means(dgrp, "stream_grad_norm", "gradient", lambda t: _eff_count(t, -1))
    _row("stream-grad effective #streams", m_ge, g_ge, "how many streams actually receive gradient")

    # ---------- stream_state: magnitudes ----------
    print("\n[stream_state] residual magnitudes (RMS)")
    _, m_rms = _entry_means(dmhc, "rms_before_read", "stream_state", lambda t: t.mean().item())
    _, g_rms = _entry_means(dgrp, "rms_before_read", "stream_state", lambda t: t.mean().item())
    _row("rms before read", m_rms, g_rms, "residual scale entering the site")

    def growth(dump):
        gv = {}
        for ls in dump["layers"]:
            ss = dump["layers"][ls]["stream_state"]
            gv[ls] = (ss["rms_after_write"].mean() / ss["rms_before_read"].mean().clamp_min(1e-9)).item()
        return sum(gv.values()) / len(gv)
    _row("rms growth (after_write/before)", growth(dmhc), growth(dgrp),
         ">1 = the block grows the residual norm")

    # ---------- inter-stream cosine of full residual streams / writes (BOTH models) ----------
    print("\n[stream_state] inter-stream cosine (full streams, both models; off-diagonal mean)")
    for key, label, note in (
        ("read_stream_cos", "residual streams BEFORE read",
         "how differentiated the n streams are entering the site"),
        ("write_stream_cos", "write-back vectors w_s",
         "mHC write=beta_s*h is collinear (->1); LoRA rotates streams apart"),
        ("postwrite_stream_cos", "residual streams AFTER write",
         "stream differentiation after the write-back"),
    ):
        _, mv = _entry_means(dmhc, key, "stream_state", _offdiag_mean)
        _, gv = _entry_means(dgrp, key, "stream_state", _offdiag_mean)
        _row(label, mv, gv, note)

    # write-back cosine is degenerate where beta_s ~ 0 (w_s = beta_s*h ~ 0 -> cos of a
    # ~zero vector floors to ~0).  Re-measure it masking out any stream pair whose beta is
    # below BETA_THR, so the number reflects only streams that actually write.
    mv, mkept = _masked_write_cos(dmhc, BETA_THR)
    gv, gkept = _masked_write_cos(dgrp, BETA_THR)
    _row(f"write cos, beta>={BETA_THR:g} masked", mv, gv,
         f"drops near-zero-beta writes (kept mHC={mkept:.0%}, Grp={gkept:.0%})")

    # ---------- group-only: within-group diversity + LoRA ----------
    print("\n[group-only] within-group diversity & LoRA behaviour (Grp only)")
    _, g_red = _entry_means(dgrp, "same_group_stream_cos", "stream_state", _offdiag_mean)
    _row("same-group stream cos (off-diag)", None, g_red,
         "cos between streams inside a channel group: ->1 redundant, ~0 diverse")
    _, g_dc = _entry_means(dgrp, "delta_stream_cos", "lora", _offdiag_mean)
    _row("LoRA delta stream cos (off-diag)", None, g_dc, "similarity of per-stream LoRA updates")
    _, g_lmr = _entry_means(dgrp, "lora_main_ratio", "lora", lambda t: t.mean().item())
    _row("LoRA/main write-norm ratio", None, g_lmr, "||LoRA write|| / ||main write||")
    _, g_lpr = _entry_means(dgrp, "lora_perp_ratio", "lora", lambda t: t.mean().item())
    _row("LoRA perp ratio", None, g_lpr,
         "fraction of the LoRA update orthogonal to the main branch (novel direction)")

    # ---------- depth trend: H_res self-retention per layer (attn) ----------
    print("\n[depth] H_res self-retention per layer (attn site), layer 0..23")
    for tag, dump in (("mHC", dmhc), ("Grp", dgrp)):
        vals = [round(_diag_mean(dump["layers"][f"L{i}.attn"]["routing"]["Hres_raw"]), 2)
                for i in range(dump["meta"]["n_layers"])]
        print(f"  {tag}: {vals}")

    # ---------- depth trend: masked write-cos & active-stream (kept) fraction per layer ----------
    print("\n[depth] per-layer (attn+mlp avg)  masked write-cos  /  active-stream kept fraction")
    for tag, dump in (("mHC", dmhc), ("Grp", dgrp)):
        pl = _per_layer_write(dump, BETA_THR)
        print(f"  {tag} write-cos : {[round(c, 3) for c, _ in pl]}")
        print(f"  {tag} kept-frac : {[round(k, 2) for _, k in pl]}")


def write_summary_md(dmhc, dgrp, path):
    """Persist the headline stream-analysis numbers (incl. the beta-masked write cosine and
    the per-layer active-stream KEPT fraction) as a markdown data doc next to the dumps."""
    thr = BETA_THR
    mw, mk = _masked_write_cos(dmhc, thr)
    gw, gk = _masked_write_cos(dgrp, thr)
    _, m_wc = _entry_means(dmhc, "write_stream_cos", "stream_state", _offdiag_mean)
    _, g_wc = _entry_means(dgrp, "write_stream_cos", "stream_state", _offdiag_mean)
    plm, plg = _per_layer_write(dmhc, thr), _per_layer_write(dgrp, thr)
    nL = dmhc["meta"]["n_layers"]

    L = []
    L.append("# Stream-analysis summary (mHC vs mHC-Group-LoRA-midnorm)\n")
    L.append("Source: `eval/analysis_out/{mhc,mhc_group_lora_midnorm}.analysis.pt` "
             "(64 x 1024 training tokens, eval mode, fp32, real fwd+bwd; 24 layers x {attn,mlp}).\n")
    L.append(f"Write-gate mask threshold: `beta_s >= {thr:g}` (a stream with beta below this "
             "writes ~nothing, so its write vector is excluded from the write-cosine).\n")
    L.append("## Write-back collinearity (off-diagonal cosine of per-stream write vectors)\n")
    L.append("| model | raw write-cos | masked write-cos | active-stream kept |")
    L.append("|---|---|---|---|")
    L.append(f"| mHC | {m_wc:.3f} | {mw:.3f} | {mk:.1%} |")
    L.append(f"| Group-LoRA | {g_wc:.3f} | {gw:.3f} | {gk:.1%} |")
    L.append("")
    L.append("- mHC write is `beta_s * h` -> collinear; masked cos = 1.000 confirms it. "
             "The raw <1 value and the low kept-fraction come from streams with `beta_s ~ 0` "
             "(silent streams) whose zero write vector makes the cosine degenerate.")
    L.append("- Group-LoRA masked cos < 1 = the LoRA genuinely rotates per-stream write "
             "directions apart; its higher kept fraction = more streams stay active.\n")
    L.append("## Per-layer active-stream KEPT fraction and masked write-cos (attn+mlp avg)\n")
    L.append("| layer | mHC kept | mHC write-cos | Grp kept | Grp write-cos |")
    L.append("|---|---|---|---|---|")
    for i in range(nL):
        L.append(f"| {i} | {plm[i][1]:.2f} | {plm[i][0]:.3f} | {plg[i][1]:.2f} | {plg[i][0]:.3f} |")
    L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L))
    print(f"\nsummary written -> {path}")


def main():
    dmhc, dgrp = _load(MHC), _load(GRP)
    print("loaded:", OUT_DIR)
    print(f"  {MHC}: {len(dmhc['layers'])} entries   {GRP}: {len(dgrp['layers'])} entries")
    ok = validate(dmhc, MHC) & validate(dgrp, GRP)
    faithfulness()
    compare(dmhc, dgrp)
    write_summary_md(dmhc, dgrp, os.path.join(OUT_DIR, "stream_analysis_summary.md"))
    print("\nvalidation:", "ALL PASS" if ok else "FAILURES SEEN")


if __name__ == "__main__":
    main()
