"""Build S/M/L markdown tables from eval/results/bulk/<branch>__<dir>.json.

Per checkpoint, picks the branch giving the LOWEST OWT-val CE: the branch whose
code matches how the ckpt was trained forwards correctly (low CE); a mismatched
branch inflates CE. Any ckpt whose best CE is still abnormally high is flagged as
a probable branch mismatch (its true training branch was not among those tried).
"""
import os
import json
import glob
import argparse


def tier_of(d, n_layer, n_embd):
    if "large" in d or (n_layer, n_embd) == (24, 1024):
        return "L"
    if "medium" in d or (n_layer, n_embd) == (12, 768):
        return "M"
    if "small" in d or (n_layer, n_embd) == (6, 512):
        return "S"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="eval/results/bulk")
    ap.add_argument("--out", default="eval/eval_report.md")
    ap.add_argument("--mismatch_ce", type=float, default=4.0)
    args = ap.parse_args()

    best, all_recs = {}, {}
    for fp in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        r = json.load(open(fp))
        d = r.get("dir")
        all_recs.setdefault(d, []).append(r)
        ce = r.get("owt_val_ce")
        if r.get("error") or ce is None:
            continue
        if d not in best or ce < best[d]["owt_val_ce"]:
            best[d] = r
    for d, recs in all_recs.items():
        best.setdefault(d, recs[0])  # dirs with only error records

    tiers = {"L": [], "M": [], "S": [], "?": []}
    for d, r in best.items():
        tiers[tier_of(d, r.get("n_layer"), r.get("n_embd"))].append(r)

    L = ["# mHC checkpoint evaluation - Tier-1 (OWT-val + WikiText-103 PPL)", "",
         "Deterministic full-pass PPL, fp32, eval mode. For each checkpoint the branch giving the",
         "lowest OWT-val CE is reported (a branch-mismatched forward inflates CE).", "",
         f"- **branch**: git branch used for the reported (min-CE) result.",
         f"- **note**: flagged when best OWT-val CE > {args.mismatch_ce} (likely wrong code branch; needs its training branch).", ""]
    for t, title in [("L", "Large (24L / 1024d)"), ("M", "Medium (12L / 768d)"), ("S", "Small (6L / 512d)")]:
        rows = tiers[t]
        rows.sort(key=lambda r: r.get("owt_val_ppl") if r.get("owt_val_ppl") is not None else 1e9)
        L += [f"## {title}", "",
              "| variant | type | iter | branch | OWT-val PPL | OWT-val CE | WT103 PPL | note |",
              "|---|---|---:|---|---:|---:|---:|---|"]
        for r in rows:
            d = r.get("dir")
            ce = r.get("owt_val_ce")
            typ, it, br = r.get("hyper_conn_type", "-"), r.get("iter", "-"), r.get("branch", "-")
            if ce is None:
                err = (r.get("error") or "no result").splitlines()[0]
                L.append(f"| {d} | {typ or '-'} | {it} | - | - | - | - | {err} |")
                continue
            ppl, wt = r.get("owt_val_ppl"), r.get("wt103_ppl")
            if isinstance(it, int) and it < 100:
                L.append(f"| {d} | {typ} | {it} | {br} | {ppl} | {round(ce, 4)} | {wt} | smoke ckpt ({it} steps) - not comparable |")
            elif ce > args.mismatch_ce:
                L.append(f"| {d} | {typ} | {it} | {br} | n/a | n/a | n/a | unresolved: needs exact training branch/commit |")
            else:
                L.append(f"| {d} | {typ} | {it} | {br} | {ppl} | {round(ce, 4)} | {wt} |  |")
        L.append("")
    if tiers["?"]:
        L += ["## (untiered)", ""] + [f"- {r.get('dir')}" for r in tiers["?"]] + [""]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write("\n".join(L))
    print("wrote", args.out, "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
