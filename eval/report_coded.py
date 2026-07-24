"""Build the three S/M/L tables for the checkpoints that carry an experiment code
(实验代号) in testckpt.md. Uses the min-CE branch-matched value from eval/results/bulk.

Previously-untestable checkpoints (unresolved group-lora, missing ckpt, wrong-vocab,
smoke, uncoded rows) are intentionally omitted. Rows whose code implies a beta-mode
that disagrees with the branch the weights actually forward correctly on are flagged.
"""
import os
import json
import glob
import argparse

# (code, dir) per tier -- only the coded + testable checkpoints
CODED = {
    "L": [
        ("mhc", "out-owt-large-mhc"),
        ("group", "out-owt-large-mhc-group-embedding"),
        ("GLfake-2", "out-owt-large-mhc-group-lora-bs4-20000step"),
        ("GL-midnorm-inbeta", "out-owt-large-mhc-group-lora-midnorm-bs4-20000step"),
        ("mhc-lite", "out-owt-large-mhc-lite"),
        ("lora-nonorm", "out-owt-large-mhc-lora-residual"),
        ("lora-midnorm-outbeta", "out-owt-large-mhc-lora-residual-midnorm-bs8-20000step"),
        ("lora-midnorm-inbeta", "out-owt-large-mhc-lora-residual-midnorm-inbeta-bs8-20000step"),
    ],
    "M": [
        ("mhc", "out-owt-medium-mhc"),
        ("Group", "out-owt-medium-mhc-group-embedding"),
        ("GL-midnorm-outbeta", "out-owt-medium-mhc-group-lora-midnorm-bs16-10000step"),
        ("GL-midnorm-inbeta", "out-owt-medium-mhc-group-lora-midnorm-inbeta-bs16-10000step"),
        ("mhc-lite", "out-owt-medium-mhc-lite"),
        ("lora-nonorm", "out-owt-medium-mhc-lora-residual"),
        ("lora-midnorm-inbeta", "out-owt-medium-mhc-lora-residual-midnorm-bs16-10000step"),
    ],
    "S": [
        ("mhc", "out-owt-small-mhc"),
        ("Group", "out-owt-small-mhc-group-embedding"),
        ("GL-out1", "out-owt-small-mhc-group-lora"),
        ("mhc-lite", "out-owt-small-mhc-lite"),
        ("lora-inbeta", "out-owt-small-mhc-lora-residual-bs24-10000step"),
        ("gl-midnorm-inbeta", "out-owt-small-mhc-group-lora-midnorm-inbeta-bs24-10000step"),
    ],
}


def load_min_ce(bulk_dir):
    best = {}
    for fp in glob.glob(os.path.join(bulk_dir, "*.json")):
        r = json.load(open(fp))
        d, ce = r.get("dir"), r.get("owt_val_ce")
        if r.get("error") or ce is None:
            continue
        if d not in best or ce < best[d]["owt_val_ce"]:
            best[d] = r
    return best


def flag(code, branch):
    c = code.lower()
    if "inbeta" in c and branch != "inbeta":
        return f"code says inbeta, but weights forward correctly under `{branch}` (out-beta)"
    if "outbeta" in c and branch == "inbeta":
        return f"code says outbeta, but matched `inbeta`"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="eval/results/bulk")
    ap.add_argument("--out", default="eval/eval_report_coded.md")
    args = ap.parse_args()
    best = load_min_ce(args.dir)

    L = ["# Coded checkpoints - OWT-val + WikiText-103 PPL (Tier-1)", "",
         "Deterministic full-pass PPL (fp32), branch-matched (min-CE across in-beta / midnorm / group_lora).",
         "Only checkpoints with an experiment code in testckpt.md; previously-untestable ones omitted.", ""]
    for tier, title in [("L", "Large (24L / 1024d)"), ("M", "Medium (12L / 768d)"), ("S", "Small (6L / 512d)")]:
        rows = []
        for code, d in CODED[tier]:
            r = best.get(d)
            if r is None:
                rows.append((1e9, code, d, "n/a", "n/a", "-", "no valid result"))
                continue
            rows.append((r["owt_val_ppl"], code, d, r["owt_val_ppl"], r.get("wt103_ppl", "-"),
                         r.get("branch", "-"), flag(code, r.get("branch", "-"))))
        rows.sort(key=lambda x: x[0])
        L += [f"## {title}", "",
              "| 实验代号 | 文件夹 | OWT-val PPL | WT103 PPL | matched branch | note |",
              "|---|---|---:|---:|---|---|"]
        for _, code, d, ppl, wt, br, note in rows:
            L.append(f"| `{code}` | {d} | {ppl} | {wt} | {br} | {note} |")
        L.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write("\n".join(L))
    print("\n".join(L))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
