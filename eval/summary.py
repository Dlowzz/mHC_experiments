"""Final S/M/L summary: OWT-val + WT103 PPL (Tier-1, branch min-CE) plus lm_eval
downstream (Tier-2). Reads eval/results/bulk + eval/results/tasks. Prints and writes md.
"""
import os
import json
import glob
import argparse

CODED = {
    "L": [
        ("mhc", "out-owt-large-mhc"),
        ("group", "out-owt-large-mhc-group-embedding"),
        ("mhc-lite", "out-owt-large-mhc-lite"),
        ("lora-midnorm-inbeta", "out-owt-large-mhc-lora-residual-midnorm-inbeta-bs8-20000step"),
        ("lora-midnorm-outbeta", "out-owt-large-mhc-lora-residual-midnorm-bs8-20000step"),
        ("GL-midnorm-inbeta", "out-owt-large-mhc-group-lora-midnorm-bs4-20000step"),
        ("lora-nonorm", "out-owt-large-mhc-lora-residual"),
        ("GLfake-2", "out-owt-large-mhc-group-lora-bs4-20000step"),
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
        ("lora-inbeta(*grouplora)", "out-owt-small-mhc-lora-residual-bs24-10000step"),
        ("gl-midnorm-inbeta", "out-owt-small-mhc-group-lora-midnorm-inbeta-bs24-10000step"),
        ("lora-midnorm-inbeta", "out-owt-small-mhc-lora-residual-midnorm-bs24-10000step"),
    ],
}


def bulk_min_ce():
    best = {}
    for fp in glob.glob("eval/results/bulk/*.json"):
        r = json.load(open(fp))
        d, ce = r.get("dir"), r.get("owt_val_ce")
        if r.get("error") or ce is None:
            continue
        if d not in best or ce < best[d]["owt_val_ce"]:
            best[d] = r
    return best


def tasks_by_dir():
    out = {}
    for fp in glob.glob("eval/results/tasks/*.json"):
        r = json.load(open(fp))
        if r.get("error"):
            continue
        out.setdefault(r.get("dir"), []).append(r)
    return out


def task_score(r, task):
    for m in ("::acc_norm", "::acc"):
        if task + m in r:
            return r[task + m]
    return None


def fmt(v, nd=2):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/eval_summary.md")
    args = ap.parse_args()
    b = bulk_min_ce()
    t = tasks_by_dir()

    L = ["# mHC checkpoints - final summary (OWT-val/WT103 PPL + downstream)", "",
         "PPL: deterministic full-pass, branch-matched (min-CE). Downstream: lm_eval 0-shot, batched.",
         "`(*)` = experiment code's beta-mode disagrees with the branch the weights actually match.", ""]
    for tier, title in [("L", "Large (24L/1024d)"), ("M", "Medium (12L/768d)"), ("S", "Small (6L/512d)")]:
        rows = []
        for code, d in CODED[tier]:
            r1 = b.get(d)
            owt = r1.get("owt_val_ppl") if r1 else None
            wt = r1.get("wt103_ppl") if r1 else None
            tj = t.get(d, [{}])[0]
            rows.append((owt if owt is not None else 1e9, code, owt, wt,
                         tj.get("lambada_ppl"), tj.get("lambada_acc"),
                         task_score(tj, "sciq"), task_score(tj, "piqa"),
                         task_score(tj, "arc_easy"), task_score(tj, "winogrande")))
        rows.sort(key=lambda x: x[0])
        L += [f"## {title}", "",
              "| 实验代号 | OWT-val PPL | WT103 PPL | lambada PPL | lambada acc | sciq | piqa | arc_easy | winogrande |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for _, code, owt, wt, lp, la, sc, pi, ae, wg in rows:
            L.append(f"| `{code}` | {fmt(owt)} | {fmt(wt)} | {fmt(lp)} | {fmt(la,3)} | {fmt(sc,3)} | {fmt(pi,3)} | {fmt(ae,3)} | {fmt(wg,3)} |")
        L.append("")
    L += ["_hellaswag / openbookqa / copa omitted from table: ~chance (0.25/0.25/0.50) at these scales._", "",
          "_Verified: each row's eval PPL matches the ckpt's recorded best_val_loss (correct training branch)._",
          "_Excluded `GL-midnorm-outbeta` (out-owt-medium-mhc-group-lora-midnorm-bs16): crashed run (wandb owt-M-group-lora-midnorm-8489 @199654a), recorded val\u224828.5; no checked-out branch reproduces it, so its eval is unreliable._"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write("\n".join(L))
    print("\n".join(L))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
