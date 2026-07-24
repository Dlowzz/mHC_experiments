"""Aggregate eval/results/*.json (merged per checkpoint) into one comparison table.

Merges all JSONs that share the same "ckpt" (e.g. the PPL json + the lm_eval json)
into a single row. Usage: python eval/collect.py [--dir eval/results]
"""
import os
import json
import glob
import argparse


def size_of(nl, ne):
    return {(6, 512): "S", (12, 768): "M", (24, 1024): "L"}.get((nl, ne), f"{nl}L/{ne}d")


def ckpt_name(ckpt, fallback):
    if not ckpt:
        return os.path.splitext(os.path.basename(fallback))[0]
    d = os.path.dirname(ckpt)
    return os.path.basename(d) if d else os.path.basename(ckpt)


def cell(v, kind, width):
    if v is None:
        s = "-"
    elif kind in ("f4", "f3", "f2") and isinstance(v, (int, float)):
        s = f"{v:.{ {'f4': 4, 'f3': 3, 'f2': 2}[kind] }f}"
    else:
        s = str(v)
    if len(s) > width - 1:
        s = s[:width - 2] + "\u2026"
    return f"{s:<{width}}"


COLS = [
    ("name", 34, "s"), ("size", 5, "s"), ("iter", 7, "s"),
    ("owt_ppl", 9, "f2"), ("wt103_ppl", 10, "f2"),
    ("lambada_ppl", 12, "f2"), ("lambada_acc", 12, "f3"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="eval/results")
    args = ap.parse_args()

    merged = {}
    for fp in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        try:
            d = json.load(open(fp))
        except Exception as e:
            print(f"[skip] {fp}: {e}")
            continue
        merged.setdefault(d.get("ckpt", fp), {}).update(d)
    if not merged:
        print("no result JSONs found in", args.dir)
        return

    rows = []
    for key, d in merged.items():
        by_task = {}
        for k, v in d.items():
            if "::" in k:
                t, m = k.split("::", 1)
                by_task.setdefault(t, {})[m] = v
        rows.append({
            "name": ckpt_name(d.get("ckpt"), key),
            "size": size_of(d.get("n_layer"), d.get("n_embd")),
            "iter": d.get("iter"),
            "owt_ppl": d.get("owt_val_ppl"),
            "wt103_ppl": d.get("wt103_ppl"),
            "lambada_ppl": d.get("lambada_ppl"),
            "lambada_acc": d.get("lambada_acc"),
            "by_task": by_task,
        })

    hdr = "".join(f"{c[0]:<{c[1]}}" for c in COLS)
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["size"], r["name"])):
        print("".join(cell(r.get(n), k, w) for n, w, k in COLS))
        parts = []
        for task in sorted(r["by_task"]):
            if task == "lambada_openai":
                continue  # shown in the main columns
            m = r["by_task"][task]
            val = m.get("acc_norm", m.get("acc"))
            if val is not None:
                parts.append(f"{task}={val}")
        if parts:
            print("    downstream(acc_norm|acc): " + "  ".join(parts))
    print("\n[Tier-1] owt/wt103 = deterministic full-pass PPL (main discriminator).")
    print("[aux]    lambada/downstream = loglikelihood via lm_eval (HF mirror).")
    print("[note]   cross-branch baselines (outside-beta / no-norm) must be evaluated on their own branch.")


if __name__ == "__main__":
    main()
