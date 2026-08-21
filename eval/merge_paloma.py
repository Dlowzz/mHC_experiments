"""Merge parallel eval_paloma.py output dirs into one summary CSV.

The full S/M/L/XL x 4-variant sweep is split across several GPUs, each writing
its own --output-dir (to avoid concurrent writers racing on one CSV). This
rebuilds a single sorted paloma_summary.csv from every tidy `<label>.json`
(the per-checkpoint files that carry a "results" block), and can also collect
all tidy + raw JSONs into one directory.

Usage:
  python eval/merge_paloma.py --dirs eval/results/paloma eval/results/paloma_p2 \
      eval/results/paloma_p3 --out eval/results/paloma
"""
import os
import sys
import csv
import glob
import json
import shutil
import argparse

CORPUS_ORDER = ["c4", "dolma", "falcon", "redpajama", "wikitext"]
TIER_ORDER = {"S": 0, "M": 1, "L": 2, "XL": 3}
VAR_ORDER = {"mhc": 0, "group": 1, "lora": 2, "grouplora": 3}


def sort_key(label):
    tier, _, var = label.partition("-")
    return (TIER_ORDER.get(tier, 9), VAR_ORDER.get(var, 9), label)


def load_rows(dirs):
    rows = {}
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                doc = json.load(open(path))
            except Exception:
                continue
            if not isinstance(doc, dict) or "results" not in doc or "label" not in doc:
                continue                        # skip raw/ dumps and anything else
            rows[doc["label"]] = doc            # last write wins per label
    return rows

def write_csv(rows, out_csv):
    header = ["model", "iter", "n_layer", "n_embd"]
    header += [f"{a}_ppl" for a in CORPUS_ORDER]
    header += [f"{a}_bpb" for a in CORPUS_ORDER]
    header += [f"{a}_byteppl" for a in CORPUS_ORDER]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for label in sorted(rows, key=sort_key):
            doc = rows[label]
            res = doc.get("results", {})
            row = {"model": label, "iter": doc.get("iter"),
                   "n_layer": doc.get("n_layer"), "n_embd": doc.get("n_embd")}
            for a in CORPUS_ORDER:
                d = res.get(a, {})
                row[f"{a}_ppl"] = d.get("word_perplexity", "")
                row[f"{a}_bpb"] = d.get("bits_per_byte", "")
                row[f"{a}_byteppl"] = d.get("byte_perplexity", "")
            w.writerow({k: row.get(k, "") for k in header})
    return header


def collect_files(dirs, out_dir):
    for d in dirs:
        if os.path.abspath(d) == os.path.abspath(out_dir):
            continue
        for path in glob.glob(os.path.join(d, "*.json")):
            shutil.copy2(path, os.path.join(out_dir, os.path.basename(path)))
        rawd = os.path.join(d, "raw")
        if os.path.isdir(rawd):
            os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
            for path in glob.glob(os.path.join(rawd, "*.json")):
                shutil.copy2(path, os.path.join(out_dir, "raw", os.path.basename(path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="output dir (holds paloma_summary.csv)")
    ap.add_argument("--collect", action="store_true", help="also copy per-ckpt JSONs into --out")
    args = ap.parse_args()

    rows = load_rows(args.dirs)
    os.makedirs(args.out, exist_ok=True)
    if args.collect:
        collect_files(args.dirs, args.out)
    out_csv = os.path.join(args.out, "paloma_summary.csv")
    write_csv(rows, out_csv)
    print(f"[merge] {len(rows)} models -> {out_csv}")
    for label in sorted(rows, key=sort_key):
        print(" ", label, json.dumps(rows[label].get("results", {})))


if __name__ == "__main__":
    main()
