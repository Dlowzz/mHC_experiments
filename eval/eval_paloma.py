"""Standardized Paloma perplexity evaluation for the mHC GPT checkpoints.

Runs EleutherAI/lm-evaluation-harness' *official* Paloma tasks
(output_type=loglikelihood_rolling) against the project's own model through the
`mhc_gpt` adapter (eval/lm_adapter.py). No re-implementation of the C4 / Dolma /
RefinedWeb / RedPajama / WikiText-103 perplexity math -- the harness owns the
data loading, rolling windows and word/byte/bit aggregation; we only supply the
model's log-likelihoods via its own tokenizer.

Five corpora (short alias -> official paloma task):
    c4        -> paloma_c4_en
    dolma     -> paloma_dolma-v1_5
    falcon    -> paloma_falcon-refinedweb
    redpajama -> paloma_redpajama
    wikitext  -> paloma_wikitext_103

Per checkpoint we emit (under --output-dir, default eval/results/paloma):
    raw/<label>.json    -- the full lm-eval results dict, untouched
    <label>.json        -- tidy {task: {word_ppl, byte_ppl, bpb}} + metadata
    paloma_summary.csv   -- one accumulated row per checkpoint

`allenai/paloma` is GATED on HF. Without access the paloma_* tasks cannot
download; use `--tasks wikitext_public` (public standalone lm-eval `wikitext`,
same loglikelihood_rolling + word/byte/bpb pipeline) to smoke-test the plumbing.
HF mirror env, when downloads are needed:
    HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_TRUST_REMOTE_CODE=1

Usage:
    # smoke test (100 docs/corpus) on one checkpoint
    python eval/eval_paloma.py --checkpoint data/test/out-owt-large-mhc \
        --device cuda:0 --batch-size 8 --limit 100

    # full 5-corpus Paloma on one checkpoint
    python eval/eval_paloma.py --checkpoint data/test/out-owt-large-mhc --device cuda:0

    # batch: many checkpoints (S/M/L/XL x 4 variants), labelled from model_args
    python eval/eval_paloma.py --checkpoints DIR1 DIR2 ... --device cuda:0
"""
import os
import sys
import csv
import json
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from loader import load_ckpt
from lm_adapter import MHCLMAdapter  # registers @register_model("mhc_gpt")

# short alias -> official paloma task name (the 5 corpora requested)
PALOMA = {
    "c4": "paloma_c4_en",
    "dolma": "paloma_dolma-v1_5",
    "falcon": "paloma_falcon-refinedweb",
    "redpajama": "paloma_redpajama",
    "wikitext": "paloma_wikitext_103",
}
# alias for a public smoke test (standalone lm-eval `wikitext`, same rolling pipeline);
# kept distinct from the "wikitext" corpus alias above, which is paloma_wikitext_103
PROXY = {"wikitext_public": "wikitext"}
# task name -> corpus column (both the paloma task and the standalone task map to "wikitext")
ALIAS = {**{v: k for k, v in PALOMA.items()}, "wikitext": "wikitext"}
METRICS = ("word_perplexity", "byte_perplexity", "bits_per_byte")
CORPUS_ORDER = ["c4", "dolma", "falcon", "redpajama", "wikitext"]

TIER = {6: "S", 12: "M", 24: "L", 28: "XL"}
VARIANT = {
    "mhc": "mhc",
    "mhc_group_embedding": "group",
    "mhc_lora_residual_midnorm": "lora",
    "mhc_group_lora_midnorm": "grouplora",
}


def resolve_ckpt(p, name="ckpt.pt"):
    """A ckpt dir -> <dir>/<name>; a file path is returned as-is."""
    return os.path.join(p, name) if os.path.isdir(p) else p


def label_for(ck, fallback):
    """Human label from model_args: e.g. 'L-mhc', 'M-grouplora'."""
    ma = ck.get("model_args", {}) or {}
    tier = TIER.get(ma.get("n_layer"), f"n{ma.get('n_layer')}")
    var = VARIANT.get(ma.get("hyper_conn_type"), ma.get("hyper_conn_type") or "?")
    if not ma:
        return fallback
    return f"{tier}-{var}"


def resolve_tasks(spec):
    """Comma list of aliases/raw names -> official task names (order preserved)."""
    names = []
    for t in spec.split(","):
        t = t.strip()
        if not t:
            continue
        names.append(PALOMA.get(t, PROXY.get(t, t)))
    return names

@torch.inference_mode()
def run_one(ckpt, task_names, device, batch_size, limit):
    """Evaluate one checkpoint on task_names; return (meta, tidy, raw_results)."""
    import lm_eval

    model, ck = load_ckpt(ckpt, device=device)
    adapter = MHCLMAdapter(model=model, device=device, batch_size=batch_size)
    res = lm_eval.simple_evaluate(model=adapter, tasks=task_names,
                                  limit=limit, bootstrap_iters=0)
    tidy = {}
    for task in task_names:
        r = res["results"].get(task, {})
        d = {}
        for m in METRICS:
            v = r.get(f"{m},none")
            if isinstance(v, (int, float)) and not math.isnan(v):
                d[m] = round(float(v), 6)
        tidy[ALIAS.get(task, task)] = d
    ma = ck.get("model_args", {}) or {}
    meta = {
        "ckpt": ckpt,
        "label": label_for(ck, os.path.basename(os.path.dirname(ckpt)) or ckpt),
        "iter": int(ck.get("iter_num", -1)),
        "hyper_conn_type": ma.get("hyper_conn_type"),
        "n_layer": ma.get("n_layer"),
        "n_embd": ma.get("n_embd"),
        "block_size": ma.get("block_size"),
        "vocab_size": ma.get("vocab_size"),
        "batch_size": batch_size,
        "limit": limit,
    }
    del adapter, model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return meta, tidy, res


def write_summary(path, meta, tidy):
    """Upsert one row per model label into the accumulating summary CSV."""
    header = ["model", "iter", "n_layer", "n_embd"]
    header += [f"{a}_ppl" for a in CORPUS_ORDER]      # word_perplexity
    header += [f"{a}_bpb" for a in CORPUS_ORDER]      # bits_per_byte
    header += [f"{a}_byteppl" for a in CORPUS_ORDER]  # byte_perplexity
    row = {"model": meta["label"], "iter": meta["iter"],
           "n_layer": meta["n_layer"], "n_embd": meta["n_embd"]}
    for a in CORPUS_ORDER:
        d = tidy.get(a, {})
        row[f"{a}_ppl"] = d.get("word_perplexity", "")
        row[f"{a}_bpb"] = d.get("bits_per_byte", "")
        row[f"{a}_byteppl"] = d.get("byte_perplexity", "")
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("model") != row["model"]]
    rows.append(row)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})

def main():
    ap = argparse.ArgumentParser(description="Official Paloma PPL for mHC checkpoints")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", help="one ckpt file or its out-* directory")
    g.add_argument("--checkpoints", nargs="+",
                   help="many ckpt dirs/files (batch: S/M/L/XL x variants)")
    ap.add_argument("--model-config", default=None,
                    help="informational only; model_args are read from the checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--output-dir", default="eval/results/paloma")
    ap.add_argument("--limit", type=int, default=None,
                    help="docs per corpus (smoke test, e.g. 100); omit for full eval")
    ap.add_argument("--tasks", default=",".join(PALOMA),
                    help="aliases c4,dolma,falcon,redpajama,wikitext (or raw task names); "
                         "use 'wikitext' alone for a public smoke test")
    ap.add_argument("--ckpt-name", default="ckpt.pt",
                    help="file inside a ckpt dir (ckpt.pt | ckpt_last.pt)")
    args = ap.parse_args()

    task_names = resolve_tasks(args.tasks)
    ckpts = [args.checkpoint] if args.checkpoint else list(args.checkpoints)
    os.makedirs(os.path.join(args.output_dir, "raw"), exist_ok=True)
    summary = os.path.join(args.output_dir, "paloma_summary.csv")

    print(f"[eval_paloma] tasks={task_names} device={args.device} "
          f"batch_size={args.batch_size} limit={args.limit}")
    for c in ckpts:
        ckpt = resolve_ckpt(c, args.ckpt_name)
        if not os.path.exists(ckpt):
            print(f"[skip] missing checkpoint: {ckpt}")
            continue
        meta, tidy, raw = run_one(ckpt, task_names, args.device, args.batch_size, args.limit)
        label = meta["label"]
        with open(os.path.join(args.output_dir, "raw", f"{label}.json"), "w") as f:
            json.dump(raw, f, indent=2, default=str)
        with open(os.path.join(args.output_dir, f"{label}.json"), "w") as f:
            json.dump({**meta, "results": tidy}, f, indent=2)
        write_summary(summary, meta, tidy)
        print(f"[done] {label}  {json.dumps(tidy)}")
    print(f"[eval_paloma] summary CSV -> {summary}")


if __name__ == "__main__":
    main()


