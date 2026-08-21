"""Driver: run the paper-analysis evaluation recorder on the trained mHC and
mHC-Group-LoRA (midnorm) L checkpoints and dump the 4 analysis categories.

Fixed experiment (identical for both models):
  * trained checkpoint, ``model.eval()``, full float32 forward/backward (NO autocast),
  * the first ``NUM_SEQ`` NON-overlapping training sequences of length ``SEQ_LEN`` taken
    from the very start of ``train.bin`` -- same tokens, same order for both models,
  * loss = the model's own mean cross-entropy (``GPT.forward`` with targets),
  * a single real ``loss.backward()`` per mini-batch -> UNSCALED gradients
    (no gradient clipping, no optimizer step -- see analysis_recorder.Recorder).

Configuration is via module constants / environment variables (this tool takes NO
argparse flags).  Overridable env vars:
    ANALYSIS_OUT_DIR, ANALYSIS_MICRO_BATCH, ANALYSIS_NUM_SEQ, ANALYSIS_SEQ_LEN,
    ANALYSIS_DEVICE, ANALYSIS_DATA_BIN, ANALYSIS_MHC_CKPT, ANALYSIS_GROUP_CKPT.

Run from the repo root::

    python -m eval.record_analysis
"""
import os

import numpy as np
import torch

from model import GPT, GPTConfig
from eval.analysis_recorder import Recorder

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)


def _env(name, default):
    return os.environ.get(name, default)


NUM_SEQ = int(_env("ANALYSIS_NUM_SEQ", "64"))
SEQ_LEN = int(_env("ANALYSIS_SEQ_LEN", "1024"))
MICRO_BATCH = int(_env("ANALYSIS_MICRO_BATCH", "2"))
DEVICE = _env("ANALYSIS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DATA_BIN = _env("ANALYSIS_DATA_BIN", os.path.join(_REPO, "data", "openwebtext", "train.bin"))
OUT_DIR = _env("ANALYSIS_OUT_DIR", os.path.join(_HERE, "analysis_out"))

# the two trained L checkpoints being compared (name -> ckpt.pt)
MODELS = {
    "mhc": _env(
        "ANALYSIS_MHC_CKPT",
        "/home/work/data/guotianzizhe/data/test/out-owt-large-mhc-bs8-30000step/ckpt.pt",
    ),
    "mhc_group_lora_midnorm": _env(
        "ANALYSIS_GROUP_CKPT",
        "/home/work/data/guotianzizhe/data/test/"
        "out-owt-large-mhc-group-lora-midnorm-wsd-bs8-30000step/ckpt.pt",
    ),
}


def load_model(ckpt_path, device):
    """Rebuild the GPT for a checkpoint and load its weights in float32 eval mode."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["model_args"])
    model = GPT(cfg)
    state_dict = ckpt["model"]
    prefix = "_orig_mod."          # torch.compile wraps params under this prefix
    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            state_dict[k[len(prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval().to(device=device, dtype=torch.float32)
    return model, ckpt.get("model_args", {}), ckpt.get("iter_num")


def make_sequences(bin_path, num_seq, seq_len):
    """First ``num_seq`` NON-overlapping length-``seq_len`` blocks from the start of the
    token stream: x[i] = data[i*L : i*L+L], y[i] = data[i*L+1 : i*L+L+1] (next-token).

    Deterministic and independent of the model -> identical data/order for both models."""
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    need = num_seq * seq_len + 1
    assert len(data) >= need, f"{bin_path}: {len(data)} tokens < required {need}"
    xs, ys = [], []
    for i in range(num_seq):
        s = i * seq_len
        xs.append(torch.from_numpy(data[s:s + seq_len].astype(np.int64)))
        ys.append(torch.from_numpy(data[s + 1:s + 1 + seq_len].astype(np.int64)))
    return torch.stack(xs), torch.stack(ys)          # [num_seq, seq_len] each


def record_model(name, ckpt_path, x, y, device, micro_batch, out_dir):
    print(f"[{name}] loading {ckpt_path}")
    model, model_args, iter_num = load_model(ckpt_path, device)
    print(f"[{name}] type={model_args.get('hyper_conn_type')} n_streams={model_args.get('hyper_conn_n')} "
          f"iter={iter_num} params={model.get_num_params()/1e6:.1f}M")
    rec = Recorder(model, name).install()

    n = x.size(0)
    losses = []
    for start in range(0, n, micro_batch):
        xb = x[start:start + micro_batch].to(device)
        yb = y[start:start + micro_batch].to(device)
        model.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)              # full float32 forward, model's own mean CE
        loss.backward()                      # real unscaled gradients (no clip / no step)
        rec.collect()                        # offload fwd + .grad to CPU, free the graph
        losses.append(float(loss))
        print(f"[{name}] seqs {start:3d}-{start + xb.size(0) - 1:3d}  loss {float(loss):.4f}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.analysis.pt")
    rec.save(out_path)
    rec.remove()
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    mean_loss = sum(losses) / len(losses)
    print(f"[{name}] saved -> {out_path}  (mean loss {mean_loss:.4f})")
    return out_path, mean_loss


def main():
    print(f"config: NUM_SEQ={NUM_SEQ} SEQ_LEN={SEQ_LEN} MICRO_BATCH={MICRO_BATCH} "
          f"DEVICE={DEVICE}\n        DATA_BIN={DATA_BIN}\n        OUT_DIR={OUT_DIR}")
    x, y = make_sequences(DATA_BIN, NUM_SEQ, SEQ_LEN)
    print(f"built {x.size(0)} sequences of length {x.size(1)} "
          f"(tokens {x[0,0].item()}..; identical for both models)")
    results = {}
    for name, ckpt_path in MODELS.items():
        with torch.enable_grad():
            results[name] = record_model(name, ckpt_path, x, y, DEVICE, MICRO_BATCH, OUT_DIR)
    print("\nDONE:")
    for name, (path, ml) in results.items():
        print(f"  {name:26s} mean_loss={ml:.4f}  {path}")


if __name__ == "__main__":
    main()
