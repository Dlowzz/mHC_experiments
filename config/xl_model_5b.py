"""XL (~620M) on OpenWebText for a ~5B token budget.

Token budget: bs 6 x grad_accum 10 x block 1024 = 61,440 tokens/iter,
80,000 iters = 4.92B tokens (the 30k round saw 2.0B).

LR: warmup 2,000 iters, cosine 2e-4 -> 2e-5 over the first 50,000 iters, then
held at min_lr for the remaining 30,000 (train.py's get_lr returns min_lr once
it > lr_decay_iters).

Micro-batch: bs=8 (which would keep the 65,536 tokens/iter of the 30k runs with
grad_accum 8) OOMs on 2xL20 -- 42.9 GiB allocated of 44.4 GiB usable and it still
wants ~0.8 GiB more -- so the micro-batch stays at 6.  Only the XL runs have to be
mutually aligned, so all four XL variants (mhc / mhc_group_embedding /
mhc_lora_residual_midnorm / mhc_group_lora_midnorm) share this file and are
compared at equal tokens.

Runs start from scratch; this is not a continuation of the 30k checkpoints (their
schedule had already decayed to min_lr at a different peak).
"""
wandb_group = "xl"
out_prefix_model = "xl"

n_layer = 28
n_head = 20
n_embd = 1280
dropout = 0.0

learning_rate = 2e-4
min_lr = 2e-5
max_iters = 80000
lr_decay_iters = 50000
warmup_iters = 2000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# 6 x 10 x 1024 = 61,440 tokens/iter (DDP divides grad_accum by the world size,
# so the effective batch does not depend on the number of GPUs)
batch_size = 6
gradient_accumulation_steps = 10
