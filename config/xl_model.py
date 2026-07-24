wandb_group = "xl"
out_prefix_model = "xl"

n_layer = 28
n_head = 20
n_embd = 1280
dropout = 0.0

learning_rate = 2e-4
min_lr = 2e-5
max_iters = 30000
lr_decay_iters = 30000
warmup_iters = 200
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# ~600M target on 2xL20: micro-batch 4 x grad_accum 16 -> 65,536 tokens/iter
batch_size = 4
gradient_accumulation_steps = 16
