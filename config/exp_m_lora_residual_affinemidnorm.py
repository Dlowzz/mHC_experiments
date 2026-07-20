# M-scale LoRA-residual-affine-midnorm (rank-dim AFFINE RMSNorm), SINGLE GPU
# affine RMSNorm scale/bias get NO cosine LR decay (constant LR).
# bs=16, grad_accum=8 -> effective batch 131,072 tokens/iter (== dual-card M), 10000 steps
# combine with: config/train_owt.py config/medium_model.py config/with_mhc_lora_residual_affinemidnorm.py

batch_size = 16
gradient_accumulation_steps = 8
max_iters = 10000
lr_decay_iters = 10000

wandb_run_name = 'M-lora_residual_affinemidnorm-owt-bs16ga8-10kstep'
