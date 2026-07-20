# L-scale mHC-group-LoRA-midnorm (rank-dim param-free RMSNorm), DUAL card (2/3)
# micro-batch 4 x grad_accum 16 -> effective batch 65,536 tokens/iter (bs8 OOMs on 46GB L20),
# 20000 steps, cosine to 20000. Same L setting as other L runs.
# combine with: config/train_owt.py config/large_model.py config/with_mhc_group_lora_midnorm.py

batch_size = 4
gradient_accumulation_steps = 16
max_iters = 20000
lr_decay_iters = 20000

wandb_run_name = 'L-group_lora_midnorm-owt-bs4ga16-20kstep'
