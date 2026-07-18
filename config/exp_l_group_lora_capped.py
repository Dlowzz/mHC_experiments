# L-scale mHC-group-LoRA-capped experiment on OpenWebText
# SAME setting as the group_lora L run (for fair before/after comparison):
#   effective batch = 4 * grad_accum16 = 65,536 tokens/iter, 20000 steps, cosine to 20000
# combine with: config/train_owt.py config/large_model.py config/with_mhc_group_lora_capped.py

batch_size = 4
gradient_accumulation_steps = 16
max_iters = 20000
lr_decay_iters = 20000

wandb_run_name = 'L-mhc_group_lora_capped-owt-bs8eff-20kstep'
