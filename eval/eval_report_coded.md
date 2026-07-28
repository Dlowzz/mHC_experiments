# Coded checkpoints - OWT-val + WikiText-103 PPL (Tier-1)

Deterministic full-pass PPL (fp32), branch-matched (min-CE across in-beta / midnorm / group_lora).
Only checkpoints with an experiment code in testckpt.md; previously-untestable ones omitted.

## XL (28L / 1280d)

| 实验代号 | 文件夹 | OWT-val PPL | WT103 PPL | matched branch | note |
|---|---|---:|---:|---|---|
| `group` | out-owt-xl-mhc-group-embedding-bs8-30000step | 21.0420 | 32.6361 | inbeta | iter 30000 / 30000; bs8×ga8 |
| `lora-midnorm-inbeta` | out-owt-xl-mhc-lora-residual-midnorm-bs4-30000step | 21.0796 | 32.8556 | inbeta | iter 28000 / 30000 |
| `mhc` | out-owt-xl-mhc-bs4-30000step | 21.1896 | 33.5740 | inbeta | iter 28500 / 30000 |
| `GL-midnorm-inbeta` | out-owt-xl-mhc-group-lora-midnorm-bs4-30000step | 21.4107 | 33.6073 | inbeta | iter 28000 / 30000 |

## Large (24L / 1024d)

| 实验代号 | 文件夹 | OWT-val PPL | WT103 PPL | matched branch | note |
|---|---|---:|---:|---|---|
| `group` | out-owt-large-mhc-group-embedding | 23.5872 | 37.6221 | inbeta |  |
| `lora-midnorm-inbeta` | out-owt-large-mhc-lora-residual-midnorm-inbeta-bs8-20000step | 23.7061 | 37.5143 | inbeta |  |
| `mhc` | out-owt-large-mhc | 23.7264 | 40.6394 | inbeta |  |
| `mhc-lite` | out-owt-large-mhc-lite | 23.7853 | 39.4057 | inbeta |  |
| `lora-midnorm-outbeta` | out-owt-large-mhc-lora-residual-midnorm-bs8-20000step | 23.8386 | 39.0073 | midnorm |  |
| `GL-midnorm-inbeta` | out-owt-large-mhc-group-lora-midnorm-bs4-20000step | 24.3826 | 37.3373 | inbeta |  |
| `lora-nonorm` | out-owt-large-mhc-lora-residual | 24.5095 | 41.7512 | midnorm |  |
| `GLfake-2` | out-owt-large-mhc-group-lora-bs4-20000step | 28.6492 | 50.1269 | midnorm |  |

## Medium (12L / 768d)

| 实验代号 | 文件夹 | OWT-val PPL | WT103 PPL | matched branch | note |
|---|---|---:|---:|---|---|
| `GL-midnorm-outbeta` | out-owt-medium-mhc-group-lora-midnorm-bs16-10000step | 25.3922 | 42.1238 | midnorm | legacy crashed run; unreliable and excluded from main result |
| `lora-midnorm-inbeta` | out-owt-medium-mhc-lora-residual-midnorm-bs16-10000step | 26.4126 | 48.4726 | inbeta | ca1c90f; this is the valid branch-matched result |
| `mhc-lite` | out-owt-medium-mhc-lite | 25.87 | 47.1103 | inbeta |  |
| `Group` | out-owt-medium-mhc-group-embedding | 25.9358 | 43.8231 | inbeta |  |
| `mhc` | out-owt-medium-mhc | 25.9976 | 43.2997 | inbeta |  |
| `GL-midnorm-inbeta` | out-owt-medium-mhc-group-lora-midnorm-inbeta-bs16-10000step | 26.1925 | 43.9056 | inbeta |  |
| `lora-nonorm` | out-owt-medium-mhc-lora-residual | 29.1801 | 55.8303 | midnorm |  |

## Small (6L / 512d)

| 实验代号 | 文件夹 | OWT-val PPL | WT103 PPL | matched branch | note |
|---|---|---:|---:|---|---|
| `gl-midnorm-inbeta` | out-owt-small-mhc-group-lora-midnorm-inbeta-bs24-10000step | 30.3654 | 60.9367 | inbeta |  |
| `mhc` | out-owt-small-mhc | 30.5009 | 61.0535 | inbeta |  |
| `lora-inbeta` | out-owt-small-mhc-lora-residual-bs24-10000step | 30.5226 | 63.9056 | group_lora | code says inbeta, but weights forward correctly under `group_lora` (out-beta) |
| `Group` | out-owt-small-mhc-group-embedding | 30.5494 | 63.9427 | inbeta |  |
| `mhc-lite` | out-owt-small-mhc-lite | 30.7714 | 60.843 | inbeta |  |
| `GL-out1` | out-owt-small-mhc-group-lora | 31.7147 | 64.5914 | midnorm |  |
