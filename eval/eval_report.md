# mHC checkpoint evaluation - Tier-1 (OWT-val + WikiText-103 PPL)

Deterministic full-pass PPL, fp32, eval mode. For each checkpoint the branch giving the
lowest OWT-val CE is reported (a branch-mismatched forward inflates CE).

- **branch**: git branch used for the reported (min-CE) result.
- **note**: flagged when best OWT-val CE > 4.0 (likely wrong code branch; needs its training branch).

## Large (24L / 1024d)

| variant | type | iter | branch | OWT-val PPL | OWT-val CE | WT103 PPL | note |
|---|---|---:|---|---:|---:|---:|---|
| out-owt-large-mhc-group-embedding | mhc_group_embedding | 19500 | inbeta | 23.5872 | 3.1607 | 37.6221 |  |
| out-owt-large-mhc-lora-residual-midnorm-inbeta-bs8-20000step | mhc_lora_residual_midnorm | 18500 | inbeta | 23.7061 | 3.1657 | 37.5143 |  |
| out-owt-large-mhc | mhc | 19500 | inbeta | 23.7264 | 3.1666 | 40.6394 |  |
| out-owt-large-mhc-lite | mhc_lite | 19500 | inbeta | 23.7853 | 3.1691 | 39.4057 |  |
| out-owt-large-mhc-lora-residual-midnorm-bs8-20000step | mhc_lora_residual_midnorm | 20000 | midnorm | 23.8386 | 3.1713 | 39.0073 |  |
| out-owt-large-mhc-group-lora-midnorm-bs4-20000step | mhc_group_lora_midnorm | 16000 | inbeta | 24.3826 | 3.1939 | 37.3373 |  |
| out-owt-large-mhc-lora-residual | mhc_lora_residual | 18500 | midnorm | 24.5095 | 3.1991 | 41.7512 |  |
| out-owt-large-mhc-group-lora-bs4-20000step | mhc_group_lora | 11500 | midnorm | 28.6492 | 3.3551 | 50.1269 |  |
| out-owt-large-mhc-group-lora | mhc_group_lora | 20000 | inbeta | n/a | n/a | n/a | unresolved: needs exact training branch/commit |
| out-owt-large-mhc-group-lora-bs8-20000step | - | - | - | - | - | - | missing ckpt.pt |

## Medium (12L / 768d)

| variant | type | iter | branch | OWT-val PPL | OWT-val CE | WT103 PPL | note |
|---|---|---:|---|---:|---:|---:|---|
| out-owt-medium-mhc-group-lora-midnorm-bs16-10000step | mhc_group_lora_midnorm | 10000 | midnorm | 25.3922 | 3.2344 | 42.1238 |  |
| out-owt-medium-mhc-lora-residual-midnorm-bs16-10000step | mhc_lora_residual_midnorm | 10000 | midnorm | 25.7201 | 3.2473 | 43.598 |  |
| out-owt-medium-mhc-lora-residual-local-jul13 | mhc_lora_residual | 10000 | group_lora | 25.8425 | 3.252 | 42.7151 |  |
| out-owt-medium-mhc-lite | mhc_lite | 10000 | inbeta | 25.87 | 3.2531 | 47.1103 |  |
| out-owt-medium-mhc-group-embedding | mhc_group_embedding | 10000 | inbeta | 25.9358 | 3.2556 | 43.8231 |  |
| out-owt-medium-mhc | mhc | 10000 | inbeta | 25.9976 | 3.258 | 43.2997 |  |
| out-owt-medium-mhc-group-lora-midnorm-inbeta-bs16-10000step | mhc_group_lora_midnorm | 9500 | inbeta | 26.1925 | 3.2655 | 43.9056 |  |
| out-owt-medium-hc-bs16-10000step | hc | 10000 | inbeta | 26.2579 | 3.268 | 46.1988 |  |
| out-owt-medium-mhc-lora-residual | mhc_lora_residual | 7500 | midnorm | 29.1801 | 3.3735 | 55.8303 |  |
| out-owt-medium-mhc-group-lora | mhc_group_lora | 9500 | inbeta | n/a | n/a | n/a | unresolved: needs exact training branch/commit |
| out-owt-medium-mhc-lora-residual-scalar | - | - | - | - | - | - | ValueError: Invalid hyper connection type: mhc_lora_residual_scalar |

## Small (6L / 512d)

| variant | type | iter | branch | OWT-val PPL | OWT-val CE | WT103 PPL | note |
|---|---|---:|---|---:|---:|---:|---|
| out-owt-small-mhc | mhc | 10000 | inbeta | 30.5009 | 3.4178 | 61.0535 |  |
| out-owt-small-mhc-lora-residual-bs24-10000step | mhc_lora_residual | 9500 | group_lora | 30.5226 | 3.4185 | 63.9056 |  |
| out-owt-small-mhc-group-embedding | mhc_group_embedding | 10000 | inbeta | 30.5494 | 3.4193 | 63.9427 |  |
| out-owt-small-mhc-lite | mhc_lite | 10000 | inbeta | 30.7714 | 3.4266 | 60.843 |  |
| out-owt-small-mhc-group-lora | mhc_group_lora | 10000 | midnorm | 31.7147 | 3.4568 | 64.5914 |  |
| smoke-owt-small-mhc-lite | mhc_lite | 20 | inbeta | 4825.9042 | 8.4818 | 16606.312 | smoke ckpt (20 steps) - not comparable |
| out-owt-small-mhc-group-lora-bs24-10000step | mhc_group_lora | 9500 | inbeta | n/a | n/a | n/a | unresolved: needs exact training branch/commit |
| out--small-mhc-lite | mhc_lite | 20 | - | - | - | - | RuntimeError: CUDA error: device-side assert triggered |
