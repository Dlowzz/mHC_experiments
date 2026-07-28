# mHC checkpoints - final summary (OWT-val/WT103 PPL + downstream)

PPL: deterministic full-pass, branch-matched (min-CE). Downstream: lm_eval 0-shot, batched.
`(*)` = experiment code's beta-mode disagrees with the branch the weights actually match.

## XL (28L/1280d)

| 实验代号 | OWT-val PPL | WT103 PPL | lambada PPL | lambada acc | sciq | piqa | arc_easy | winogrande |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `group` | 21.04 | 32.64 | 68.45 | 0.267 | 0.641 | 0.598 | 0.400 | 0.510 |
| `lora-midnorm-inbeta` | 21.08 | 32.86 | 54.46 | 0.285 | 0.634 | 0.605 | 0.385 | 0.515 |
| `mhc` | 21.19 | 33.57 | 57.85 | 0.270 | 0.613 | 0.590 | 0.391 | 0.517 |
| `GL-midnorm-inbeta` | 21.41 | 33.61 | 66.96 | 0.258 | 0.633 | 0.602 | 0.385 | 0.513 |

_XL `group` 是唯一跑满 30k 的 run（bs8×ga8，有效 batch 与其它 XL 相同），其余三个 best-val 停在 28–28.5k；其 PPL 优势含额外训练量因素。_

## Large (24L/1024d)

| 实验代号 | OWT-val PPL | WT103 PPL | lambada PPL | lambada acc | sciq | piqa | arc_easy | winogrande |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `group` | 23.59 | 37.62 | 93.23 | 0.252 | 0.583 | 0.597 | 0.375 | 0.504 |
| `lora-midnorm-inbeta` | 23.71 | 37.51 | 87.95 | 0.241 | 0.627 | 0.585 | 0.372 | 0.527 |
| `mhc` | 23.73 | 40.64 | 88.38 | 0.246 | 0.624 | 0.600 | 0.379 | 0.510 |
| `mhc-lite` | 23.79 | 39.41 | 90.05 | 0.252 | 0.604 | 0.596 | 0.375 | 0.515 |
| `lora-midnorm-outbeta` | 23.84 | 39.01 | 85.17 | 0.248 | 0.634 | 0.588 | 0.368 | 0.495 |
| `GL-midnorm-inbeta` | 23.44 | 35.26 | 84.85 | 0.256 | 0.618 | 0.588 | 0.366 | 0.499 |
| `lora-nonorm` | 24.51 | 41.75 | 102.71 | 0.239 | 0.592 | 0.578 | 0.366 | 0.500 |
| `GLfake-2` | 28.65 | 50.13 | 167.42 | 0.211 | 0.540 | 0.587 | 0.356 | 0.509 |

## Medium (12L/768d)

| 实验代号 | OWT-val PPL | WT103 PPL | lambada PPL | lambada acc | sciq | piqa | arc_easy | winogrande |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mhc-lite` | 25.87 | 47.11 | 102.10 | 0.242 | 0.587 | 0.590 | 0.352 | 0.506 |
| `Group` | 25.94 | 43.82 | 101.24 | 0.253 | 0.586 | 0.589 | 0.361 | 0.505 |
| `mhc` | 26.00 | 43.30 | 107.78 | 0.248 | 0.609 | 0.585 | 0.353 | 0.532 |
| `GL-midnorm-inbeta` | 26.19 | 43.91 | 94.10 | 0.237 | 0.619 | 0.588 | 0.361 | 0.473 |
| `lora-midnorm-inbeta` | 26.41 | 48.47 | 93.03 | 0.238 | 0.620 | 0.577 | 0.351 | 0.508 |
| `lora-nonorm` | 29.18 | 55.83 | 162.72 | 0.215 | 0.599 | 0.578 | 0.360 | 0.519 |
| `GL-midnorm-outbeta` | - | - | - | - | - | - | - | - | legacy crashed run; excluded |

## Small (6L/512d)

| 实验代号 | OWT-val PPL | WT103 PPL | lambada PPL | lambada acc | sciq | piqa | arc_easy | winogrande |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lora-midnorm-inbeta` | 30.34 | 55.39 | 196.94 | 0.205 | 0.557 | 0.578 | 0.349 | 0.522 |
| `gl-midnorm-inbeta` | 30.37 | 60.94 | 174.81 | 0.215 | 0.593 | 0.579 | 0.353 | 0.504 |
| `mhc` | 30.50 | 61.05 | 168.47 | 0.221 | 0.605 | 0.580 | 0.355 | 0.485 |
| `lora-inbeta(*grouplora)` | 30.52 | 63.91 | 163.42 | 0.217 | 0.593 | 0.582 | 0.346 | 0.518 |
| `Group` | 30.55 | 63.94 | 162.80 | 0.218 | 0.582 | 0.583 | 0.347 | 0.496 |
| `mhc-lite` | 30.77 | 60.84 | 216.47 | 0.191 | 0.587 | 0.566 | 0.358 | 0.520 |
| `GL-out1` | 31.71 | 64.59 | 236.43 | 0.194 | 0.589 | 0.565 | 0.343 | 0.494 |

_hellaswag / openbookqa / copa omitted from table: ~chance (0.25/0.25/0.50) at these scales._

_Verified: each row's eval PPL matches the ckpt's recorded best_val_loss (correct training branch)._
_Excluded `GL-midnorm-outbeta` (out-owt-medium-mhc-group-lora-midnorm-bs16): crashed run (wandb owt-M-group-lora-midnorm-8489 @199654a), recorded val≈28.5; no checked-out branch reproduces it, so its eval is unreliable._