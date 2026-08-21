# Stream-analysis summary (mHC vs mHC-Group-LoRA-midnorm)

Source: `eval/analysis_out/{mhc,mhc_group_lora_midnorm}.analysis.pt` (64 x 1024 training tokens, eval mode, fp32, real fwd+bwd; 24 layers x {attn,mlp}).

Write-gate mask threshold: `beta_s >= 0.001` (a stream with beta below this writes ~nothing, so its write vector is excluded from the write-cosine).

## Write-back collinearity (off-diagonal cosine of per-stream write vectors)

| model | raw write-cos | masked write-cos | active-stream kept |
|---|---|---|---|
| mHC | 0.998 | 1.000 | 75.8% |
| Group-LoRA | 0.926 | 0.925 | 90.3% |

- mHC write is `beta_s * h` -> collinear; masked cos = 1.000 confirms it. The raw <1 value and the low kept-fraction come from streams with `beta_s ~ 0` (silent streams) whose zero write vector makes the cosine degenerate.
- Group-LoRA masked cos < 1 = the LoRA genuinely rotates per-stream write directions apart; its higher kept fraction = more streams stay active.

## Per-layer active-stream KEPT fraction and masked write-cos (attn+mlp avg)

| layer | mHC kept | mHC write-cos | Grp kept | Grp write-cos |
|---|---|---|---|---|
| 0 | 0.45 | 1.000 | 0.67 | 0.957 |
| 1 | 0.13 | 1.000 | 0.69 | 0.935 |
| 2 | 0.04 | 1.000 | 0.82 | 0.916 |
| 3 | 0.55 | 1.000 | 1.00 | 0.944 |
| 4 | 0.58 | 1.000 | 0.48 | 0.969 |
| 5 | 0.33 | 1.000 | 0.95 | 0.956 |
| 6 | 0.47 | 1.000 | 0.78 | 0.864 |
| 7 | 0.50 | 1.000 | 0.95 | 0.915 |
| 8 | 0.99 | 1.000 | 0.72 | 0.932 |
| 9 | 1.00 | 1.000 | 1.00 | 0.923 |
| 10 | 1.00 | 1.000 | 1.00 | 0.944 |
| 11 | 0.77 | 1.000 | 0.98 | 0.884 |
| 12 | 1.00 | 1.000 | 1.00 | 0.905 |
| 13 | 1.00 | 1.000 | 0.97 | 0.874 |
| 14 | 1.00 | 1.000 | 1.00 | 0.912 |
| 15 | 0.75 | 1.000 | 0.89 | 0.856 |
| 16 | 1.00 | 1.000 | 1.00 | 0.945 |
| 17 | 0.99 | 1.000 | 1.00 | 0.902 |
| 18 | 1.00 | 1.000 | 1.00 | 0.915 |
| 19 | 0.75 | 1.000 | 0.94 | 0.960 |
| 20 | 0.91 | 1.000 | 0.98 | 0.907 |
| 21 | 1.00 | 1.000 | 0.97 | 0.938 |
| 22 | 0.99 | 1.000 | 0.93 | 0.990 |
| 23 | 1.00 | 1.000 | 0.95 | 0.983 |
