# mHC 实验 ↔ Git 版本 对照表

> 目的：把每个 checkpoint / wandb run 对应到**训练时的 git commit**，以免以后忘记某个 ckpt 该用哪份代码评测。
> 数据来源：wandb（run 的 `commit` 字段 = 权威）+ `git show`（commit 含义）。生成日期 2026-07-23。

## 0. 最重要的两条结论

1. **判断一个 ckpt 用的什么代码版本，唯一可靠依据是它 wandb run 的 `commit` 字段**（`wandb.Api().run(id).commit`）。
   - ckpt 里的 `config` **不记录 git commit，也不记录 `lora_lambda`**（后者是 hyper_conn 模块内部参数，非 train.py config 变量）→ **不能**靠 ckpt config 推断代码版本。
   - 要正确评测某 ckpt：先 `git checkout <它的 commit>`，再跑 eval。用错 commit 的前向会得到垃圾/虚高 PPL（甚至下游 NaN），看着像"灾难性退化"，其实是评测错分支。

2. **`out-owt-medium-mhc-lora-residual-midnorm-bs16-10000step`（run `owt-M-lora-res-midnorm-inbeta-2241`）确实是 in-beta 代码**：
   - commit = **`ca1c90f`**（= "Move LoRA write inside beta"，当前 `midnorm-lora-in-beta` 分支 HEAD）。
   - wandb 记录 val/loss = **3.2733**（PPL≈26.4），**完全健康**，就在 M 档正常区间。
   - 在当前主树（ca1c90f）上重评得 **OWT-val PPL 26.41 / WT103 48.47**，与训练记录吻合。
   - 之前出现的 54 / 28434 / 下游 NaN 都是**评测用错分支**的假象，**不是模型退化**。

## 1. Git commit 谱系（`git show -s`）

| commit | 日期 | 含义 / 引入的东西 | 分支 |
|---|---|---|---|
| `ea19775` | 01-12 | README（基线，早期 mhc/mhc_lite/embedding 用它） | main |
| `2380851` | 07-12 | mHC-embedding 低秩分支 | - |
| `df7dc15` | 07-13 | **加 mHC-LoRA-Residual + OrthogonalDiff（最初版，无 norm）** | - |
| `f0c2d53` | 07-14 | mHC-group-embedding（H_pre/H_post 读写） | - |
| `4b88713` | 07-14 | group-embedding 仅 H_pre 读 | - |
| `e1b228f` | 07-15 | **加 mHC-group-LoRA（per-stream LoRA 写回）** | - |
| `7a9d957` | 07-18 | **LoRA output/mid RMSNorm 变体（final_v0）**；`mhc_lora_residual_midnorm` 诞生于此 | final_v0 |
| `82637da` | 07-20 | 加 lora-residual affine-midnorm 变体 + rmsnorm 参数不做 cosine decay | midnorm |
| `199654a` | 07-20 | 修 midnorm LoRA A 的 fan_in 初始化 + 把尺度不变的 A 排除出 weight decay | **midnorm (HEAD)** |
| `4942e3c` | 07-21 | midnorm cleanups（rmsnorm affine 正常 LR、group-midnorm 单次初始化） | midnorm |
| `ca1c90f` | 07-21 | **LoRA 写进 beta 内：`u_s = β_s(h + λ·δ_s)`，新增 `lora_lambda`(默认1.0)** | **midnorm-lora-in-beta (HEAD, 当前)** |
| `d340322` | ? | （未取到 message）"lorabeta" 变体，疑似 group-lora in-beta | - |

## 2. wandb run ↔ commit ↔ 设置 ↔ val（按 commit 归组）

### ca1c90f — LoRA-in-beta（当前分支；评测用主树即可）
| run | hyper_conn_type | bs / lr / iters | 档 | val |
|---|---|---|---|---|
| L-lora_residual_midnorm-inbeta | mhc_lora_residual_midnorm | 8 / 3e-4 / 20k | L | 3.1634 |
| **owt-M-lora-res-midnorm-inbeta-2241** | mhc_lora_residual_midnorm | 16 / 6e-4 / 10k | M | **3.2733** |
| owt-S-lora-res-midnorm-inbeta-1338 | mhc_lora_residual_midnorm | 24 / 1e-3 / 10k | S | 3.4176 |
| M-group_lora_midnorm-inbeta | mhc_group_lora_midnorm | 16 / 6e-4 / 10k | M | 3.2601 |
| S-group_lora_midnorm-inbeta | mhc_group_lora_midnorm | 24 / 1e-3 / 10k | S | 3.418 |
| owt-L-group-lora-midnorm-ffn-9709 | mhc_group_lora_midnorm_ffn | 4 / 3e-4 / 20k | L | 3.3512 (running) |
| S-mhc-owt-bs24-seed1338 | mhc | 24 / 1e-3 / 10k | S | 3.4016 |

### 199654a — midnorm 分支 HEAD（outbeta；评测用 `git checkout midnorm`）
| run | hyper_conn_type | bs/lr/iters | 档 | val |
|---|---|---|---|---|
| exp-2519（= L midnorm outbeta 基线） | mhc_lora_residual_midnorm | 8 / 3e-4 / 20k | L | 3.1798 |

### 7a9d957 — final_v0（output/mid RMSNorm 变体；`git checkout final_v0` 或该 commit）
| run | hyper_conn_type | bs/lr/iters | 档 | val |
|---|---|---|---|---|
| L-lora_residual-outRMSNorm-final_v0 | mhc_lora_residual | 8 / 3e-4 / 20k | L | 3.1975 |
| M-lora_residual_midnorm | mhc_lora_residual_midnorm | 16 / 6e-4 / 10k | M | 3.239 |
| M-lora_residual-scalar0.01 | mhc_lora_residual_scalar | 16 / 6e-4 / 10k | M | 3.2505 |
| M-lora_residual_affinemidnorm | mhc_lora_residual_affinemidnorm | 16 / 6e-4 / 10k | M | 3.2671 |
| M-lora_residual-affineRMSNorm(-nobias) | mhc_lora_residual_affine(_nobias) | 16 / 6e-4 / 10k | M | 3.2915 / 3.2975 |
| M/owt-M-lora-res-*-ffn | mhc_lora_residual_affine_ffn | 16 / 6e-4 / 10k | M | 3.28~3.58 (多为 crashed) |

### e1b228f — mHC-group-LoRA（per-stream LoRA 写回）
| run | hyper_conn_type | bs/lr/iters | 档 | val |
|---|---|---|---|---|
| L-mhc-group-embedding-Hpre-only | mhc_group_embedding | 8 / 3e-4 / 20k | L | 3.1513 |
| L-mhc-group-lora-scale0.1 | mhc_group_lora | 8 / 3e-4 / 20k | L | 3.1708 |
| L-mhc_group_lora(-bs8eff) | mhc_group_lora | 4~8 / 3e-4 / 20k | L | 3.2069 / exp-7084 3.1798 |
| S-mhc-group-embedding / S-mhc_group_lora | mhc_group_embedding / mhc_group_lora | 24 / 1e-3 / 10k | S | 3.411 / 3.4161 |

### df7dc15 — 最初的 LoRA-Residual（无 norm）
| run | hyper_conn_type | bs/lr/iters | 档 | val |
|---|---|---|---|---|
| L-mhc-lora-residual-20k（原始 no-norm） | mhc_lora_residual | 8 / 3e-4 / 20k | L | 3.1714 |
| M-hc | hc | 16 / 6e-4 / 10k | M | 3.2718 |
| S-mhc_lora_residual | mhc_lora_residual | 24 / 1e-3 / 10k | S | 3.4193 |

### 其它
- `4b88713`：M-mhc-group-lora-10k（mhc_group_lora，val 3.2439）
- `f0c2d53`：M-mhc-group-embedding-Hpre-only（3.2595）、M-shc（3.2451）
- `2380851`：LORA-M（mhc_lora_residual，3.2523）、shc-ablation-M（mhc_orthogonal_diff，3.253）
- `ea19775`：owt-L-mhclite（3.1591）、M-mhc-embedding（3.2779）、mhc[S 基线]（3.4105）
- `d340322`：L-group_lora_midnorm_lorabeta（mhc_group_lora_midnorm，bs4/20k，**val 3.15，L 档最好**）— commit message 未取到，疑似 group-lora in-beta 变体，值得回头确认。
- **commit 未记录（wandb 没抓到 git）**：L-mhc-20k-replay(3.1552)、M-mhc-10k-replay(3.2612)、owt-M-mhclite(3.2567)、owt-S-mhclite(3.4257)。

## 3. 正确评测某 ckpt 的流程
1. 在 wandb 找到该 ckpt 对应的 run（按 `out_prefix_model` + `hyper_conn_type` + bs/step 匹配），读 `run.commit`。
2. `git checkout <commit>`（或对应分支：ca1c90f=midnorm-lora-in-beta，199654a=midnorm，7a9d957=final_v0）。
3. 用 `eval/run_all.py`（PPL）/ `eval/run_tasks.py`（下游）评测。
4. 用 wandb 记录的 `val/loss` 做 sanity check：对得上才说明分支选对了。
