# mHC-lite 训练交接文档（给其他 comate 进程）

> 目的：让并行的其他 comate 进程了解当前实验现状，并**与现有训练设置保持一致**。
> 初版实验是 **S / M / L 三档**（已完成，ckpt 已评测）；现在**新增 XL 档（~620M）**，正在训练。
> 分支：`final`（LoRA 变体 = **midnorm + in-beta**，`u_s = β_s(h + λ·δ_s)`，commit `ca1c90f` 起）。
>
> 具体的**训练/评测怎么跑**、**评测脚本**、**ckpt↔git版本对照**都已有专门文档，本文件**不重复**，只做设置对齐 + 环境/tmux 细节，并指向它们：
> - 训练 & 评测 how-to：`TRAIN_EVAL.md`
> - 每个 run ↔ commit ↔ 设置 ↔ val 对照、以及给某 ckpt 选对评测分支：`eval/experiment_commit_map.md`

---

## 一、环境

- venv（**不要污染**，直接用）：`source /home/work/data/guotianzizhe/venv/bin/activate`
  （已含 torch / lm_eval / datasets / tiktoken / wandb）
- 硬件：2×L20（46GB），DDP：`torchrun --standalone --nproc_per_node=2`
- 训练卡：**当前 XL 用 `CUDA_VISIBLE_DEVICES=2,3`**（0,1 留空/他用；起新实验前先 `nvidia-smi` 确认目标卡空闲）
- wandb：在线，`wandb_project='mhc-lite'`，user `guotzzz`
- 数据：`data/openwebtext/{train,val}.bin`（GPT-2 BPE, vocab 50304, block 1024）
- 评测拉数据集走 HF 镜像：`HF_ENDPOINT=https://hf-mirror.com`

## 二、tmux 约定

- 每个训练跑在**独立 detached tmux 会话**里，会话名 = 实验短名（如 `xl-mhc`）。
- 起法：
  ```bash
  tmux new-session -d -s <name> -c /home/work/data/guotianzizhe/project/mhc-lite
  tmux send-keys -t <name> "source /home/work/data/guotianzizhe/venv/bin/activate" C-m
  tmux send-keys -t <name> "<训练命令，见第四节> 2>&1 | tee logs/<name>.log" C-m
  ```
- 查看：`tmux ls` / `tmux attach -t <name>`（detach：`Ctrl-b d`）；或直接看 `logs/<name>.log`。
- 训练日志同时 `tee` 到 `logs/<实验名>.log`，方便不进 tmux 直接 `tail`。
- **停止**：`tmux kill-session -t <name>`（清理全部：`tmux kill-server`）。

## 三、各档实验设置

共享（全部档位一致）：`dropout=0.0`、`warmup_iters=200`、`weight_decay=0.1`、`beta1=0.9`、`beta2=0.95`、`grad_clip=1.0`、`block_size=1024`、`dtype=bfloat16`、`vocab=50304`。

| 档 | 配置文件 | n_layer / n_head / n_embd | 参数量(mhc) | lr → min_lr | batch_size | grad_accum | steps | 有效 tokens/iter |
|---|---|---|---|---|---|---|---|---|
| **S** | `config/small_model.py`  | 6 / 8 / 512   | ~30M  | 1e-3 → 1e-4 | 24 | 8 | 10k | 196,608 |
| **M** | `config/medium_model.py` | 12 / 12 / 768 | ~124M | 6e-4 → 6e-5 | 16 | 8 | 10k | 131,072 |
| **L** | `config/large_model.py`  | 24 / 16 / 1024| ~350M | 3e-4 → 3e-5 | 8  | 8 | 20k | 65,536  |
| **XL**| `config/xl_model.py`     | 28 / 20 / 1280| ~620M | 2e-4 → 2e-5 | 4  | 16| 30k | 65,536  |

> **有效 batch = grad_accum × batch_size × block**（DDP 会把 `grad_accum` 除以卡数，world_size 抵消，故与卡数无关）。
> tokens/iter：S 196,608 / M 131,072 / L 65,536 / XL 65,536。

### ⚠️ S/M/L 的配置文件与实际跑的设置**不完全一致**（务必看这里，否则复现不上）

`small/medium/large_model.py` **只设了** `n_layer/n_head/n_embd/lr/max_iters/warmup/wd/betas/grad_clip`，**没有设 `batch_size`**（默认继承 `train_owt.py` 的 `batch_size=16`、`gradient_accumulation_steps=8`）。初版真实 run（见 `eval/experiment_commit_map.md`）用了**命令行覆盖**：

- **S**：`--batch_size=24`（配置默认是 16 → 必须覆盖）；steps 10k 与配置一致。
- **M**：无需覆盖（bs=16、10k 恰好等于默认/配置）。
- **L**：`--batch_size=8 --max_iters=20000 --lr_decay_iters=20000`（**配置文件里写的是 10000，实际跑 20000**，必须覆盖）。
- **XL**：**全部已写进 `config/xl_model.py`**（bs=4、ga=16、30k），**无需任何覆盖**。

> 个别显存重的变体（如 L 档 group-lora）会把 bs 再降到 4，逐 run 的准确 bs 以 `eval/experiment_commit_map.md` 为准。

## 四、启动命令（config 顺序：train_owt → size → method）

统一模板见 `TRAIN_EVAL.md` 第三节；下面给**与初版一致**的可复现命令（`<method>` 见第五节）：

```bash
source /home/work/data/guotianzizhe/venv/bin/activate

# S 档（注意 --batch_size=24）
WANDB_MODE=online CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 train.py \
  config/train_owt.py config/small_model.py config/with_<method>.py \
  --batch_size=24 --compile=True --wandb_run_name=S-<method>-owt

# M 档（无需覆盖）
WANDB_MODE=online CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 train.py \
  config/train_owt.py config/medium_model.py config/with_<method>.py \
  --compile=True --wandb_run_name=M-<method>-owt

# L 档（注意 bs=8 且 20k 步）
WANDB_MODE=online CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 train.py \
  config/train_owt.py config/large_model.py config/with_<method>.py \
  --batch_size=8 --max_iters=20000 --lr_decay_iters=20000 \
  --compile=True --wandb_run_name=L-<method>-owt

# XL 档（bs/ga/steps 已在配置里，无需覆盖）
WANDB_MODE=online CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 train.py \
  config/train_owt.py config/xl_model.py config/with_<method>.py \
  --compile=True --wandb_run_name=XL-<method>-owt-bs4ga16-30kstep
```

- 输出目录自动命名：`out-owt-<size>-<method>-bs<bs>-<iters>step/ckpt.pt`（`always_save_checkpoint=False`，只存 best-val）。
- 若同名目录会被覆盖，用 `--out_prefix_method=<method>-<tag>` 加后缀区分。

## 五、主实验 4 类变体（`config/with_<method>.py`）

| 类别 | config | hyper_conn_type |
|---|---|---|
| **mhc**（基线） | `with_mhc.py` | `mhc` |
| **mhc-group** | `with_mhc_group_embedding.py` | `mhc_group_embedding` |
| **mhc-lora** | `with_mhc_lora_residual_midnorm.py` | `mhc_lora_residual_midnorm`（midnorm + in-beta） |
| **mhc-group-lora** | `with_mhc_group_lora_midnorm.py` | `mhc_group_lora_midnorm`（midnorm + in-beta） |

> LoRA 系变体前向**依赖分支代码**：`final`(=in-beta) 上训的 ckpt 必须在 in-beta 代码下评测；老 ckpt 按其训练 commit 评测。详见 `eval/experiment_commit_map.md`。

## 六、XL 实验与评测状态（更新于 2026-07-28）

- 四个 XL 主实验 checkpoint 已集中到 `/home/work/data/guotianzizhe/data/test/` 并完成统一评测：OWT-val、WikiText-103 以及 8 个 lm-eval 0-shot 任务。
- **XL-mhc**：`out-owt-xl-mhc-bs4-30000step`，best-val checkpoint 保存于 28500 step；OWT PPL 21.1896，WT103 PPL 33.5740。
- **XL-mhc-group**：`out-owt-xl-mhc-group-embedding-bs8-30000step`，**bs8×ga8**（有效 batch 65,536 与其它 XL 一致），跑满 30000 step；OWT PPL 21.0420，WT103 PPL 32.6361。
- **XL-mhc-lora**：`out-owt-xl-mhc-lora-residual-midnorm-bs4-30000step`，in-beta，保存于 28000 step；OWT PPL 21.0796，WT103 PPL 32.8556。
- **XL-mhc-group-lora**：`out-owt-xl-mhc-group-lora-midnorm-bs4-30000step`，in-beta，保存于 28000 step；OWT PPL 21.4107，WT103 PPL 33.6073。
- ⚠️ 跨 XL 比较时注意：只有 group 跑满 30k，其余三个 best-val 停在 28–28.5k，PPL 差距含训练量因素。
- L 档 `mhc-group-lora-midnorm-ffn` 是效果不佳后中途停止的中间实验，不纳入评测。

## 七、参考文档（本文件不重复其内容）

- `TRAIN_EVAL.md` — 训练 & 评测完整 how-to（环境、config 组合、评测流水线、分支匹配校验）。
- `eval/experiment_commit_map.md` — 每个 run ↔ git commit ↔ 设置 ↔ val 对照；给某 ckpt 选对评测分支的流程。
- `eval/eval_summary.md` / `eval/eval_report_coded.md` — 最新 S/M/L 评测结果表。
