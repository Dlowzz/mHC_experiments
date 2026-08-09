# mHC-lite — 训练 & 评测指南 (branch `final`)

nanoGPT 派生的 mHC (Manifold-constrained Hyper-Connections) 消融实验代码库。
本分支 `final` 的 LoRA 变体默认是 **midnorm + in-beta**（LoRA 写入 beta 内：`u_s = β_s(h + λ·δ_s)`）。

## 环境
- venv：`/home/work/data/guotianzizhe/venv`（含 torch / lm_eval / datasets / tiktoken / wandb）
- 硬件：2×L20 DDP；wandb 在线；评测拉数据集走 HF 镜像 `HF_ENDPOINT=https://hf-mirror.com`
- 数据：`data/openwebtext/{train,val}.bin`（GPT-2 BPE, vocab 50304, block 1024）

## 一、模型档位 `config/<size>_model.py`
| 档 | n_layer / n_embd / n_head | 参数量 | lr | bs / ga | steps |
|---|---|---|---|---|---|
| S  | 6 / 512 / 8   | ~30M  | 1e-3 | 24 / - | 10k |
| M  | 12 / 768 / 12 | ~124M | 6e-4 | 16 / - | 10k |
| L  | 24 / 1024 / 16| ~350M | 3e-4 | 8 / -  | 20k |
| XL | 28 / 1280 / 20| ~620M | 2e-4 | 4 / 16 | 30k |

## 二、实验变体 `config/with_<method>.py`（主实验 4 类）
| 类别 | config | hyper_conn_type | 说明 |
|---|---|---|---|
| **mhc**（基线） | `with_mhc.py` | `mhc` | 标准 mHC |
| **mhc-group** | `with_mhc_group_embedding.py` | `mhc_group_embedding` | 分组 H_pre/H_post 读写 |
| **mhc-lora** | `with_mhc_lora_residual_midnorm.py` | `mhc_lora_residual_midnorm` | 新版 `ca1c90f` 为 midnorm + in-beta；旧 `199654a` 为 out-beta |
| **mhc-group-lora** | `with_mhc_group_lora_midnorm.py` | `mhc_group_lora_midnorm` | 新版 `ca1c90f` 为 midnorm + in-beta；旧 checkpoint 按训练 commit 判断 |

> 本分支（`perf`）只保留这 4 类主实验 + `mhc_lite` + `none`（plain GPT）共 6 个可选 `hyper_conn_type`；`hc` / `mhc_embedding` / `mhc_orthogonal_diff` / `mhc_group_lora_capped` / `mhc_lora_residual`(no-norm) / `mhc_group_lora`(no-norm) / `mhc_lora_residual_affinemidnorm` / `analysis` 及其 config、单测都已删除，传这些类型会直接报错。要训练或评测它们的旧 ckpt 请切到 `final` 分支。
> `mhc_lora_residual.py` 与 `mhc_group_lora.py` 两个文件仍保留，因为两个 midnorm 变体继承自它们（但类型本身不可选）。
> **重要**：LoRA 系变体的前向依赖分支代码。`final`(=in-beta) 上的 LoRA ckpt 必须在 in-beta 代码下评测；老 ckpt 要按其训练 commit 评测，见 `eval/experiment_commit_map.md`。

## 三、训练
```bash
source /home/work/data/guotianzizhe/venv/bin/activate
WANDB_MODE=online CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 train.py \
  config/train_owt.py config/<size>_model.py config/with_<method>.py \
  --compile=True --wandb_run_name=<size>-<method>-owt
```
- config 组合顺序：`train_owt.py`（数据/默认）→ `<size>_model.py`（尺寸/lr/steps/bs）→ `with_<method>.py`（变体）；`--key=val` 可再覆盖。
- 有效 batch = `ga × world_size × bs × block`（DDP 会把 `ga` 除以卡数）。
- 输出目录自动命名：`out-owt-<size>-<method>-bs<bs>-<iters>step/ckpt.pt`（best-val 保存）。

XL mhc 示例（已完成；新 XL 主实验结果见 `data/test/ckpt_experiment_map.md`）：
```bash
... torchrun ... train.py config/train_owt.py config/xl_model.py config/with_mhc.py \
  --compile=True --wandb_run_name=XL-mhc-owt-bs4ga16-30kstep
```

## 四、评测 `eval/`（不需重训，读 ckpt 评测）
| 脚本 | 作用 |
|---|---|
| `eval/loader.py` | 载入 ckpt（去 `_orig_mod.` 前缀）+ `full_logits`（targets-path 取全序列 logits） |
| `eval/run_all.py` | **Tier-1**：OWT-val 全量确定性 PPL + WikiText-103 rolling PPL，逐 ckpt 存 `eval/results/bulk/` |
| `eval/run_tasks.py` | **Tier-2**：lm-eval 下游（lambada + hellaswag/piqa/arc_easy/winogrande/sciq/openbookqa/copa），存 `eval/results/tasks/` |
| `eval/lm_adapter.py` | lm-eval 自定义 `LM`（批处理 loglikelihood；mHC 非 HF 架构，不能用 `--model hf`） |
| `eval/make_report.py` / `eval/summary.py` | 汇总成 S/M/L markdown 表 |
| `eval/report_coded.py` | 带实验代号的表 |

评测命令（走 HF 镜像）：
```bash
HF_ENDPOINT=https://hf-mirror.com python eval/run_all.py  --dirs <ckpt_dir> --wt103 --device cuda:0 --out_dir eval/results/bulk
HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_TRUST_REMOTE_CODE=1 \
  python eval/run_tasks.py --dirs <ckpt_dir> --device cuda:0 --out_dir eval/results/tasks
python eval/summary.py    # 生成 eval/eval_summary.md
```
**分支匹配校验**：每个 ckpt 应在其训练 commit 下评测；用 ckpt 记录的 `best_val_loss` 做 sanity check（评出的 PPL ≈ exp(best_val_loss) 才说明选对了代码）。详见 `eval/experiment_commit_map.md`。

## 五、参考文档
- `eval/experiment_commit_map.md` — 每个 run ↔ git commit ↔ 设置 的对照，以及"怎么给某 ckpt 选对评测 commit"。
- `eval/eval_summary.md` / `eval/eval_report_coded.md` — 最新 S/M/L/XL 评测结果。
- `/home/work/data/guotianzizhe/data/test/ckpt_experiment_map.md` — `data/test` 全部 checkpoint、实验名、训练进度、beta 语义和评测状态。
