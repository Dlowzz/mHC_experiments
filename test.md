
“真正 test”其实有两层含义

1. 独立的语言建模 test loss

也就是与 OWT validation 同分布，但从未参与：

训练
checkpoint 选择
超参数选择
消融决策

这才是严格意义上的 OWT test。

你当前的数据处理中不存在这个集合。而且不能在训练结束后从现有 train.bin 再抽一部分当 test，因为模型已经看过那些 token。

下一轮实验建议预先固定为：

train: 99.90%
val:    0.05%
test:   0.05%

按目前规模，val 和 test 都约有 4000 篇文档、440 万 token，已经足够稳定地估计 loss。流程应当是：

full_train
├── train       # 只用于参数更新
├── validation  # 选 checkpoint、调超参数
└── test        # 所有实验完成后只跑一次

必须在文档级别先划分，再分别 tokenize，不要对连续 token 流随机切窗口，否则相邻内容可能跨 split 泄漏。

2. 下游 benchmark 的“测试”

LLM 论文里说的 test benchmark，并不一定真的使用名为 test 的 split。很多数据集的公开 test 标签不可见，所以统一评测框架会使用公开 validation：

HellaSwag：使用 validation
PIQA：使用 validation
ARC-Easy：使用 test
SciQ：使用 test
OpenBookQA：使用 test
LAMBADA OpenAI：使用 test

所以论文中的“zero-shot test”通常表示下游泛化评测，而不保证底层 split 一定叫 test。

最适合你当前实验的评测组合

我建议分成三层。

A. 必做：HC-compatible 零样本评测

直接沿用 HC 的七任务组合：

hellaswag
piqa
arc_easy
winogrande
sciq
openbookqa
copa

主要报告：

每个任务的 accuracy
支持时优先报告 acc_norm
七任务 macro average
baseline、HC、mHC、你的方法逐任务对比

这组任务对 340M 模型仍然能产生比较明显的区分，而且和 HC 论文的评价口径最接近。

B. 强烈建议：语言建模泛化

加入：

LAMBADA OpenAI test
WikiText-103 test

其中：

LAMBADA 同时报告最后一个词预测 accuracy 和 perplexity，能观察长上下文条件利用能力；lm-eval 的 lambada_openai 明确使用 test split。
WikiText-103 test 用于观察从网页文本训练迁移到百科文本的 OOD language-modeling loss/PPL。HC 本身也把 WikiText-103 纳入了跨领域 validation suite。

注意，lm-eval 当前内置名为 wikitext 的任务实际上是 WikiText-2，而不是 WikiText-103；WikiText-103 最好单独写 rolling-loss evaluator，别被名字坑了。

C. 次要挑战集

再补两个：

arc_challenge
mmlu

但建议单列，不要混入主平均分：

ARC-Challenge 可以观察架构收益是否迁移到更难推理。
MMLU 对当前模型大概率接近随机水平，可以作为参考曲线，但不适合决定哪种残差连接更优。
GSM8K、MATH、BBH、DROP 暂时没必要跑全套，等训练 token 或模型规模上去再加入。
我建议你最终固定成这套
主表
OWT validation loss
LAMBADA accuracy / PPL
HellaSwag acc_norm
PIQA acc_norm
ARC-Easy acc_norm
WinoGrande accuracy
SciQ acc_norm
OpenBookQA acc_norm
COPA accuracy
7-task average
附表
WikiText-103 test loss / PPL
ARC-Challenge acc_norm
MMLU 5-shot accuracy
gradient norm max / mean / std
loss spike count
tokens/s
peak memory

下一轮重新预处理 OWT 后，再在主表中增加：

OWT test loss
OWT test PPL

这会形成一个比较完整的闭环：

OWT val 看优化过程，OWT test 看同分布泛化，WikiText/LAMBADA 看语言建模迁移，HC 七任务看实际能力，gradient/activation 指标看架构稳定性。

使用 lm-eval 的任务命令

将 nanoGPT checkpoint 导出成 Hugging Face 格式，或者给 lm-eval 写一个自定义 model adapter 后，可以固定为零样本：

lm-eval run 
  --model hf 
  --model_args pretrained=/path/to/hf_checkpoint 
  --tasks hellaswag,piqa,arc_easy,winogrande,sciq,openbookqa,copa,lambada_openai 
  --num_fewshot 0 
  --batch_size auto 
  --device cuda:0 
  --output_path eval_results

lm-evaluation-harness 支持本地 Hugging Face checkpoint，并提供统一的任务 prompt、split 和指标实现，比较适合把之后所有 mHC 消融固定在同一个协议下。

对你当前已有的 checkpoint，最优先补跑的是：

HellaSwag + PIQA + ARC-Easy + WinoGrande + SciQ + LAMBADA

这六个已经能明显弥补只看 OWT val loss 的不足；下一轮正式重跑时，再增加预先隔离的 test.bin。
