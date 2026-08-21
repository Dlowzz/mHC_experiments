# ## Paper Context

# ## Target venue
# ICLR 2027

# ## Task
# Improve multi-stream residual interaction in Hyper-Connections / mHC
# for language model pretraining.

# ## Existing method
# mHC maintains multiple residual streams.

# During the read stage:
# - all channel dimensions share the same residual-stream mixing weights.

# During the write stage:
# - each stream receives the same branch feature vector,
# - only scalar gating differs across streams.

# ## Technical limitation
# This creates two forms of coarse-grained interaction:

# 1. Different feature subspaces cannot independently select residual streams.
# 2. Residual-stream updates are directionally coupled / collinear.

# ## Our hypothesis
# Residual streams should interact with different feature subspaces
# at finer granularity.

# ## Method
# We introduce xxx.

# Module A:
# group-wise H_pre
# ...

# Module B:
# ...
    
# ## Main experimental findings

# 500M model, OWT, xxxB tokens.

# baseline mHC:
# loss = ...

# ours:
# loss = ...

# Parameter overhead:
# ~1%

# Throughput:
# ...

# Important ablations:
# - group number
# - LoRA rank
# - normalization
# ...

# ## Main intended claim
# Fine-grained feature-stream interaction improves representation diversity
# while preserving the stable multi-stream residual structure of mHC.






# abstract


# 一、introduction
# 1.残差流及其优化介绍
# 残差连接作为模型优化的重要手段，通过在网络的前向传播过程中引入残差跳跃连接，缓解了梯度消失问题，同时保留了特征图的原始信息，从而在训练深度网络时取得了更好的收敛性能。
# hc通过把残差扩展到多流，从而让不同特征子空间可以adjust the strength of connections between features at different depths。增强跨层信息传递，其优化直接决定浅层特征能否在深度扩展中被有效保留与利用。
#现在的深层模型彰显出了令人震撼的性能，其scaling机制are tremendously benefit from残差连接，which在深层模型中通过保留浅层的特征，在训练深度网络时取得了更好的收敛性能。
#然而残差本身会遇到逐层信息聚合稀释的问题，这使得【和hc同级别的残差优化方法，例如hc，attention residual等】通过不同的角度来优化残差流的传播，attention residual通过block-wise 逐层信息的加权和 xxxxxx 来选择性聚合浅层信息，而hc则通过把residual扩展到多条流上，从而让不同特征子空间可以adjust the strength of connections between features at different depths。
# 2.hc优化方法
# 现有基于 HC 优化主要聚焦于残差流混合算子的几何与谱性质（mhc、mhc-lite、shc、kromhc），通过约束其线性组合的矩阵可行域，改善多流残差连接的计算效率与跨层传播稳定性。
# mhc通过sk矩阵迭代的方式来将xxxxx近似为双随机矩阵，让xxxx在深层网络中保持xxxxx，解决hc的xxxxx问题。
# 3.特征流没有被充分研究
# 在对残差流进行详尽研究，取得谱范数和表达能力权衡的同时，特征流作为单独的一侧却没有得到同等的待遇（俏皮）。
# attention/ffn层在读取特征时，所有隐藏通道共享同一组权重，不同特征子空间被迫采用相同的跨层信息聚合策略。而在写回每条流的过程中，多条 residual streams 仅通过标量门控接收同一个 branch 特征，各流更新方向彼此共线。
# 在深层网络中，这种完全共线的写回一定程度上导致了特征梯度的不稳定性。
# 4.提出的方法
# in this work，我们对

# 二、method
#在这章中，我们在2.1章对hc和mhc的架构进行回顾，并分析当前绝大多数hc相关架构中尚未被发掘的潜在缺陷，针对这些缺陷，我们提出了极其轻量级的架构更新方案，在2.2章提出groupwise读取，在2.3章提出rankwise写回，在2.4章讲到硬件优化。
# 2.1 preliminary：
# 2.1.1 hc
#（hc里的基本公式，x怎么通过Hpre，Hpost，Hres得到新的x，x=hpost（hpre（x））+hres（x））（根据真实公式验证写出）
#残差可以通过两条流的不同视角看待：residual流包含了原始的浅层信息聚合，而branch流则通过attention/ffn的信息处理，产生更深层的特征聚合信息。
#hc将残差x_i(i是layer数目)从(1,C)扩展到(n,C)，while n是残差数，而C是残差dimension，如式所示
#公式：x_i+1=hpost(branch(hpre(x_i))+hres(x_i)
#其中branch代表attention或mlp层，hpre维度1，n，hpost维度n，1，表示写入branch的流间加权系数和写回系数，通过线性运算和tanh函数得到【一句带过，按照hc论文的hpre/hpost公式为准（我的表述可能不准）】。hres维度n，n，通过线性层得到系数【一句带过，按照hc论文的hres公式为准（我的表述可能不准）】
#
#2.1.2 mhc
#（mhc的修改）
# mhc通过对Hres系数进行近似sinkhorn操作，约束hres近似双随机矩阵，奇异值【我说的不一定准确，以mhc的核心数学原理为准】，缓解了深层中，浅层residual信息逐层乘性叠加，信息爆炸的问题。
#同时mhc把hpre和hpost写成线性组合和sigmoid的形式xxxxx
#x～_i=rmsnor(flatten(xi)),hres，hpre，hpost的公式
#其中x～_i nC代表flatten后的残差，xxxxxx【公式里的其他解释】，here 2*sigma是为了初始化时为1，与原始残差流保持一致
#2.1.3 mhc的问题
#由【hres公式】可知，mhc通过sinkhorn约束残差流，达到更稳定的训练和更好的跨层信息聚合能力。这种写回方式与残差的本质概念无比相符，通过线性聚合直接向深层传递浅层信息。
# 而直观地，特征流，区别于残差流的设计，bold（理应包含更多的非共线的特征提取能力）。尤其是在这类多流融合的架构中，流与流之间的非线性特征关系也理应被更多的发掘和leveraged。
# 但following 简单的branch聚合策略 derived from hc（as【hc基本公式】所示）mhc的hpre和hpost均仅包含n个线性缩放标量，这种简单策略暴露出了两个问题：
# 一、branch流的读入是针对多条流的线性缩放相加，这会导致单条流的特征子空间被迫采用相同的信息聚合策略。这不仅限制了信息融合的灵活性，同时也造成潜在的类似moe的梯度坍塌的问题，某一条流过度更新，导致这种多信息聚合的范式不能被最大化利用。
# 二、while branch写回各条stream时，mhc仅用标量乘法把共线的特征向量写回不同stream，这种方式会造成各条流中信息更新方向完全一致，无法利用多流结构间的非线性特征关系。同时，这会造成在residual数目n进一步增长（如从4到16）时，mhc的收益与计算量的增长不成正比，while xhc也提及了这个问题。他们通过历史token的特征提取和剔除同向信息的计算来解决这个问题，while我们采用了更简单更直观和更低参数量的方法。
# 2.2 groupwise读入
# this section主要为了解决2.1.3中的问题一，为了enable特征子空间的细粒度读取，我们将每条残差在embedding维上切分为n个group
# 这里group数精确等于residual数n，这一设计使得每个group对应的信息量和residual数严格相同。这一点xxxxx【编一点优势】
#to be exactly，我们将Hpre写为
# 【hpre新公式】
#不同于hc对flatten后的nc维残差做线性变换，映射到n个权重标量，我们将每个组的embedding进行逐流的concate，产生每一组embedding专属的n个权重标量


# 2.3 lora方向写回

# 我们在rank维应用了一个无参数的midnorm，这一操作使得A矩阵具有数量无关性，仅通过加工特征来得到方向向量，写回强度则完全由B矩阵控制

# 2.4 硬件优化


# 三 experiments

# main result

# 余弦相似度

# 不同group的特征表示+不同层的特征表示

# ablation of loranorm，densegroup，mutiple n，

四 结论和不足

五/附录 related work


