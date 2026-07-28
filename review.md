
## 第一优先：梯度解剖实验

在当前约 4k step 的 checkpoint 上，固定 16–32 个 batch，只做 forward/backward，不更新参数。必须在 AMP `unscale_` 之后、gradient clipping 之前记录：

* `global_grad_norm_preclip`
* `A_s_grad_norm`
* `B_s_grad_norm`
* `beta_grad_norm`
* mHC 主干 grad norm
* Transformer 主干 grad norm
* `A_s`、`B_s` 的 parameter RMS
* `grad_norm / param_norm`
* clipping coefficient
* 实际 Adam `update_norm / param_norm`
* `RMS(hA)`
* norm 后 rank 向量 RMS
* `RMS(delta)`
* beta 加权后的实际写入：

rwrite=RMS⁡(βλδ)RMS⁡(βh)r_
================================

\frac{\operatorname{RMS}(\beta\lambda\delta)}
{\operatorname{RMS}(\beta h)}**r**write=**RMS**(**β**h**)**RMS**(**β**λ**δ**)**
尤其要比较：

raw grad normvs实际 Adam update norm\text{raw grad norm}
\quad\text{vs}\quad
\text{实际 Adam update norm}**raw grad norm**vs**实际** Adam update norm
判断方式：

* 只有 `A_s` grad norm 不同，但 update norm、clip coefficient、delta RMS 相近：基本是无害的参数化现象。
* global norm 不同且 clipping coefficient 明显不同：会影响整个模型，后期 loss 可能分叉。
* `B_s`、beta、主干梯度也明显不同：不是单纯的 AA**A** 尺度现象，未来更可能分叉。
* rwrite≪1%r_{\rm write}\ll1\%**r**write≪**1%**：LoRA 目前几乎没参与模型行为，loss 重合完全合理。

## 第二优先：直接验证 RMSNorm 尺度不变性

在相同 checkpoint 和固定 batch 上，不训练，只把全部或某一层的 AsA_s**A**s 分别乘：

c∈{0.5,1,2,10}c\in\{0.5,1,2,10\}**c**∈**{**0.5**,**1**,**2**,**10**}**
分别测：

* loss
* logits 最大差异或 KL
* delta RMS
* `A_s` grad norm
* `B_s` grad norm
* global grad norm

midnorm 理论预期：

δ(cA)≈δ(A)\delta(cA)\approx\delta(A)**δ**(**c**A**)**≈**δ**(**A**)
∥∇AL∥cA≈1c∥∇AL∥A\|\nabla_A L\|_{cA}
\approx
\frac{1}{c}\|\nabla_A L\|_A**∥**∇**A****L**∥**c**A≈**c**1∥**∇**A****L**∥**A****
而 `B_s` 梯度和 loss 应当基本不变。

如果实验完全符合这个关系，就能确认：

> loss 一致、grad norm 不一致，主要是 RMSNorm 参数化造成的，不代表模型功能已经不同。

注意 c=0.1c=0.1**c**=**0.1** 可能使 `RMS(hA)` 接近 RMSNorm 的 epsilon 区域，所以先用 0.5–10 更干净。

## 第三优先：检查 global clipping 是否传染主干

先统计最近 200–500 步：

* clipping 触发比例；
* 平均 clip coefficient；
* 最小 clip coefficient；
* LoRA 参数占 global grad norm 平方和的比例：

ρLoRA=∥gLoRA∥2∥gglobal∥2\rho_
==================================

\frac{\|g_{\rm LoRA}\|^2}
{\|g_{\rm global}\|^2}**ρ**LoRA=**∥**g**global****∥**2**∥**g**LoRA∥**2****
如果 clipping 几乎不触发，例如低于 1%–5%，暂时不用做额外实验。

如果频繁触发，例如超过 20%，从同一个 checkpoint 分叉训练 300–500 步：

1. 当前 global clipping；
2. 主干和 LoRA 分组 clipping；
3. 暂时使用很大的 clipping threshold。

然后比较：

* 主干实际 update norm；
* train/val loss；
* delta RMS；
* 是否出现数值 spike。

如果取消或分组 clipping 后，主干 update 明显变大且 loss 开始分叉，说明问题不是 midnorm 的 raw grad 本身，而是：

> LoRA 梯度抬高 global norm，通过 global clipping 间接压慢了整个模型。

这是最需要修的情况。
