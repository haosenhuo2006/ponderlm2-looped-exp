# PROBE — h 的 logit-lens 探针：latent thought 里有没有可读的预测信息

## 0. 为什么这个探针有必要做

EXP-10 的问题 6：h 是“假设”还是“噪声”？已有证据（EXP-02 的一致率、EXP-06 的逐轮 NLL）只能说明 h 参与了预测，不能区分“h 编码了信息”与“h 只是多了一步计算”。logit-lens 是唯一便宜的直接检验：把 H^(k) 过 lm_head，看它的 top-1 能不能读出下一 token。

考虑过的更便宜替代：没有比这更便宜的了——本探针无新训练，64 个验证 chunk 只用了约 4 秒 GPU。不选的替代：训练探针分类器（需要额外训练与调参；logit-lens 已能回答“可读性”问题，只有当 logit-lens 显示 h 完全不可读时才需要）。

## 1. 方法

- 对象：T1（EXP-01，D0）与 T2（EXP-03，D1）的 seed 42、600 步 checkpoint。
- 样本：验证集前 64 个 chunk（固定子集，16320 个目标 token）。
- 对每个 chunk：token-only 前向得 H^(0)；3 轮 interleaved refinement 得 H^(1..3) 与每轮 thought 位置输出；最后再一轮得最终 thought logits。
- 指标：每个 H^(k) 与每轮 thought 输出过 lm_head 后的 top-1，对真实下一 token 的准确率、与最终 thought-logit top-1 的一致率。
- 基线：(i) 打乱——沿序列维随机 permute hidden state（固定 seed 123）；(ii) token hidden state——token-only 前向的 hidden state 过 lm_head（注：它与 H^(0) 是同一量，数字应当相同，是个 sanity check）；(iii) 零向量过 lm_head（只剩 lm_head bias）。
- 实现：`src/probe_h.py`（复用 `src/diagnose.py` 的 backbone/interleaved 与 `src/ponder.py` 的 embeddings；测试 `tests/test_probe_h.py`，28 项测试全过的一部分）。

## 2. 结果

### T1（D0）

| 量 | next-token top-1 准确率 | 与最终预测的一致率 |
|---|---:|---:|
| H^(0) | 14.28% | 44.88% |
| H^(1) | 15.22% | 47.29% |
| H^(2) | 15.22% | 47.16% |
| H^(3) | 15.25% | 47.43% |
| 第 1 轮 thought 输出 | 15.07% | 49.99% |
| 第 2 轮 thought 输出 | 15.48% | 69.79% |
| 第 3 轮 thought 输出 | 15.80% | 81.56% |
| 最终 thought logits | 15.65% | — |

基线：打乱 1.66–2.07%；token hidden 14.28%；零向量 0%。

### T2（D1）

| 量 | next-token top-1 准确率 | 与最终预测的一致率 |
|---|---:|---:|
| H^(0) | 16.42% | 48.65% |
| H^(1) | 16.53% | 48.85% |
| H^(2) | 16.53% | 48.85% |
| H^(3) | 16.53% | 48.85% |
| 第 1 轮 thought 输出 | 16.60% | 80.22% |
| 第 2 轮 thought 输出 | 16.51% | 100% |
| 第 3 轮 thought 输出 | 16.51% | 100% |
| 最终 thought logits | 16.51% | — |

基线：打乱 1.71–2.03%；token hidden 16.42%；零向量 0%。

## 3. 结论

1. **h 不是噪声，但精炼几乎没增加可读信息**。H^(0)（只从 x 算出、还没经过任何 thought 精炼）已经能读出 14.28%（T1）/ 16.42%（T2）的下一 token，远高于打乱基线（~2%）；3 轮精炼只带来 +1pp（T1）/ +0.1pp（T2）。“思考”的迭代过程没有让 h 本身变得更会预测。
2. **h 的 top-1 与最终预测只有约一半一致**（T1 47%、T2 49%），且精炼轮次不改善这个数字；与之相对，thought 位置的输出与最终预测的一致率逐轮上升（T1：50% → 70% → 82%；T2：第 2 轮起 100%）。也就是说：模型在 thought 位置上越算越收敛，但喂给下一轮的 h 并没有跟着收敛——h 与最终决策之间存在系统性的错位。Jacobi 迭代的“定点”在输出层面成立，在 h 本身的可读内容上并不成立。
3. **T2 的 h 比 T1 更会预测**（16.5% vs 15.2%），与 T2 的 NLL 更好一致，但差距很小；且 T2 的 thought 输出第 2 轮起与最终完全一致（100%），说明无前序 h 时精炼很快达到不动点——这与 T2 训练时更少通路、更容易优化是一致的。
4. **Sanity check 通过**：token hidden 基线的准确率与 H^(0) 完全相同（14.28%/16.42%），符合 H^(0) 就是 token hidden state 的定义。

## 4. 问题与局限性

- 只有 64 个 chunk（16320 个 token），单种子、单 checkpoint；top-1 准确率是粗指标（NLL 会更敏感），数字只用于比较相对大小，不宜当作绝对能力。
- logit-lens 只测可读性：h 能读出信息 ≠ 模型在因果上使用了这些信息；反过来，读不出也不等于没用（信息可能以 lm_head 读不出的形式编码）。本探针只能排除“h 是纯噪声”这个极端说法，不能证明 h 的因果作用。
- T2 第 2/3 轮 thought 输出与最终 100% 一致，可能是 D1 mask 下精炼快速收敛的表现，也可能与评估用的固定 K=3 有关；未深究。
- 全部结论限于 Pythia-70M + WikiText-2 + 600 步的 tiny-scale 设置。
