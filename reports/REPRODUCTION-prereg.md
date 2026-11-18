# Looped Transformer 完整复现：预注册

**版本：** v1.0（2026-11-18 深夜，按 Codex 核实修订后冻结）
**状态：** 待用户批准后生效；生效后冻结，任何改动须新版本号 + 用户批准
**设计文档：** `~/workspace/research_notes/ponderlm2-looped-20261113/reproduction-design.md`
**核实报告：** `~/workspace/codex-audit/reproduction/codex-verification.md`
**仓库：** `/home/lenovo_jason/ponderlm2-looped-exp`（HEAD `278a163`；存入 `reports/REPRODUCTION-prereg.md`，与审计修复协调员的推送错开提交）
**前置条件：** 审计修复（评测入口 + metadata）落地后才开训；新模块单元测试全过（见 §6）。

---

## 1. 冻结的实验臂

### 1.1 诊断臂（训前做，无训练）

| 臂 | 内容 | 冻结值 |
|---|---|---|
| D1 | EXP-25 checkpoint R-scaling 评测 | r ∈ {1, 2, 4, 8}，验证集 NLL |
| D2 | EXP-25 checkpoint 隐状态漂移诊断 | token 相关性与范数随轮次变化 |
| D3 | 新实现零步基线（160M CPT 图） | 零步验证集 NLL；底座 4.3733、旧 looped 图 5.3841 为参照 |
| D4 | 采样器模拟（CPU） | t=2、s=2、σ=0.5、cap 12；确认 E[r]≈4.93、E[k]≈1.94 |

### 1.2 LM 臂（160M，WikiText-2，seq 256，batch 16 序列/步）

| 臂 | 实现 | 步数 | 种子 | 主指标 |
|---|---|---|---|---|
| L1-looped | CPT 注入 looped（§2 规格），(3,6,3) | 1200 | 42/43/44 | 验证集 NLL，评测 r=4 |
| L1-T0 | vanilla CPT | **2006**（等 FLOPs） | 42/43/44 | 验证集 NLL |
| L2-vanilla | 从零训 vanilla | 1200 | 42/43/44 | 验证集 NLL |
| L2-looped | 从零训注入 looped（§2 规格），(3,6,3) | 1200 | 42/43/44 | 验证集 NLL，评测 r=4 |

### 1.3 Yang 臂（从零训，batch 64，Adam lr 1e-4，无 wd、无 clip）

| 任务 | d | k | b | T | 步数 | 种子 | 主指标 |
|---|---|---|---|---|---|---|---|
| 线性回归 | 20 | 40 | 20 | 15 | 50k | 42/43 | 归一化 MSE（分母 20） |
| 稀疏线性回归 | 20（s=3） | 40 | 20 | 10 | 50k | 42/43 | 归一化 MSE（分母 3） |
| 决策树 | 20（depth 4） | 100 | 70 | 15 | 20k | 42/43 | 归一化 MSE（分母 1） |
| 两层 ReLU 网络 | 20（100 隐单元） | 100 | 12 | 5 | 20k | 42/43 | 归一化 MSE（分母 20） |

模型：looped 1 层 GPT（D=256、8 头，847,873 参数）vs 标准 12 层 GPT（D=256、8 头，9,535,233 参数）。评测：每任务 1280 条 prompt，最终轮预测，90% bootstrap CI（1000 次）。

### 1.4 加法臂 A1（70M，从零训，batch 256，lr 1e-3，warmup 100，8000 步）

- 训练：**严格 3 位数**（两操作数都 100–999）；测试：3/4/5–6 位，每种 500 条，seed 1234。
- 模型：vanilla（R=1）vs 新实现 looped（(2,2,2)，§2 规格），种子 42/43。
- 初始化按 §2.3（l_eff：looped 12、vanilla 6）；emb_scale=√512，两臂一致。

### 1.5 条件下游（仅 L1  verdict (a) 时触发）

ARC-Easy（570）/ HellaSwag（2000）/ PIQA（1838）/ SciQ（1000）/ LAMBADA（229）/ Pile PPL（500 块），R1 主、R2 次（Step 0/EXP-26 口径）。

---

## 2. 冻结的实现规格

### 2.1 新 looped 前向（`src/looped2.py`，不动 `src/looped.py`）

- 结构 prelude → core（循环）→ coda；注入 `h = h + e`（"add"，每轮同一 prelude 输出 e）；s_0 每序列每前向新采样，截断正态**截断前 std=√(2/5)，截断 ±3σ**，**与 γ 解耦直接采样**；评测用固定种子 s_0，另做 3 种子 s_0 方差诊断。
- Sandwich norms：仅 core 层，`x̂ = n2(x + Attn(n1(x)))`、`x' = n4(x̂ + MLP(n3(x̂)))`；n1/n3 复用已有 LayerNorm，n2/n4 新增（weight=1、bias=0）；**core 由并行残差改为串联**（本地适配，Huginn 三段都 sandwich RMSNorm）。
- 边界 ln_f：core 循环后、coda 前过一次 `final_layer_norm`，coda 后再过一次接 lm_head。
- 目标：只监督最终输出 CE。

### 2.2 随机 R 与截断反传

- 采样器：log-normal Poisson，σ=0.5，名义 t=2、s=2；`r = 1 + Poisson(λ)`，E[λ]=4；`k=min(s,r)`、`n=r−k`（前 n 轮 no_grad、后 k 轮带梯度）。**实际 E[r]=5（无 cap）、E[r]≈4.93（cap 12）、E[k]≈1.94**。总轮次 cap 12。
- 等 FLOPs：比值 1.6721（cap 12），T0 步数 **2006**。事后用 logged (n,k) + profiler 核验，偏离超 ±5% 加注。
- 评测 r：主指标 r=4；R-scaling r ∈ {1, 2, 4, 8}。

### 2.3 从零训初始化（Huginn Sec 4.1 口径）

- 权重截断正态 std=√(2/(5h))（±3σ）；bias=0；LayerNorm 权重=1、bias=0；attention/MLP 输出投影 std=1/√(5·h·l_eff)，**l_eff 取名义值**：160M looped 30、vanilla 12；70M looped 12、vanilla 6。非绑定 lm_head 用普通权重尺度。embedding ×√h（CPT 臂 γ=1）。
- 实现：新文件后处理重初始化（不改 vendored 文件），构造后、优化器建立前执行，CPT 臂跳过。
- 优化：从零训 AdamW lr 3e-4、warmup 100、cosine 到 10%、betas (0.9,0.95)、wd 0.01、clip 1.0；CPT lr 1e-4、warmup 20（余同）。

### 2.4 Yang 实现

- 新文件 port 官方任务采样器与模型；loop 函数 `z=f(x+z)`，零张量起步；截断 loss（最后 T 轮 MSE 均值）；**固定 b/T，不用课程表**（本地偏差）；无 tying 对照。

---

## 3. 冻结的判定规则

### 3.1 L1（Q2，主问题）：looped ×1200 vs T0×2006，等 FLOPs

- 逐种子配对差 `d_i = NLL_looped,i − NLL_T0,i`（i=42,43,44）；报告配对均值、逐种子差、SD、SE、95% 配对 t 区间、双侧 p。
- **(a) looped 赢：** 配对均值 < −0.05 **且** 三种子方向一致。区间跨 0 时仍判 (a)，但注明“未达统计确证”。
- **(b) looped 不赢：** 配对均值 ≥ −0.05。looped 线收尾，不再开新臂。
- (a) 时触发条件下游；任何情况不自动补种子（补 seed 45/46 须另行授权）。

### 3.2 L2（Q3）：从零训 vanilla vs looped，同 token 数

- 同 3.1 的规则与阈值，但**不叫“等 FLOPs”**（同 1200 步，looped 每步 FLOPs 仍高约 1.67×）。Q2/Q3 四种组合按组合解读，不事先选一种。

### 3.3 Yang（Q4）：每任务独立裁决

- 某任务上 looped 赢：looped 归一化 MSE 均值 < 标准归一化 MSE 均值的 90% **且** 两种子方向一致。四任务不合并总判。

### 3.4 A1（Q5）：加法

- Sanity 门：3 位数准确率 ≥90%（每模型），不过门则泛化比较判“不可解释”。
- 主裁决：4 位数上 looped − vanilla > 5pp **且** 两种子方向一致 → “新实现下 looped 在 4 位数泛化上占优”（整套新配方的效果）。5–6 位只报告、不裁决。

### 3.5 诊断（Q1）：不裁决，只报告

- D3 门槛：新实现零步损伤 ≥ +1.0 NLL（相对底座 4.3733）时，**先向用户汇报、再决定是否开训**，不自动开。

---

## 4. 冻结的成本与执行顺序

- 诊断 D1–D4：约 0.25 h GPU（D4 为 CPU）。
- L1：looped 约 1.3 h + T0×2006 约 1.45 h；L2：约 0.9 h + 1.3 h；A1：约 1.2 h；Yang：约 3.0 h（FLOPs 估算，timing pilot 校准）；条件下游约 0.6 h。
- **合计约 9.4 h GPU**（不含条件下游；Yang 不确定性另计）。
- 执行顺序：D1–D4 → L1 → A1 → L2 → Y1–Y4 → 条件下游。
- Yang 成本上限（预注册的条件分支，不须重新批准）：timing pilot（每任务-模型 200 步）预计总计 >3 h 时，依次执行：(1) 决策树 20k→10k 步；(2) Yang 全线降为 1 种子；(3) 稀疏 50k→20k 步。

---

## 5. 不做的事（冻结）

410M；thought 线任何新实验；T4 式顶部循环新变体；noisy 线性回归、OpenML、奇偶校验、减法、乘法；只加注入的消融（完整版赢了再考虑）；单种子；600 步短训探路（主臂）。

## 6. 开跑前核验清单（逐项打勾，缺一项不开训）

- [ ] Codex 核实完成，设计按核实修订（本预注册 v1.0 即修订后版本）。
- [ ] 新模块单元测试通过（注入公式、s_0 确定性/随机性、s_0 与 γ 解耦、`eval_r`、截断反传梯度、新 norm checkpoint 往返、FLOPs 记账偏差 < ±1%）。
- [ ] D4 模拟确认 E[r]≈4.93、E[k]≈1.94、cap 12 合理。
- [ ] vendor 相关核对针对线上仓库重验（核对用的是审计补充镜像）。
- [ ] 审计修复已落地；加法评测走修复后入口。
- [ ] 本预注册存入仓库 `reports/REPRODUCTION-prereg.md`。

## 7. 已知的局限（结论措辞须遵守）

- 规模鸿沟：160M/约 500 万 token vs 论文 1.4B–3.5B/300B–800B token；结论不外推大规模。
- PPL 不是论文的赢点；NLL 结论只在我们配方内成立。
- Sandwich 只改 core、用 LayerNorm、并行改串联：本地适配，已记录。
- Yang 是官方 500k 步的 1/10–1/25 缩减预算；不赢时不能排除“欠训”。
- 稀疏线性 k=40 跟 YAML/notebook，论文 §5.4 正文写 k=100（分歧已记录）。
- EXP-24-new 内部比较估计整套新配方的效果；跨 EXP-23/24 比较无法单独识别分布或实现的影响。
- Pile 评测 Books3 偏斜 87.2%（已知）；R4 偏离未决（沿用 R1 主、R2 次）。

---

**冻结签署：** 用户批准本预注册后生效。生效后冻结；训练按本文件执行，改动须新版本号 + 用户批准。
