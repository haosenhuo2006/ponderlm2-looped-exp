# EXPERIMENT_PLAN.md — PonderLM-2 tiny-scale 复现 + 诊断（EXP-01 / EXP-02）

版本：v1.0，2026-11-13（北京时间）
仓库：https://github.com/haosenhuo2006/ponderlm2-looped-exp
论文：PonderLM-2: Pretraining LLM with Latent Thoughts in Continuous Space，Zeng 等，SJTU LUMIA Lab，arXiv:2509.23184v4，ICML 2026 Spotlight。
精读笔记：`~/workspace/research/ponderlm2/ponderlm2-deep-read.md`；研究总笔记：`~/workspace/research_notes/ponderlm2-looped-20261113/notes.md`。

## 0. 研究问题（以皓森 2026-11-09 / 11-12 GitLab 日志为准）

1. PonderLM-2 的收益来自哪里？是“预测前多算一次”（把 h 再喂回去预测下一个 token），还是“前序 h 向后传递信息”（h 参与后续位置的计算）？
2. Jacobi 迭代收敛很快（论文 Fig.9：RMSE 约 2–3e-2 → 3e-5，约 7 轮基本收敛），而序列长 2048 时理论上至多需 2048 轮才保证与顺序计算一致——模型是否并没有充分利用前序 h 的层层加工，而更多依靠原始 token x？
3. 导师意见（待确认两点，实验前须与导师确认，不自行定论）：
   - “每次迭代之后都优化”：是指每轮加 next-token 监督、汇总损失后统一反向传播，还是每轮立刻执行一次优化器更新？皓森日志第 5 条明确写的是前者（汇总后统一更新）。
   - “h 不用 x”：是指保留当前 x_i、只屏蔽历史 x_<i，还是只在初始化时允许读 x？本计划 D2 采用前者（保留当前 x_i）。
4. 本阶段不做 T2/T3/T4/T5（训练版屏蔽、Looped 对照、中间监督）。只做 EXP-01（T0 vs T1 复现与基线）与 EXP-02（D0–D2 推理诊断）。看完这两篇报告后由皓森再定下一步。

## 1. 模型分工（皓森定的规矩 + 本次上网核实）

| 任务 | 路由 | 依据（2026-11-13 网上核实） |
|---|---|---|
| 复杂代码：最小实现、Jacobi 训练循环、诊断 mask、训练脚本 | Codex，gpt-6.1-sol 为主 | OpenAI 定位 Sol 为“复杂编码、专业工作”；Terminal-Bench 4.0 上 Sol 55.9% vs Terra 50.3%（max effort）；Codex Coding Agent Index v2.0：Sol 54 vs Terra 48。Sol 有 Ultrafast 模式，Terra 没有。 |
| 简单/中等代码：小修补、脚本、测试、数据处理 | Codex，gpt-6.1-terra；或 DeepSeek | Terra 便宜一半（API $1/$5 vs Sol $2/$10 每百万 token）、速度约 3 倍，适合输入输出可核查的定点任务；DeepSeek-V3（671B MoE，37B 激活）在 HumanEval 上约 78.6%、SWE-bench Verified 42%，擅长 Python 脚本与重构，成本极低，适合简单代码与批量小任务。 |
| 普通非代码任务：调研、写报告、整理 | DeepSeek harness | 按皓森规矩。 |
| 编排、验证、监控、写报告 | zura（协调员）本人 | 不写成规模的代码；只写胶水/命令级别的少量内容。委托后必须自己读改动文件、跑相关测试验证，才能汇报完成。 |

**禁令：绝对不要用 gpt-6-astra**（消耗 Codex 额度极高，皓森 2026-11-08 明确叫停）。
Codex 调用：助手 VM 上的 Codex 0.163.0 在 `~/.local/bin`，每次用前 `export PATH="$HOME/.local/bin:$PATH"`；用法 `codex exec --skip-git-repo-check --approve-for-me - < prompt.md`（prompt 走 stdin 传 `-`）。

## 2. 硬件与软件基线（实验开始时如实记录，变化即更新本节）

- 硬件：单张 NVIDIA GeForce RTX 4060 Laptop GPU，8 GB 显存；WSL2 Ubuntu 24.04.4，Python 3.12.3；磁盘剩余约 846 GB（2026-11-13 核实）。
- 软件（已核实）：torch 2.11.0+cu130，`torch.cuda.is_available() == True`，位于 `/home/lenovo_jason/GPT2_Latent_Reasoning/.venv/bin/python`；系统 python 无 torch；uv 在 `/home/lenovo_jason/.local/bin/uv`；gh 在 `/home/lenovo_jason/.local/bin/gh`。
- GPT2_Latent_Reasoning 仓库（`/home/lenovo_jason/GPT2_Latent_Reasoning`）的工程规范可作参考（uv + .venv，`ruff check`、`ruff format --check`、`pytest -q -m "not lab"`、smoke_test），但**不要改动该仓库**。本实验用独立 venv（`uv venv` + `uv sync --frozen` 或 `uv pip install`，以实现者记录为准）。
- 每次实验报告必须记录：git hash、种子、torch/transformers/datasets 版本、GPU 型号与显存、batch/序列长/步数、耗时、命令全文。

## 3. 规模诚实声明：这不是原规模复现

| 项 | 论文原版 | 本次 tiny-scale |
|---|---|---|
| 模型 | Pythia-1.4B / LLaMA-1.4B，另有 410M / 1.26B / 3B CPT | Pythia-70M（官方 quickstart 底座）或 GPT-2-small 级 tiny 模型（二选一，见 §4 决策门） |
| 训练数据 | The Pile 300B tokens；Table 3 对照 26B tokens | minipile 子集 / 小规模公开语料，量级待实现者按 8GB 显存与可接受时间定，**先定一个小预算并写死在报告里**（例如 ≤1B tokens，实际以报告为准） |
| 序列长 | 2048 | 先从 256 或 512 起步，显存允许再升到 1024；不得声称用了 2048 而实际没有 |
| 训练成本 | 1.4B/300B 共 73,047 GPU 小时；PonderLM-2 训练 FLOPs 约 8× vanilla | 单卡 8GB，目标是数小时内跑完一个可判断机制的训练，不是复现论文绝对数字 |
| 目标 | 复现论文的绝对性能数字 | **验证机制**：Jacobi 是否收敛（r_k、d_k）、thought 在匹配推理 FLOPs 下是否带来收益、诊断 D0–D2 中前序 h 与当前 h 各贡献多少 |

任何报告不得把 tiny-scale 的数字写成与论文同等规模的复现；所有偏差在本节与各实验报告中逐项列出。

## 4. 路线决策门（先试官方，再决定是否走最小实现）

### 4.1 先试官方路线
官方 quickstart：`scripts/quickstart_ponderlm2_pythia_70m.sh`，Pythia-70M + minipile，核心 flag：
`--patch_method ours --recurrent_model true --scale_embeds true --interpolation false --num_latent_thoughts 1 --num_jacobi_iterations 3 --random_jacobi_iterations true`，
另有 `--cutoff_len 2048 --per_device_train_batch_size 8 --gradient_accumulation_steps 2 --bf16 --flash_attn fa2 --deepspeed ds_z0_config.json`，目标机器是 8×RTX4090。
在单卡 8GB 上试跑，允许的调整（须记录）：batch 减小、序列长缩短、梯度累积、关闭 flash-attn / deepspeed、关闭 wandb（`--report_to none`）。
**决策门**：若官方 LLaMA-Factory fork 在单卡上跑不通（依赖装不上、OOM 无法通过合理缩参解决、或 monkey-patch 与现有 transformers 版本冲突），**停下来汇报**，经皓森确认后改走 4.2，不得自行悄悄换路线。

### 4.2 最小实现（仅在官方路线走不通时）
在 GPT-2-small / Pythia-70M 上按论文 §3 自己实现：
- 交错序列 S = [e(x1), h1, e(x2), h2, …]，thought 复用对应 token 的 position id；
- Jacobi 训练：初始前向得 H^(0)，K ∈ {2,3} 随机，每轮一次并行前向更新全部位置；
- 损失：只在 thought 位置算 CE 预测下一个真实 token，无辅助损失；
- 推理：prefill 做 Jacobi、decoding 序贯（论文与代码的不对称要注明）。
报告中必须写明“改走最小实现”的决定、原因、与官方实现的差异（文件、函数、超参对照表），不要假装跑的是官方代码。

### 4.3 已知的论文–代码出入（实现时必须处理并记录选择）
1. GPT-NeoX 版随机 Jacobi 取 {2,3,4}（`modeling_gpt_neox.py` L1357，`randint(2,5)`），与论文 {2,3} 不符；Llama 版是 {2,3}。本次以论文 {2,3} 为准，偏离须注明。
2. `scale_embeds=true` 时 token embedding 乘 √2.5d（PonderLM-2 分支），该系数只在代码里、论文未提；记录是否开启。
3. K 的双重含义：Jacobi 轮数 vs thoughts/token；代码与报告中一律用 `num_jacobi_iterations` 与 `num_latent_thoughts` 全名，不得只写 K。
4. 推理不对称：prefill 做 Jacobi、decoding 序贯多 stage；无自适应停止，thought 数与 Jacobi 轮数均为固定超参。

## 5. EXP-01：复现与基线（T0 vs T1）

- **T0 普通 LM 基线**：同一底座、同一数据顺序、同一训练 token 数、同一优化器与评测设置，标准自回归训练（不插 thought）。
- **T1 完整 PonderLM-2**：每 token 后插 1 个 thought（`num_latent_thoughts=1`），Jacobi K ∈ {2,3} 随机，`num_jacobi_iterations=3`，其余设置与 T0 完全相同。
- **固定量**：种子（主种子 42；结论性差异再加种子）、tokenizer、数据顺序、训练 token 数、评测集、优化器（AdamW，lr、warmup、weight decay 写死在配置里）。
- **必须验证的机制点**：
  - Jacobi 收敛曲线：r_k = RMSE(H^(k), H^(k-1)) 与 d_k = RMSE(H^(k), H_seq)，列表或画图；与论文 Fig.9 的形态对照（约 7 轮基本收敛、前 5 轮消除大部分误差），不要求数值相同。
  - loss 对比：训练 loss 曲线（T0 vs T1，相同 token 数口径 + 相同训练 FLOPs 口径各一张/一表）；验证集 NLL/PPL。
  - 推理成本：每真实 token 的推理 FLOPs（T1 约 2× T0）、实际延迟（tok/s）、KV-cache 占用。
- **评测**：验证集 NLL/PPL（held-out 切片，划分方式写死）；固定下游小任务（规模须与 tiny 底座匹配，例如 LAMBADA 子集 / ARC-Easy 子集，具体任务与样本数由实现者在训练前定稿并写入报告，不得事后挑任务）。
- **通过标准（事前定）**：T1 训练能跑通、loss 不发散、Jacobi r_k 随轮数下降；T1 vs T0 的 PPL 差异如实报告，无论正负都算有效结果，不得只报好看的。

## 6. EXP-02：诊断 D0–D2（用 T1 的 checkpoint，不重训）

只改推理时 attention 可见范围，不重训、不改权重：

| 组 | 定义 | 回答的问题 |
|---|---|---|
| D0 完整基准 | 原始计算，全部可见 | 基线 |
| D1 屏蔽前序 h | 当前这一对 (x_i, h_i) 无法读取 h_<i；保留真实 token 与当前 h_i 的生成与使用 | 前序 thought 是否参与有效预测；仅“当前 thought 的额外计算”能带来多少收益 |
| D2 屏蔽前序 x | 当前这一对 (x_i, h_i) 无法读取 x_<i；保留当前 x_i 与前序 h | 历史信息能否通过 h 支持预测 |

- 测：验证集 NLL/PPL、预测变化（与 D0 的预测一致率 / top-1 变化比例）、下游准确率（与 EXP-01 同一套任务）。
- **局限性必须写进报告**：推理时突然改 mask 会造成分布偏移（模型没在这种 mask 下训练过），因此 D0–D2 只能做诊断、说明已有模型对各路径的依赖程度，**不能直接当作“从训练起就屏蔽”的结论**；后者是下一步 T2/T3 的训练实验。
- 实现注意：D1/D2 的 mask 必须精确到“位置对”级别（x_i 与 h_i 各自的可见范围），实现者须在报告中画出/写出 mask 定义表，并用小例子单测验证 mask 生效（例如屏蔽后某位置的 attention 权重确为 0）。

## 7. 成本与稳定性记录（每个实验必记）

- 训练 FLOPs（按公式与实测二者，公式：PonderLM-2 训练约 3+2K 倍 vanilla，K 平均 2.5 时约 8×；实测以 profiler 或步数×单步时间为准，注明口径）。
- 每真实输出 token 的推理 FLOPs 与实际延迟、吞吐（tok/s）、KV-cache。
- 训练 token 数、优化器设置、Jacobi 轮数、thought 数、可训练参数量。
- 稳定性：loss 曲线、梯度异常、OOM、训练失败与恢复方式。

## 8. 可复现性规则

- 固定种子；固定数据与评测设置；记录硬件/软件版本/提交 hash；每个结果可复现（命令 + 配置 + 日志）。
- 代码、配置、日志、checkpoint 索引（checkpoint 本身不进 git，大文件用 GitHub Release 或注明本地路径）一律进仓库；`runs/` 下按 `EXP-01-T0-<seed>` 这样的名字分目录存放。
- 目录结构（v1.0）：
  ```
  ponderlm2-looped-exp/
  ├── README.md
  ├── EXPERIMENT_PLAN.md        # 本文件
  ├── LICENSE
  ├── .gitignore
  ├── src/                       # 最小实现 / 诊断脚本（若走官方路线则放封装与配置）
  ├── configs/                   # 每个实验的配置（json/yaml，含种子、数据、超参）
  ├── scripts/                   # 训练 / 评测 / 诊断的入口脚本
  ├── tests/                     # mask 单测、Jacobi 收敛单测等
  ├── runs/                      # 日志与结果（小文件进 git，大 checkpoint 不进）
  └── reports/
      ├── EXP-01-baseline-reproduction.md
      └── EXP-02-diagnostics-D0-D2.md
  ```
- 每个实验一篇详细报告（见 §9），写完就提交并推送到 GitHub。

## 9. 报告要求（每做完一个实验一篇）

放在 `reports/EXP-*.md`，结构：
1. 方法（用了什么方法、为什么用这个方法、与论文/官方实现的关系）
2. 做了什么（命令、配置、git hash、种子、硬件、耗时、数据）
3. 结果（表 + 数字 + 图/曲线；正负结果都写）
4. 问题与局限性（重要的写出来，不要求面面俱到；失败、OOM、loss 不收敛都如实记录原因和下一步）
5. 下一步建议
**不许编数字、不许把未跑通写成跑通。**失败也要写报告。

## 10. 实验顺序与决策门

1. EXP-01 复现与基线（T0 vs T1）→ 报告 → 汇报。
2. EXP-02 诊断 D0–D2 → 报告 → 汇报。
3. T2/T3/T4/T5 不在本阶段铺开，等皓森看完 EXP-01/EXP-02 再定。
4. 遇到需要皓森决定的事（底座换模型、预算超支、官方路线走不通要改最小实现、评测任务换集）先停下来汇报，不要自己拍板换方向。

## 11. 第二阶段（2026-11-14 皓森确认：今晚一直跑，不要等看完再定）

EXP-01/EXP-02 之后直接继续，不停下来等汇报。每个实验仍是一篇详细报告，写完就提交推送。

### T2 无前序 h（训练版 D1）
- 从训练起就用 D1 的 mask：每个位置只能看到当前 thought 的生成与使用，看不到前序 h_<i；保留真实 token 的可见性。
- 与 T1 对照：T1 − T2 = 前序 h 的信息传递带来的收益（训练口径，不是推理干预口径）。
- 其余设置与 T1 完全相同（底座、数据顺序、token 数、优化器、种子）。

### T3 历史信息走 h（训练版 D2）
- 从训练起就用 D2 的 mask：保留当前 x_i，屏蔽历史 x_<i 的直接访问；历史信息只能通过前序 h 传递。
- 与 T1 对照：模型能否在限制直接读历史 x 的条件下学会有效的连续状态传递；以及这种限制是损害还是改善表现。
- 注意：D2/T3 的“屏蔽历史 x”不包括当前 x_i；实现时 mask 定义须与 EXP-02 的 D2 完全一致（同一份 mask 定义表）。

### T4 Looped Transformer 对照
- 初期用结构简单的循环模型，先不引入 Huginn 的输入注入、随机状态、sandwich 归一化等设计。
- 循环范围与次数按实际 FLOPs 选，尽量接近 T1 的每输出 token 推理成本；**注意 PonderLM-2 增加了序列位置，不能只凭“都多跑一次”判断预算相等**——须按 §7 的口径实测 FLOPs/延迟并列表对照。
- 建议的简单实现：同一底座，取顶部若干层（或中间一段层，须在报告中写明选了哪几层、为什么）作为共享块重复运行 R 次，其余层只跑一次；R 的选择以匹配 T1 推理 FLOPs 为准。
- 与 T1 对照：两种计算组织方式（序列内插 thought vs 深度方向重复共享层）在相近预算下的实际差别。

### T5 中间监督（在 T1 上加中间轮次监督）
- 保持结构、Jacobi 轮数、数据和最终预测位置不变，只增加中间轮次的 next-token 损失：L = (1/K) Σ_{k=1..K} L_CE^(k)，各轮等权重，**汇总损失后统一反向传播、统一更新参数**。
- 对照 T1（只监督最终轮）：相同训练 token 数下收敛速度与最终验证损失是否改善；中间轮的预测是否逐步改善；加入额外监督后训练耗时与计算量增加多少。
- **先不做“每轮立刻更新优化器”的版本**：那是另一个变体（参数在 Jacobi 中途变化，定点本身也在动，稳定性风险不同），须单独注明、单独做实验，不得与本变体混为一谈。

### 第二阶段报告
- `reports/EXP-03-T2-no-prev-h.md`、`reports/EXP-04-T3-history-via-h.md`、`reports/EXP-05-T4-looped.md`、`reports/EXP-06-T5-intermediate-supervision.md`
- 每篇结构同 §9；T4 报告须附 FLOPs/延迟对照表，T5 报告须附中间轮次的逐轮 loss/预测曲线。
- 训练时间长就让它在后台跑，监控进度、处理失败（OOM、loss 不收敛等如实记录），不要空等。
- 决策门不变：底座换模型、预算/时间超支到做不完、官方路线走不通要改最小实现，先停下来汇报，不要自己换方向。

## 第三阶段（2026-11-14）：EXP-07 T6

T6 每个数据 batch 采样 K=2/3，逐轮对16个 microbatch 累加 CE_k 梯度，
clip 后立即 AdamW 更新，轮间 H detach，前缀以当前参数重算。
K轮更新后用最终参数监测 T1 separate final pass。LR 按600个数据batch索引，
同batch共享LR，总1494次更新。per_round_optimizer_update 与 intermediate_supervision 联合启用。
T5 是K轮CE均值后统一更新；同CE定义，比较更新时机，但需承认T6截断梯度。
T6/T1同时混杂监督方式和更新时机。
先100步pilot：有限loss、无发散、梯度同量级、无OOM才继续600步；失败则停止并报告。
后续只列计划：多种子配对、下游benchmark、梯度与Jacobi诊断、验证学习曲线、
更大语料/模型/预算；本阶段不展开做。
