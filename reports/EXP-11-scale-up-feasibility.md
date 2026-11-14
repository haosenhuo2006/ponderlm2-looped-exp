# EXP-11 — 规模放大可行性测算（未跑新训练，仅测算）

## 0. 范围

本篇只做第二阶段第 5 项要求的前半步：**可行性测算**。没有跑任何放大后的训练。结论是“能不能跑、要多久、要多少显存”，底座选择须由皓森决定，本篇不擅自开新实验。

测算时间：2026-11-14。硬件：单张 NVIDIA GeForce RTX 4060 Laptop，8GB（8585 MB）。软件：仓库 `.venv`（torch 2.11.0+cu130，transformers 4.46.1）。测算脚本为一次性脚本（`/tmp/scale_probe.py`、`/tmp/scale_probe3.py`，未进仓库），方法见 §1。

## 1. 方法

- 候选：Pythia-160M、Pythia-410M（与 70M 同系列、非 deduped）；以及 70M 步数翻倍（1200/2400 步）。
- 实测口径：batch 1、seq 256、fp32 权重 + AdamW（betas (0.9,0.95)、weight decay 0.01，与训练配置一致）、bf16 autocast、每个 microbatch 一次 forward + backward + clip + optimizer step；预热 2 步、取后 5 步均值；峰值显存为 `torch.cuda.max_memory_allocated`。
- 160M/410M 的 ponder（T1 式、K=3）无法直接实测：仓库的 `src/masked.py` → `src/ponder.py` 依赖官方 PonderLM-2 的定制 modeling（`extract_hidden_for_computation`、`scale_embeds` 等配置属性），只针对 70M 接线；直接对 160M/410M 调用会报 `AttributeError`（两次探测均已证实：先是 `config.scale_embeds` 缺失，补上后又是 `extract_hidden_for_computation` 缺失）。因此 ponder 成本用 70M 的实测比例外推，并在 §3 写明该外推的局限。
- 外推系数（70M 实测）：T1/T0 训练耗时比 906/243 ≈ 3.73×；T1 峰值显存比 T0 高 1685 − 1392 = 293 MiB。

## 2. 结果

| 候选 | vanilla 实测 s/microbatch | vanilla 峰值 MiB | vanilla 600 步×16 耗时 | ponder 外推 600 步×16 耗时 | ponder 外推峰值 MiB |
|---|---:|---:|---:|---:|---:|
| Pythia-160M（162M 参数） | 0.106 | 3203 | 约 16.9 分钟 | 约 63 分钟 | 约 3496 |
| Pythia-410M（405M 参数） | 0.281 | 7880 | 约 45 分钟 | 约 168 分钟 | 约 8173 |
| Pythia-70M 步数翻倍（实测 600 步 906 s） | — | 1685（T1 实测） | — | 1200 步约 30 分钟；2400 步约 60 分钟 | 1685（不变） |

补充：
- 下载：160M 权重在第一次探测时已下载缓存（第二次加载 3.2 s）；410M 下载耗时 113 s。磁盘余量 804G，不是瓶颈。
- 70M 步数翻倍的耗时按 600 步实测线性外推，显存与 600 步相同（checkpoint 数量不变、每步峰值不变）。

## 3. 结论与待定事项

- **Pythia-160M：可行。** ponder 外推约 63 分钟、约 3.5GB，显存余量充足；是三个候选中最稳妥的放大方向。但须先做一次代码接线（官方定制 modeling 适配 160M，见下），接线本身须走 Codex + 测试。
- **Pythia-410M：按现有训练方式不可行。** vanilla 单 microbatch 已占 7880 MiB（约 96% 显存），ponder 外推 8173 MiB 已超上限；且外推的 +293 MiB 是 70M 的增量，410M 的 hidden size 是 70M 的 2 倍（1024 vs 512），ponder 额外 activation 的实际增量会更大，外推值是乐观下限。要跑 410M 须先改内存口径（gradient checkpointing、8-bit 优化器或 bf16 权重等），属于额外工程，不在本篇测算内。
- **70M 步数翻倍：可行且最便宜。** 1200 步约 30 分钟、2400 步约 60 分钟，显存不变；不需要任何代码改动。EXP-10 的实验 3（延长训练看差距是否缩小）可直接用此口径。
- **代码前置工作（任何换底座都须做）**：官方 PonderLM-2 的定制 modeling（`extract_hidden_for_computation`、`embed_out`、相关配置属性）目前只对 70M 接线；换 160M/410M 须把该 modeling 适配到新尺寸并通过现有测试，工作量未在本篇测算。
- **待皓森决定**：放大方向三选一——（a）Pythia-160M（约 63 分钟/模型，须先接线）；（b）70M 步数翻倍（最便宜、无代码改动）；（c）410M（须先改内存口径，成本最高）。本篇不替他选。

## 4. 局限性

- ponder 耗时与显存为**外推估计**（70M 比例 × 新尺寸 vanilla 实测），非直接实测；比例在不同尺寸下可能变化，160M 的 63 分钟应理解为量级估计，误差可能在 ±30% 以上。
- 显存外推的 +293 MiB 同样来自 70M，未按 hidden size 缩放；410M 的“不可行”结论是保守的（真实需求只会更高）。
- 未测算数据侧成本（WikiText-2 只有 2.46M tokens；步数翻倍意味着多 epoch，须另行说明训练口径）。
- 未做 profiler 实测 FLOPs；本篇数字均为 wall time 与 allocated bytes 实测/外推。
