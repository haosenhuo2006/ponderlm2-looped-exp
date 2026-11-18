# 诊断 D1 / D2 / D4：结果报告（只报告，不裁决）

**日期：** 2026-11-18 晚
**口径：** 设计 `reproduction-design.md` §7.5——本文件只报告数字，不下裁决。
**Checkpoint：** `runs/EXP-25-loopmid160-s42/checkpoints/step-1200`（EXP-25 looped，seed 42，160M，block layers 3–8，训练 repetitions=2）
**仓库 HEAD：** `6386c8e`（另有未提交的修改与新实现文件，与本次诊断无关）
**GPU：** NVIDIA GeForce RTX 4060 Laptop GPU；评测用仓库 `.venv/bin/python`（torch 2.11.0+cu130）
**约束遵守：** 未跑 D3、未跑任何训练、未改任何 `src/` 文件、未提交。D1 未走旧 `evaluate_looped.py`（它会覆写该 run 的 `evaluation.json`），结果只写 `runs/reproduction/`。

## D1：R-scaling 评测

方法：与 `src/evaluate_looped.py` 的并行 NLL 计算完全一致（bf16 autocast、FP32 CE、`logits[:, :-1]` 对 `x[:, 1:]` 求和；WikiText-2 验证集全部 977 块、seq 256、batch 1、249,135 个预测 token）。r=8 的绕开方式：`looped.configure()` 校验 R∈1..4，先用 checkpoint 自带的 R=2 正常 configure，再直接设 `model.config.loop_repetitions = 8`（`layer_schedule()` 只读 config、不校验）；`src/looped.py` 未动。脚本：`runs/reproduction/d1_rscaling.py`，结果：`runs/reproduction/d1_rscaling.json`。

| r | NLL | PPL | 用时 |
|---|-----|-----|------|
| 1 | 4.952071 | 141.47 | 9.1 s |
| 2 | 4.572619 | 96.80 | 14.3 s |
| 4 | 5.432877 | 228.81 | 17.2 s |
| 8 | 9.106902 | 9017.32 | 33.4 s |

记录：

- r=2 的 NLL 4.572619 与该 run 已存 `evaluation.json` 的 4.572618549242881 一致到小数点后 6 位——checkpoint 加载与评测口径复核通过。
- 曲线呈 U 形、最低在训练所用的 r=2；r=4 比 r=2 差 +0.860 NLL，r=8 差 +4.534 NLL（PPL 9017）。
- r=1（不重复、即 12 层直通）也比 r=2 差 +0.379 NLL：该 checkpoint 的权重是在 r=2 下训的，r=1 同样不在训练分布内。

## D2：隐状态漂移诊断

方法：现有 `src/diagnose.py`、`src/probe_h.py`、`src/jacobi_check.py` 均为 thought 线（ponder/Jacobi）工具——`load_model("ponder")`、走 interleaved thought schedule，无法跑 looped 前向，故未复用。新脚本 `runs/reproduction/d2_drift.py` 手动单步 looped 层，逐字复现 `src/looped.py` 的 `forward()`（同一 `embed_in`、同一因果 mask、同一份 rotary、同一逐层调用签名），在 prelude 之后（round 0）与 loop block 每重复一次后（rounds 1–8）截取 h（coda 之前）。验证集前 32 块、batch 1、bf16 autocast、FP32 计量。结果：`runs/reproduction/d2_drift.json`。

| round | token 两两余弦均值 | 与上一轮余弦均值 | 范数均值 | 相对变化 ‖hᵢ−hᵢ₋₁‖/‖hᵢ₋₁‖ |
|------:|------------------:|------------------:|---------:|------------------:|
| 0（prelude 后） | 0.3732 | — | 22.68 | — |
| 1 | 0.6138 | 0.6430 | 49.22 | 1.6486 |
| 2 | 0.5577 | 0.9167 | 59.24 | 0.5013 |
| 3 | 0.5555 | 0.9684 | 70.87 | 0.3551 |
| 4 | 0.5743 | 0.9850 | 83.75 | 0.2780 |
| 5 | 0.5977 | 0.9914 | 97.48 | 0.2299 |
| 6 | 0.6203 | 0.9945 | 111.77 | 0.1951 |
| 7 | 0.6407 | 0.9962 | 126.46 | 0.1691 |
| 8 | 0.6584 | 0.9972 | 141.47 | 0.1488 |

记录（描述性，不裁决）：

- token 两两余弦：round 0 为 0.373，第 1 轮跳到 0.614，第 2–3 轮回落到 ~0.556，此后单调爬升，round 8 到 0.658。**未冲向 1.0**——token 状态在变相似，但远未变成同一向量。
- 与上一轮的余弦：0.643 → 0.917 → 0.968 → … → round 8 为 0.997，逐轮方向变化越来越小，接近不动点方向。
- 范数：22.68 → 141.47，单调增长，round 2 之后每轮约 +12~+15，无饱和迹象；round 8 约为 round 0 的 6.2 倍。
- 相对变化：每轮仍有 15%（round 8），方向趋稳但幅度仍在向外漂移。

## D4：采样器模拟（CPU）

方法：独立重跑。Huginn log-normal Poisson：`log λ ~ N(ln(4) − 0.125, 0.25)`，`r = 1 + Poisson(λ)`，`r = min(r, 12)`，`k = min(r, 2)`；NumPy `default_rng(42)`，2,000,000 次抽样、分批（先批量抽 log-normal 再抽 Poisson）。脚本：`runs/reproduction/d4_sampler.py`，结果：`runs/reproduction/d4_sampler.json`。

| 量 | 本次（2,000,000 抽） | 设计预期 | Codex 核实报告 C3（100,000 抽） |
|---|---:|---:|---:|
| E[r]（无 cap） | 4.999566 | 5（理论精确值） | 4.98803 |
| E[r]（cap 12） | 4.931741 | ≈4.93 | 4.91792 |
| E[k] | 1.94226 | ≈1.94 | 1.94218 |
| P(r=1) | 5.774% | — | 5.782% |
| P(r≥8) | 16.722% | — | 16.516% |
| P(r>12)（cap 前） | 2.191% | 约 2.2% | 2.225% |

截断后 r 的分布（cap 12）：r=1…12 的概率依次为 5.77%、12.75%、16.33%、16.17%、13.78%、10.70%、7.77%、5.50%、3.74%、2.52%、1.67%、3.29%（r=12 一格含全部尾部，即 2.19% 的截尾质量 + 原 r=12 的 ~1.1%）。

记录：E[r]≈4.93、E[k]≈1.94 与设计一致；cap 12 只截去约 2.2% 的步，对 E[k] 无影响（k≤2 已限幅），cap 是合理的时延上限。

## 偏离与说明

- D2 未复用任何现有诊断脚本（原因见 D2 方法段）；D1 的 r=8 绕开校验的方式如 D1 方法段所述，`src/` 无改动。
- D2 只用了验证集前 32 块（漂移量逐轮稳定、单块间差异小；32 块 1.9 秒跑完，全量 977 块无必要）。
- D3 未跑（按任务分工，由协调员验收新代码后亲自跑）。
- 新增文件（均在 `runs/reproduction/`，未提交）：`d1_rscaling.py`、`d1_rscaling.json`、`d2_drift.py`、`d2_drift.json`、`d4_sampler.py`、`d4_sampler.json`、`diagnostics-d1d2d4.md`。原 `runs/EXP-25-loopmid160-s42/evaluation.json` 未动。
