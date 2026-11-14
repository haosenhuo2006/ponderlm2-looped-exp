# EXP-14 — Pythia-160M：T1−T0 的差距是否随模型规模缩小

## 0. 为什么这个实验有必要做

EXP-10 的问题 4（放大底座）：H2（分布偏移与容量竞争）预言，thought 机制的成本随模型规模增大而摊薄，差距应随规模缩小。EXP-12 已证明“多训”能让差距缩小 21%；EXP-14 检验另一条轴——“更大模型”是否也有同样效果。

考虑过的更便宜替代：无。规模问题只能靠真的换底座回答；EXP-11 的测算（63 分钟/模型）是外推，须实测验证。160M 是本机 8GB 显存下无需改内存口径的最大可行底座（410M 的 vanilla 单 microbatch 已占 96% 显存）。

一个纠正：EXP-11 报告说换底座“须先做 modeling 接线”，是错的。协调员实测 `src/ponder.py` 的 `load_model("ponder")` 在 160M 上直接工作（162,322,944 参数，`scale_embeds`、`extract_hidden_for_computation` 等全都有，ponder forward 与 masked_forward 都能跑）。EXP-11 的 AttributeError 走的不是 `load_model` 路径。接线成本为零；唯一的前置工作是：160M 的 HF 缓存快照里本来没有 tokenizer 文件（只有 config + 权重），须先下载 tokenizer 文件（已做，见 §1）；以及把 600 步预算从 3600 秒放宽到 5400 秒（T1 预估约 63 分钟，防撞线）。

## 1. 方法

- 设置：Pythia-160M（revision `50f5173d932e8e61f858120bcb800b97af589f46`，hidden 768、12 层、12 头）+ WikiText-2、seq 256、seed 42、600 步、其余与 EXP-01 完全一致。
- 模型切换：`src/ponder.py` 的 `MODEL`/`REVISION` 改为读环境变量 `PYTHIA_MODEL`/`PYTHIA_REVISION`（默认仍是 70M）；`src/data.py` 的 tokenizer 加载共用这两个常量，一起切换。run 脚本从 config 读 model/revision 并 export；预检：meta-device 加载核对参数量（162,322,944，在 160M–165M 内）+ tokenizer 一致性检查（70M 与 160M 的 tokenizer 对同一段英文的 ids 完全相同，`runs/exp14-tokenizer-parity.json`，parity=true）。
- 数据顺序：与 EXP-01 完全相同（同一 seed、同一 600 步、9600 项，汇总脚本已校验）。
- 评测：step-600 checkpoint，验证集固定 977 chunk、249135 targets。
- 实现：`configs/exp14_{t0,t1}.json`（与 EXP-01 相比只改 model/revision/run_dir/run_budget_s）、`scripts/run_exp14.sh`、`scripts/summarize_exp14.py`、`scripts/check_exp14.py`、测试 `tests/test_ponder_model_override.py`（29 项测试全过）。

## 2. 结果

| 模型 | 160M NLL | 70M NLL（EXP-01） | 规模带来的改进 |
|---|---:|---:|---:|
| T0 vanilla | 4.7960 | 4.9271 | −0.1311 |
| T1 完整 PonderLM-2 | **5.3888** | 6.0070 | **−0.6182** |

差距 T1−T0：70M 为 +1.0799；160M 为 **+0.5928**，缩小 0.4871（约 45%）。T1 从规模中得到的收益（0.6182）是 T0（0.1311）的 4.7 倍。

Sequential NLL（8×128）：T0 4.6447、T1 5.2762，差距 0.6315，与并行验证的方向一致。

训练成本：T0 529 s、T1 1821 s（约 30 分钟）；峰值显存 T0 3181 MiB、T1 3868 MiB。EXP-11 外推的 T1 耗时（63 分钟）比实测高约一倍，显存外推（3496 MiB）比实测低约 10%——外推的方向一正一负，今后引用时以实测为准。

## 3. 结论

1. **规模是强杠杆，差距几乎减半**。160M 下 T1−T0 从 1.08 缩到 0.59（−45%），比 EXP-12 的“多训一轮”（−21%）效果更强。H2 的规模依赖性得到支持：thought 机制的成本随模型增大而摊薄。
2. **三条证据收敛到同一个结论**：EXP-12（多训：T1 多好 0.41 vs T0 的 0.18）、EXP-13（多更新：T7 比 T1 好 0.48）、EXP-14（放大：T1 多好 0.62 vs T0 的 0.13）——T1 的落后首先是资源问题（优化量 + 模型规模），而不是机制本身不行。论文在 1.4B 上看到的收益，与 tiny-scale 下看到的亏损，是同一条规模/预算曲线的两端。
3. **但 T1 在 160M 下仍然输 0.59**。单靠“160M + 600 步”翻不了盘。自然的下一步是看两条杠杆叠加（160M × 1200 步）能把差距压到多少——如果叠加后差距继续大幅缩小，追求规模（410M 需先改内存口径，或换更大 GPU）就值得；如果叠加后差距不动，说明规模与预算的收益已到头，该回头做机制诊断。这是 EXP-15 要回答的问题。
4. **EXP-11 的一个结论被纠正**：换底座不需要 modeling 接线（见 §0）。410M 的障碍只剩显存（vanilla 单 microbatch 已占 96%），不是代码。

## 4. 问题与局限性

- 单种子（seed 42）；WikiText-2 唯一语料；无下游 benchmark。
- 160M 仍是 tiny-scale（论文是 410M–1.4B）；结论不能外推到论文规模。
- T1 的 600 步预算与 70M 的 EXP-01 相同（token 预算相同，不是 FLOPs 预算相同——160M 每 token 的训练成本更高）。
- FLOPs 为解析估计；耗时为实际 wall。
