# D3：新实现（looped2）零步基线 — 门槛触发

日期：2026-11-18 晚（执行协调员亲自跑）
仓库 HEAD：`8442e22`；GPU RTX 4060 Laptop；仓库 `.venv/bin/python`。

## 方法
- 零步 checkpoint：`runs/reproduction/d3_zero_step.py`——加载 Pythia-160M 底座权重，
  `looped2.resolve_config(widths=[3,6,3])` + `apply_config`，`model.save_pretrained` 到
  `runs/reproduction/d3-zero-step/`（与 `train.py` 同一保存路径）。未跑任何训练步。
- 评测：官方入口 `.venv/bin/python -m src.evaluate_looped2 lm --checkpoint
  runs/reproduction/d3-zero-step --eval-r 4 --output runs/reproduction/d3-zero-step.json`。
  WikiText-2 验证集 977 块、249,135 个预测 token。

## 结果
| 评测 | NLL | 与底座 4.3733 之差 |
|---|---|---|
| looped2 零步，r=4，s0_seed=0 | **10.0225** | **+5.65** |
| looped2 零步，r=4，s0_seed=1234（复核） | 10.0336 | +5.66 |
| looped2 零步，r=2，s0_seed=1234 | 10.0770 | +5.70 |
| looped2 零步，r=1，s0_seed=1234 | 12.0266 | +7.65 |
| 旧 looped 图零步（Step 0，对照） | 5.3841 | +1.01 |
| Pythia-160M 底座零步（Step 0） | 4.3733 | — |

## 判定：门槛触发，停训等裁决
设计 D3 门槛：新图零步损伤 ≥ +1.0 NLL 即停下汇报。实测 +5.65，远超门槛。
**L1 及之后全部训练臂暂停，bonus L1 未跑，GPU smoke 未跑，等用户裁决。**

## 机制（已读代码核实，不是测量错误）
- `src/looped2.py forward()`：`h = s0`——循环初态就是纯随机噪声 s_0
  （截断正态，截断前 std=√(2/5)≈0.63，768 维，范数约 17）；prelude 输出只在每轮
  开头以 `h = h + e` 注入，从不作为初态。这是设计 §3.1 第 2 点写死的做法
  （Huginn `like-init` 口径，s_0 与 γ 解耦），代码与设计一致。
- 损伤不在重复次数：r=1 时最差（12.03），r 越大反而越好（注入多轮后部分挽回），
  两个 s_0 种子在 r=4 的结果只差 0.01——不是种子偶然，也不是评测口径问题。
- 与旧图（+1.01）的差额主要来自 s_0 初态：旧图从 prelude 输出直接起循环，
  新图从纯噪声起循环；另有 sandwich 串联改动与新增 n2/n4（设计 §3.1 第 4 点与
  line 268 已预见零步函数会变，但未预见损伤有这么大）。
- 设计 line 268 已写明已知局限：“CPT 训练要同时学会用新 norms，1200 步可能不够”。
  D3 的问题正在于此：CPT 臂要从 −5.65 的坑里爬出来，而旧图只需爬 −1.01。

## 待裁决的问题（供用户定）
1. 是否按原设计继续跑 L1（接受 +5.65 起点，赌 1200 步能爬回）？
2. 还是改初态设计（如 h_0 = prelude 输出 + s_0，或 h_0 = prelude 输出）——
   这偏离 Huginn/冻结设计，需改设计、重核实、重跑 D3？
3. 从零训臂（L2、A1）与 Yang 不受 CPT 损伤影响（权重从头学着用随机 s_0），
   是否先放行、只冻结 CPT 臂 L1？按门槛字面意思，全部训练臂目前都停着，等裁决。
