# Step 1 小结：新实现（looped2 + Yang）验收与全套测试

日期：2026-11-18（执行协调员记录）

## 1. 交付文件（commit 待填）

| 文件 | 说明 |
|---|---|
| `src/looped2.py`（新建，约 340 行） | prelude→core→coda；每轮 `h = h + e` 注入；s_0 截断正态（截断前 std=√(2/5)，与 γ 解耦）；仅 core sandwich LayerNorm（n2/n4 新增，串联残差）；双次共用 `final_layer_norm`；log-normal Poisson 采样器（t=s=2、σ=0.5、cap 12）；截断反传；FLOPs 记账；从零重初始化；严格 checkpoint 恢复 |
| `src/evaluate_looped2.py`（新建） | `lm` 子命令（多 eval_r 独立输出、跨 r 复用同一 s_0、s_0 方差诊断）与 `addition` 子命令（逐 r 贪心评测、`id_acc` 仅 3 位准确率） |
| `src/train.py`（221→约 300 行） | 新 `looped2`/`from_scratch` 分支（与旧模式互斥）；每 microbatch 采样一次；记录 (n,k,r)、FLOPs、采样 RNG 状态；优化器建立前注册新 norm / 重初始化 |
| `src/train_addition.py`（144→约 230 行） | EXP-24-new：`strict_3digit`、`looped2`（2,2,2）、`from_scratch_init`（l_eff looped 12 / vanilla 6）、emb_scale=√512；默认行为不变 |
| `src/addition.py`（+5 行） | 仅新增 `sample_operands_strict3`，旧函数未动 |
| `src/yang_tasks.py` / `src/yang_model.py` / `src/train_yang.py`（新建） | port Yang 官方任务采样器与模型结构；截断 loss；固定 b/T；Adam lr 1e-4、无 wd、无裁剪、batch 64；评测归一化 MSE + 90% bootstrap CI；`--pilot-steps` |
| `tests/test_looped2.py`（新建，12 项）、`tests/test_yang.py`（新建，13 项） | CPU 单元测试 |

`src/looped.py`、vendor/、reports/ 未动。Yang 评测主指标定为查询点（最终轮、最后一点），与官方 utils.py 口径一致；全点数据亦返回、可重聚合。

## 2. 协调员验收（逐项对照设计 §3 与核实报告）

- `looped2.py` 全文已读：注入、s_0 解耦、sandwich 串联、双次 ln_f、仅最终 CE、采样器公式、eval_r、FLOPs 公式、重初始化、checkpoint 加载均与设计一致。
- `train.py` / `train_addition.py` diff 已读：互斥校验、emb_scale 规则（CPT γ=1、从零训 γ=√h）、l_eff 名义值、采样器 RNG 与 metadata 记录、旧路径保留均符合。
- `evaluate_looped2.py` 全文已读：跨 r 同 s_0、s_0 方差诊断、id_acc 仅 3 位、vanilla hook 缩放、loop_source=checkpoint 均符合。
- `math` 导入已核查（train.py、train_addition.py 均有）；14/14 新测试协调员重跑通过。
- 唯一小 wart：`_greedy_vanilla` 内联重复而非复用 `addition.greedy_answer`（等价已核对，后续清理）。

## 3. 全套测试

`.venv/bin/python -m unittest discover -s tests -v`：**Ran 119 tests in 622.6s，OK**（94 旧 + 13 Yang + 12 looped2）。日志 `runs/reproduction/full-tests-impl.log`。

## 4. 待 GPU 验证（训练前）
- GPU smoke（新训练分支端到端）与 FLOPs 记账 profiler 复核（< ±1%）。
- D3 新实现零步基线（见诊断小结）。
