# Looped 完整复现 — 续跑状态（STATUS）

更新：2026-11-18 晚约 22:00（执行协调员）
仓库：`/home/lenovo_jason/ponderlm2-looped-exp`，HEAD `8442e22`（见下，D3/STATUS 待提交）
设计：`~/workspace/research_notes/ponderlm2-looped-20261113/reproduction-design.md`
预注册：仓库 `reports/REPRODUCTION-prereg.md`（v1.0 冻结，sha256 `a8259083…`）

## 当前状态：D3 门槛触发，全部训练臂暂停，等用户裁决
**不要擅自开训。** 明天 9:30 续跑前先看用户对 D3 的裁决（见“D3 门槛”一节）。
WSL/Windows 今晚约 23:00 关机，明天 9:30 恢复；本文件提交推送后协调员交回。

## 已完成（今晚，全部提交推送）
| # | 内容 | 证据 / commit |
|---|---|---|
| 0 | 预注册 v1.0 入库 + EXP-25 配套脚本 | `06a585c`；`reports/REPRODUCTION-prereg.md` |
| 0 | Step-0 小结 | `6386c8e`；`runs/reproduction/step0-prereg.md` |
| 1 | 新实现 looped2 + Yang + 加法臂，协调员逐项验收 | `8442e22`；`runs/reproduction/implementation-summary.md` |
| 1 | 全套测试 **119/119 通过**（622.6 s） | `8442e22`；本地日志 `runs/reproduction/full-tests-impl.log`（gitignored） |
| 2 | 诊断 D1/D2/D4 | `8442e22`；`runs/reproduction/diagnostics-d1d2d4.md` |
| 2 | Profiler FLOPs 复核：**通过**（比值偏差 +0.001%/+0.420%/+0.652%，均 < ±1%）；`expected_flops_ratio` = 1.672074（设计 1.6721） | `runs/reproduction/profile-flops-looped2.json`（待提交） |
| 2 | D3 新实现零步基线：**门槛触发**（见下） | `runs/reproduction/d3-zero-step.md`、`d3-zero-step.json`、`d3-confirm.json`（待提交） |

## D3 门槛：触发（+5.65 NLL）
- 新 looped2 图零步 NLL（160M，验证集，r=4）：**10.0225**；底座 4.3733；损伤 **+5.65**。
- 复核：s0_seed=1234 时 r=4 为 10.0336、r=2 为 10.0770、r=1 为 12.0266——不是种子偶然，
  损伤不在重复次数。旧 looped 图零步损伤仅 +1.01。
- 机制（已读代码核实）：`looped2.forward()` 里 `h = s0`，循环初态是纯随机噪声
  （截断正态 std=√(2/5)）；prelude 输出只每轮注入一次。这是设计 §3.1 写死的做法，
  代码与设计一致，不是实现 bug。设计 line 268 已预见“CPT 训练要同时学会用新 norms，
  1200 步可能不够”。
- 后果：**bonus L1 未跑；GPU smoke 未跑**（smoke 也是训练，门槛触发后一律停）。

## 诊断 D1/D2/D4 数字（只报告，不裁决）
- D1 R-scaling（EXP-25 checkpoint，seed 42）：r=1: 4.952 / r=2: 4.573 / r=4: 5.433 /
  r=8: 9.107；U 形，最低在训练 r=2；r=2 与已存 evaluation.json 一致到 1e-6。
- D2 漂移：范数 22.68 → 141.47（6.2 倍，无饱和）；与上一轮余弦 →0.997（round 8）；
  token 两两余弦 0.373 → 0.658（未冲向 1.0）。
- D4 采样器（2M 抽）：E[r]=4.9317、E[k]=1.94226，与设计一致。

## 下一臂（待裁决后）
- 若用户裁决**按原设计继续**：先跑 GPU smoke（205 步，配置
  `runs/reproduction/smoke-looped2.json`，验证 GPU 训练路径 + step-200 checkpoint
  保存），再开 L1（looped×1200 vs T0×2006，160M，种子 42/43/44，评测 r=4），
  然后 A1 → L2 → Yang → 条件下游（按设计顺序）。
- 若用户裁决**改初态设计**：先改设计文档 + 预注册附录、重核实、重跑 D3，D3 通过后再开训。
- 若用户裁决**只放行从零训臂**：L2/A1/Yang 可先行（权重从头学着用随机 s_0，
  不受 CPT 损伤影响），L1 仍冻结；但 smoke 仍须先跑。
- 无论哪种裁决：开训前先跑 GPU smoke；`train.py` 每 200 步存 checkpoint 但无续跑入口，
  run 被中断需从头重跑，不要赌关机时间。

## 续跑前检查清单
- [ ] 用户对 D3 的裁决已拿到（继续 / 改设计 / 只放行从零训臂）
- [ ] 仓库 HEAD 与本文件一致，无未提交的改动（`git status` 干净，除 runs/ 下的实验输出）
- [ ] GPU 空闲，`.venv/bin/python` 可用，HF 离线缓存完好
- [ ] 先跑 GPU smoke（205 步），确认 step-200 checkpoint 落盘且 `metadata.json` 里
      `expected_flops_ratio` ≈ 1.6721，再开 L1
- [ ] L1 第一跑先用 seed 42，跑完立即评测 r=4 并与设计核对，再开后续种子

## 待提交（本文件同批）
`runs/reproduction/profile_flops_looped2.py`、`profile-flops-looped2.json`、
`d3_zero_step.py`、`d3-zero-step.json`、`d3-confirm.json`、
`d3-zero-step.md`、`smoke-looped2.json`、`STATUS.md`。
（`runs/reproduction/d3-zero-step/` checkpoint 权重约数百 MB，只留本地，不提交；
明日协调员可在 WSL 上直接复用该目录重评。）

## s0=0 消融（2026-11-18 晚，D3 拆分；新增）

- 方法：`runs/reproduction/s0_zero_ablation.py`，未改 `src/`（monkeypatch `sample_s0` 返回全零），
  官方评测器复用。结果 `s0-zero-ablation.json` / `.md`。
- s0=0：r=1 NLL 9.1932，r=4 NLL 9.6375（随机 s0：r=1 12.0266，r=4 10.0225；底座 4.3733）。
- 拆分（r=1）：随机 s0 部分 2.83（37%），sandwich 结构部分 4.82（63%）。
  r=4 时去掉噪声只挽回 0.39，残留 +5.26 几乎全是结构部分。
- **结论：只改初态救不了 L1**——最好的初态修复在 r=4 仍剩 +5.26 的坑；
  损伤大头是 sandwich 结构改动（并行→串联 + core 边界多一次 final_layer_norm）。
  D3 三条路里"改初态设计"单独走不通；要救 L1 得连结构一起重新设计（超出冻结预注册）。
  从零训臂（L2/A1/Yang）不受影响。
- 明早 9:30 续跑前，先等用户在 D3 三条路上的裁决（本消融供裁决参考）。
