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

## 2026-11-18 晚：smoke 通过 + Yang timing pilot（D3 已裁决后）
- smoke（205 步，seed 42，160M looped2）：跑完 283 秒，loss 4.69（零步 10.02），step-200 与 step-205 checkpoint 均落盘，metadata 里 expected_flops_ratio=1.672074（设计 1.6721），峰值显存 3.3GB。训练路径已验证，明早可直接开正式臂。
- smoke 两次启动失败的教训已写入 TOMORROW-QUEUE.md：须设 PYTHIA_MODEL/PYTHIA_REVISION（train.py 不读 config 的 model 字段，默认 70M）与 EXP08_RUN_DIR（每 run 一个目录，否则撞 exp01-order.json 缓存报 AssertionError）。
- Yang timing pilot（每任务-模型 200 步，seed 42，7/8 完成）：decision_tree looped 一次超时（120s 上限），已重跑（timeout 300s），结果见 yang-pilot-dt-looped.log。
  - linear：looped 0.219 s/步，50k 步约 3.04h；standard 0.156 s/步，约 2.17h
  - sparse：looped 0.193 s/步，约 2.68h；standard 0.173 s/步，约 2.40h
  - decision_tree：standard 0.469 s/步，20k 步约 2.60h；looped 待补测
  - relu_2nn：looped 0.266 s/步，20k 步约 1.48h；standard 0.424 s/步，约 2.35h
  - 已知 7 项两种子合计约 33.4h，再加 decision_tree looped（待补测），是设计估算 3.0h 的 11 倍以上。pilot 不含评测开销，实际只会更慢。
- 成本上限规则（预注册已定，不须重新批准）：pilot 预计 >3h 时依次 (1) 决策树 20k→10k 步；(2) Yang 全线降为 1 种子；(3) 稀疏 50k→20k 步。三项全执行后单种子仍约 13.5–15h（视 decision_tree looped 补测值），仍超 3h 上限——预注册的削减分支已用尽。Yang 全量装不进明天的 GPU 窗口，明早协调员先跑 A1→L2，Yang 等用户另作裁决（只跑核心任务 / 再降步数 / 暂缓 Yang），不擅自开全量。

## 2026-11-18 晚补：decision_tree looped 补测完成，Yang 预算定稿
- decision_tree looped：1.273 s/步，20k 步约 7.07h（比 standard 的 2.60h 慢 2.7 倍，是最贵的一组）。Yang pilot 8/8 完成（见 yang-pilot.log 与 yang-pilot-dt-looped.log）。
- Yang 全量（两种子，满步数）约 47.6h；单种子约 23.8h。设计估算 3.0h，实测约 16 倍。pilot 不含评测开销，实际只会更慢。
- 预注册三项削减全执行后（决策树 10k 步、单种子、稀疏 20k 步）仍约 15.9h，仍超 3h 上限 5 倍以上。削减分支已用尽，Yang 须用户另作裁决；明早先跑 A1→L2，Yang 不擅自开。

## 2026-11-19 上午：开始执行 A1→L2（用户 09:25 电脑已开）
- 用户 09:29 最新指示：Yang 无所谓，跑完 A1→L2 就停，不用再跑 Yang 填显卡时间。重点改到写报告：每臂完成时就写一份方法、结果、问题、局限齐全的报告，并把该臂回答的问题写清楚，不要只报数字。A1/L2 结果里若有重要发现，继续设计后续实验（设计与问题写出来汇报），但开跑前必须想好、验证好；新实验未经设计论证和批准不要开。
- 执行顺序：A1（4 个 run）→ L2（6 个 run），背靠背；L1 冻结；Yang 不跑；条件下游不触发。
- 评测入口纠正：A1 两臂一律走 src.evaluate_looped2 addition（--eval-r 4；vanilla 走 emb_scale 挂钩的 plain forward）。不要用 src.evaluate_addition.py——它只认旧 loop 配置与 looped.configure，遇到新 looped2 checkpoint 会静默按 vanilla（R=1）评测，且不处理 emb_scale。L2 两臂一律走 src.evaluate_looped2 lm（--eval-r 4；vanilla 从零训走 emb_scale 的 plain forward）。config_from_checkpoint 已核对：只有含 looped2 三段的 checkpoint 才走 looped 分支。
- 10 个 config 已全部生成并核对（runs/reproduction/a1-*.json、l2-*.json；A1：70M，8000 步，batch 256，lr 1e-3，warmup 100，严格 3 位数，from_scratch_init，emb_scale √512；L2：160M，1200 步，lr 3e-4，warmup 100，from_scratch）。驱动脚本 runs/reproduction/run_a1.py、run_l2.py 已备好并通过语法检查；每 run 训练加评测加提交，失败即停。

## 2026-11-19 10:12 A1 驱动被 SIGTERM 杀掉，s43 训练存活，已恢复（A1 仍在跑）
- A1 驱动（run_a1.py，后台 exec proc_7b5cbc178940）在 10:12 被 SIGTERM 终止，死在 a1-vanilla-s43 训练中段。原因不明（exec 会话被终止，非训练本身失败；训练 loss/进度此前均正常）。
- 已完成的两个 s42 run 不受影响，已提交（本地，未推送，待 resume 驱动最后统一 push）：
  - a1-vanilla-s42：8000 步，终 loss 0.700，评测 3 位数 161/500=0.322，4 位数 0，5-6 位数 0。提交 a87d3b2。
  - a1-looped-s42：8000 步，终 loss 0.723，评测 3 位数 161/500=0.322（与 vanilla 完全相同），4 位数 0，5-6 位数 0。提交 b8634ef。
- **s42 关键发现（先报告，不下机制结论）**：两模型 3 位数均只有 0.322，远低于预注册 sanity 门 90%（设计 §7.4）。按冻结规则，4 位数泛化比较判"不可解释"，只报数字。Loss 平台 ~0.70（EXP-23 混训终 loss 0.012），两模型都没把任务学透。**vanilla 也只有 0.322**，所以失败原因不在 looped 架构（vanilla 没有 looped），在训练设置本身（严格 3 位数-only、无课程：EXP-23 混训 1-3 位数达 97%；或新从零训初始化+emb_scale 的影响——两者同时变了，无法区分）。161/500 完全相同提示两模型学到了相同的部分逐位技能（逐位 ~70% 正确率 → 0.7^3 ≈ 0.34 精确匹配）。s43 复制后若仍 ~32%，该发现稳固；届时在 A1 报告里写清，并设计后续实验（只设计、不开跑）。
- 恢复：a1-vanilla-s43 训练进程在 SSH 会话死后存活为孤儿（PID 3260），健康推进（10:16 已到 step 2300，约 10:28 完成）。已启动 resume 驱动 runs/reproduction/run_a1_resume.py（PID 3599，setsid 脱离会话，只写日志文件、不向 stdout 打印），等 s43 vanilla 的 complete.json → 评测+提交 → 训 s43 looped → 评测+提交 → git push。预计 A1 约 10:53 完成。
- 教训：长时驱动必须只写日志文件、不向 stdout 打印（stdout 管道断裂会杀死驱动），且用 setsid nohup 脱离 exec 会话启动；否则 exec 会话一旦被终止，驱动随之死、只剩子进程存活。

## 2026-11-19 上午：A1 全部完成，L2 已启动（新协调员接手）
- A1 四组全部完成（训练+评测+本地提交；推送因 github.com DNS 故障暂未成功，见下）：
  - a1-vanilla-s42：8000 步，终 loss 0.6999，3 位数 161/500=0.322，4 位数 0，5-6 位数 0。
  - a1-looped-s42：8000 步，终 loss 0.7230，3 位数 161/500=0.322，4 位数 0，5-6 位数 0。
  - a1-vanilla-s43：8000 步，终 loss 0.7239，3 位数 133/500=0.266，4 位数 0，5-6 位数 0。
  - a1-looped-s43：8000 步，终 loss 0.6804，3 位数 139/500=0.278，4 位数 0，5-6 位数 0。
- s43 复制 s42 的结论：3 位数均远低于 90% sanity 门；4 位数全 0。两种子 3 位数均值：vanilla 0.294，looped 0.300；配对差 s42 为 0，s43 为 +0.012。按冻结规则，4 位数泛化比较判不可解释，只报数字。
- 推送阻塞：github.com 从 WSL 的 DNS 解析失败（Tailscale MagicDNS 100.100.100.100 返回 Temporary failure），10:50 与 10:55 两次 git push 均 rc=128；经 HTTP 代理的 SSH（nc CONNECT 到 22 端口）也被拒，ssh.github.com:443 直连挂起。4 个 A1 提交（a87d3b2、b8634ef、83915bd、e32efca）暂只在本地，待 DNS 恢复后补推。HTTPS 经代理可通（curl 200），但 SSH 是唯一已配好的推送通道。
- L2 已启动：驱动 runs/reproduction/run_l2.py（PID 5261，setsid 脱离会话，只写日志），6 个 run 背靠背（vanilla/looped × 42/43/44，各 1200 步，160M 从零训）。run_l2.py 的 log() 已去掉 stdout 打印（只写日志文件），与 resume 驱动一致。预计约 2.5 小时，约 13:30 完成。L1 冻结，Yang 不跑。
- A1 报告撰写中（L2 训练期间并行写），中文版与英文版完成后提交推送。

## 2026-11-19 下午：L2 全部完成，判分支 (b)

- 6 组全部完成并评测：vanilla s42 5.2810、looped s42 5.2338、vanilla s43 5.2785、looped s43 5.2353、vanilla s44 5.2709、looped s44 5.2253。
- 配对差：s42 -0.0472、s43 -0.0432、s44 -0.0456；配对均值 -0.0453，SD 0.0020，SE 0.0012，t=-39.0 (df=2)，p=0.00066，95% 区间 [-0.0503, -0.0403]。
- 裁决（设计 §7.2）：分支 (b)，looped 未赢。均值 -0.0453 未达 -0.05 阈值（差 0.0047）；方向 3/3 一致。注意：L2 是同 token 数对照（looped FLOPs 约 1.676× vanilla），非等 FLOPs。
- 执行：vanilla-s44 第一遍死在 1197/1200（电脑重启），resume 驱动 run_l2_resume.py（setsid）重跑；looped-s44 背靠背完成。推送成功（git push rc=0，DNS 已恢复）。
- 存储：按 2026-11-19 新规矩，每组评测完删中间 checkpoint、只留最终 step-1200（s44 两组各腾出 9.74 GB；watcher trim_l2_s44.py 执行）。
- 报告：中文 ~/workspace/research_notes/ponderlm2-looped-20261113/l2-zh.md，英文仓库 reports/L2-from-scratch-looped-vs-vanilla.md。marker 已写 runs/reproduction/l2-complete.marker。
- 后续：先做零成本 R-scaling 诊断（D-L2-4）；补种子（D-L2-1）须授权；L1 冻结、Yang 不跑；不开没过论证关的实验。
