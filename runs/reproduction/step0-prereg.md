# Step 0 小结：预注册入库 + EXP-25 脚本 + vendor 重验

日期：2026-11-18（执行协调员记录）

## 1. 预注册入库（commit 06a585c，已推 main）

- `reports/REPRODUCTION-prereg.md`：冻结 v1.0 **原文入库**，145 行，sha256 `a8259083d5dd7199be81cab3f61e7f9e0110866ba38372573f482af84d2ba481`，与 `~/workspace/research_notes/ponderlm2-looped-20261113/reproduction-prereg.md` 逐字节一致。
- EXP-25 配套脚本（此前 untracked，现入库）：`scripts/check_exp25.py`（48 行）、`scripts/run_exp25.sh`（100 行）、`scripts/summarize_exp25.py`（148 行），语法检查通过（py_compile / bash -n）。
- 提交 `06a585c`（`docs: frozen reproduction prereg v1.0 + EXP-25 companion scripts`），`origin/main` 已核实为 `06a585c`。
- 注意：预注册正文首部写设计时 HEAD `278a163`；入库时仓库 HEAD 为 `236c31a`（审计修复）。冻结文本**未改**，差异记于提交信息。

## 2. §10 vendor 重验（针对线上仓库，2026-11-18）

核实报告用的是审计补充镜像（无 vendor 文件与 Git 元数据），实现前已对线上仓库重验：

- vendor 文件 SHA-256：`5d685b38cde9fb44a8dea33eb886cc38ccaf5ecb5091fe6b666053ce3560212a`，与审计镜像一致。
- `_is_attention_output` 标记：`modeling_gpt_neox.py:135`；`_is_mlp_output` 标记：`:679`。与核对报告 D3 一致。
- `_init_weights` 公式（:82–105）：截断正态 ±3σ，`std=sqrt(2/(5h))`；attention/MLP 输出投影再除 `sqrt(2·num_layers·(K+1))`。与审计结论一致。
- 审计修复已落地：HEAD 含 `236c31a`。
- 行号漂移（实现须重锚）：`src/train.py` 现为 221 行（核对报告引用时为 210 行）；`src/ponder.py` 59 行（原 40 行）；`src/evaluate_addition.py` 115 行（原 79 行）。Codex 实现时以当前文件为准，不得沿用旧行号。

## 3. 下一步

实现（Step 1）：`src/looped2.py` 新模块 + `train.py` 新模式分支 + 加法臂 + Yang 独立文件；Codex 写码，协调员逐项对照设计 §3 与核实报告验收；完整测试套件全过才开训。
