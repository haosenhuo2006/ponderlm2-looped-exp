# EXP-21：seed 42 下游评测

## 1 方法

沿用 [EXP-09 下游协议](EXP-09-downstream.md)（reports/EXP-09-downstream.md）；EXP-09 数字仅引用，不重新评测。固定 seed 42 的四个新 checkpoint，不重训，不调 prompt、不挑样本。分词器 EleutherAI/pythia-70m，revision a39f36b100fe8a5377810d56c3f4789b9c53ac42，add_special_tokens=False。batch=1，bf16 autocast，直接使用各入口返回的 FP32 loss（未用另算的 CE 替换）；固定 seed=42、确定性算法，无采样。
ARC-Easy：validation 全量 570 题，不访问 test。固定 prompt：`{question} Answer: {answer}`。每个选项调用自身 loss 入口，total NLL = loss × (完整序列 token 数−1)，包含问题与答案的全部 shifted CE；再除以完整编码中与答案字符区间相交的 token 数，取最小值（等价于最大负归一化 NLL；平局取源顺序第一项）。同时记录 raw total NLL 和其准确率。它并非仅答案条件似然。
LAMBADA：标准 plain_text test 原顺序前 500 条。原文 rstrip 后用最后一个非空白词定义末词；offset mapping 定位所有相交 token（包括其可能携带的前导空格）。完整文本 teacher forcing，logits[j−1] 的 top-1 对照末词 token j。主指标只用单 token 末词；包含口径要求多 token 末词所有 token 在 teacher forcing 下均正确，不等价于自由生成整词准确率。保留标点，不截断上下文。
T0_1200：src.ponder.forward 的标准 vanilla forward（labels=x）；T8_1200：同入口的 ponder 最终预测，K=3；LOOPMID600/LOOPMID1200：src.looped.configure + src.looped.forward(labels=x)，与 src/evaluate_looped.py 同入口。
looped-start 修复：中间层 checkpoint 使用其保存的 loop_block_start=2 进行评测，由新增单元测试验证（tests/test_downstream.py、tests/test_looped.py）。

### 数据可行性探测与来源

EXP-09 的官方 ARC S3 探测返回 HTTP 404，使用 HF 同数据集固定 revision。本次从 runs/EXP-09/data/ 复用两个原始 parquet，不重新下载；--prepare 重新校验固定 SHA256，选取数据另记 SHA256。

- arc: ready; URL: https://huggingface.co/datasets/allenai/ai2_arc/resolve/210d026faf9955653af8916fad021475a3f00453/ARC-Easy/validation-00000-of-00001.parquet; revision: 210d026faf9955653af8916fad021475a3f00453; SHA256: ed890ff1e4cef7a7140d3a30dcea3ed2c9d467c6458f447ad9ef0176d8dcbb74; 样本数: 570; 选取文件 SHA256: 86cf60991f22840a3bd78df89c29ca99e5978350ec0ce0a98d95b667f297c113.

- lambada: ready; URL: https://huggingface.co/datasets/cimec/lambada/resolve/5953bd97664b64b95754f299b2309ecfbfbe81b9/plain_text/test-00000-of-00001.parquet; revision: 5953bd97664b64b95754f299b2309ecfbfbe81b9; SHA256: 9ec8a3b32099a8bb590515352e09a9fc790e044162c6d1af35c26efd810860e9; 样本数: 500; 选取文件 SHA256: ed39d04cb5f8872e737268b33a75747f81f793a4698d4753317410dcb532f122.

## 2 做了什么

命令：`nohup bash scripts/run_exp21.sh > runs/EXP-21/pipeline.log 2>&1 &`；配置：configs/exp21_downstream.json。逐 checkpoint、逐任务串行运行，各项日志位于 runs/EXP-21/logs/；完成后执行 scripts/summarize_exp21.py。已完整完成且协议指纹匹配的启动前首项结果可复用，仍记录原始耗时。
执行信息：`{"started_at": "2026-11-16T03:01:07.828401+00:00", "hardware": "NVIDIA GeForce RTX 4060 Laptop GPU", "torch_version": "2.11.0+cu130", "git_hash": "d2b1def25ca9e289ffb95361fcc8f78a2c1f15f2", "git_status": " M src/downstream.py\n M tests/test_downstream.py\n M tests/test_looped.py\n?? configs/exp21_downstream.json\n?? runs/EXP-12-T0/\n?? runs/EXP-12-T1/\n?? runs/EXP-12-T2/\n?? runs/EXP-13-T7/\n?? runs/EXP-14-T0/\n?? runs/EXP-14-T1/\n?? runs/EXP-15-T0/\n?? runs/EXP-15-T1/\n?? runs/EXP-16-T6-pilot/\n?? runs/EXP-16-T6/\n?? runs/EXP-17-T8-pilot/\n?? runs/EXP-17-T8/\n?? runs/EXP-18-T0-1200-s43/\n?? runs/EXP-18-T0-1200-s44/\n?? runs/EXP-18-T8-1200-s42/\n?? runs/EXP-18-T8-1200-s43/\n?? runs/EXP-18-T8-1200-s44/\n?? runs/EXP-19-loopmid-pilot/\n?? runs/EXP-19-loopmid-s42/\n?? runs/EXP-19-loopmid-s43/\n?? runs/EXP-19-loopmid-s44/\n?? runs/EXP-20-loopmid-pilot/\n?? runs/EXP-20-loopmid-s42/\n?? runs/EXP-20-loopmid-s43/\n?? runs/EXP-20-loopmid-s44/\n?? runs/EXP-21/\n?? runs/PROBE-2/\n?? runs/exp12-summary.json\n?? runs/exp12.lock\n?? runs/exp13-summary.json\n?? runs/exp13.lock\n?? runs/exp14-summary.json\n?? runs/exp14-tokenizer-parity.json\n?? runs/exp14.lock\n?? runs/exp15-summary.json\n?? runs/exp15-tokenizer-parity.json\n?? runs/exp15.lock\n?? runs/exp16-summary.json\n?? runs/exp16-watch-result.txt\n?? runs/exp16.lock\n?? runs/exp17-summary.json\n?? runs/exp17-watch-result.txt\n?? runs/exp17.lock\n?? runs/exp18-summary.json\n?? runs/exp18.lock\n?? runs/exp19-summary.json\n?? runs/exp19.lock\n?? runs/exp20-summary.json\n?? runs/exp20.lock\n?? runs/exp21.lock\n?? runs/probe-h-T1.json\n?? runs/probe-h-T2.json\n?? scripts/run_exp21.sh\n?? scripts/summarize_exp21.py\n", "source_sha256": {"src/downstream.py": "6f3a4c2c3d5ae3f56f169bb44d3b0405c5da2c9af6ac3cfe1bbdb677dbbf6e54", "scripts/summarize_exp21.py": "1fc24c7dc7dcf1089ca107a8583548d8eb1b018d819c1ede82517555bf592091", "scripts/run_exp21.sh": "f029d4afde177c929881ffedfe608977e00cf725baf82cdfc284ec1f3662e0aa", "configs/exp21_downstream.json": "59342c588ee858d789b673d126368c9d570fb9c74f518f3f4177530d6439851a", "tests/test_downstream.py": "aed72c5d76de5cb463238801f0f7959300f69cfc9b3181dde1e47cfee0ce2468", "tests/test_looped.py": "e566537ec10b8fdd23ab13c73f7166d29200188a33732fe84bed83650d5d6d64"}, "protocol_hash": "6182f40ba01257b9b41c51bc944369b26cd7b69f3eed5163c396ceb0530e93c6", "command": "nohup bash scripts/run_exp21.sh > runs/EXP-21/pipeline.log 2>&1 &", "environment": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, "tests_log": "runs/EXP-21/logs/tests.log"}`。
逐项实测评测耗时合计 266.280 秒（不含下载、加载权重和测试）。CPU/GPU 异步在计时边界同步；峰值为 PyTorch allocated bytes。训练评测不并行。所有结果写入 EXP-21 新目录，旧 checkpoint/结果只读。

## 3 结果

arc：LOOPMID600 对 T0_1200：负（准确率差 -0.1754%）；LOOPMID1200 对 T0_1200：负（准确率差 -0.8772%）；T8_1200 对 T0_1200：胜（准确率差 +0.7018%）。 lambada：LOOPMID600 对 T0_1200：负（准确率差 -0.8734%）；LOOPMID1200 对 T0_1200：负（准确率差 -0.4367%）；T8_1200 对 T0_1200：负（准确率差 -0.4367%）。 ARC 570 题，2pp 以内的差距在噪声范围内。LAMBADA 在 70M 上接近地板，区分度很低，不单独当结论。

对照列为同一 seed42 checkpoint 的 WikiText-2 validation NLL，非三种子均值。准确率越高越好，NLL 越低越好。

| checkpoint | WikiText-2 NLL | ARC 归一化准确率 | ARC raw 准确率 | LAMBADA 单 token 主指标 | LAMBADA 含多 token |
|---|---:|---:|---:|---:|---:|
| LOOPMID600 | 4.943286857487846 | 27.0175% | 30.8772% | 0.8734% | 0.4000% |
| LOOPMID1200 | 4.933984027116636 | 26.3158% | 31.7544% | 1.3100% | 0.8000% |
| T0_1200 | 4.742789367863197 | 27.1930% | 32.2807% | 1.7467% | 0.8000% |
| T8_1200 | 4.820668799691464 | 27.8947% | 31.5789% | 1.3100% | 0.6000% |

**arc：LOOPMID600 对 T0_1200：负（准确率差 -0.1754%）；LOOPMID1200 对 T0_1200：负（准确率差 -0.8772%）；T8_1200 对 T0_1200：胜（准确率差 +0.7018%）。**

**lambada：LOOPMID600 对 T0_1200：负（准确率差 -0.8734%）；LOOPMID1200 对 T0_1200：负（准确率差 -0.4367%）；T8_1200 对 T0_1200：负（准确率差 -0.4367%）。**

| checkpoint/task | 主指标正确/样本 | 全部正确/样本 | 多 token 正确/样本 | 实测秒 | 峰值 allocated bytes |
|---|---:|---:|---:|---:|---:|
| LOOPMID600/arc | 154/570 | 154/570 | —/— | 39.749 | 491300352 |
| LOOPMID600/lambada | 2/229 | 2/500 | 0/271 | 7.645 | 508252672 |
| LOOPMID1200/arc | 150/570 | 150/570 | —/— | 41.361 | 491300352 |
| LOOPMID1200/lambada | 3/229 | 4/500 | 1/271 | 7.958 | 508252672 |
| T0_1200/arc | 155/570 | 155/570 | —/— | 26.162 | 491506176 |
| T0_1200/lambada | 4/229 | 4/500 | 0/271 | 5.519 | 508560384 |
| T8_1200/arc | 159/570 | 159/570 | —/— | 116.756 | 596880384 |
| T8_1200/lambada | 3/229 | 3/500 | 0/271 | 21.131 | 615721984 |

完整逐题 raw NLL、分母、预测、gold 与末词 token 结果保存在 runs/EXP-21/results/；汇总数字来源与 runs/exp21-summary.json 一致。

## 4 问题与局限性

tiny-scale 70M、仅一个训练种子、WikiText-2 训练；ARC 570 与 LAMBADA 500 样本较小，前500并非随机代表性样本。不能由一个 seed 或小幅准确率差宣称普遍机制收益或统计显著。
ARC 用整序列 NLL/答案 token 数，问题损失也进入分子，容易受答案长度影响；raw 总 NLL 也有长度偏置。本次口径事先固定，不根据结果改格式。
LAMBADA 单 token 子集依赖当前 tokenizer，须结合表中子集大小阅读；多 token 包含口径为 teacher-forced 全 token 命中，不能当作自由生成末词成功率。标点及前导空格按原文本编码。
耗时为本机实测，包含逐样本评分与 host 处理，不含权重加载；本实验没有实测 FLOPs，也不提供下游 FLOPs 估计。以往每输出 token 的缓存推理估计不能直接当作当前整序列 Jacobi 或 looped 的实际 FLOPs。
首项验证和最终串行运行同卡进行，不重训。任务间结论相同也只算本设置的方向复核，平局不算严格复现。

数据下载与所有评测均完成，未发现执行失败。

## 5 下一步建议

先检查逐题预测、答案长度偏置与任务间差异；保留 ARC test，后续预注册答案条件 NLL 对照。再扩展训练种子及 LAMBADA 样本并进行配对不确定性分析。更大语料/模型与等实测计算预算对照另立实验，本次不据结果重选 prompt。
