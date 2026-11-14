# EXP-09：seed 42 下游评测

## 1 方法

固定 seed 42 的 step-600 checkpoint，不重训，不调 prompt、不挑样本。分词器 EleutherAI/pythia-70m，revision a39f36b100fe8a5377810d56c3f4789b9c53ac42，add_special_tokens=False。batch=1，bf16 autocast，直接使用各入口返回的 FP32 loss（未用另算的 CE 替换）；固定 seed=42、确定性算法，无采样。
ARC-Easy：validation 全量 570 题，不访问 test。固定 prompt：`{question} Answer: {answer}`。每个选项调用自身 loss 入口，total NLL = loss × (完整序列 token 数−1)，包含问题与答案的全部 shifted CE；再除以完整编码中与答案字符区间相交的 token 数，取最小值（等价于最大负归一化 NLL；平局取源顺序第一项）。同时记录 raw total NLL 和其准确率。它并非仅答案条件似然。
LAMBADA：标准 plain_text test 原顺序前 500 条。原文 rstrip 后用最后一个非空白词定义末词；offset mapping 定位所有相交 token（包括其可能携带的前导空格）。完整文本 teacher forcing，logits[j−1] 的 top-1 对照末词 token j。主指标只用单 token 末词；包含口径要求多 token 末词所有 token 在 teacher forcing 下均正确，不等价于自由生成整词准确率。保留标点，不截断上下文。
T0：src.ponder.forward 的标准 vanilla forward（labels=x）；T1/T5：同入口的 ponder 最终预测，K=3；T2：src.masked.forward，D1 mask、K=3，与 src/evaluate.py --mask D1 一致；T4：src.looped.configure + src.looped.forward(labels=x)，与 src/evaluate_looped.py 同入口。未改训练或已有评测代码。

### 数据可行性探测与来源

官方 ARC S3 探测返回 HTTP 404，按授权改用 HF 同数据集固定 revision。两个任务独立下载；失败任务跳过并在 §4 披露。固定原始文件 SHA256 后重新校验，选取数据另记 SHA256。

- arc: ready; URL: https://huggingface.co/datasets/allenai/ai2_arc/resolve/210d026faf9955653af8916fad021475a3f00453/ARC-Easy/validation-00000-of-00001.parquet; revision: 210d026faf9955653af8916fad021475a3f00453; SHA256: ed890ff1e4cef7a7140d3a30dcea3ed2c9d467c6458f447ad9ef0176d8dcbb74; 样本数: 570; 选取文件 SHA256: 86cf60991f22840a3bd78df89c29ca99e5978350ec0ce0a98d95b667f297c113.

- lambada: ready; URL: https://huggingface.co/datasets/cimec/lambada/resolve/5953bd97664b64b95754f299b2309ecfbfbe81b9/plain_text/test-00000-of-00001.parquet; revision: 5953bd97664b64b95754f299b2309ecfbfbe81b9; SHA256: 9ec8a3b32099a8bb590515352e09a9fc790e044162c6d1af35c26efd810860e9; 样本数: 500; 选取文件 SHA256: ed39d04cb5f8872e737268b33a75747f81f793a4698d4753317410dcb532f122.

## 2 做了什么

命令：`nohup bash scripts/run_exp09.sh > runs/EXP-09/pipeline.log 2>&1 &`；配置：configs/exp09_downstream.json。逐 checkpoint、逐任务串行运行，各项日志位于 runs/EXP-09/logs/；完成后执行 scripts/summarize_exp09.py。已完整完成且协议指纹匹配的启动前首项结果可复用，仍记录原始耗时。
执行信息：`{"started_at": "2026-11-14T06:44:25.782054+00:00", "hardware": "NVIDIA GeForce RTX 4060 Laptop GPU", "torch_version": "2.11.0+cu130", "git_hash": "fd84a4f03e32db625493025fbba8a02d523f587a", "git_status": " M EXPERIMENT_PLAN.md\n?? configs/exp09_downstream.json\n?? runs/EXP-09/\n?? scripts/run_exp09.sh\n?? scripts/summarize_exp09.py\n?? src/downstream.py\n?? tests/test_downstream.py\n", "source_sha256": {"src/downstream.py": "5235cda40b7f80239dad1cd32bdaa6cce6ecade4f4e2a4433d35a71f3afc861a", "scripts/summarize_exp09.py": "a1225178f53a349835bc20ae48be19d2d5872aa26d2152f22754b6a9f1bdc142", "scripts/run_exp09.sh": "1130f45e440162c205753710187cea036d8edb1e70141caf633d3bfba9b0194f", "configs/exp09_downstream.json": "59f3f21f61b582112c00e631d54c984f5f84497cdd5a4a4c318f6f4fc470bf43", "tests/test_downstream.py": "8d8cedd75f0ca9567a60d5daf9ba98e22118006a8199136d59b2b657d176f37c"}, "protocol_hash": "ced2301f31f7d176295eb6ba4a3c875f1409a52a2a04e2cc70ead83dbf49602a", "command": "nohup bash scripts/run_exp09.sh > runs/EXP-09/pipeline.log 2>&1 &", "environment": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, "tests_log": "runs/EXP-09/logs/tests.log"}`。
逐项实测评测耗时合计 502.519 秒（不含下载、加载权重和测试）。CPU/GPU 异步在计时边界同步；峰值为 PyTorch allocated bytes。训练评测不并行。所有结果写入 EXP-09 新目录，旧 checkpoint/结果只读。

## 3 结果

**两个任务的方向结论不一致，不支持将 WikiText-2 NLL 排名直接推广到下游。**

对照列为同一 seed42 checkpoint 的 WikiText-2 validation NLL，非三种子均值。准确率越高越好，NLL 越低越好。

| checkpoint | WikiText-2 NLL | ARC 归一化准确率 | ARC raw 准确率 | LAMBADA 单 token 主指标 | LAMBADA 含多 token |
|---|---:|---:|---:|---:|---:|
| T0 | 4.927076670631949 | 26.6667% | 33.1579% | 3.9301% | 2.0000% |
| T1 | 6.0069609707008125 | 24.9123% | 31.2281% | 0.0000% | 0.0000% |
| T2 | 5.810691901694708 | 25.7895% | 34.0351% | 0.0000% | 0.0000% |
| T4 | 5.485511056242881 | 25.4386% | 33.1579% | 0.4367% | 0.2000% |
| T5 | 5.835645400468433 | 25.0877% | 31.4035% | 0.4367% | 0.2000% |

**arc：T1 对 T0：负（准确率差 -1.7544%）；T2 对 T1：胜（+0.8772%）。**

**lambada：T1 对 T0：负（准确率差 -3.9301%）；T2 对 T1：平（+0.0000%）。**

| checkpoint/task | 主指标正确/样本 | 全部正确/样本 | 多 token 正确/样本 | 实测秒 | 峰值 allocated bytes |
|---|---:|---:|---:|---:|---:|
| T0/arc | 152/570 | 152/570 | —/— | 25.925 | 491506176 |
| T0/lambada | 9/229 | 10/500 | 1/271 | 5.763 | 508560384 |
| T1/arc | 142/570 | 142/570 | —/— | 115.507 | 596880384 |
| T1/lambada | 0/229 | 0/500 | 0/271 | 21.592 | 615721984 |
| T2/arc | 147/570 | 147/570 | —/— | 120.119 | 491937280 |
| T2/lambada | 0/229 | 0/500 | 0/271 | 24.040 | 509202944 |
| T4/arc | 145/570 | 145/570 | —/— | 41.309 | 491300352 |
| T4/lambada | 1/229 | 1/500 | 0/271 | 10.255 | 508252672 |
| T5/arc | 143/570 | 143/570 | —/— | 114.156 | 596880384 |
| T5/lambada | 1/229 | 1/500 | 0/271 | 23.853 | 615721984 |

完整逐题 raw NLL、分母、预测、gold 与末词 token 结果保存在 runs/EXP-09/results/；汇总数字来源与 runs/exp09-summary.json 一致。

## 4 问题与局限性

tiny-scale 70M、仅一个训练种子、WikiText-2 训练；ARC 570 与 LAMBADA 500 样本较小，前500并非随机代表性样本。不能由一个 seed 或小幅准确率差宣称普遍机制收益或统计显著。
ARC 用整序列 NLL/答案 token 数，问题损失也进入分子，容易受答案长度影响；raw 总 NLL 也有长度偏置。本次口径事先固定，不根据结果改格式。
LAMBADA 单 token 子集依赖当前 tokenizer，须结合表中子集大小阅读；多 token 包含口径为 teacher-forced 全 token 命中，不能当作自由生成末词成功率。标点及前导空格按原文本编码。
耗时为本机实测，包含逐样本评分与 host 处理，不含权重加载；本实验没有实测 FLOPs，也不提供下游 FLOPs 估计。以往每输出 token 的缓存推理估计不能直接当作当前整序列 Jacobi 或 looped 的实际 FLOPs。
首项验证和最终串行运行同卡进行，不重训。任务间结论相同也只算本设置的方向复核，平局不算严格复现。

数据下载与所有评测均完成，未发现执行失败。

## 5 下一步建议

先检查逐题预测、答案长度偏置与任务间差异；保留 ARC test，后续预注册答案条件 NLL 对照。再扩展训练种子及 LAMBADA 样本并进行配对不确定性分析。更大语料/模型与等实测计算预算对照另立实验，本次不据结果重选 prompt。
