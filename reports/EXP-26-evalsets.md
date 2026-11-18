# EXP-26 评测报告

数据不完整，不能判断；已完成且通过校验的结果与对照差值见报告。

## 方法

0-shot likelihood；seed=42，jacobi_iterations=3，batch=1。CUDA bf16 autocast，out.loss.float() 转 FP32 后读取 loss；禁用 TF32 与 cudnn benchmark，启用确定性算法，CUBLAS_WORKSPACE_CONFIG=:4096:8。

统一 prompt：`{question} Answer: {answer}`；prefix 为 question + ` Answer: `。不加特殊 token，offset_mapping 中 end > len(prefix) 的 token 计为答案 token。每题每选项一次前向，同时取 total（prompt + 答案的 CE 之和）与 A（仅答案 span 的 CE 之和）。主分数 R1 选 A 最小项（= lm-eval `acc`）；次分数 R2 选 A / 答案 UTF-8 字节数最小项（字节归一化，= lm-eval `acc_norm` 字节版）。旧主分数 R3（total / 答案 token 数最小）已废弃——它系统性选最长选项；同时保存 R3 与 raw total 最小项的准确率仅作诊断。平局取源顺序第一项。上下文溢出直接报错，不截断或按长度挑题。

SciQ：support_excluded: true；support 保留在数据 JSON 备查，prompt 不含 support。选项固定为 correct_answer、distractor1、distractor2、distractor3，gold=0；每选项独立打分，固定 gold 位置不引入位置偏置。

Pile：规范来源 EleutherAI/pile 的 The Eye validation URL `https://the-eye.eu/public/AI/pile/val.jsonl.zst` 已不可达：证书 2025-12-21 签发、2026-03-21 过期，curl exit 60；-k range 探测返回 404，HF 讨论亦称已下架。因此使用注明备份该 validation 文件的 mit-han-lab/pile-val-backup 自动 parquet。仅用 shard 0；128000 token 即使每非空文档只有 1 token 也仅需 128000 文档，远少于该分片行数；实际不足则报错，绝不取 train 或其他分片。prepare 校验前 1000 行 pile_set_name 至少两类，分布与行数见 provenance。

Pile 按存储顺序、1000 行 batch 流式 tokenize，不加特殊 token；跨文档拼接，取满 500 个 256-token 块。每块预测 255 token，mean NLL 为块 loss 的等权均值，PPL=exp(mean NLL)，predicted_tokens=127500；total_nll_sum 为每块 NLL × 255 的总和。记录消耗文档数（跳过零 token 文档）、最后一篇剩余 token 与消耗文档来源计数。

每个 checkpoint-task 完成后立即原子写入结果（.tmp 后 replace）。全量仅在 protocol_hash 与 count 均匹配时复用；smoke 永远重跑。

### 数据来源与 pin

- **hellaswag**：dataset=`Rowan/hellaswag`，split=`validation`；URL：https://huggingface.co/datasets/Rowan/hellaswag/resolve/218ec52e09a7e7462a5400043bb9a69a41d06b76/data/validation-00000-of-00001.parquet；revision=`218ec52e09a7e7462a5400043bb9a69a41d06b76`；SHA256=`899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd`；selection：first 2000 validation rows in source order。
- **piqa**：dataset=`ybisk/piqa`，split=`validation`；URL：https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip；revision=`2e8ac2dffd59bac8c3c6714948f4c551a0848bb0`；SHA256=`54d32a04f59a7e354396f321723c8d7ec35cc6b08506563d8d1ffcc15ce98ddd`；selection：all 1838 validation rows (dev.jsonl + dev-labels.lst) in source order。
- **sciq**：dataset=`allenai/sciq`，split=`validation`；URL：https://huggingface.co/datasets/allenai/sciq/resolve/2c94ad3e1aafab77146f384e23536f97a4849815/data/validation-00000-of-00001.parquet；revision=`2c94ad3e1aafab77146f384e23536f97a4849815`；SHA256=`455dd9f1d725cd3ecbce369799a2fbbdbbfecf51ab84a86d56ba3370dc847b8a`；selection：all 1000 validation rows in source order。
- **pile**：dataset=`EleutherAI/pile`，split=`validation`；URL：https://huggingface.co/api/datasets/mit-han-lab/pile-val-backup/parquet/default/validation/0.parquet；revision=`2f5e46ae6a69cf0dce4b12f78241c408936ca0e4`；SHA256=`1a829b612e3723f4035852cf95aad1418a0cc5d291111b64c0fa7777a48f62d4`；selection：first 500 full 256-token blocks from validation shard 0 in storage order。
  mirror_dataset=`mit-han-lab/pile-val-backup`；mirror_revision=`2f5e46ae6a69cf0dce4b12f78241c408936ca0e4`；shard_index=0。

### Checkpoint 与分词器

| checkpoint | mode | tokenizer model | revision |
|---|---|---|---|
| T0_600 | vanilla | EleutherAI/pythia-70m | a39f36b100fe8a5377810d56c3f4789b9c53ac42 |
| T0_1200 | vanilla | EleutherAI/pythia-70m | a39f36b100fe8a5377810d56c3f4789b9c53ac42 |
| T8_1200 | ponder | EleutherAI/pythia-70m | a39f36b100fe8a5377810d56c3f4789b9c53ac42 |
| LOOPMID600 | looped | EleutherAI/pythia-70m | a39f36b100fe8a5377810d56c3f4789b9c53ac42 |
| LOOPMID1200 | looped | EleutherAI/pythia-70m | a39f36b100fe8a5377810d56c3f4789b9c53ac42 |
| T0_160M_600 | vanilla | EleutherAI/pythia-160M | 50f5173d932e8e61f858120bcb800b97af589f46 |
| LOOPMID160_600 | looped | EleutherAI/pythia-160M | 50f5173d932e8e61f858120bcb800b97af589f46 |

looped 参数从各 checkpoint 自身 config.json 读取。70M 与 160M 分词器环境变量逐项切换。

## 结果

| checkpoint | 任务 | accuracy 或 mean NLL | correct/count 或 PPL |
|---|---|---:|---:|
| T0_600 | hellaswag | 缺失或校验失败 | — |
| T0_600 | piqa | 缺失或校验失败 | — |
| T0_600 | sciq | 0.429000 | 429/1000 |
| T0_600 | pile | 缺失或校验失败 | — |
| T0_1200 | hellaswag | 缺失或校验失败 | — |
| T0_1200 | piqa | 缺失或校验失败 | — |
| T0_1200 | sciq | 0.423000 | 423/1000 |
| T0_1200 | pile | 缺失或校验失败 | — |
| T8_1200 | hellaswag | 缺失或校验失败 | — |
| T8_1200 | piqa | 缺失或校验失败 | — |
| T8_1200 | sciq | 0.314000 | 314/1000 |
| T8_1200 | pile | 缺失或校验失败 | — |
| LOOPMID600 | hellaswag | 缺失或校验失败 | — |
| LOOPMID600 | piqa | 缺失或校验失败 | — |
| LOOPMID600 | sciq | 0.420000 | 420/1000 |
| LOOPMID600 | pile | 缺失或校验失败 | — |
| LOOPMID1200 | hellaswag | 缺失或校验失败 | — |
| LOOPMID1200 | piqa | 缺失或校验失败 | — |
| LOOPMID1200 | sciq | 0.428000 | 428/1000 |
| LOOPMID1200 | pile | 缺失或校验失败 | — |
| T0_160M_600 | hellaswag | 缺失或校验失败 | — |
| T0_160M_600 | piqa | 缺失或校验失败 | — |
| T0_160M_600 | sciq | 0.436000 | 436/1000 |
| T0_160M_600 | pile | 缺失或校验失败 | — |
| LOOPMID160_600 | hellaswag | 缺失或校验失败 | — |
| LOOPMID160_600 | piqa | 缺失或校验失败 | — |
| LOOPMID160_600 | sciq | 0.416000 | 416/1000 |
| LOOPMID160_600 | pile | 缺失或校验失败 | — |

通过校验的评测耗时总和：600.628 秒。

## 对照差值

MC 为 candidate accuracy − T0 accuracy，正值更好；Pile 为 candidate mean NLL − T0 mean NLL，负值更好，方向相反。

| 任务 | 对照 | 差值 | 说明 |
|---|---|---:|---|
| hellaswag | LOOPMID600_minus_T0_600 | — | 数据不完整，不能判断 |
| hellaswag | LOOPMID1200_minus_T0_1200 | — | 数据不完整，不能判断 |
| hellaswag | T8_1200_minus_T0_1200 | — | 数据不完整，不能判断 |
| hellaswag | LOOPMID160_600_minus_T0_160M_600 | — | 数据不完整，不能判断 |
| piqa | LOOPMID600_minus_T0_600 | — | 数据不完整，不能判断 |
| piqa | LOOPMID1200_minus_T0_1200 | — | 数据不完整，不能判断 |
| piqa | T8_1200_minus_T0_1200 | — | 数据不完整，不能判断 |
| piqa | LOOPMID160_600_minus_T0_160M_600 | — | 数据不完整，不能判断 |
| sciq | LOOPMID600_minus_T0_600 | -0.009000 | accuracy 差值越高越好（正值为改善） |
| sciq | LOOPMID1200_minus_T0_1200 | +0.005000 | accuracy 差值越高越好（正值为改善） |
| sciq | T8_1200_minus_T0_1200 | -0.109000 | accuracy 差值越高越好（正值为改善） |
| sciq | LOOPMID160_600_minus_T0_160M_600 | -0.020000 | accuracy 差值越高越好（正值为改善） |
| pile | LOOPMID600_minus_T0_600 | — | 数据不完整，不能判断 |
| pile | LOOPMID1200_minus_T0_1200 | — | 数据不完整，不能判断 |
| pile | T8_1200_minus_T0_1200 | — | 数据不完整，不能判断 |
| pile | LOOPMID160_600_minus_T0_160M_600 | — | 数据不完整，不能判断 |

## 问题与局限性

tiny-scale 模型、单一种子与小样本限制外推；HellaSwag 仅取前 2000 题且用 ` Answer: ` 模板而非标准 continuation；SciQ 固定选项顺序，虽独立打分无位置偏置，仍须披露；Pile 只用 shard 0 的前 500 块，并不代表完整 validation。

缺失或无效结果不会计入对照或耗时。tests_log 仅记录测试日志位置，不代表测试已执行；未完成项及日志尾部如下。

```json
{
  "checkpoint": "T0_600",
  "task": "hellaswag",
  "error": "FileNotFoundError(2, 'No such file or directory')",
  "log_tail": "日志不存在"
}
```

```json
{
  "checkpoint": "T0_1200",
  "task": "hellaswag",
  "error": "FileNotFoundError(2, 'No such file or directory')",
  "log_tail": "日志不存在"
}
```

```json
{
  "checkpoint": "T8_1200",
  "task": "hellaswag",
  "error": "FileNotFoundError(2, 'No such file or directory')",
  "log_tail": "日志不存在"
}
```

```json
{
  "checkpoint": "LOOPMID600",
  "task": "hellaswag",
  "error": "FileNotFoundError(2, 'No such file or directory')",
  "log_tail": "日志不存在"
}
```

```json
{
  "checkpoint": "LOOPMID1200",
  "task": "hellaswag",
  "error": "FileNotFoundError(2, 'No such file or directory')",
  "log_tail": "日志不存在"
}
```

```json
{
  "checkpoint": "T0_160M_600",
  "task": "hellaswag",
  "error": "FileNotFoundError(2, 'No such file or directory')",
  "log_tail": "日志不存在"
}
```

```json
{
  "checkpoint": "LOOPMID160_600",
  "task": "hellaswag",
  "error": "FileNotFoundError(2, 'No such file or directory')",
  "log_tail": "日志不存在"
}
```

```json
{
  "checkpoint": "T0_600",
  "task": "piqa",
  "error": "FileNotFoundError(2, 'No such file or directory')",
  "log_tail": "日志不存在"
}
```

```json
{
  "checkpoint": "T0_1200",
  "task": "piqa",
  "error": "FileNotFoundError(2, 'No such file or directory')",
  "log_tail": "日志不存在"
}
```

```json
{
  "checkpoint": "T8_1200",
  "task": "piqa",
  "error": "FileNotFoundError(2, 'No such file or directory')",
  "log_tail": "日志不存在"
}
```

```json
{
  "checkpoint": "LOOPMID600",
  "task": "piqa",
  "error": "FileNotFoundError(2, 'No such file or directory')",
  "log_tail": "日志不存在"
}
```

```json
{
  "checkpoint": "LOOPMID1200",
  "task": "piqa",
  "error": "FileNotFoundError(2, 'No such file or directory')",
  "log_tail": "日志不存在"
}
```

```json
{
  "checkpoint": "T0_160M_600",
  "task": "piqa",
  "error": "FileNotFoundError(2, 'No such file or directory')",
  "log_tail": "日志不存在"
}
```

```json
{
  "checkpoint": "LOOPMID160_600",
  "task": "piqa",
  "error": "FileNotFoundError(2, 'No such file or directory')",
  "log_tail": "日志不存在"
}
```

```json
{
  "checkpoint": "T0_600",
  "task": "pile",
  "error": "FileNotFoundError(2, 'No such file or directory')",
  "log_tail": "日志不存在"
}
```

```json
{
  "checkpoint": "T0_1200",
  "task": "pile",
  "error": "FileNotFoundError(2, 'No such file or directory')",
  "log_tail": "日志不存在"
}
```

```json
{
  "checkpoint": "T8_1200",
  "task": "pile",
  "error": "FileNotFoundError(2, 'No such file or directory')",
  "log_tail": "日志不存在"
}
```

```json
{
  "checkpoint": "LOOPMID600",
  "task": "pile",
  "error": "FileNotFoundError(2, 'No such file or directory')",
  "log_tail": "日志不存在"
}
```

```json
{
  "checkpoint": "LOOPMID1200",
  "task": "pile",
  "error": "FileNotFoundError(2, 'No such file or directory')",
  "log_tail": "日志不存在"
}
```

```json
{
  "checkpoint": "T0_160M_600",
  "task": "pile",
  "error": "FileNotFoundError(2, 'No such file or directory')",
  "log_tail": "日志不存在"
}
```

```json
{
  "checkpoint": "LOOPMID160_600",
  "task": "pile",
  "error": "FileNotFoundError(2, 'No such file or directory')",
  "log_tail": "日志不存在"
}
```

