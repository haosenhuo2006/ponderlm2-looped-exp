# EXP-26 评测报告

28 项全量结果均通过协议与样本数校验；对照差值见报告，结论仅限本次评测。

## 方法

0-shot likelihood；seed=42，jacobi_iterations=3，batch=1。CUDA bf16 autocast，out.loss.float() 转 FP32 后读取 loss；禁用 TF32 与 cudnn benchmark，启用确定性算法，CUBLAS_WORKSPACE_CONFIG=:4096:8。

统一 prompt：`{question} Answer: {answer}`；prefix 为 question + ` Answer: `。不加特殊 token，offset_mapping 中 end > len(prefix) 的 token 计为答案 token。total = loss × (输入 token 数 - 1)，选择 total / 答案 token 数最小项；平局取源顺序第一项。同时保存 raw total 最小项的准确率。上下文溢出直接报错，不截断或按长度挑题。

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
| T0_600 | hellaswag | 0.293000 | 586/2000 |
| T0_600 | piqa | 0.555495 | 1021/1838 |
| T0_600 | sciq | 0.429000 | 429/1000 |
| T0_600 | pile | 7.019712 | PPL=1118.464520 |
| T0_1200 | hellaswag | 0.289500 | 579/2000 |
| T0_1200 | piqa | 0.539717 | 992/1838 |
| T0_1200 | sciq | 0.423000 | 423/1000 |
| T0_1200 | pile | 6.924483 | PPL=1016.867977 |
| T8_1200 | hellaswag | 0.277000 | 554/2000 |
| T8_1200 | piqa | 0.536997 | 987/1838 |
| T8_1200 | sciq | 0.314000 | 314/1000 |
| T8_1200 | pile | 7.398134 | PPL=1632.935160 |
| LOOPMID600 | hellaswag | 0.286500 | 573/2000 |
| LOOPMID600 | piqa | 0.542437 | 997/1838 |
| LOOPMID600 | sciq | 0.420000 | 420/1000 |
| LOOPMID600 | pile | 7.304048 | PPL=1486.303683 |
| LOOPMID1200 | hellaswag | 0.292000 | 584/2000 |
| LOOPMID1200 | piqa | 0.536453 | 986/1838 |
| LOOPMID1200 | sciq | 0.428000 | 428/1000 |
| LOOPMID1200 | pile | 7.483667 | PPL=1778.751994 |
| T0_160M_600 | hellaswag | 0.288500 | 577/2000 |
| T0_160M_600 | piqa | 0.554407 | 1019/1838 |
| T0_160M_600 | sciq | 0.436000 | 436/1000 |
| T0_160M_600 | pile | 7.035910 | PPL=1136.729157 |
| LOOPMID160_600 | hellaswag | 0.295000 | 590/2000 |
| LOOPMID160_600 | piqa | 0.548966 | 1009/1838 |
| LOOPMID160_600 | sciq | 0.416000 | 416/1000 |
| LOOPMID160_600 | pile | 7.133393 | PPL=1253.121694 |

### Pile 分领域

归属口径：每块归属于其首 token 所在文档的 pile_set_name（见 `scripts/pile_block_sets.py`）；跨文档的块不拆分，只按首 token 归属。
结论口径：Pile 结论以去维基版（排除 `Wikipedia (en)` 的块）为准；总 mean NLL / PPL 仅作对照。

局限性：500 块中 `Books3` 占 436 块（87.2%），且全部来自同一篇长文档；其余 9 集合计仅 64 块。这不是均衡的跨领域样本，分集均值（尤其小集）噪声大，跨集比较仅作描述，不作排名依据。

| checkpoint | 集 | 块数 | mean NLL |
|---|---|---:|---:|
| T0_600 | Books3 | 436 | 6.979607 |
| T0_600 | OpenWebText2 | 26 | 7.473650 |
| T0_600 | FreeLaw | 14 | 6.412529 |
| T0_600 | PubMed Abstracts | 8 | 6.811627 |
| T0_600 | StackExchange | 6 | 8.722778 |
| T0_600 | Github | 4 | 7.816927 |
| T0_600 | USPTO Backgrounds | 2 | 6.071902 |
| T0_600 | Wikipedia (en) | 2 | 7.381172 |
| T0_600 | Enron Emails | 1 | 7.785178 |
| T0_600 | Pile-CC | 1 | 9.868458 |
| T0_1200 | Books3 | 436 | 6.906245 |
| T0_1200 | OpenWebText2 | 26 | 7.220196 |
| T0_1200 | FreeLaw | 14 | 6.164437 |
| T0_1200 | PubMed Abstracts | 8 | 6.602291 |
| T0_1200 | StackExchange | 6 | 8.567987 |
| T0_1200 | Github | 4 | 7.507657 |
| T0_1200 | USPTO Backgrounds | 2 | 6.151167 |
| T0_1200 | Wikipedia (en) | 2 | 6.984223 |
| T0_1200 | Enron Emails | 1 | 7.465395 |
| T0_1200 | Pile-CC | 1 | 9.097995 |
| T8_1200 | Books3 | 436 | 7.273396 |
| T8_1200 | OpenWebText2 | 26 | 7.965773 |
| T8_1200 | FreeLaw | 14 | 6.906846 |
| T8_1200 | PubMed Abstracts | 8 | 8.080665 |
| T8_1200 | StackExchange | 6 | 10.715523 |
| T8_1200 | Github | 4 | 11.036912 |
| T8_1200 | USPTO Backgrounds | 2 | 6.990086 |
| T8_1200 | Wikipedia (en) | 2 | 7.990507 |
| T8_1200 | Enron Emails | 1 | 8.819521 |
| T8_1200 | Pile-CC | 1 | 12.193823 |
| LOOPMID600 | Books3 | 436 | 7.209936 |
| LOOPMID600 | OpenWebText2 | 26 | 7.820712 |
| LOOPMID600 | FreeLaw | 14 | 6.692316 |
| LOOPMID600 | PubMed Abstracts | 8 | 7.496265 |
| LOOPMID600 | StackExchange | 6 | 10.537091 |
| LOOPMID600 | Github | 4 | 9.614081 |
| LOOPMID600 | USPTO Backgrounds | 2 | 6.573910 |
| LOOPMID600 | Wikipedia (en) | 2 | 8.244232 |
| LOOPMID600 | Enron Emails | 1 | 8.273316 |
| LOOPMID600 | Pile-CC | 1 | 11.902277 |
| LOOPMID1200 | Books3 | 436 | 7.404735 |
| LOOPMID1200 | OpenWebText2 | 26 | 8.030698 |
| LOOPMID1200 | FreeLaw | 14 | 6.754270 |
| LOOPMID1200 | PubMed Abstracts | 8 | 7.348351 |
| LOOPMID1200 | StackExchange | 6 | 10.489796 |
| LOOPMID1200 | Github | 4 | 9.317345 |
| LOOPMID1200 | USPTO Backgrounds | 2 | 6.666057 |
| LOOPMID1200 | Wikipedia (en) | 2 | 8.819964 |
| LOOPMID1200 | Enron Emails | 1 | 8.376212 |
| LOOPMID1200 | Pile-CC | 1 | 11.668030 |
| T0_160M_600 | Books3 | 436 | 6.984397 |
| T0_160M_600 | OpenWebText2 | 26 | 7.441846 |
| T0_160M_600 | FreeLaw | 14 | 6.278035 |
| T0_160M_600 | PubMed Abstracts | 8 | 7.106205 |
| T0_160M_600 | StackExchange | 6 | 9.355043 |
| T0_160M_600 | Github | 4 | 8.345777 |
| T0_160M_600 | USPTO Backgrounds | 2 | 6.091338 |
| T0_160M_600 | Wikipedia (en) | 2 | 7.460053 |
| T0_160M_600 | Enron Emails | 1 | 7.808146 |
| T0_160M_600 | Pile-CC | 1 | 10.103569 |
| LOOPMID160_600 | Books3 | 436 | 7.049399 |
| LOOPMID160_600 | OpenWebText2 | 26 | 7.574017 |
| LOOPMID160_600 | FreeLaw | 14 | 6.624332 |
| LOOPMID160_600 | PubMed Abstracts | 8 | 7.359415 |
| LOOPMID160_600 | StackExchange | 6 | 9.940680 |
| LOOPMID160_600 | Github | 4 | 9.387535 |
| LOOPMID160_600 | USPTO Backgrounds | 2 | 6.603474 |
| LOOPMID160_600 | Wikipedia (en) | 2 | 7.650613 |
| LOOPMID160_600 | Enron Emails | 1 | 8.516981 |
| LOOPMID160_600 | Pile-CC | 1 | 10.398829 |

| checkpoint | 去维基块数 | 去维基 mean NLL | 去维基 PPL |
|---|---:|---:|---:|
| T0_600 | 498 | 7.018260 | 1116.842082 |
| T0_1200 | 498 | 6.924243 | 1016.624039 |
| T8_1200 | 498 | 7.395755 | 1629.055011 |
| LOOPMID600 | 498 | 7.300272 | 1480.702220 |
| LOOPMID1200 | 498 | 7.478301 | 1769.231618 |
| T0_160M_600 | 498 | 7.034207 | 1134.794517 |
| LOOPMID160_600 | 498 | 7.131316 | 1250.521427 |

通过校验的评测耗时总和：2303.841 秒。

## 对照差值

MC 为 candidate accuracy − T0 accuracy，正值更好；Pile 为 candidate mean NLL − T0 mean NLL，负值更好，方向相反。

| 任务 | 对照 | 差值 | 说明 |
|---|---|---:|---|
| hellaswag | LOOPMID600_minus_T0_600 | -0.006500 | accuracy 差值越高越好（正值为改善） |
| hellaswag | LOOPMID1200_minus_T0_1200 | +0.002500 | accuracy 差值越高越好（正值为改善） |
| hellaswag | T8_1200_minus_T0_1200 | -0.012500 | accuracy 差值越高越好（正值为改善） |
| hellaswag | LOOPMID160_600_minus_T0_160M_600 | +0.006500 | accuracy 差值越高越好（正值为改善） |
| piqa | LOOPMID600_minus_T0_600 | -0.013058 | accuracy 差值越高越好（正值为改善） |
| piqa | LOOPMID1200_minus_T0_1200 | -0.003264 | accuracy 差值越高越好（正值为改善） |
| piqa | T8_1200_minus_T0_1200 | -0.002720 | accuracy 差值越高越好（正值为改善） |
| piqa | LOOPMID160_600_minus_T0_160M_600 | -0.005441 | accuracy 差值越高越好（正值为改善） |
| sciq | LOOPMID600_minus_T0_600 | -0.009000 | accuracy 差值越高越好（正值为改善） |
| sciq | LOOPMID1200_minus_T0_1200 | +0.005000 | accuracy 差值越高越好（正值为改善） |
| sciq | T8_1200_minus_T0_1200 | -0.109000 | accuracy 差值越高越好（正值为改善） |
| sciq | LOOPMID160_600_minus_T0_160M_600 | -0.020000 | accuracy 差值越高越好（正值为改善） |
| pile | LOOPMID600_minus_T0_600 | +0.284336 | NLL 差值越低越好（负值为改善） |
| pile | LOOPMID600_minus_T0_600_no_wiki | +0.282011 | 去维基版 mean NLL 差值，负值更好（结论口径） |
| pile | LOOPMID1200_minus_T0_1200 | +0.559185 | NLL 差值越低越好（负值为改善） |
| pile | LOOPMID1200_minus_T0_1200_no_wiki | +0.554058 | 去维基版 mean NLL 差值，负值更好（结论口径） |
| pile | T8_1200_minus_T0_1200 | +0.473652 | NLL 差值越低越好（负值为改善） |
| pile | T8_1200_minus_T0_1200_no_wiki | +0.471513 | 去维基版 mean NLL 差值，负值更好（结论口径） |
| pile | LOOPMID160_600_minus_T0_160M_600 | +0.097483 | NLL 差值越低越好（负值为改善） |
| pile | LOOPMID160_600_minus_T0_160M_600_no_wiki | +0.097109 | 去维基版 mean NLL 差值，负值更好（结论口径） |

## 问题与局限性

tiny-scale 模型、单一种子与小样本限制外推；HellaSwag 仅取前 2000 题且用 ` Answer: ` 模板而非标准 continuation；SciQ 固定选项顺序，虽独立打分无位置偏置，仍须披露；Pile 只用 shard 0 的前 500 块，并不代表完整 validation。

缺失或无效结果不会计入对照或耗时。tests_log 仅记录测试日志位置，不代表测试已执行；未完成项及日志尾部如下。

无缺失或协议校验问题。
