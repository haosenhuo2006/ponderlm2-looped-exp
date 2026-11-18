"""汇总 EXP-26；只写 summary 与报告，不修改实验计划。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evalsets26 import RUN, config, protocol_hash, save

PAIRS = {
    'LOOPMID600_minus_T0_600': ('LOOPMID600', 'T0_600'),
    'LOOPMID1200_minus_T0_1200': ('LOOPMID1200', 'T0_1200'),
    'T8_1200_minus_T0_1200': ('T8_1200', 'T0_1200'),
    'LOOPMID160_600_minus_T0_160M_600': ('LOOPMID160_600', 'T0_160M_600'),
}


def read_optional(path, issues):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        issues.append(dict(path=str(path), error=repr(error)))
        return {}


def main():
    cfg = config()
    issues = []
    provenance = read_optional(RUN / 'data-provenance.json', issues)
    execution = read_optional(RUN / 'execution.json', issues)
    expected_hash = protocol_hash()
    results = {task: {} for task in cfg['tasks']}
    for task, task_spec in cfg['tasks'].items():
        expected_count = task_spec['blocks'] if task == 'pile' else task_spec['samples']
        for name in cfg['checkpoints']:
            path = RUN / 'results' / f'{task}-{name}.json'
            try:
                result = json.loads(path.read_text())
                if result.get('protocol_hash') != expected_hash or result.get('count') != expected_count:
                    raise ValueError('protocol_hash 或样本数不匹配，排除该结果')
                results[task][name] = {k: v for k, v in result.items() if k != "records"}
            except (OSError, ValueError) as error:
                log = RUN / 'logs' / f'{task}-{name}.log'
                tail = log.read_text(errors='replace')[-1800:] if log.exists() else '日志不存在'
                issues.append(dict(checkpoint=name, task=task, error=repr(error), log_tail=tail))
    comparisons = {}
    for task, items in results.items():
        metric = 'mean_nll' if task == 'pile' else 'accuracy'
        comparisons[task] = {}
        for label, (candidate, baseline) in PAIRS.items():
            if candidate in items and baseline in items:
                comparisons[task][label] = dict(metric=metric,
                    delta=items[candidate][metric] - items[baseline][metric],
                    message='NLL 差值越低越好（负值为改善）' if task == 'pile' else 'accuracy 差值越高越好（正值为改善）')
            else:
                comparisons[task][label] = dict(metric=metric, delta=None, message='数据不完整，不能判断')
    complete = all(len(items) == len(cfg['checkpoints']) for items in results.values())
    conclusion = ('28 项全量结果均通过协议与样本数校验；对照差值见报告，结论仅限本次评测。'
                  if complete else '数据不完整，不能判断；已完成且通过校验的结果与对照差值见报告。')
    seconds = sum(result['seconds'] for items in results.values() for result in items.values())
    summary = dict(experiment='EXP-26', generated_at=datetime.now(timezone.utc).isoformat(),
                   complete=complete, config=cfg, provenance=provenance, execution=execution,
                   results=results, comparisons=comparisons, conclusion=conclusion,
                   issues=issues, measured_evaluation_seconds=seconds)
    save(ROOT / 'runs/exp26-summary.json', summary)
    lines = ['# EXP-26 评测报告', '', conclusion, '', '## 方法', '',
             '0-shot likelihood；seed=42，jacobi_iterations=3，batch=1。CUDA bf16 autocast，'
             'out.loss.float() 转 FP32 后读取 loss；禁用 TF32 与 cudnn benchmark，启用确定性算法，'
             'CUBLAS_WORKSPACE_CONFIG=:4096:8。', '',
             f'统一 prompt：`{cfg["prompt"]}`；prefix 为 question + ` Answer: `。'
             '不加特殊 token，offset_mapping 中 end > len(prefix) 的 token 计为答案 token。'
             'total = loss × (输入 token 数 - 1)，选择 total / 答案 token 数最小项；平局取源顺序第一项。'
             '同时保存 raw total 最小项的准确率。上下文溢出直接报错，不截断或按长度挑题。', '',
             'SciQ：support_excluded: true；support 保留在数据 JSON 备查，prompt 不含 support。'
             '选项固定为 correct_answer、distractor1、distractor2、distractor3，gold=0；'
             '每选项独立打分，固定 gold 位置不引入位置偏置。', '',
             'Pile：规范来源 EleutherAI/pile 的 The Eye validation URL '
             '`https://the-eye.eu/public/AI/pile/val.jsonl.zst` 已不可达：证书 2025-12-21 签发、'
             '2026-03-21 过期，curl exit 60；-k range 探测返回 404，HF 讨论亦称已下架。'
             '因此使用注明备份该 validation 文件的 mit-han-lab/pile-val-backup 自动 parquet。'
             '仅用 shard 0；128000 token 即使每非空文档只有 1 token 也仅需 128000 文档，'
             '远少于该分片行数；实际不足则报错，绝不取 train 或其他分片。'
             'prepare 校验前 1000 行 pile_set_name 至少两类，分布与行数见 provenance。', '',
             'Pile 按存储顺序、1000 行 batch 流式 tokenize，不加特殊 token；跨文档拼接，'
             '取满 500 个 256-token 块。每块预测 255 token，mean NLL 为块 loss 的等权均值，'
             'PPL=exp(mean NLL)，predicted_tokens=127500；total_nll_sum 为每块 NLL × 255 的总和。'
             '记录消耗文档数（跳过零 token 文档）、最后一篇剩余 token 与消耗文档来源计数。', '',
             '每个 checkpoint-task 完成后立即原子写入结果（.tmp 后 replace）。'
             '全量仅在 protocol_hash 与 count 均匹配时复用；smoke 永远重跑。', '',
             '### 数据来源与 pin', '']
    for task, spec in cfg['tasks'].items():
        lines.extend([f'- **{task}**：dataset=`{spec["dataset"]}`，split=`{spec["split"]}`；'
                      f'URL：{spec["url"]}；revision=`{spec["revision"]}`；'
                      f'SHA256=`{spec["sha256"]}`；selection：{spec["selection"]}。'])
        if task == 'pile':
            lines.append(f'  mirror_dataset=`{spec["mirror_dataset"]}`；mirror_revision=`{spec["mirror_revision"]}`；shard_index=0。')
    lines.extend(['', '### Checkpoint 与分词器', '',
                  '| checkpoint | mode | tokenizer model | revision |', '|---|---|---|---|'])
    for name, spec in cfg['checkpoints'].items():
        lines.append(f'| {name} | {spec["mode"]} | {spec["model"]} | {spec["revision"]} |')
    lines.extend(['', 'looped 参数从各 checkpoint 自身 config.json 读取。70M 与 160M 分词器环境变量逐项切换。', '',
                  '## 结果', '', '| checkpoint | 任务 | accuracy 或 mean NLL | correct/count 或 PPL |', '|---|---|---:|---:|'])
    for name in cfg['checkpoints']:
        for task in cfg['tasks']:
            result = results[task].get(name)
            if result is None:
                value, detail = '缺失或校验失败', '—'
            elif task == 'pile':
                value, detail = f'{result["mean_nll"]:.6f}', f'PPL={result["ppl"]:.6f}'
            else:
                value, detail = f'{result["accuracy"]:.6f}', f'{result["correct"]}/{result["count"]}'
            lines.append(f'| {name} | {task} | {value} | {detail} |')
    lines.extend(['', f'通过校验的评测耗时总和：{seconds:.3f} 秒。', '', '## 对照差值', '',
                  'MC 为 candidate accuracy − T0 accuracy，正值更好；Pile 为 candidate mean NLL − T0 mean NLL，负值更好，方向相反。', '',
                  '| 任务 | 对照 | 差值 | 说明 |', '|---|---|---:|---|'])
    for task, pairs in comparisons.items():
        for label, item in pairs.items():
            delta = '—' if item['delta'] is None else f'{item["delta"]:+.6f}'
            lines.append(f'| {task} | {label} | {delta} | {item["message"]} |')
    lines.extend(['', '## 问题与局限性', '',
                  'tiny-scale 模型、单一种子与小样本限制外推；HellaSwag 仅取前 2000 题且用 ` Answer: ` '
                  '模板而非标准 continuation；SciQ 固定选项顺序，虽独立打分无位置偏置，仍须披露；'
                  'Pile 只用 shard 0 的前 500 块，并不代表完整 validation。', '',
                  '缺失或无效结果不会计入对照或耗时。tests_log 仅记录测试日志位置，不代表测试已执行；'
                  '未完成项及日志尾部如下。', ''])
    for issue in issues:
        lines.extend(['```json', json.dumps(issue, ensure_ascii=False, indent=2), '```', ''])
    if not issues:
        lines.append('无缺失或协议校验问题。')
    report = ROOT / 'reports/EXP-26-evalsets.md'
    report.parent.mkdir(parents=True, exist_ok=True)
    tmp = report.with_suffix('.md.tmp')
    tmp.write_text('\n'.join(lines) + '\n')
    tmp.replace(report)
    print(json.dumps(dict(complete=complete, conclusion=conclusion, report=str(report)), ensure_ascii=False))


if __name__ == '__main__':
    main()
