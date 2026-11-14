"""Generate EXP-09 JSON and report from measured result files, including failures."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.downstream import ROOT, RUN, config, save

def comparison(results, task):
    values = {n: results.get(n, {}).get(task, {}).get("primary_accuracy") for n in ("T0","T1","T2")}
    if any(v is None for v in values.values()):
        return dict(status="incomplete", message="数据不完整，不能判断 T1−T0 / T2−T1。")
    a, b = values["T1"]-values["T0"], values["T2"]-values["T1"]
    def direction(v):
        return "胜" if v > 0 else ("负" if v < 0 else "平")
    return dict(status="complete", t1_minus_t0=a, t2_minus_t1=b,
        t1_loses_t0=a < 0, t2_beats_t1=b > 0,
        message=f"T1 对 T0：{direction(a)}（准确率差 {a:+.4%}）；T2 对 T1：{direction(b)}（{b:+.4%}）。")

def main():
    cfg = config()
    provenance = json.loads((RUN/"data-provenance.json").read_text())
    results, issues, nll = {}, [], {}
    for name, spec in cfg["checkpoints"].items():
        results[name] = {}
        prior = ROOT / spec["path"]
        prior = prior.parents[1] / "evaluation.json"
        nll[name] = json.loads(prior.read_text())["nll"] if prior.exists() else None
        for task in cfg["tasks"]:
            p = RUN/"results"/f"{name}-{task}.json"
            if p.exists():
                r = json.loads(p.read_text())
                from src.downstream import protocol_hash
                if r.get("protocol_hash") != protocol_hash():
                    issues.append(f"{name}/{task}: 协议指纹不匹配，排除结果")
                elif r["count"] != cfg["tasks"][task]["samples"]:
                    issues.append(f"{name}/{task}: 样本数异常，排除结果")
                else:
                    results[name][task] = {k:v for k,v in r.items() if k != "records"}
            else:
                log = RUN/"logs"/f"{name}-{task}.log"
                tail = log.read_text(errors="replace")[-1800:] if log.exists() else "no log"
                issues.append(f"{name}/{task} 无结果。日志末尾：{tail}")
    comparisons = {task:comparison(results,task) for task in cfg["tasks"]}
    for task, source in provenance["tasks"].items():
        if source["status"] != "ready":
            issues.append(f"{task} 下载失败：{source.get('error')}；下一步修复同来源固定版本下载，不更换任务。")
    complete = all(len(r) == len(cfg["tasks"]) for r in results.values())
    both = all(c["status"] == "complete" for c in comparisons.values())
    def sign(v):
        return (v > 0) - (v < 0)
    disagreement = both and any(sign(comparisons["arc"][key]) != sign(comparisons["lambada"][key])
        for key in ("t1_minus_t0","t2_minus_t1"))
    if disagreement:
        conclusion = "**两个任务的方向结论不一致，不支持将 WikiText-2 NLL 排名直接推广到下游。**"
    elif both:
        conclusion = "两个任务方向一致。"
        conclusion += "T1 输给 T0：" + ("均成立。" if comparisons["arc"]["t1_loses_t0"] else "未在两个任务均复现。")
        conclusion += "T2 赢过 T1：" + ("均成立。" if comparisons["arc"]["t2_beats_t1"] else "未在两个任务均复现。")
    else:
        conclusion = "评测不完整，仅讨论已完成任务；不能对两个任务给出完整结论。"
    elapsed = sum(r["seconds"] for ts in results.values() for r in ts.values())
    metadata = json.loads((RUN/"execution.json").read_text()) if (RUN/"execution.json").exists() else {}
    feasibility = json.loads((RUN/"feasibility.json").read_text()) if (RUN/"feasibility.json").exists() else {}
    summary = dict(feasibility=feasibility, experiment="EXP-09", generated_at=datetime.now(timezone.utc).isoformat(),
        complete=complete, config=cfg, provenance=provenance, execution=metadata,
        results=results, wikitext_seed42_nll=nll, comparisons=comparisons,
        disagreement=disagreement, conclusion=conclusion, issues=issues, measured_evaluation_seconds=elapsed)
    save(ROOT/"runs/exp09-summary.json", summary)
    def pct(v):
        return "—" if v is None else f"{100*v:.4f}%"
    lines = ["# EXP-09：seed 42 下游评测", "", "## 1 方法", "",
        "固定 seed 42 的 step-600 checkpoint，不重训，不调 prompt、不挑样本。分词器 EleutherAI/pythia-70m，revision a39f36b100fe8a5377810d56c3f4789b9c53ac42，add_special_tokens=False。batch=1，bf16 autocast，直接使用各入口返回的 FP32 loss（未用另算的 CE 替换）；固定 seed=42、确定性算法，无采样。",
        "ARC-Easy：validation 全量 570 题，不访问 test。固定 prompt：`{question} Answer: {answer}`。每个选项调用自身 loss 入口，total NLL = loss × (完整序列 token 数−1)，包含问题与答案的全部 shifted CE；再除以完整编码中与答案字符区间相交的 token 数，取最小值（等价于最大负归一化 NLL；平局取源顺序第一项）。同时记录 raw total NLL 和其准确率。它并非仅答案条件似然。",
        "LAMBADA：标准 plain_text test 原顺序前 500 条。原文 rstrip 后用最后一个非空白词定义末词；offset mapping 定位所有相交 token（包括其可能携带的前导空格）。完整文本 teacher forcing，logits[j−1] 的 top-1 对照末词 token j。主指标只用单 token 末词；包含口径要求多 token 末词所有 token 在 teacher forcing 下均正确，不等价于自由生成整词准确率。保留标点，不截断上下文。",
        "T0：src.ponder.forward 的标准 vanilla forward（labels=x）；T1/T5：同入口的 ponder 最终预测，K=3；T2：src.masked.forward，D1 mask、K=3，与 src/evaluate.py --mask D1 一致；T4：src.looped.configure + src.looped.forward(labels=x)，与 src/evaluate_looped.py 同入口。未改训练或已有评测代码。",
        "", "### 数据可行性探测与来源", "",
        "官方 ARC S3 探测返回 HTTP 404，按授权改用 HF 同数据集固定 revision。两个任务独立下载；失败任务跳过并在 §4 披露。固定原始文件 SHA256 后重新校验，选取数据另记 SHA256。"]
    for task, p in provenance["tasks"].items():
        lines += ["", f"- {task}: {p['status']}; URL: {p['url']}; revision: {p['revision']}; SHA256: {p.get('sha256','未取得')}; 样本数: {p.get('count',0)}; 选取文件 SHA256: {p.get('selected_sha256','未取得')}."]
        if p["status"] != "ready":
            issues.append(f"{task} 下载失败：{p.get('error')}; 下一步修复同来源下载并保持固定任务，不替换任务。")
    lines += ["", "## 2 做了什么", "",
        "命令：`nohup bash scripts/run_exp09.sh > runs/EXP-09/pipeline.log 2>&1 &`；配置：configs/exp09_downstream.json。逐 checkpoint、逐任务串行运行，各项日志位于 runs/EXP-09/logs/；完成后执行 scripts/summarize_exp09.py。已完整完成且协议指纹匹配的启动前首项结果可复用，仍记录原始耗时。",
        f"执行信息：`{json.dumps(metadata,ensure_ascii=False)}`。",
        f"逐项实测评测耗时合计 {elapsed:.3f} 秒（不含下载、加载权重和测试）。CPU/GPU 异步在计时边界同步；峰值为 PyTorch allocated bytes。训练评测不并行。所有结果写入 EXP-09 新目录，旧 checkpoint/结果只读。",
        "", "## 3 结果", "", conclusion, "",
        "对照列为同一 seed42 checkpoint 的 WikiText-2 validation NLL，非三种子均值。准确率越高越好，NLL 越低越好。",
        "", "| checkpoint | WikiText-2 NLL | ARC 归一化准确率 | ARC raw 准确率 | LAMBADA 单 token 主指标 | LAMBADA 含多 token |",
        "|---|---:|---:|---:|---:|---:|"]
    for name in cfg["checkpoints"]:
        a, l = results[name].get("arc",{}), results[name].get("lambada",{})
        lines.append(f"| {name} | {nll[name] if nll[name] is not None else '—'} | {pct(a.get('primary_accuracy'))} | {pct(a.get('raw_accuracy'))} | {pct(l.get('primary_accuracy'))} | {pct(l.get('accuracy'))} |")
    for task, comp in comparisons.items():
        lines += ["", f"**{task}：{comp['message']}**"]
    lines += ["", "| checkpoint/task | 主指标正确/样本 | 全部正确/样本 | 多 token 正确/样本 | 实测秒 | 峰值 allocated bytes |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, tasks in results.items():
        for task,r in tasks.items():
            lines.append(f"| {name}/{task} | {r['primary_correct']}/{r['primary_count']} | {r['correct']}/{r['count']} | {r.get('multi_token_correct','—')}/{r.get('multi_token_count','—')} | {r['seconds']:.3f} | {r['peak_memory_bytes']} |")
    lines += ["", "完整逐题 raw NLL、分母、预测、gold 与末词 token 结果保存在 runs/EXP-09/results/；汇总数字来源与 runs/exp09-summary.json 一致。",
        "", "## 4 问题与局限性", "",
        "tiny-scale 70M、仅一个训练种子、WikiText-2 训练；ARC 570 与 LAMBADA 500 样本较小，前500并非随机代表性样本。不能由一个 seed 或小幅准确率差宣称普遍机制收益或统计显著。",
        "ARC 用整序列 NLL/答案 token 数，问题损失也进入分子，容易受答案长度影响；raw 总 NLL 也有长度偏置。本次口径事先固定，不根据结果改格式。",
        "LAMBADA 单 token 子集依赖当前 tokenizer，须结合表中子集大小阅读；多 token 包含口径为 teacher-forced 全 token 命中，不能当作自由生成末词成功率。标点及前导空格按原文本编码。",
        "耗时为本机实测，包含逐样本评分与 host 处理，不含权重加载；本实验没有实测 FLOPs，也不提供下游 FLOPs 估计。以往每输出 token 的缓存推理估计不能直接当作当前整序列 Jacobi 或 looped 的实际 FLOPs。",
        "首项验证和最终串行运行同卡进行，不重训。任务间结论相同也只算本设置的方向复核，平局不算严格复现。",
        ""]
    lines += issues or ["数据下载与所有评测均完成，未发现执行失败。"]
    lines += ["", "## 5 下一步建议", "",
        "先检查逐题预测、答案长度偏置与任务间差异；保留 ARC test，后续预注册答案条件 NLL 对照。再扩展训练种子及 LAMBADA 样本并进行配对不确定性分析。更大语料/模型与等实测计算预算对照另立实验，本次不据结果重选 prompt。", ""]
    (ROOT/"reports/EXP-09-downstream.md").write_text("\n".join(lines))
    plan = ROOT/"EXPERIMENT_PLAN.md"
    text = plan.read_text()
    marker = "### EXP-09 下游评测"
    following = ""
    if marker in text:
        before, section = text.split(marker, 1)
        if "\n### " in section:
            following = "\n### " + section.split("\n### ",1)[1]
        text = before.rstrip()+"\n\n"
    text += marker+"\n\nseed42 step-600 T0/T1/T2/T4/T5；ARC-Easy validation 全量570题；标准LAMBADA test前500条（主指标单token末词）。\n"
    text += ("完成。" if complete else "部分完成/有失败，详见报告。") + conclusion + "\n"
    plan.write_text(text + following)
    print(json.dumps(dict(complete=complete,conclusion=conclusion,report="reports/EXP-09-downstream.md"),ensure_ascii=False),flush=True)

if __name__ == "__main__":
    main()
