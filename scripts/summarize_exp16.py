"""Verify pilot/full T6 and write evidence-backed report."""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
D = Path("runs/EXP-16-T6")
P = Path("runs/EXP-16-T6-pilot")


def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(s) for s in Path(p).read_text().splitlines()]
def table(headers, data):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"] +
                     ["| " + " | ".join(str(v) for v in row) + " |" for row in data])


def pilot():
    t = rows(P / "train.jsonl")
    bases = [rows("runs/EXP-01-T1/train.jsonl"), rows("runs/EXP-06-T5/train.jsonl")]
    vals = [x for v in t for x in [v["loss"], v["final_prediction_ce"], *v["round_ce"], *v["round_grad_norm"]]]
    baseline = median(v["grad_norm"] for b in bases for v in b[:100])
    measured = median(x for v in t for x in v["round_grad_norm"])
    finite = all(math.isfinite(v) for v in vals)
    stable = mean(v["loss"] for v in t[-20:]) <= 1.5 * mean(v["loss"] for v in t[:20])
    passed = len(t) == 100 and (P / "complete.json").exists() and finite and stable and .1 <= measured / baseline <= 10
    result = dict(passed=passed, decision="continue" if passed else "stop", finite=finite,
        no_divergence=stable, no_oom=(P / "complete.json").exists(), gradient_median=measured,
        baseline_gradient_median=baseline, gradient_ratio=measured / baseline,
        max_gradient=max(x for v in t for x in v["round_grad_norm"]),
        last20_mean_ce=mean(v["loss"] for v in t[-20:]),
        last20_final_ce=mean(v["final_prediction_ce"] for v in t[-20:]),
        baseline_last20=[mean(v["loss"] for v in b[80:100]) for b in bases],
        wall_s=read(P / "complete.json")["wall_s"] if (P / "complete.json").exists() else None)
    (P / "decision.json").write_text(json.dumps(result, indent=2))
    return result


def interpretation(nll):
    if nll <= 4.9:
        return ("接近 T0×1200 的 4.8–4.9 档（或更好）", "下一步考虑机制改进的深化或 T6@160M。")
    elif nll >= 5.0:
        return ("≈5.1 的基本不动档（或更差）", "机制方向进入收益递减，重新考虑新机制思路或接受 tiny-scale 天花板。")
    return ("位于 4.8–4.9 与 ≈5.1 两档之间，尚不能归入预设两档", "先检查轨迹并做多种子配对，再决定是否深化机制；只有仍在快速改善时才补 2400 步。")


def report(gate, full=False):
    meta = read((D if full else P) / "metadata.json")
    text = """# EXP-16 — T6 × 1200 步

## 0. 必要性

- 要回答的具体问题：最好的机制（T6）再加一倍训练量，差距能不能继续缩小？机制+预算叠加后，能不能接近 T0×1200 步的 4.74？
- 为什么不能只看已有曲线：T6 训练 CE 到 600 步时 4.93，末段仍在下降（500 步 5.01 → 600 步 4.93），外推不可靠；T1 从 1200 步得了 +0.41，而 T6 的每轮更新意味着 1200 步 ≈ 3013 次更新——EXP-13 证明的更新次数杠杆（59%）会与预算杠杆复合，必须实测。
- 判定逻辑（跑之前写清）：T6×1200 ≈ 4.8–4.9（接近 T0×1200 的 4.74）→ thought 模型第一次接近打平 vanilla，机制优先被验证，下一步考虑机制改进的深化或 T6@160M；T6×1200 ≈ 5.1（相对 5.19 几乎不动）→ T6 的优势基本是 600 步现象，机制方向也进入收益递减，重新考虑（新机制思路，或接受 tiny-scale 天花板）。两种结果都改变下一步决定，所以值得跑。
- 成本：约 2×1147 s ≈ 38 分钟 + 评测；无新 src 代码（只改配置 + 数据顺序 + 脚本）。T0×1200 对照已存在（4.74），不用重跑。
- 考虑过的更便宜替代与为什么没选：只读已有训练曲线外推——T6 末段仍在下降、T1 的 1200 步收益（+0.41）与更新次数杠杆复合，外推不可靠；只做 2400 步——EXP-12 已确立“1200 步足够判定方向”的口径，只有 T6×1200 仍在快速改善时才补；T6@160M——把规模重新混进来，约 80 分钟，只有 EXP-16 很强（接近 T0×1200）时才成为下一步。
- 一个必须先说清的口径：1200 步 cosine 下的 step-600 中间点 LR 还没衰减完（约 5.6e-5，而 EXP-07 的 600 步终点 LR 是 1e-5），**中间点不能直接对标 EXP-07 的终点**；中间点只用于看轨迹，终点对比才是有效比较。

## 1. 方法

一个 optimizer step（本报告称数据 batch 步）对应 16 个 microbatch。
1200×16×256 = 4,915,200 输入 tokens。独立 random.Random(42) 每步采样 K∈{2,3}；
587 个 2、613 个 3，总优化器更新 3013 次（汇总时实时生成 RNG 序列并断言）。
seed 42、Pythia-70M（EleutherAI/pythia-70m，固定 revision a39f36b100fe8a5377810d56c3f4789b9c53ac42）、
WikiText-2 raw-v1、seq 256、microbatch 1、grad_accum 16、bf16 autocast / fp32 master weights、SDPA。
训练 9411 chunks；验证固定 977 chunks、249135 targets，固定 K=3。

对第 k=1..K 轮 refinement，用当前参数重算 init + 1..k 链，对全部 16 个 microbatch
计算 thought-logit F.cross_entropy(z[:, :-1].float(), targets[:, 1:])（展平 batch/token）。
token-only 初始化无 thought 位置，不监督。CE_k 除以 grad_accum 后 backward，
累加 microbatch，clip 1.0，立刻 optimizer.step() 并 zero_grad，再进入下一轮。
轮间 H^(k) detach；CE_{k+1} 只经过当前轮 refinement，不经过前轮图。
每轮前缀用当前 θ 在 no_grad 下重算，当前轮 token embeddings 重新构建，允许训练输入 embedding；不保留旧参数图。
K 轮更新后用最终 θ 无梯度重跑完整 T1 路径，记录 separate final CE，不计入损失。
T5 是 K 轮 CE 均值、一次反向、一次更新；T6 各轮独立反向、各更新一次。
T6/T5 的 per-round CE 定义相同，但更新时机所需的跨轮梯度截断也不同，不能声称纯控制梯度路径。
T6/T1 同时混杂监督方式与更新时机，不能单归因于更新时机。

AdamW betas (0.9,0.95)、weight decay 0.01；LR 按 1200 步 cosine 重算：warmup 20、
峰值 1e-4 → 1e-5，按数据 batch 步索引，同一步的 K 次更新共用 lr。
数据顺序：1200 步进入第 2 epoch，前 9600 项与 runs/exp01-order.json 完全相同，
第 2 epoch 为同一 seed 下的下一个 permutation；与 EXP-12 同一套顺序机制，汇总脚本已校验。
配置 configs/exp16_t6.json，pilot 副本 configs/exp16_t6_pilot.json。
pilot 的 100 步 cosine 与完整 run 不同，只是稳定性门。
K=3 的各轮链共 2+3+4=9 次 backbone 前向，最终监测再重算 5 次，共 14 次；K=2 共 9 次。
FLOPs 仅解析估计，非 profiler 测量。

从干净 main fe72f6f 开始，先 git pull --ff-only；未修改 vendor 或 GPT2_Latent_Reasoning；无新 src 代码。
只新增配置与脚本；跑前全套既有测试通过（测试输出由执行员记录）。
训练内部 3600 s 超时明确抛错；外部 timeout 3650，不截断成功。
pipeline 顺序：pilot → 决策门 → 1200 步训练 → evaluate step-1200 与 step-600 中间点
→ evaluate_rounds（T6、T1 各一次）→ jacobi_check → 汇总。pilot 不重试，满训与评测最多两轮 attempt。

~~~bash
nohup bash scripts/run_exp16.sh > runs/exp16-pipeline.log 2>&1 < /dev/null &
# pipeline 中依次执行（python 均为 .venv/bin/python）：
# EXP08_RUN_DIR 分别指定 pilot / full run 目录
 timeout 3650 .venv/bin/python -m src.train --config configs/exp16_t6_pilot.json
 .venv/bin/python scripts/summarize_exp16.py --pilot
 timeout 3650 .venv/bin/python -m src.train --config configs/exp16_t6.json
 .venv/bin/python -m src.evaluate --checkpoint runs/EXP-16-T6/checkpoints/step-1200 --mode ponder --split validation
# step-600 经临时目录 checkpoints/step-600 符号链接以相同 mode/split 评测，再移动 evaluation.json
 .venv/bin/python -m src.evaluate --checkpoint "$stage/checkpoints/step-600" --mode ponder --split validation
 .venv/bin/python -m src.evaluate_rounds --checkpoint runs/EXP-16-T6/checkpoints/step-1200 --output runs/EXP-16-T6/rounds-T6.json
 .venv/bin/python -m src.evaluate_rounds --checkpoint runs/EXP-01-T1/checkpoints/step-600 --output runs/EXP-16-T6/rounds-T1.json
 .venv/bin/python -m src.jacobi_check --checkpoint runs/EXP-16-T6/checkpoints/step-1200 --output-dir runs/EXP-16-T6
 .venv/bin/python scripts/summarize_exp16.py
~~~

## 2. 做了什么

硬件与软件 metadata：
"""
    text += json.dumps(meta, ensure_ascii=False, indent=2) + "\n"
    text += "\nPilot 决策门：全部有限、后20步均值≤前20步1.5倍、梯度中位数相对 T1/T5 前100步在0.1–10倍内、无 OOM 且完成100步。\n"
    text += json.dumps(gate, ensure_ascii=False, indent=2) + "\n\n"
    pt = rows(P / "train.jsonl")
    text += table(["Pilot block", "mean CE", "round1", "round2", "round3 (K=3)", "final CE", "max grad"],
        [[f"{s+1}-{s+20}", mean(v["loss"] for v in pt[s:s+20]),
          *[mean(v["round_ce"][j] for v in pt[s:s+20] if len(v["round_ce"]) > j) for j in range(3)],
          mean(v["final_prediction_ce"] for v in pt[s:s+20]),
          max(x for v in pt[s:s+20] for x in v["round_grad_norm"])] for s in range(0, 100, 20)])
    if not full:
        text += "\n\n## 3. 结果\n\nPilot " + gate["decision"] + "；尚无完整 T6×1200 主验证，不能填写 NLL/PPL。\n"
    else:
        t = rows(D / "train.jsonl")
        rng = random.Random(42)
        expected = [rng.choice([2, 3]) for _ in range(1200)]
        counts = {str(k): expected.count(k) for k in [2, 3]}
        updates = sum(expected)
        assert counts == {"2": 587, "3": 613} and updates == 3013
        assert [v["step"] for v in t] == list(range(1, 1201))
        assert all(len(v["round_ce"]) == v["K"] == len(v["round_grad_norm"]) for v in t)
        assert [v["K"] for v in t] == expected
        assert sum(v["optimizer_updates"] for v in t) == updates
        assert all(math.isfinite(x) for v in t for x in [v["loss"], v["final_prediction_ce"], *v["round_ce"], *v["round_grad_norm"]])
        order = read(D / "data-order.json")
        assert order["seed"] == 42 and order["chunks"] == 9411 and len(order["indices"]) == 19200
        assert order["indices"][:9600] == read("runs/exp01-order.json")["indices"]
        assert order["indices"] == read("runs/EXP-12-T1/data-order.json")["indices"]
        source = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted((ROOT / "src").glob("*.py"))}
        assert meta["source_sha256"] == source
        e = read(D / "evaluation.json")
        mid = read(D / "evaluation-step600.json")
        r = read(D / "rounds-T6.json")
        br = read(D / "rounds-T1.json")
        for ev in [e, mid]:
            assert math.isfinite(ev["nll"]) and math.isfinite(ev["ppl"])
            assert abs(math.exp(ev["nll"]) - ev["ppl"]) < 1e-8
            assert ev["predicted_tokens"] == 249135 and ev["seq"] == 256
            assert ev["mode"] == "ponder" and ev["split"] == "validation"
        difference = abs(e["nll"] - r["final_prediction_nll"])
        assert difference < 1e-6
        assert read(D / "complete.json")["steps"] == 1200
        models = [("T0×1200", Path("runs/EXP-12-T0")), ("T1×1200", Path("runs/EXP-12-T1")),
                  ("T6×600", Path("runs/EXP-07-T6")), ("T6×1200", D)]
        metrics = []
        for label, d in models:
            ev = read(d / "evaluation.json")
            tr = rows(d / "train.jsonl")
            metrics.append([label, ev["nll"], ev["ppl"], read(d / "complete.json")["wall_s"],
                            max(v["peak_memory_bytes"] for v in tr) / 2**20])
        gaps = {"T6x1200-" + key: e["nll"] - metrics[j][1]
                for j, key in enumerate(["T0x1200", "T1x1200", "T6x600"])}
        text += "\n\n## 3. 结果\n\n" + table(["Model", "主验证 NLL", "PPL", "训练 wall s", "峰值 allocated MiB"], metrics)
        text += "\n\n差距（实测 ΔNLL）：\n\n" + table(["比较", "ΔNLL"], list(gaps.items()))
        historical = read("runs/EXP-07-T6/rounds-T6.json")
        text += "\n\n" + table(["k", "T1 recomputed", "T6×600 historical", "T6×1200"],
            [[j+1, br["round_nll"][j], historical["round_nll"][j], r["round_nll"][j]] for j in range(3)])
        text += "\n\nstep-600 中间点：\n\n" + table(["Model", "中间点 NLL"],
            [[label, read(d / "evaluation-step600.json")["nll"]] for label, d in [models[0], models[1], models[3]]])
        text += "\n\n中间点 LR 未衰减完（约 5.6e-5），不可与 EXP-07 的 600 步终点（1e-5）直接比较，只用于看轨迹。\n"
        text += "\n训练曲线（每100步边界前20步均值）：\n\n" + table(["step", "mean CE", "CE1", "CE2", "CE3 (K=3)", "final CE"],
            [[end, mean(v["loss"] for v in t[end-20:end]),
              *[mean(v["round_ce"][j] for v in t[end-20:end] if len(v["round_ce"]) > j) for j in range(3)],
              mean(v["final_prediction_ce"] for v in t[end-20:end])] for end in range(100, 1201, 100)])
        same = expected[:600] == [v["K"] for v in rows("runs/EXP-07-T6/train.jsonl")]
        text += f"\n\nK 序列与 EXP-07 前600项逐项相同：{same}；{counts['2']} 个 2、{counts['3']} 个 3、{updates} 次更新。\n"
        text += "\n最终 θ 下 Jacobi 检查（8×128；r_k 为相邻状态 RMS，d_k 为距顺序固定点 RMS）：\n\n"
        text += table(["k", "r_k", "d_k"], [[v["k"], v["r_k"], v["d_k"]] for v in read(D / "jacobi_summary.json")])
        band, next_step = interpretation(e["nll"])
        text += f"\n\n实测 T6×1200 NLL={e['nll']}，对照 4.8–4.9 / ≈5.1 判定区间：{band}。{next_step}\n"
        verification = dict(steps=len(t), K_counts=counts, optimizer_updates=updates, finite=True, final_nll_difference=difference)
        summary = dict(results={"T6x1200": dict(nll=e["nll"], ppl=e["ppl"], wall_s=metrics[3][3],
            peak_memory_mib=metrics[3][4], seed=42, optimizer_updates=updates, K_counts=counts)},
            comparison={key: {"nll": metrics[j][1]} for j, key in enumerate(["T0x1200", "T1x1200", "T6x600"])}, gaps=gaps)
        text += """
## 4. 问题与局限性

单种子、WikiText-2 唯一语料、70M tiny-scale、无下游 benchmark。
1200 步进入第 2 epoch（数据重复），不是新数据；中间点不可与 EXP-07 终点直接比较。
per-round 更新截断跨轮梯度，与 T5 完整图不同。LR 按 batch 步索引保证相同数据位置 lr，
但实际 3013 次更新而非 1200 次，累计优化量/weight decay 不同。
各轮训练 CE 在不同 θ 下测，均值只是描述指标，不能当成同一图的目标。
pilot 的 100 步 cosine 与完整 run 不同，梯度门阈值是稳定性启发式。
最终 θ 的固定点诊断不代表训练中途变化 θ 时存在同一个固定点。
FLOPs 为解析估计；wall 含实际监测和 checkpoint，排除加载，历史基线未同步配对。
未做下游泛化、统计显著性或规模放大结论。

## 5. 下一步建议

"""
        text += next_step + "无论哪档，多种子配对与下游 benchmark 仍是未做项。\n"
        (D / "verification.json").write_text(json.dumps(verification, indent=2))
        Path("runs/exp16-summary.json").write_text(json.dumps(summary, indent=2))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/EXP-16-T6-1200-steps.md").write_text(text)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args()
    g = pilot()
    report(g, full=not a.pilot)
    print(json.dumps(g), flush=True)
    if not g["passed"]:
        raise SystemExit("Pilot failed: stopping before full training")
