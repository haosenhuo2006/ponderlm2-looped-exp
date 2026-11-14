"""Aggregate immutable seed-42 evaluations and EXP-08 replications."""
import json
import math
import random
import hashlib
from pathlib import Path
from statistics import mean, stdev
BASE={"T0":"EXP-01-T0","T1":"EXP-01-T1","T2":"EXP-03-T2","T4":"EXP-05-T4"}
def read(p): return json.loads(Path(p).read_text())
def table(h,rs):
    return "\n".join(["| "+" | ".join(h)+" |","|"+"|".join(["---"]*len(h))+"|"]+["| "+" | ".join(map(str,r))+" |" for r in rs])
def main():
    results={}; times=[]; provenance=[]; failures=[]; seed_orders={}
    for model,base in BASE.items():
        results[model]={}
        for seed in (42,43,44):
            d=Path("runs")/(base if seed==42 else f"EXP-08-seed{seed}-{model}")
            if not (d/"evaluation.json").exists():
                failures.append(f"{model}/seed{seed}: missing evaluation")
                continue
            e=read(d/"evaluation.json")
            assert e["predicted_tokens"]==249135 and e["seq"]==256
            assert math.isfinite(e["nll"]) and abs(math.exp(e["nll"])-e["ppl"])<1e-8
            t=[json.loads(x) for x in (d/"train.jsonl").read_text().splitlines()]
            assert [r["step"] for r in t]==list(range(1,601))
            assert all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"]) for r in t)
            rng=random.Random(seed)
            expected=[rng.choice([2,3]) if model in ("T1","T2") else 0 for _ in range(600)]
            assert [r["K"] for r in t]==expected
            results[model][seed]=e
            times.append([model,seed,f'{read(d/"complete.json")["wall_s"]:.2f}',f'{max(r["peak_memory_bytes"] for r in t)/2**20:.2f}',str({k:expected.count(k) for k in sorted(set(expected))})])
            if seed!=42:
                meta=read(d/"metadata.json")
                assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in meta["source_sha256"].items())
                data=read(d/"data-order.json")
                assert data["seed"]==seed and len(data["indices"])==9600
                if seed in seed_orders: assert data["indices"]==seed_orders[seed]
                seed_orders[seed]=data["indices"]
                import numpy as np
                rng_order=np.random.default_rng(seed)
                expected_order=[]
                while len(expected_order)<9600: expected_order.extend(rng_order.permutation(data["chunks"]).tolist())
                assert data["indices"]==expected_order[:9600]
                provenance.append(dict(model=model,seed=seed,metadata=meta,order_sha256=hashlib.sha256((d/"data-order.json").read_bytes()).hexdigest(),data=read(d/"data-train.json")))
    protected=read("runs/exp08-protected.json")
    for name, original in protected.items():
        p=Path(name)
        assert p.stat().st_size==original["size"] and p.stat().st_mtime_ns==original["mtime_ns"], name
        if original["sha256"]: assert hashlib.sha256(p.read_bytes()).hexdigest()==original["sha256"], name
    rows=[]
    for model,ev in results.items():
        row=[model]+[f'{ev[s]["nll"]:.6f} / {ev[s]["ppl"]:.6f}' if s in ev else "FAILED / missing" for s in (42,43,44)]
        for key in ("nll","ppl"):
            vals=[ev[s][key] for s in (42,43,44) if s in ev]
            row += [f"{mean(vals):.6f} ± {stdev(vals):.6f}" if len(vals)==3 else "NA",f"{min(vals):.6f}–{max(vals):.6f}" if len(vals)==3 else "NA"]
        rows.append(row)
    diffs=[]; conclusions=[]
    for a,b in (("T1","T0"),("T2","T1"),("T4","T1")):
        vals=[results[a][s]["nll"]-results[b][s]["nll"] if s in results[a] and s in results[b] else None for s in (42,43,44)]
        diffs.append([a+"−"+b]+[f"{v:+.6f}" if v is not None else "NA" for v in vals]+[f"{mean(vals):+.6f}" if None not in vals else "NA"])
        signs=["+" if v>0 else "−" if v<0 else "0" for v in vals if v is not None]
        conclusions.append(f'{a}−{b}: {"/".join(signs)} (seed 42/43/44). '+("**存在符号翻转。**" if len(set(signs))>1 else "三个种子方向一致。" if len(signs)==3 else "**未完成，不能判断稳定性。**"))
    text="""# EXP-08 多种子复核 T0 / T1 / T2 / T4

## 1. 方法

沿用仓库最小实现及 seed-42 的训练和主验证定义，只改配置 seed/run_dir。
seed 43、44 各自派生 torch 随机状态、数据排列及 Python Random 的 K 序列；
同一种子各模型共用相同数据顺序，T1/T2 共用 K∈{2,3} 序列。
T0/T4 不采样 K。不是将 seed-42 的数据次序或 K 序列强行固定后仅修改初始化。
为避免旧共享 exp01-order.json 的 seed=42 断言，EXP08_RUN_DIR 将排序与
数据元信息路由至新 run 目录；未设置时历史入口行为保持原样。
验证脚本核对全部配置除 seed/run_dir 外完全一致，并独立重复生成 K 和数据排列。
数据、vendor、历史 checkpoint 及历史结果不修改。

## 2. 做了什么

起点 main 2bfeabd90408e132b8942ba1933d43f61d0ef10a，先 git pull --ff-only 并确认干净。
配置 configs/exp08_seed{43,44}_{t0,t1,t2,t4}.json。
模型 revision、scale_embeds、seq=256、microbatch=1、grad_accum=16、600步、
lr=1e-4、warmup=20、betas=(0.9,0.95)、weight decay=.01、clip=1.0 均沿用基线。
每次 2,457,600 输入 tokens，9411 训练 chunks；固定验证 977 chunks、249135 targets。
WikiText-2 raw-v1 Salesforce 镜像固定 revision，沿用原 tokenizer、拼接与缓存数据。
逐项数据哈希、版本、GPU、启动 git hash 和源码哈希见 runs/exp08-provenance.json。
GPU 为 RTX 4060 Laptop 8GB；实际版本以 metadata 为准。
启动 git hash 为起点提交加未提交 EXP-08 修改，源码哈希精确记录实际代码。

命令（均通过助手 VM 的 tssh wsl 在仓库目录执行）：

~~~bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 .venv/bin/python -m scripts.check_exp08
nohup bash scripts/run_exp08.sh > runs/exp08-pipeline.log 2>&1 < /dev/null &
~~~

顺序 seed43 T0→T1→T2→T4，再 seed44 T0→T1→T2→T4。
每项 nohup timeout 3650 .venv/bin/python -m src.train --config 对应配置，
独立 runs/exp08-<seed>-<model>-pipeline.log；内部仍为3600秒预算。
失败保留原日志和产物，原配置从头重试一次；连续失败记录，不填补结果。
T0: src.evaluate --mode vanilla --split validation；
T1: src.evaluate --mode ponder --split validation；
T2: src.evaluate --mode ponder --mask D1 --split validation；
T4: src.evaluate_looped。均传 --checkpoint <run>/checkpoints/step-600。
T1/T2 在现有入口中硬编码固定 K=3，未增加不同评测口径。
seed42 直接读取 EXP-01/03/05 evaluation.json，未重跑。

## 3. 结果

主表每格 NLL / PPL；均值与样本标准差 n=3，范围为 min–max。

"""
    if all(s in results["T1"] and s in results["T0"] and s in results["T2"] for s in (42,43,44)):
        d10=[results["T1"][s]["nll"]-results["T0"][s]["nll"] for s in (42,43,44)]
        d21=[results["T2"][s]["nll"]-results["T1"][s]["nll"] for s in (42,43,44)]
        conclusions.append("T1 输给 T0："+("三个种子下均成立，方向稳定。" if all(v>0 for v in d10) else "**并非三个种子下均成立，不能声称稳定。**"))
        conclusions.append("T2 赢过 T1："+("三个种子下均成立，方向稳定。" if all(v<0 for v in d21) else "**并非三个种子下均成立，不能声称稳定。**"))
    text+=table(["模型","seed42 NLL/PPL","seed43 NLL/PPL","seed44 NLL/PPL","NLL 均值±SD","NLL范围","PPL 均值±SD","PPL范围"],rows)
    text+="\n\n差值为 NLL，正值表示左侧模型更差。\n\n"+table(["差值","seed42","seed43","seed44","均值"],diffs)
    text+="\n\n"+"\n\n".join(conclusions)
    text+="\n\n训练 wall time（内部计时，排除加载，包含 checkpoint）与峰值 allocated 显存：\n\n"+table(["模型","seed","wall秒","峰值MiB","K计数"],times)
    text+="\n\n完成状态："+str(8-sum("seed42" not in f for f in failures))+"/8 个新训练及评测有完整结果。\n"
    text+="\n失败记录："+("; ".join(failures) if failures else "无最终失败；逐项尝试状态见各 run 的 attempts.jsonl；完整训练日志保留。")+"\n"
    text+="""
## 4. 问题与局限性

仍是 70M tiny-scale，WikiText-2 为唯一语料，3个种子只能看出波动量级，
不能做严格显著性检验，不能推广到论文规模或判定机制的普遍优劣。
T3/T5/T6 未复核。FLOPs 沿用解析估计，未做 profiler；
相同输入 tokens 和步数不等于相同训练 FLOPs。
种子控制数据顺序和 K 采样；GPU 数值运算不承诺逐位确定性。
训练耗时为各 run 实际测量，不是并行配对硬件计时。
汇总校验全部600步、有限 loss/梯度、实际 K 序列和种子数据排列；
历史实验文件大小/mtime及小文件SHA256与启动保护快照一致。

## 5. 下一步建议

先据逐种子符号决定结论措辞；如有翻转，增加预注册种子与学习曲线分析。
另行安排下游评测、更大语料与预算；T5/T6 的复核须单独立项。
"""
    Path("reports/EXP-08-multi-seed.md").write_text(text)
    Path("runs/exp08-provenance.json").write_text(json.dumps(provenance,indent=2))
    Path("runs/exp08-summary.json").write_text(json.dumps(dict(results=results,differences=diffs,failures=failures),indent=2))
    plan=Path("EXPERIMENT_PLAN.md")
    original=plan.read_text().split("\n### EXP-08 多种子复核")[0]
    plan.write_text(original+"\n### EXP-08 多种子复核\n\n只改 seed/run_dir，复核 T0/T1/T2/T4 的 seed43/44，各600步，与既有seed42配对。\n"+" ".join(conclusions)+"\n详见 reports/EXP-08-multi-seed.md。\n")
    print(table(["模型","seed42","seed43","seed44","NLL均值±SD","范围","PPL均值±SD","范围"],rows))
    print("\n".join(conclusions))
if __name__=="__main__": main()
