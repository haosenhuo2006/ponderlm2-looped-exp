"""Verify pilot/full T6 and write evidence-backed report."""
import hashlib
import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean, median
D=Path("runs/EXP-07-T6")
P=Path("runs/EXP-07-T6-pilot")
def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(s) for s in Path(p).read_text().splitlines()]
def table(headers, data):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|"]+
        ["| "+" | ".join(str(v) for v in row)+" |" for row in data])
def pilot():
    t=rows(P/"train.jsonl")
    bases=[rows("runs/EXP-01-T1/train.jsonl"),rows("runs/EXP-06-T5/train.jsonl")]
    vals=[x for v in t for x in [v["loss"],v["final_prediction_ce"],*v["round_ce"],*v["round_grad_norm"]]]
    baseline=median(v["grad_norm"] for b in bases for v in b[:100])
    measured=median(x for v in t for x in v["round_grad_norm"])
    # Explicit conservative gate: same order of magnitude, finite, no late loss explosion.
    finite=all(math.isfinite(v) for v in vals)
    stable=mean(v["loss"] for v in t[-20:]) <= 1.5*mean(v["loss"] for v in t[:20])
    passed=len(t)==100 and (P/"complete.json").exists() and finite and stable and .1<=measured/baseline<=10
    result=dict(passed=passed,decision="continue" if passed else "stop",finite=finite,
        no_divergence=stable,no_oom=(P/"complete.json").exists(),gradient_median=measured,
        baseline_gradient_median=baseline,gradient_ratio=measured/baseline,
        max_gradient=max(x for v in t for x in v["round_grad_norm"]),
        last20_mean_ce=mean(v["loss"] for v in t[-20:]),
        last20_final_ce=mean(v["final_prediction_ce"] for v in t[-20:]),
        baseline_last20=[mean(v["loss"] for v in b[80:100]) for b in bases],
        wall_s=read(P/"complete.json")["wall_s"])
    (P/"decision.json").write_text(json.dumps(result,indent=2))
    return result
def report(gate, full=False):
    meta=read((D if full else P)/"metadata.json")
    text="""# EXP-07 — T6: Per-round optimizer update

## 1. Method

一个 optimizer step（本报告称数据 batch 步）对应一个数据 batch = 16 个 microbatch。
600 步 × 16 × 256 = 2,457,600 输入 tokens。每步以独立 random.Random(42)
采样 K∈{2,3}；600 步应为 306 个 2、294 个 3，总优化器更新 1494 次。

对第 k=1..K 轮 refinement：用当前参数重算 init + 1..k 链，
对全部 16 个 microbatch 计算 thought-logit
F.cross_entropy(z[:, :-1].float(), targets[:, 1:])（实现展平 batch/token 维）。
token-only 初始化无 thought 位置，不监督。
CE_k 除以 grad_accum 后 backward，累加全部 microbatch，clip 1.0，
立刻 optimizer.step() 并 zero_grad，再进入下一轮。
轮与轮之间把隐状态 H^(k) detach；CE_{k+1} 的梯度只经过第 k+1 轮
refinement 前向，不经过前面轮次的图。这是有意的截断。
每轮前缀在当前 θ 下以 no_grad 重算，H 显式 detach；当前轮 token embeddings
重新构建以允许该轮训练输入 embedding。没有保留旧参数的计算图。

K 轮更新后用最终 θ 无梯度重跑完整 T1 路径，记录 separate final CE，
不计入损失。T5 的“每轮优化”是 K 轮 CE 取均值、一次反向、一次更新；
T6 是各轮独立反向、各更新一次。T6 与 T5 用相同 per-round CE 定义。
按实验对照设计，T6 vs T5 隔离更新时机；但实现此更新时机所需的跨轮
梯度截断也不同，不能声称纯粹控制了梯度路径。T6 vs T1 同时混杂监督方式
与更新时机，不得把差异单归因于更新时机。

AdamW betas (0.9,0.95)、weight decay .01；沿用 T1 schedule 公式，
LR 按数据 batch 步 1..600 索引，同 batch 的 K 次更新共用 lr：
warmup 20、峰值 1e-4、cosine 到 1e-5。旧配置缺省 flag=false，行为不变。
pilot 唯一额外差异为 steps=100，因此原公式 cosine 终点在 100；
它是稳定性门，不能作为完整 run step-100 的同 schedule 对照。

K=3 时训练各轮链为 2+3+4=9 次 backbone 前向（T5 连最终 pass 为 5）。
为保证最终 θ 下完整 T1 监测，再重算 5 次，总计 14 次；
K=2 总计 2+3+4=9 次。FLOPs 仅解析估计，不是 profiler 测量。

## 2. 做了什么

从干净 main 38a7d3b 开始，先 git pull --ff-only。全部仓库操作通过助手 VM
的 tssh wsl 通道执行。未修改 vendor 或 GPT2_Latent_Reasoning。
实现、配置和报告的最终版本为包含本报告的提交；启动源 SHA256 在 metadata.json。

命令：
~~~bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 .venv/bin/python -m unittest discover -s tests -v
nohup bash scripts/run_exp07.sh > runs/exp07-pipeline.log 2>&1 < /dev/null &
~~~
pipeline 顺序：100 batch pilot → 决策门 → 600 batch 训练 → src.evaluate
--mode ponder → src.evaluate_rounds（T6、T1 各一次）→ src.jacobi_check → 汇总。
训练内部 3600s 超时明确抛错；外部 timeout 3650，不截断成功。
配置 configs/exp07_t6.json，pilot 副本 configs/exp07_t6_pilot.json。
seed 42、seq 256、microbatch 1、grad_accum 16，同 T1/T5 WikiText-2 raw-v1、
相同 tokenizer/固定 revision 与数据排序。数据处理沿用 EXP-06：
训练 9411 chunks；验证 977 chunks、249135 targets，固定 K=3。
数据镜像与拼接限制见 EXP-06。检查点保存在 gitignored runs 下。

首次 pilot 在训练前因共享顺序文件要求9600项而失败，未执行更新。
修复为T6 pilot读取同一600步顺序的前缀，不改动排序文件；全套测试重跑。

新 GPU 测试输入 1×64，覆盖 K=2/3 更新次数、首轮参数变化、round-2 输出扰动
归因、强制注入 requires_grad 的 round-1 状态并检查 CE_2 反向无该状态梯度、
当前 head 梯度非零、600 步 K 序列。完整测试输出 reports/exp07-tests.txt。

硬件与软件：
"""
    text+=json.dumps(meta,ensure_ascii=False,indent=2)+"\n"
    text+="\nPilot 决策门：全部有限、后20步均值≤前20步1.5倍、梯度中位数相对 T1/T5 前100步在0.1–10倍内、无 OOM 且完成100步。\n"
    text+=json.dumps(gate,ensure_ascii=False,indent=2)+"\n"
    pt=rows(P/"train.jsonl")
    text+=table(["Pilot block","mean CE","round1","round2","round3 (K=3)","final CE","max grad"],
        [[f"{s+1}-{s+20}",mean(v["loss"] for v in pt[s:s+20]),
          *[mean(v["round_ce"][j] for v in pt[s:s+20] if len(v["round_ce"])>j) for j in range(3)],
          mean(v["final_prediction_ce"] for v in pt[s:s+20]),
          max(x for v in pt[s:s+20] for x in v["round_grad_norm"])] for s in range(0,100,20)])
    text+="\n\n## 3. 结果\n\n"
    if full:
        t=rows(D/"train.jsonl")
        rng=random.Random(42); expected=[rng.choice([2,3]) for _ in range(600)]
        protected=read("runs/EXP-06-T5/provenance.json")["protected_sha256"]
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in meta["source_sha256"].items())
        assert [v["step"] for v in t]==list(range(1,601))
        assert all(len(v["round_ce"])==v["K"]==len(v["round_grad_norm"]) for v in t)
        assert [v["K"] for v in t]==expected
        assert sum(v["optimizer_updates"] for v in t)==1494
        assert all(math.isfinite(x) for v in t for x in [v["loss"],v["final_prediction_ce"],*v["round_ce"],*v["round_grad_norm"]])
        e=read(D/"evaluation.json"); r=read(D/"rounds-T6.json"); br=read(D/"rounds-T1.json")
        assert abs(e["nll"]-r["final_prediction_nll"])<1e-6
        models=[Path("runs/EXP-01-T1"),Path("runs/EXP-06-T5"),D]
        metrics=[]
        for d in models:
            ev=read(d/"evaluation.json"); tr=rows(d/"train.jsonl")
            metrics.append([d.name,ev["nll"],ev["ppl"],read(d/"complete.json")["wall_s"],max(v["peak_memory_bytes"] for v in tr)/2**20])
        text+=table(["Model","主验证 NLL","PPL","训练 wall s","峰值 allocated MiB"],metrics)
        text+="\n\n"+table(["k","T1 recomputed","T5 historical","T6"],
             [[j+1,br["round_nll"][j],read("runs/EXP-06-T5/rounds-T5.json")["round_nll"][j],r["round_nll"][j]] for j in range(3)])
        text+="\n\n训练曲线（每100步边界前20步均值）：\n\n"+table(["step","mean CE","CE1","CE2","CE3 (K=3)","final CE"],
            [[end,mean(v["loss"] for v in t[end-20:end]),
              *[mean(v["round_ce"][j] for v in t[end-20:end] if len(v["round_ce"])>j) for j in range(3)],
              mean(v["final_prediction_ce"] for v in t[end-20:end])] for end in range(100,601,100)])
        text+="\n\nK 序列与 T1 逐项核对："+str(expected==[v["K"] for v in rows("runs/EXP-01-T1/train.jsonl")])+"；306 个 2、294 个 3，1494 次更新。\n"
        text+="186 个历史保护文件及所有启动训练源哈希均核对未变。完整测试19项通过。\n"
        text+="训练曲线基线（同样为边界前20步均值；T1 final CE、T5 mean CE）：\n\n"
        text+=table(["step","T1 final CE","T5 mean CE"],
            [[end,*[mean(v["loss"] for v in rows(d/"train.jsonl")[end-20:end]) for d in models[:2]]] for end in range(100,601,100)])
        text+="\n最终 θ 下 Jacobi 检查（8×128，r_k 为相邻状态 RMS，d_k 为距顺序固定点 RMS）：\n\n"
        text+=table(["k","r_k","d_k"],[[v["k"],v["r_k"],v["d_k"]] for v in read(D/"jacobi_summary.json")])
        text+="\n\nΔNLL T6−T5="+str(e["nll"]-metrics[1][1])+"；T6−T1="+str(e["nll"]-metrics[0][1])+"。正负结果均以此测量为准。耗时为各次实际 wall，历史基线未同步配对。\n"
        text+="\n本单种子固定输入token预算下，T6主验证优于T1/T5；这不证明纯更新时机的因果优势。T6逐轮NLL不单调，首轮已最低；更多轮次并未持续改善预测。Jacobi r_k到第7轮下降、第8轮反弹，d_k在第4轮后振荡，不能声称严格收敛。训练无OOM、非有限损失/梯度或超时。\n"
        (D/"verification.json").write_text(json.dumps(dict(steps=600,K_counts={"2":306,"3":294},optimizer_updates=1494,finite=True,final_nll_difference=abs(e["nll"]-r["final_prediction_nll"])),indent=2))
    else:
        text+="Pilot "+gate["decision"]+"；尚无完整 T6 主验证，不能填写 NLL/PPL。\n"
    text+="""
## 4. 问题与局限性

单种子、WikiText-2 唯一语料、70M tiny-scale、无下游 benchmark。
FLOPs 为解析估计；wall time 含实际监测和 checkpoint，排除加载。
per-round 更新截断跨轮梯度，与 T5 完整图不同。LR 按 batch 步索引保证
相同数据位置 lr，但实际 1494 次更新而非600次，累计优化量/weight decay 不同。
各轮训练 CE 在不同 θ 下测，均值只是描述指标，不能当成同一图的目标。
pilot 的100步 cosine schedule 与完整run不同；梯度门阈值是稳定性启发式。
最终 θ 的固定点诊断不代表训练中途变化 θ 时存在同一个固定点。
未做下游泛化、统计显著性或规模放大结论。

## 5. 下一步建议

多种子配对 T1/T5/T6，补充固定 schedule 的诊断；区分截断梯度与更新时机效应；
下游 benchmark、验证学习曲线、梯度/固定点诊断；随后扩展语料、模型与训练规模。
"""
    Path("reports/EXP-07-T6-per-round-optimizer-update.md").write_text(text)
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--pilot",action="store_true"); a=p.parse_args()
    g=pilot()
    report(g,full=not a.pilot)
    print(json.dumps(g),flush=True)
    if not g["passed"]: raise SystemExit("Pilot failed: stopping before full training")
