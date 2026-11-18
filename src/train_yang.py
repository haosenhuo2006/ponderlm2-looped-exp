"""Yang 合成任务复现，依据 Yang arXiv:2311.12424v3 与官方提交 6b3bdb2。

冻结的任务参数

| task | d | k | b (n_loops) | T (n_loop_window) | steps |
|---|---|---|---|---|---|
| linear_regression | 20 | 40 | 20 | 15 | 50000 |
| sparse_linear_regression | 20 (s=3) | 40 | 20 | 10 | 50000 |
| decision_tree | 20 (depth 4) | 100 | 70 | 15 | 20000 |
| relu_2nn_regression | 20 (100 hidden) | 100 | 12 | 5 | 20000 |

n_points = k + 1，最后一个点为查询点。种子 42/43，batch 64，Adam lr=1e-4，
weight_decay=0.0，无梯度裁剪。

(1) 固定 b/T、不用课程表——论文 §5.1 主结论支持影响不大，但附录 E 只比了 b 排程且决策树用排程略好，三者（d/k/b）同时取消的效果未核实；
(2) 步数是官方 500k 的缩减预算（线性/稀疏 50k、决策树/两层 ReLU 20k），属“缩减预算下的机制检验”，不赢不能排除欠训。
稀疏线性 k=40 跟 YAML/notebook，论文 §5.4 正文写 k=100（分歧已记录）。

从仓库根目录运行 python -m src.train_yang。
"""

import argparse
import json
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from src.yang_model import TransformerModelLooped, build_yang_model
from src.yang_tasks import get_task_sampler, squared_error


TASK_CONFIGS = {
    "linear_regression": dict(d=20, k=40, n_loops=20, n_loop_window=15, steps=50000),
    "sparse_linear_regression": dict(
        d=20, k=40, n_loops=20, n_loop_window=10, steps=50000, sparsity=3,
    ),
    "decision_tree": dict(d=20, k=100, n_loops=70, n_loop_window=15, steps=20000, depth=4),
    "relu_2nn_regression": dict(
        d=20, k=100, n_loops=12, n_loop_window=5, steps=20000, hidden_layer_size=100, sparsity=100,
    ),
}
NORMALIZATION = {
    "linear_regression": 20,
    "sparse_linear_regression": 3,
    "decision_tree": 1,
    "relu_2nn_regression": 20,
}
BATCH_SIZE = 64
LEARNING_RATE = 1e-4


def truncation_horizon(n_loops, T):
    """Return the index of the first loop in the gradient window."""
    return max(0, n_loops - T)


def train_step(model, xs, ys, optimizer, *, model_type, n_loops, n_loop_window):
    if model_type == "looped":
        horizon_start = truncation_horizon(n_loops, n_loop_window)
        pred_list = model(xs, ys, horizon_start, n_loops)
        loss = (torch.cat([ys] * len(pred_list)) - torch.cat(pred_list)).square().mean()
        last_pred = pred_list[-1]
    elif model_type == "standard":
        last_pred = model(xs, ys)
        loss = (ys - last_pred).square().mean()
    else:
        raise ValueError(f"Unknown model type {model_type!r}")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.detach(), last_pred.detach()


def sample_batch(task_name, config, batch_size, device, *, generator=None):
    """Draw fresh prompts at the frozen dimension and context length."""
    sampler = get_task_sampler(
        task_name, batch_size, config["k"] + 1, config["d"], config["d"], device,
        sparsity=config.get("sparsity"),
    )
    task_kwargs = {key: config[key] for key in ("depth", "hidden_layer_size") if key in config}
    batch = sampler(generator=generator, **task_kwargs)
    return batch.xs.float(), batch.ys.float()


def evaluate(model, task_name, *, n_eval=1280, batch_size=128, eval_seed=1234, device):
    """Score the final prediction at every point and aggregate the query errors.

    Bootstrap uses the sorted means and indices from the official notebook's
    aggregate_metrics, with an independent CPU Generator in place of global
    RNG. The same Generator draws the evaluation prompts on CPU before they
    are moved to the model's device.
    """
    if n_eval <= 0 or batch_size <= 0:
        raise ValueError("n_eval and batch_size must be positive")

    config = TASK_CONFIGS[task_name]
    generator = torch.Generator(device="cpu").manual_seed(eval_seed)
    is_looped = isinstance(model, TransformerModelLooped)
    point_errors = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, n_eval, batch_size):
                xs, ys = sample_batch(task_name, config, min(batch_size, n_eval - start),
                                      "cpu", generator=generator)
                xs, ys = xs.to(device), ys.to(device)
                if is_looped:
                    prediction = model(xs, ys, 0, config["n_loops"])[-1]
                else:
                    prediction = model(xs, ys)
                point_errors.append(squared_error(prediction, ys).cpu())
    finally:
        model.train(was_training)

    point_sq_error = torch.cat(point_errors)
    query_sq_error = point_sq_error[:, -1]
    raw_mse_query = query_sq_error.mean().item()
    denominator = NORMALIZATION[task_name]
    bootstrap_trials = 1000
    indices = torch.randint(n_eval, (bootstrap_trials, n_eval), generator=generator, device="cpu")
    bootstrap_means = query_sq_error[indices].mean(dim=1).sort().values

    return {
        "per_prompt_query_sq_error": query_sq_error,
        "per_prompt_point_sq_error": point_sq_error,
        "raw_mse_query": raw_mse_query,
        "normalized_mse": raw_mse_query / denominator,
        "bootstrap_low": bootstrap_means[int(0.05 * bootstrap_trials)].item() / denominator,
        "bootstrap_high": bootstrap_means[int(0.95 * bootstrap_trials)].item() / denominator,
    }


def _resolve_device(device):
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _init_training(model_type, seed, device, lr):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = build_yang_model(model_type).to(device=device, dtype=torch.float32).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    return model, optimizer


def _run_steps(task, model_type, model, optimizer, steps, batch_size, device):
    config = TASK_CONFIGS[task]
    for step in range(1, steps + 1):
        start = perf_counter()
        xs, ys = sample_batch(task, config, batch_size, device)
        loss, _ = train_step(
            model, xs, ys, optimizer, model_type=model_type,
            n_loops=config["n_loops"], n_loop_window=config["n_loop_window"],
        )
        # Reading the loss on the host also waits for the device's training step.
        loss_value = loss.item()
        yield step, loss_value, perf_counter() - start


def train_one(task, model_type, seed, *, steps=None, out_dir=None, device=None,
              save_every=10000, log_every=100, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    """Train one arm and save checkpoints indexed by completed optimizer steps."""
    config = TASK_CONFIGS[task]
    steps = config["steps"] if steps is None else steps
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")

    device = _resolve_device(device)
    model, optimizer = _init_training(model_type, seed, device, lr)
    out_dir = Path(out_dir) if out_dir is not None else Path("runs/yang") / f"{task}_{model_type}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = dict(
        config, task=task, model_type=model_type, seed=seed, steps=steps,
        b=config["n_loops"], T=config["n_loop_window"], batch_size=batch_size, lr=lr,
    )
    elapsed = 0.0
    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log:
        for step, loss, sec_per_step in _run_steps(task, model_type, model, optimizer,
                                                  steps, batch_size, device):
            elapsed += sec_per_step
            if log_every > 0 and step % log_every == 0:
                row = json.dumps(dict(step=step, loss=loss, sec_per_step=sec_per_step))
                print(row, flush=True)
                log.write(row + "\n")
                log.flush()
            if step == steps or (save_every > 0 and step % save_every == 0):
                torch.save({"model": model.state_dict(), "config": run_config, "step": step},
                           out_dir / f"model_{step}.pt")

    return {"final_loss": loss, "mean_sec_per_step": elapsed / steps, "out_dir": str(out_dir)}


def pilot_steps(task, model_type, *, n_steps=200, seed=42, device=None,
                batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    """Time training steps and extrapolate to the frozen budget for this task."""
    if n_steps <= 0 or batch_size <= 0:
        raise ValueError("n_steps and batch_size must be positive")

    full_steps = TASK_CONFIGS[task]["steps"]
    device = _resolve_device(device)
    model, optimizer = _init_training(model_type, seed, device, lr)
    mean_sec = sum(sec for _, _, sec in _run_steps(
        task, model_type, model, optimizer, n_steps, batch_size, device,
    )) / n_steps
    result = dict(mean_sec_per_step=mean_sec, n_steps=n_steps,
                  extrapolated_hours_full_run=full_steps * mean_sec / 3600)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run the Yang synthetic regression experiments.")
    parser.add_argument("--task", choices=TASK_CONFIGS, required=True)
    parser.add_argument("--model", choices=("looped", "standard"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training and pilot batch size.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--out-dir", type=Path, help="Defaults to runs/yang/<task>_<model>_s<seed>.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pilot-steps", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    if args.pilot_steps > 0:
        pilot_steps(args.task, args.model, n_steps=args.pilot_steps, seed=args.seed,
                    device=args.device, batch_size=args.batch_size, lr=args.lr)
    elif args.eval_only:
        if args.checkpoint is None:
            parser.error("--eval-only requires --checkpoint")
        device = _resolve_device(args.device)
        model = build_yang_model(args.model).to(device=device, dtype=torch.float32)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        metrics = evaluate(model, args.task, device=device)
        print(json.dumps({key: metrics[key] for key in
                          ("normalized_mse", "bootstrap_low", "bootstrap_high")}))
    else:
        train_one(args.task, args.model, args.seed, steps=args.steps, out_dir=args.out_dir,
                  device=args.device, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
