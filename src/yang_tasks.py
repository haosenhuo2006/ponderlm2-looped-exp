"""Synthetic regression prompts from Yang et al. (official commit 6b3bdb2).

Each prompt has k + 1 points, with the query in the last position. The task
metrics return raw squared errors.
"""

import math
import random

import torch
from torch.nn import functional as F


def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return squared_error(ys_pred, ys).mean()


class Task:
    def __init__(self, batch_size, n_points, n_dims, n_dims_truncated):
        self.b_size = batch_size
        self.n_points = n_points
        self.n_dims = n_dims
        self.n_dims_truncated = n_dims_truncated

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearRegression(Task):
    def __init__(self, batch_size, n_points, n_dims, n_dims_truncated, device,
                 sparsity=None, *, generator=None):
        super().__init__(batch_size, n_points, n_dims, n_dims_truncated)
        self.xs = torch.randn(batch_size, n_points, n_dims, device=device, generator=generator)
        self.xs[..., n_dims_truncated:] = 0
        self.w_b = torch.randn(batch_size, n_dims, 1, device=device, generator=generator)
        self.w_b[:, n_dims_truncated:] = 0
        self.ys = (self.xs @ self.w_b).sum(-1)


class SparseLinearRegression(LinearRegression):
    def __init__(self, batch_size, n_points, n_dims, n_dims_truncated, device,
                 sparsity=3, *, generator=None):
        super().__init__(batch_size, n_points, n_dims, n_dims_truncated, device,
                         generator=generator)
        self.sparsity = sparsity
        for w in self.w_b:
            # The official mask draws from all d coordinates on CPU.
            mask = torch.ones(n_dims, dtype=torch.bool, device="cpu")
            perm = torch.randperm(n_dims, device="cpu", generator=generator)
            mask[perm[:sparsity]] = False
            w[mask.to(w.device)] = 0
        self.ys = (self.xs @ self.w_b).sum(-1)


class Relu2nnRegression(Task):
    def __init__(self, batch_size, n_points, n_dims, n_dims_truncated, device,
                 sparsity=100, hidden_layer_size=100, *, generator=None):
        super().__init__(batch_size, n_points, n_dims, n_dims_truncated)
        self.hidden_layer_size = hidden_layer_size
        self.W1 = torch.randn(batch_size, n_dims, hidden_layer_size,
                              device=device, generator=generator)
        self.W2 = torch.randn(batch_size, hidden_layer_size, 1,
                              device=device, generator=generator)
        if sparsity < hidden_layer_size:
            mask = torch.zeros(hidden_layer_size, device=device)
            mask[random.sample(range(hidden_layer_size), sparsity)] = 1
            self.W1 = self.W1 * mask[None, None, :]
            self.W2 = self.W2 * mask[None, :, None]

        self.xs = torch.randn(batch_size, n_points, n_dims, device=device, generator=generator)
        self.xs[..., n_dims_truncated:] = 0
        self.ys = self.evaluate(self.xs)

    def evaluate(self, xs):
        return (F.relu(xs @ self.W1) @ self.W2)[:, :, 0] * math.sqrt(2 / self.hidden_layer_size)


class DecisionTree(Task):
    def __init__(self, batch_size, n_points, n_dims, n_dims_truncated, device,
                 sparsity=None, depth=4, *, generator=None):
        super().__init__(batch_size, n_points, n_dims, n_dims_truncated)
        self.depth = depth
        shape = (batch_size, 2 ** (depth + 1) - 1)
        self.dt_tensor = torch.randint(0, n_dims, shape, device=device, generator=generator)
        self.target_tensor = torch.randn(shape, device=device, generator=generator)
        self.xs = torch.randn(batch_size, n_points, n_dims, device=device, generator=generator)
        self.ys = self.evaluate(self.xs)

    def evaluate(self, xs):
        dt_tensor = self.dt_tensor.to(xs.device)
        target_tensor = self.target_tensor.to(xs.device)
        ys = target_tensor.new_zeros(xs.shape[:2])
        point_indices = torch.arange(xs.shape[1], device=xs.device)
        for i, xs_prompt in enumerate(xs):
            # A single tree can be reused across prompts.
            tree_index = 0 if self.b_size == 1 else i
            dt = dt_tensor[tree_index]
            xs_bool = xs_prompt > 0
            cur_nodes = torch.zeros_like(point_indices)
            for _ in range(self.depth):
                cur_coords = dt[cur_nodes]
                decisions = xs_bool[point_indices, cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + decisions
            ys[i] = target_tensor[tree_index, cur_nodes]
        return ys


def get_task_sampler(task_name, batch_size, n_points, n_dims, n_dims_truncated,
                     device, sparsity=None):
    """Bind the batch shape and return a callable that draws a fresh task."""
    task_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
    }
    if task_name not in task_classes:
        raise NotImplementedError(f"Unknown task {task_name!r}")
    task_class = task_classes[task_name]
    defaults = {} if sparsity is None else {"sparsity": sparsity}
    return lambda **args: task_class(batch_size, n_points, n_dims, n_dims_truncated,
                                     device, **{**defaults, **args})
