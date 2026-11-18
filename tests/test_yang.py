"""CPU-only checks for the Yang synthetic regression tasks and models."""

import math
import unittest

import torch

from src import train_yang, yang_model, yang_tasks


DEVICE = "cpu"
LINEAR = "linear_regression"


class YangTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(min(threads, 4))

    def test_sampler_shapes_and_metrics(self):
        for task, config in train_yang.TASK_CONFIGS.items():
            with self.subTest(task=task):
                n_points = config["k"] + 1
                xs, ys = train_yang.sample_batch(task, config, 4, DEVICE)
                self.assertEqual(xs.shape, (4, n_points, 20))
                self.assertEqual(ys.shape, (4, n_points))
                self.assertTrue(torch.equal(ys[:, -1], ys[:, config["k"]]))
                self.assertEqual(xs.device.type, DEVICE)
                self.assertEqual(ys.device.type, DEVICE)
                sampler = yang_tasks.get_task_sampler(task, 4, n_points, 20, 20, DEVICE)
                batch = sampler()
                errors = batch.get_metric()(torch.zeros_like(batch.ys), batch.ys)
                self.assertTrue(torch.equal(errors, batch.ys.square()))
                self.assertEqual(batch.get_training_metric()(torch.zeros_like(batch.ys), batch.ys),
                                 errors.mean())

    def test_linear_dimension_truncation(self):
        task = yang_tasks.LinearRegression(4, 41, 20, 7, DEVICE)
        self.assertEqual(torch.count_nonzero(task.xs[..., 7:]).item(), 0)
        self.assertEqual(torch.count_nonzero(task.w_b[:, 7:]).item(), 0)
        self.assertTrue(torch.equal(task.ys, (task.xs @ task.w_b).sum(-1)))

    def test_sparse_linear_has_three_nonzero_weights_per_prompt(self):
        task = yang_tasks.SparseLinearRegression(4, 41, 20, 20, DEVICE)
        self.assertEqual(torch.count_nonzero(task.w_b.squeeze(-1), dim=1).tolist(), [3] * 4)
        self.assertTrue(torch.equal(task.ys, (task.xs @ task.w_b).sum(-1)))

    def test_relu_target_scaling(self):
        task = yang_tasks.Relu2nnRegression(4, 101, 20, 20, DEVICE)
        expected = (torch.relu(task.xs @ task.W1) @ task.W2)[:, :, 0] * math.sqrt(2 / 100)
        self.assertTrue(torch.equal(task.ys, expected))

    def test_decision_tree_leaves_and_seed(self):
        def sample(seed):
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            return yang_tasks.DecisionTree(4, 101, 20, 20, DEVICE, generator=generator)

        first, same, different = sample(42), sample(42), sample(43)
        for ys, targets in zip(first.ys, first.target_tensor):
            leaves = targets[2 ** first.depth - 1:]
            self.assertTrue(torch.isin(ys, leaves).all().item())
        self.assertTrue(torch.equal(first.xs, same.xs))
        self.assertTrue(torch.equal(first.ys, same.ys))
        self.assertFalse(torch.equal(first.xs, different.xs))
        self.assertFalse(torch.equal(first.ys, different.ys))

    def test_parameter_counts(self):
        for model_type, count in (("looped", 847873), ("standard", 9535233)):
            with self.subTest(model_type=model_type):
                model = yang_model.build_yang_model(model_type).to(DEVICE)
                self.assertEqual(sum(p.numel() for p in model.parameters()), count)

    def test_truncation_horizon(self):
        for b, window, start in ((20, 15, 5), (12, 5, 7), (5, 10, 0), (5, 5, 0)):
            with self.subTest(b=b, window=window):
                self.assertEqual(train_yang.truncation_horizon(b, window), start)

    def test_prediction_window_and_truncated_loss(self):
        model = yang_model.TransformerModelLooped(n_embd=32, n_head=4).to(DEVICE)
        task = yang_tasks.LinearRegression(2, 5, 20, 20, DEVICE)
        xs, ys = task.xs, task.ys
        for b, window in ((4, 2), (3, 3), (3, 5)):
            with self.subTest(b=b, window=window):
                start = train_yang.truncation_horizon(b, window)
                grad_modes = []

                def record_grad_mode(module, inputs, output):
                    grad_modes.append(torch.is_grad_enabled())

                with model._backbone.register_forward_hook(record_grad_mode):
                    predictions = model(xs, ys, start, b)
                self.assertEqual(len(predictions), min(b, window))
                self.assertEqual(grad_modes, [False] * start + [True] * (b - start))
                self.assertTrue(all(pred.shape == ys.shape for pred in predictions))

        with torch.no_grad():
            predictions = model(xs, ys, 2, 4)
            expected_loss = torch.stack([(pred - ys).square().mean() for pred in predictions]).mean()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
        loss, last_pred = train_yang.train_step(
            model, xs, ys, optimizer, model_type="looped", n_loops=4, n_loop_window=2,
        )
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(last_pred, predictions[-1])
        self.assertFalse(loss.requires_grad)
        self.assertFalse(last_pred.requires_grad)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_evaluate_normalization(self):
        denominators = {
            LINEAR: 20, "sparse_linear_regression": 3, "decision_tree": 1, "relu_2nn_regression": 20,
        }
        self.assertEqual(train_yang.NORMALIZATION, denominators)
        model = yang_model.TransformerModel(n_layer=1, n_embd=32, n_head=4).to(DEVICE)
        for task, denominator in denominators.items():
            with self.subTest(task=task):
                metrics = train_yang.evaluate(model, task, n_eval=64, batch_size=48, device=DEVICE)
                point_errors = metrics["per_prompt_point_sq_error"]
                query_errors = metrics["per_prompt_query_sq_error"]
                self.assertEqual(point_errors.shape, (64, train_yang.TASK_CONFIGS[task]["k"] + 1))
                self.assertEqual(query_errors.shape, (64,))
                self.assertTrue(torch.equal(query_errors, point_errors[:, -1]))
                self.assertEqual(metrics["raw_mse_query"], query_errors.mean().item())
                self.assertEqual(metrics["normalized_mse"], metrics["raw_mse_query"] / denominator)

    def test_model_and_batch_seed(self):
        config = train_yang.TASK_CONFIGS[LINEAR]

        def build_and_sample():
            torch.random.default_generator.manual_seed(42)
            model = yang_model.build_yang_model("looped").to(DEVICE)
            return model.state_dict(), train_yang.sample_batch(LINEAR, config, 64, DEVICE)

        first_state, first_batch = build_and_sample()
        second_state, second_batch = build_and_sample()
        for name in first_state:
            with self.subTest(parameter=name):
                self.assertTrue(torch.equal(first_state[name], second_state[name]))
        for first, second in zip(first_batch, second_batch):
            self.assertTrue(torch.equal(first, second))

    def test_three_training_steps_replay_exactly(self):
        config = train_yang.TASK_CONFIGS[LINEAR]

        def losses(seed):
            torch.random.default_generator.manual_seed(seed)
            # Three training steps with the full model take a few seconds on CPU.
            model = yang_model.build_yang_model("looped").to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
            values = []
            for _ in range(3):
                xs, ys = train_yang.sample_batch(LINEAR, config, 64, DEVICE)
                loss, _ = train_yang.train_step(
                    model, xs, ys, optimizer, model_type="looped",
                    n_loops=config["n_loops"], n_loop_window=config["n_loop_window"],
                )
                values.append(loss.item())
            return values

        expected = losses(42)
        self.assertTrue(all(math.isfinite(loss) for loss in expected))
        self.assertEqual(losses(42), expected)

    def test_single_loop_matches_manual_injection(self):
        model = yang_model.build_yang_model("looped").to(DEVICE)
        task = yang_tasks.LinearRegression(2, 5, 20, 20, DEVICE)
        xs, ys = task.xs, task.ys
        with torch.no_grad():
            embeds = model._read_in(model._combine(xs, ys))
            output = model._backbone(inputs_embeds=torch.zeros_like(embeds) + embeds)
            expected = model._read_out(output)[:, 0::2, 0]
            actual = model(xs, ys, 0, 1)[-1]
            changed_ys = ys.clone()
            changed_ys[:, -1] += 100
            changed = model(xs, changed_ys, 0, 1)[-1]
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual[:, -1], changed[:, -1]))
        with self.assertRaises(ValueError):
            yang_model.build_yang_model("invalid")

    def test_evaluation_seed_is_independent_of_global_rng(self):
        task = "sparse_linear_regression"
        model = yang_model.build_yang_model("looped", n_embd=32, n_head=4).to(DEVICE)
        first = train_yang.evaluate(model, task, n_eval=64, batch_size=32, eval_seed=1234, device=DEVICE)
        torch.rand(17, device=DEVICE)
        rng_state = torch.random.get_rng_state()
        second = train_yang.evaluate(model, task, n_eval=64, batch_size=32, eval_seed=1234, device=DEVICE)
        self.assertTrue(torch.equal(first["per_prompt_query_sq_error"], second["per_prompt_query_sq_error"]))
        self.assertEqual(first["bootstrap_low"], second["bootstrap_low"])
        self.assertEqual(first["bootstrap_high"], second["bootstrap_high"])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state))
        self.assertTrue(model.training)

        generator = torch.Generator(device=DEVICE).manual_seed(1234)
        for _ in range(2):
            train_yang.sample_batch(task, train_yang.TASK_CONFIGS[task], 32, DEVICE, generator=generator)
        query_errors = first["per_prompt_query_sq_error"]
        indices = torch.randint(64, (1000, 64), generator=generator, device=DEVICE)
        means = query_errors[indices].mean(dim=1).sort().values
        self.assertEqual(first["bootstrap_low"], means[50].item() / 3)
        self.assertEqual(first["bootstrap_high"], means[950].item() / 3)


if __name__ == "__main__":
    unittest.main()
