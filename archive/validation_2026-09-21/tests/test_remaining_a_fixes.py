"""Behavioral regression checks for review items A5, A6, A8 and A9."""
import ast
import contextlib
import csv
import io
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_notebook_regressions import (
    ANALYSES, EXPERIMENTS, definitions, experiment_definitions, notebook_tree,
    ride, scripted_env,
)


class ActivityTimeTests(unittest.TestCase):
    def test_pickup_and_passenger_steps_include_the_final_dropoff(self):
        for filename in EXPERIMENTS:
            ns = experiment_definitions(filename)
            for pickup, dropoff, expected_steps in ((.05, .1, 2), (.3, .5, 7)):
                with self.subTest(notebook=filename, trip=(pickup, dropoff)):
                    request = ride(ns, (pickup, 0), (dropoff, 0))
                    env = scripted_env(ns, [(4, 0)], initial=[(request, 100)],
                                       v=.19, dt=.5, r_pick_alpha=1.)
                    reward = env.step(1)[1]
                    steps = 1
                    while env.to_pickup is not None or env.with_passenger:
                        reward += env.step(0)[1]
                        steps += 1
                        self.assertLess(steps, 20)
                        self.assertAlmostEqual(env.time_rides_min, steps * env.dt)
                        self.assertEqual(env.time_delivery_min, 0)
                    self.assertEqual(steps, expected_steps)
                    self.assertAlmostEqual(reward, env.rt * (dropoff - pickup), places=6)
                    self.assertAlmostEqual(env.time_rides_min, env.t)
                    self.assertIsNone(env.drop_target)
                    env.step(0)
                    self.assertEqual(env.time_delivery_min, env.dt)
                    self.assertAlmostEqual(env.time_rides_min + env.time_delivery_min, env.t)

    def test_reservation_and_rejected_actions_still_count_delivery_time(self):
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename):
                ns = experiment_definitions(filename)
                request = ride(ns, (.1, 0), (0, .3))
                other = ride(ns, (.1, 0), (0, .4))
                env = scripted_env(ns, [(.05, 0), (.1, 0), (0, 1)],
                                   initial=[(request, 10), (other, 10)], v=.19, dt=.5)
                self.assertTrue(env._accept_ride(request, defer_pickup=True))
                env.step(1)  # Another request is visible, but a reservation blocks accepting it.
                env.step(0)
                self.assertEqual(env.time_delivery_min, 1.)
                self.assertEqual(env.time_rides_min, 0)
                self.assertIs(env.pending_ride, request)
                self.assertTrue(env.pkg_delivered[:2].all())
                self.assertTrue(env._start_reserved_ride())
                env.step(0)
                self.assertEqual(env.time_rides_min, env.dt)
                self.assertEqual(env.time_delivery_min, 1.)
                self.assertAlmostEqual(env.time_rides_min + env.time_delivery_min, env.t)

                free_env = scripted_env(ns, [(4, 0)], v=.19, dt=.5)
                free_env.step(99)  # Invalid action falls back to package delivery.
                self.assertEqual(free_env.time_delivery_min, free_env.dt)
                self.assertEqual(free_env.time_rides_min, 0)


class ScatterSizeTests(unittest.TestCase):
    def test_only_target_densities_are_selected_with_float_tolerance(self):
        for filename in ANALYSES:
            with self.subTest(notebook=filename):
                select = definitions(filename, {"select_scatter_package_sizes"})["select_scatter_package_sizes"]
                frame = pd.DataFrame({"GAMMA_PACK": [.33, .50, .67, .83, 1., .50 + 1e-10, .50 + 1e-4, np.nan],
                                      "row_id": range(8)}, index=[8, 7, 7, 5, 4, 3, 2, 1])
                original = frame.copy(deep=True)
                selected = select(frame)
                self.assertEqual(selected["row_id"].tolist(), [1, 3, 5])
                self.assertEqual(selected["n_label"].tolist(), [30, 50, 30])
                pd.testing.assert_frame_equal(frame, original)
                self.assertTrue(select(frame.iloc[0:0]).empty)
                self.assertTrue(select(frame.iloc[[0, 2, 4]]).empty)

    def test_actual_scatter_cell_plots_only_matching_rows(self):
        class Axes:
            def __init__(self): self.xs, self.title = [], ""
            def scatter(self, xs, ys, **kwargs): self.xs.extend(xs.tolist())
            def set_title(self, title): self.title = title
            def set_xlabel(self, label): pass
            def set_ylabel(self, label): pass
            def grid(self, *args, **kwargs): pass
            def get_legend_handles_labels(self): return [], []

        frame = pd.DataFrame([dict(GAMMA_PACK=gamma, LAMBDA=lam, algo=algo,
                                   terminal_time=100 * gamma + lam + offset, rate=1.)
                              for gamma in (.33, .50, .67, .83, 1.)
                              for lam in (10, 40)
                              for offset, algo in enumerate(("PURE", "HEUR_VOR", "DRL"))])
        for filename in ANALYSES:
            with self.subTest(notebook=filename):
                nb, _ = notebook_tree(filename)
                source = next("".join(c["source"]) for c in nb["cells"]
                              if c["cell_type"] == "code" and "def select_scatter_package_sizes(" in "".join(c["source"]))
                tree = ast.parse(source)
                tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
                axes = np.array([[Axes(), Axes()], [Axes(), Axes()]], dtype=object)
                figure = types.SimpleNamespace(legend=lambda *args, **kwargs: None)
                plotting = types.SimpleNamespace(subplots=lambda *args, **kwargs: (figure, axes),
                                                 tight_layout=lambda: None, show=lambda: None)
                ns = dict(np=np, pd=pd, plt=plotting, results_path=lambda name: name,
                          result_file=lambda name: name)
                with patch.object(pd, "read_csv", return_value=frame.copy(deep=True)):
                    exec(compile(tree, filename, "exec"), ns)
                for i, gamma in enumerate((.50, .83)):
                    for j, lam in enumerate((10, 40)):
                        expected = [100 * gamma + lam + offset for offset in range(3)]
                        np.testing.assert_allclose(axes[i, j].xs, expected)
                        self.assertIn("Expected packages ≈", axes[i, j].title)


class FinalEvaluationTests(unittest.TestCase):
    def test_final_model_metrics_and_actual_log_agree_without_duplicate_evaluation(self):
        for filename in EXPERIMENTS:
            ns = experiment_definitions(filename)
            _, tree = notebook_tree(filename)
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TRAIN_LOG_COLUMNS" for t in node.targets):
                    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), ns)
            ns.update(csv=csv, os=os)
            definitions(filename, {"append_training_log"}, ns)

            class Policy:
                def __init__(self, *args): self.updates = 0
                def to(self, *args): return self
                def train(self): pass
                def parameters(self): return []

            def update(policy, *args):
                policy.updates += 1
                return 0., 0., 0., 0.

            evaluated = []
            def evaluate(env, policy, **kwargs):
                evaluated.append(policy.updates)
                metric = dict(avg_reward=policy.updates, avg_t=1., avg_rate=policy.updates,
                              avg_ep_rate=policy.updates, finish_rate=1., avg_finish_time=1.)
                return {name: metric.copy() for name in ("drl", "heur", "heur_vor", "four_zone", "pure", "pure_or")}

            ns.update(ActorCritic=Policy, DEVICE="cpu", PPO_EPOCHS=1,
                      optim=types.SimpleNamespace(Adam=lambda *args, **kwargs: object()),
                      collect_rollout=lambda *args: (None, [], 0., 0.),
                      make_minibatches=lambda *args: [None], ppo_update=update, evaluate_all=evaluate)
            definitions(filename, {"train_policy_brief"}, ns)
            for updates, interval, expected in ((7, 5, [5, 7]), (10, 5, [5, 10]), (3, 10, [3]),
                                                (3, 0, [3]), (0, 5, [0]), (1, 1, [1])):
                with self.subTest(notebook=filename, updates=updates, interval=interval):
                    evaluated.clear()
                    with tempfile.TemporaryDirectory(prefix="logistics-a8-") as directory:
                        log_path = str(Path(directory) / "training.csv")
                        with contextlib.redirect_stdout(io.StringIO()):
                            policy, metrics = ns["train_policy_brief"](
                                ns["CoModalEnv"](seed=1, lam=0), updates=updates, eval_every=interval,
                                log_path=log_path, combo_meta={"combo_id": 19})
                        self.assertEqual(evaluated, expected)
                        self.assertEqual(policy.updates, updates)
                        self.assertEqual(metrics["drl"]["avg_reward"], updates)
                        with open(log_path, newline="") as stream:
                            rows = list(csv.DictReader(stream))
                        evaluation_rows = [r for r in rows if r["kind"] == "eval"]
                        self.assertEqual([int(r["update"]) for r in evaluation_rows], expected)
                        self.assertEqual([float(r["reward"]) for r in evaluation_rows], expected)
                        self.assertTrue(all(int(r["combo_id"]) == 19 for r in rows))
                        self.assertEqual(len([r for r in rows if r["kind"] == "train"]), updates)


class EvaluationProfileTests(unittest.TestCase):
    def test_hourly_profile_is_copied_and_drives_the_same_hourly_arrival_rates(self):
        filename = "NonStationary/experiment2.ipynb"
        ns = experiment_definitions(filename)
        definitions(filename, {"_make_eval_env_from"}, ns)
        for profile in (None, np.linspace(.5, 1.5, 24)):
            with self.subTest(custom_profile=profile is not None):
                options = {} if profile is None else dict(hourly_multiplier=profile)
                training = ns["CoModalEnv"](seed=3, lam=12., **options)
                evaluation = ns["_make_eval_env_from"](training)
                np.testing.assert_array_equal(evaluation.hourly_multiplier, training.hourly_multiplier)
                self.assertFalse(np.shares_memory(evaluation.hourly_multiplier, training.hourly_multiplier))

                class ArrivalRecorder:
                    def __init__(self): self.rates = []
                    def poisson(self, rate):
                        self.rates.append(rate)
                        return 0

                for env in (training, evaluation):
                    env.rng = ArrivalRecorder()
                    for hour in range(24):
                        env.t = hour * 60.
                        env._sample_rides_this_step()
                np.testing.assert_allclose(evaluation.rng.rates, training.rng.rates)
                np.testing.assert_allclose(evaluation.rng.rates, 12 * training.hourly_multiplier * training.dt)
                old = training.hourly_multiplier.copy()
                evaluation.hourly_multiplier[3] += 7.
                np.testing.assert_array_equal(training.hourly_multiplier, old)


if __name__ == "__main__":
    unittest.main()
