"""Real CPU-tensor rollout checks and automatic alpha-equivalence regression tests."""
import ast
import contextlib
import csv
import io
import json
import math
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
import torch

from test_notebook_regressions import (
    ANALYSES, EXPERIMENTS, definitions, experiment_definitions, notebook_tree,
    ride, scripted_env,
)
from experiment_support import (
    RolloutState, alpha_metadata, build_sweep_plan, with_effective_alpha,
    summarize_alpha_groups, best_equivalent_alpha_rate,
)

torch.set_num_threads(1)


def training_definitions(filename):
    ns = experiment_definitions(filename)
    ns.update(torch=torch, nn=torch.nn, optim=torch.optim, DEVICE="cpu", csv=csv, os=os,
              Path=Path, json=json)
    definitions(filename, {"ActorCritic", "collect_rollout", "ppo_update", "make_minibatches",
                           "_make_eval_env_from", "evaluate_all", "run_episode",
                           "baseline_nearby_rule", "baseline_nearby_rule_voronoi",
                           "train_policy_brief", "append_training_log", "run_param_sweep"}, ns)
    _, tree = notebook_tree(filename)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TRAIN_LOG_COLUMNS" for t in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), ns)
    return ns


class ConstantPolicy(torch.nn.Module):
    def __init__(self, actions=1, value=3.):
        super().__init__()
        self.actions, self.value = actions, value

    def forward(self, obs):
        logits = torch.full((len(obs), self.actions), -1e9)
        logits[:, 0] = 0
        return logits, torch.full((len(obs), 1), self.value)


class ClockEnv:
    """A known seven-step episode isolates bookkeeping and boundary returns."""
    def __init__(self, length=7, dt=.25):
        self.length, self.dt, self.reset_count = length, dt, 0

    def observation(self):
        return np.array([self.step_count], np.float32), np.ones(1, np.float32)

    def reset(self):
        self.reset_count += 1
        self.step_count, self.t = 0, 0.
        return self.observation()

    def step(self, action):
        self.step_count += 1
        self.t += self.dt
        obs, mask = self.observation()
        return obs, 1., self.step_count >= self.length, {}, mask


class RolloutContinuationTests(unittest.TestCase):
    def test_split_episode_statistics_and_bootstrap_boundaries(self):
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename):
                ns = training_definitions(filename)
                collect = ns["collect_rollout"]
                env, state, policy = ClockEnv(), RolloutState(), ConstantPolicy()
                first, completed, step_reward, step_rate = collect(env, policy, 5, state)
                self.assertEqual(completed, [])
                self.assertEqual(env.reset_count, 1)
                self.assertEqual((state.episode_return, state.episode_steps), (5., 5))
                self.assertEqual((step_reward, step_rate), (1., 4.))
                self.assertAlmostEqual(first[3][-1].item(), 1 + ns["DISCOUNT"] * 3, places=5)

                second, completed, _, _ = collect(env, policy, 5, state)
                self.assertEqual(second[0][:, 0].tolist(), [5., 6., 0., 1., 2.])
                self.assertEqual(env.reset_count, 2)
                self.assertEqual(completed, [dict(reward=7., steps=7, terminal_time_min=1.75, rate=4.)])
                self.assertEqual((state.episode_return, state.episode_steps), (3., 3))
                self.assertAlmostEqual(second[3][1].item(), 1., places=5)  # Real terminal: no bootstrap.
                self.assertAlmostEqual(second[3][-1].item(), 1 + ns["DISCOUNT"] * 3, places=5)

                third, completed, _, _ = collect(env, policy, 4, state)
                self.assertEqual(len(completed), 1)
                self.assertEqual(completed[0]["steps"], 7)
                self.assertEqual((state.episode_return, state.episode_steps), (0., 0))
                self.assertEqual(env.reset_count, 3)
                self.assertAlmostEqual(third[3][-1].item(), 1., places=5)
                single, _, _, _ = collect(env, policy, 1, state)
                self.assertTrue(torch.isfinite(single[4]).all())
                with self.assertRaises(ValueError):
                    collect(ClockEnv(), policy, 2, state)
                with self.assertRaises(ValueError):
                    collect(env, policy, 0, state)

    def test_real_environments_continue_past_4096_to_the_existing_horizon(self):
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename):
                ns = training_definitions(filename)
                env = scripted_env(ns, [(4, 0)], v=1e-6, dt=.5)
                state = RolloutState()
                policy = ConstantPolicy(env.act_dim)
                ns["collect_rollout"](env, policy, 8192, state)
                self.assertEqual(env.t, 4096.)
                batch, completed, _, _ = ns["collect_rollout"](env, policy, 3330, state)
                self.assertEqual(batch[0][0, 2].item(), np.float32(4096 / 5760))
                self.assertEqual(len(completed), 1)
                self.assertEqual(completed[0]["terminal_time_min"], 5760.)
                self.assertEqual(completed[0]["steps"], 11520)
                self.assertEqual(state.episode_steps, 2)
                self.assertEqual(env.t, 1.)

    def test_actual_ppo_updates_and_evaluation_preserve_training_progress(self):
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename):
                ns = training_definitions(filename)
                ns.update(PPO_STEPS=8, PPO_EPOCHS=1, PPO_MINI_BATCH=8,
                          HORIZON_MIN=10., STEPS_PER_EP=20, EVAL_EPISODES=1)
                # Solver implementations have separate real-OR tests. Keep this PPO check short.
                ns["baseline_pure_ortools"] = lambda env, seed=None: ns["run_episode"](env, seed=seed)
                ns["baseline_four_zone"] = lambda env, seed=None: ns["run_episode"](env, seed=seed)
                env = ns["CoModalEnv"](seed=1, v=1e-6, lam=0., gamma_pack=0.)
                real_collect, real_eval, real_update = ns["collect_rollout"], ns["evaluate_all"], ns["ppo_update"]
                starts, changes, evaluated_at = [], [], []

                def collect(env, policy, steps, state):
                    starts.append(state.episode_steps)
                    return real_collect(env, policy, steps, state)

                def evaluate(env, policy, **kwargs):
                    before = (env.t, env.pos.copy(), env.pkg_delivered.copy(), torch.random.get_rng_state())
                    result = real_eval(env, policy, **kwargs)
                    self.assertEqual(env.t, before[0])
                    np.testing.assert_array_equal(env.pos, before[1])
                    np.testing.assert_array_equal(env.pkg_delivered, before[2])
                    self.assertTrue(torch.equal(torch.random.get_rng_state(), before[3]))
                    evaluated_at.append(env.t)
                    return result

                def update(policy, *args):
                    before = [p.detach().clone() for p in policy.parameters()]
                    loss = real_update(policy, *args)
                    changes.append(any(not torch.equal(old, p) for old, p in zip(before, policy.parameters())))
                    self.assertTrue(all(math.isfinite(float(x)) for x in loss))
                    return loss

                ns.update(collect_rollout=collect, evaluate_all=evaluate, ppo_update=update)
                with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                    log = str(Path(directory) / "train.csv")
                    policy, metrics = ns["train_policy_brief"](env, updates=3, eval_every=2, log_path=log)
                    rows = pd.read_csv(log)
                self.assertEqual(starts, [0, 8, 16])
                self.assertEqual(evaluated_at, [8., 2.])
                self.assertTrue(all(changes))
                train = rows[rows.kind == "train"]
                self.assertEqual(train.episodes.tolist(), [0, 0, 1])
                self.assertEqual(train.steps.iloc[-1], 20.)
                self.assertEqual(train.terminal_time_min.iloc[-1], 10.)
                self.assertTrue(math.isfinite(metrics["drl"]["avg_rate"]))


def plan(**changes):
    options = dict(lambda_values=[10], alpha_values=[.1, .2, .25, .3, .35, .4],
                   gamma_values=[.5], ttl_values=[5], max_visible_values=[5],
                   grace_values=[6], rt_values=[6], R=5.5, v=.19, dt=.5, train_seeds=[42, 43])
    options.update(changes)
    return build_sweep_plan(**options)


def frame_for_rates():
    return pd.DataFrame([dict(algo="DRL", LAMBDA=10, GAMMA_PACK=.5, RIDE_TTL_MIN=5,
                              RT=6., R=5.5, V=.19, DT=.5, MAX_VISIBLE_RIDES=5,
                              SWITCH_GRACE_STEPS=6, R_PICK_ALPHA=a, rate=r)
                         for a, r in ((.1, 6.), (.25, 2.), (.4, 8.))])


class AlphaEquivalenceTests(unittest.TestCase):
    def test_dynamic_thresholds_discretization_and_paired_seed_plan(self):
        default = plan()
        self.assertEqual(len(default), 6)  # Three effective alphas, two seeds each.
        last = [r for r in default if r["R_PICK_ALPHA"] == .25]
        self.assertEqual([r["train_seed"] for r in last], [42, 43])
        self.assertEqual(json.loads(last[0]["ALPHA_MEMBERS"]), [.25, .3, .35, .4])
        self.assertAlmostEqual(last[0]["ALPHA_SATURATION"], .95 * math.sqrt(2) / 5.5)
        self.assertEqual(default, plan(alpha_values=[.4, .2, .1, .35, .3, .25]))
        self.assertEqual(len(plan(ttl_values=[10])), 12)
        self.assertEqual(len(plan(v=.38)), 12)
        self.assertEqual(len(plan(R=2.75)), 12)
        self.assertEqual(alpha_metadata(.4, R=5.5, v=.19, dt=3, ttl_minutes=5)["RIDE_TTL_STEPS"], 2)
        self.assertEqual(len(plan(dt=3)), 8)  # alpha=.25 now below the cap (~.293).
        with self.assertRaises(ValueError):
            plan(train_seeds=[42, 42])
        with self.assertRaises(ValueError):
            plan(dt=0)

    def test_real_visibility_matches_groups_and_changes_with_ttl(self):
        for filename in EXPERIMENTS:
            ns = experiment_definitions(filename)
            for ttl, expected in ((5, [False, False]), (10, [False, True])):
                results = []
                for alpha in (.25, .4):
                    request = ride(ns, (1.2, 0), (0, 1))
                    env = scripted_env(ns, [(0, 0), (2, 0)], initial=[(request, int(ttl / .5))],
                                       R=5.5, v=.19, dt=.5, r_pick_alpha=alpha, ride_ttl_minutes=ttl)
                    results.append(bool(env._visible_rides()))
                self.assertEqual(results, expected)
            envs = [ns["CoModalEnv"](seed=91, lam=5, gamma_pack=.5, r_pick_alpha=a) for a in (.25, .4)]
            for step in range(120):
                action = 1 if step % 2 else 0
                first, second = [e.step(action) for e in envs]
                np.testing.assert_array_equal(first[0], second[0])
                np.testing.assert_array_equal(first[4], second[4])
                self.assertEqual(first[1:4], second[1:4])

    def test_group_means_variance_and_all_best_alpha_helpers(self):
        frame = frame_for_rates()
        original = frame.copy(deep=True)
        summary = summarize_alpha_groups(frame)
        saturated = summary[summary.ALPHA_EFFECTIVE > .2].iloc[0]
        self.assertEqual(saturated.rate_mean, 5.)
        self.assertEqual(saturated.n_runs, 2)
        self.assertAlmostEqual(saturated.rate_std, math.sqrt(18))
        self.assertEqual(best_equivalent_alpha_rate(frame), 6.)
        pd.testing.assert_frame_equal(frame, original)
        for filename in ANALYSES:
            ns = definitions(filename, {"best_alpha_series", "best_alpha_by_lambda_gamma", "best_alpha_rate"})
            self.assertEqual(ns["best_alpha_series"](frame, .5, "DRL")[1].tolist(), [6.])
            self.assertEqual(ns["best_alpha_by_lambda_gamma"](frame, 10, .5, "DRL"), 6.)
            self.assertEqual(ns["best_alpha_rate"](frame, 10, .5, "DRL"), 6.)
            ratio_only = definitions(filename, {"best_alpha_rate"})
            self.assertEqual(ratio_only["best_alpha_rate"](frame, 10, .5, "DRL"), 6.)

    def test_different_scenarios_and_legacy_metadata_are_kept_distinct(self):
        base = frame_for_rates().iloc[1:].copy()
        variants = [base]
        for col, value in (("LAMBDA", 20), ("GAMMA_PACK", .83), ("RT", 5.5),
                           ("RIDE_TTL_MIN", 10), ("V", .38), ("DT", 3), ("R", 2.75)):
            changed = base.copy()
            changed[col] = value
            variants.append(changed)
        summary = summarize_alpha_groups(pd.concat(variants, ignore_index=True))
        self.assertEqual(summary.n_runs.sum(), 16)
        self.assertEqual(len(summary), 12)  # Four unsaturated scenarios each retain two alphas.
        legacy = base.drop(columns=["R", "V", "DT"])
        with self.assertRaises(ValueError):
            with_effective_alpha(legacy)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            restored = with_effective_alpha(legacy, legacy_geometry={"R": 5.5, "V": .19, "DT": .5})
        self.assertEqual(len(seen), 3)
        self.assertEqual(restored.ALPHA_EFFECTIVE.nunique(), 1)
        # Recorded metadata takes precedence over an outdated historical fallback.
        changed = base.assign(RIDE_TTL_MIN=10)
        result = with_effective_alpha(changed, legacy_geometry={"R": 100, "V": 100, "DT": 100})
        self.assertEqual(result.ALPHA_EFFECTIVE.nunique(), 2)

    def test_sweep_executes_representatives_and_writes_seed_and_summary_metadata(self):
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename):
                ns = training_definitions(filename)
                captured, seeded = [], []
                ns["set_global_seeds"] = seeded.append
                ns["TRAIN_LOG_PATH"] = "unused-test-path"

                def train(env, **kwargs):
                    meta = kwargs["combo_meta"]
                    captured.append(meta)
                    self.assertEqual(env.ride_ttl_steps, meta["RIDE_TTL_STEPS"])
                    self.assertAlmostEqual(env.r_pick, meta["R_PICK_ALPHA"] * env.R / math.sqrt(2))
                    metric = dict(avg_reward=2., avg_t=1., avg_rate=float(meta["train_seed"]),
                                  avg_ep_rate=2., avg_finish_time=1., finish_rate=1.)
                    return None, {a: metric.copy() for a in ("drl", "heur", "heur_vor", "four_zone", "pure", "pure_or")}

                ns["train_policy_brief"] = train
                with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                    path = Path(directory) / "runs.csv"
                    df = ns["run_param_sweep"]([10], [.1, .25, .4], [.5], [5, 10], [5], [6], [6],
                                              train_seeds=[7, 8], csv_path=path)
                    saved = pd.read_csv(path)
                    summary = pd.read_csv(path.with_name("runs_alpha_summary.csv"))
                self.assertEqual(len(captured), 10)
                self.assertEqual(seeded, [7, 8] * 5)
                self.assertEqual(len(saved), 60)
                self.assertEqual(len(summary), 30)
                self.assertTrue((summary.n_runs == 2).all())
                self.assertTrue((summary.rate_mean == 7.5).all())
                self.assertTrue((summary.rate_std > 0).all())
                self.assertEqual(df.train_seed.unique().tolist(), [7, 8])
                self.assertTrue(all("DEMAND_PROFILE" in m for m in captured))

    def test_training_log_records_new_metadata_and_rejects_old_schema(self):
        for filename in EXPERIMENTS:
            ns = training_definitions(filename)
            with tempfile.TemporaryDirectory() as directory:
                log = Path(directory) / "train.csv"
                meta = dict(plan()[0], TRAIN_UPDATES=3, DEMAND_PROFILE="stationary")
                ns["append_training_log"](log, 1, "train", meta=meta)
                saved = pd.read_csv(log)
                self.assertEqual(saved.train_seed.iloc[0], 42)
                self.assertEqual(saved.TRAIN_UPDATES.iloc[0], 3)
                old = Path(directory) / "old.csv"
                old.write_text("update,kind,rate\n1,train,2\n", encoding="utf-8")
                before = old.read_bytes()
                with self.assertRaises(ValueError):
                    ns["append_training_log"](old, 2, "train", meta=meta)
                self.assertEqual(old.read_bytes(), before)

    def test_experiment_plots_receive_equivalent_group_means(self):
        frame = frame_for_rates()
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename):
                plots = []
                def plot(xs, ys, **kwargs):
                    plots.append((list(xs), list(ys)))
                plotting = types.SimpleNamespace(plot=plot, **{
                    name: (lambda *args, **kwargs: None)
                    for name in ("figure", "xlabel", "ylabel", "title", "legend", "grid", "tight_layout", "show")})
                ns = experiment_definitions(filename)
                ns["plt"] = plotting
                definitions(filename, {"plot_rate_vs_lambda_optimal_alpha_by_gamma", "plot_single_param_trend"}, ns)
                with patch.object(pd, "read_csv", return_value=frame.copy()):
                    ns["plot_rate_vs_lambda_optimal_alpha_by_gamma"]("unused", gamma_values=[.5], algos=["DRL"])
                self.assertEqual(plots, [([10], [6.])])
                plots.clear()
                ns["plot_single_param_trend"](frame, "R_PICK_ALPHA", algos=["DRL"])
                self.assertEqual(plots[0][1], [6., 5.])
                self.assertAlmostEqual(plots[0][0][1], .95 * math.sqrt(2) / 5.5)


if __name__ == "__main__":
    unittest.main()
