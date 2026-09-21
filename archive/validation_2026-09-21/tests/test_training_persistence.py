"""Interruption/recovery tests with actual CPU PPO and small deterministic evaluations."""
import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from test_b1_b5 import training_definitions
from test_notebook_regressions import EXPERIMENTS, ride
from experiment_support import RolloutState
from training_persistence import (TrainingCheckpoint, TrainingProgress, SweepStore,
                                  atomic_json, atomic_write, training_settings)


def seed_all(seed, ns):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ns["rng"] = np.random.default_rng(seed)


def small_training(filename):
    ns = training_definitions(filename)
    ns.update(PPO_STEPS=4, PPO_MINI_BATCH=4, PPO_EPOCHS=1, HORIZON_MIN=8., EVAL_EPISODES=1)
    ns["set_global_seeds"] = lambda seed: seed_all(seed, ns)
    evaluated = []

    def evaluate(env, policy, **kwargs):
        evaluated.append(env.t)
        score = float(next(policy.parameters()).detach().sum())
        metric = dict(avg_reward=score, avg_t=1., avg_rate=score, avg_ep_rate=score,
                      avg_finish_time=float("nan"), finish_rate=0.)
        return {key: metric.copy() for key in ("drl", "heur", "heur_vor", "four_zone", "pure", "pure_or")}

    ns["evaluate_all"] = evaluate
    return ns, evaluated


def environment(ns):
    return ns["CoModalEnv"](R=1., v=.01, dt=.5, lam=3., gamma_pack=2., seed=31)


def load_file(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def assert_nested(test, a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for key in a:
            assert_nested(test, a[key], b[key])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_nested(test, x, y)
    elif isinstance(a, float) and np.isnan(a):
        test.assertTrue(np.isnan(b))
    else:
        test.assertEqual(a, b)


class CheckpointTests(unittest.TestCase):
    def test_interrupted_optimizer_recovers_exact_training_and_rewinds_only_uncommitted_logs(self):
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename), tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                root = Path(directory)
                ns, _ = small_training(filename)
                seed_all(17, ns)
                reference, _ = ns["train_policy_brief"](environment(ns), 5, eval_every=2,
                    checkpoint_dir=root / "reference", checkpoint_every=2, show_progress=False)
                ns, _ = small_training(filename)
                seed_all(17, ns)
                actual_update, calls = ns["ppo_update"], []

                def fail_mid_update(*args):
                    result = actual_update(*args)
                    calls.append(1)
                    if len(calls) == 4:
                        raise KeyboardInterrupt("simulated stop after optimizer mutation")
                    return result

                ns["ppo_update"] = fail_mid_update
                with self.assertRaises(KeyboardInterrupt):
                    ns["train_policy_brief"](environment(ns), 5, eval_every=2,
                        checkpoint_dir=root / "resume", checkpoint_every=2, show_progress=False)
                self.assertEqual(load_file(root / "resume/latest.pt")["update"], 2)
                self.assertIn(3, pd.read_csv(root / "resume/training_log.csv").query("kind == 'train'")["update"].tolist())
                # New definitions simulate a restarted notebook: no old RideReq class can be unpickled.
                ns, _ = small_training(filename)
                seed_all(998, ns)
                recovered, _ = ns["train_policy_brief"](environment(ns), 5, eval_every=2,
                    checkpoint_dir=root / "resume", checkpoint_every=2, resume=True, show_progress=False)
                assert_nested(self, reference.state_dict(), recovered.state_dict())
                before, after = [load_file(root / p / "final.pt") for p in ("reference", "resume")]
                for key in ("optimizer", "environment", "rollout", "torch_rng", "numpy_rng", "python_rng", "notebook_rng", "metrics"):
                    assert_nested(self, before[key], after[key])
                # Independent runs have different identities; their resumed numeric
                # trajectories and log records must still match exactly.
                pd.testing.assert_frame_equal(pd.read_csv(root / "reference/training_log.csv").drop(columns="run_id"),
                                              pd.read_csv(root / "resume/training_log.csv").drop(columns="run_id"))
                status = json.loads((root / "resume/progress.json").read_text())
                self.assertEqual((status["phase"], status["update"], status["saved_update"]), ("complete", 5, 5))

    def test_final_evaluation_failure_resumes_without_retraining_or_duplicate_evaluation(self):
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename), tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                ns, evaluated = small_training(filename)
                evaluate = ns["evaluate_all"]
                ns["evaluate_all"] = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("evaluation failed"))
                with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                    ns["train_policy_brief"](environment(ns), 3, eval_every=0,
                        checkpoint_dir=directory, checkpoint_every=5, show_progress=False)
                saved = load_file(Path(directory) / "latest.pt")
                self.assertEqual(saved["update"], 3)
                self.assertIsNone(saved["last_eval"])
                ns["evaluate_all"] = evaluate
                ns["ppo_update"] = lambda *args: self.fail("Completed updates must not repeat")
                ns["train_policy_brief"](environment(ns), 3, eval_every=0,
                    checkpoint_dir=directory, resume=True, show_progress=False)
                self.assertEqual(len(evaluated), 1)
                # Even a fully complete re-open must not re-evaluate or append rows.
                log = (Path(directory) / "training_log.csv").read_bytes()
                (Path(directory) / "final.pt").unlink()  # Simulate final-file publication failing.
                ns["train_policy_brief"](environment(ns), 3, eval_every=0,
                    checkpoint_dir=directory, resume=True, show_progress=False)
                self.assertEqual(len(evaluated), 1)
                self.assertEqual(log, (Path(directory) / "training_log.csv").read_bytes())
                self.assertTrue((Path(directory) / "final.pt").is_file())

    def test_zero_update_checkpoint_and_configuration_rejection(self):
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename), tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                ns, evaluated = small_training(filename)
                ns["train_policy_brief"](environment(ns), 0, checkpoint_dir=directory, show_progress=False)
                ns["train_policy_brief"](environment(ns), 0, checkpoint_dir=directory, resume=True, show_progress=False)
                self.assertEqual(len(evaluated), 1)
                checkpoint = Path(directory) / "latest.pt"
                old = checkpoint.read_bytes()
                changed = environment(ns)
                changed.r_pick *= 2
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    ns["train_policy_brief"](changed, 0, checkpoint_dir=directory, resume=True, show_progress=False)
                ns["LR"] *= 2
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    ns["train_policy_brief"](environment(ns), 0, checkpoint_dir=directory, resume=True, show_progress=False)
                self.assertEqual(checkpoint.read_bytes(), old)
                with self.assertRaises(FileExistsError):
                    ns["train_policy_brief"](environment(ns), 0, checkpoint_dir=directory, show_progress=False)

    def test_accepted_requests_rng_and_observation_round_trip(self):
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename), tempfile.TemporaryDirectory() as directory:
                ns, _ = small_training(filename)
                env = environment(ns)
                pending = ride(ns, (.1, .2), (-.2, .1))
                visible = ride(ns, (.3, 0), (0, -.3))
                env.pending_ride = pending
                env.ride_buffer = [(visible, 4)]
                obs, mask = env._get_obs()
                state = RolloutState(env, obs, mask, 2.5, 7)
                policy = ns["ActorCritic"](env.obs_dim, env.act_dim)
                optimizer = torch.optim.Adam(policy.parameters())
                checkpoint = TrainingCheckpoint(directory, env, training_settings(ns), {})
                checkpoint.save(policy, optimizer, env, state, ns, update=2, last_eval=None, metrics=None)
                fresh = environment(ns)
                _, _, _, restored = checkpoint.load(policy, optimizer, fresh, ns, ns["RideReq"])
                self.assertIsInstance(fresh.pending_ride, ns["RideReq"])
                self.assertIsInstance(fresh.ride_buffer[0][0], ns["RideReq"])
                self.assertIs(restored.env, fresh)
                np.testing.assert_array_equal(fresh._get_obs()[0], obs)
                np.testing.assert_array_equal(env.rng.random(10), fresh.rng.random(10))
                self.assertEqual((restored.episode_return, restored.episode_steps), (2.5, 7))


class SweepPersistenceTests(unittest.TestCase):
    def test_sweep_saves_each_group_and_resumes_real_partial_training_without_duplicates(self):
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename), tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                ns, _ = small_training(filename)
                ns.update(R=1., V=.01)
                real_train, calls = ns["train_policy_brief"], []
                def train(env, **kwargs):
                    combo = kwargs["combo_meta"]["combo_id"]
                    calls.append(combo)
                    update = ns["ppo_update"]
                    if combo == 2:
                        count = [0]
                        def fail(*args):
                            count[0] += 1
                            if count[0] == 2:
                                raise RuntimeError("group 2 interrupted")
                            return update(*args)
                        ns["ppo_update"] = fail
                    try:
                        return real_train(env, **kwargs)
                    finally:
                        ns["ppo_update"] = update
                ns["train_policy_brief"] = train
                args = ([0, 2], [.1], [2.], [5], [3], [6], [5.5])
                path = Path(directory) / "results.csv"
                kwargs = dict(train_updates_per_combo=2, csv_path=path, train_seeds=[42], checkpoint_every=1, show_progress=False)
                with self.assertRaisesRegex(RuntimeError, "group 2 interrupted"):
                    ns["run_param_sweep"](*args, **kwargs)
                folder = next((Path(directory) / "results_runs").iterdir())
                self.assertEqual(len(pd.read_csv(path)), 6)
                self.assertTrue((folder / "combo_0001/final.pt").is_file())
                self.assertEqual(load_file(folder / "combo_0002/latest.pt")["update"], 1)
                ns["train_policy_brief"] = real_train
                resumed = ns["run_param_sweep"](*args, resume_dir=folder, **kwargs)
                self.assertEqual(len(resumed), 12)
                self.assertFalse(resumed.duplicated(["combo_id", "algo"]).any())
                self.assertEqual(json.loads((folder / "progress.json").read_text())["completed_groups"], 2)
                self.assertEqual(resumed.attrs["run_directory"], str(folder))
                ns["train_policy_brief"] = lambda *args, **kwargs: self.fail("Committed group retrained")
                resumed_again = ns["run_param_sweep"](*args, resume_dir=folder, **kwargs)
                pd.testing.assert_frame_equal(resumed, resumed_again)
                with self.assertRaisesRegex(ValueError, "Sweep settings differ"):
                    ns["run_param_sweep"](*args, resume_dir=folder, **dict(kwargs, train_updates_per_combo=3))

    def test_atomic_failure_keeps_old_file_and_group_commit_rebuilds_csv_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "existing.csv"
            path.write_bytes(b"original")
            with patch("training_persistence.os.replace", side_effect=OSError("locked")):
                with self.assertRaises(OSError):
                    atomic_write(path, lambda stream: stream.write(b"new"))
            self.assertEqual(path.read_bytes(), b"original")
            self.assertEqual(list(root.glob("*.tmp")), [])
            plan = [dict(combo_id=1)]
            store = SweepStore(root / "results.csv", plan, {})
            rows = [dict(combo_id=1, algo=str(i), R_PICK_ALPHA=.1, RIDE_TTL_MIN=5,
                         R=5.5, V=.19, DT=.5, rate=1.) for i in range(6)]
            with patch.object(store, "export", side_effect=OSError("export failed")):
                with self.assertRaises(OSError):
                    store.commit(1, rows)
            restored = SweepStore(root / "results.csv", plan, {}, resume_dir=store.directory)
            self.assertEqual(restored.completed, {1})
            self.assertEqual(len(pd.read_csv(root / "results.csv")), 6)
            # Fresh runs never reuse this folder or its scoped training logs.
            fresh = SweepStore(root / "results.csv", plan, {})
            self.assertNotEqual(fresh.directory, restored.directory)

    def test_progress_fallback_and_persistent_stage(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as output:
            path = Path(directory) / "progress.json"
            updates = []
            with patch.dict("sys.modules", {"tqdm.auto": None}):
                bar = TrainingProgress(10, enabled=True, path=path, callback=updates.append)
            bar.report(3, "evaluating", saved_update=2)
            bar.close()
            result = json.loads(path.read_text())
            self.assertEqual((result["percent"], result["phase"], result["saved_update"]), (30., "evaluating", 2))
            self.assertIn("3/10 30%", output.getvalue())
            self.assertEqual(len(updates), 1)


if __name__ == "__main__":
    unittest.main()
