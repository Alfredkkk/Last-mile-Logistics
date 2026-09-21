"""Accepted-trip/TTL observations and unchanged code-to-paper geometry conventions."""
import contextlib
import io
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from test_notebook_regressions import EXPERIMENTS, experiment_definitions, ride, scripted_env, set_packages
from test_b1_b5 import training_definitions
from experiment_support import alpha_metadata, summarize_alpha_groups, with_effective_alpha


def core_length(filename):
    return 9 if filename.startswith("NonStationary/") else 7


def visible_features(filename, env):
    obs, _ = env._get_obs()
    start = core_length(filename) + 3 * env.k_pack
    return obs[start:-7].reshape(env.max_visible, 7)


class TripObservationTests(unittest.TestCase):
    def test_previously_identical_passenger_states_now_expose_the_destination(self):
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename):
                ns = experiment_definitions(filename)
                envs = []
                for drop in (.05, 2.):
                    env = scripted_env(ns, [(4, 0)], v=.19, dt=.5)
                    env.with_passenger = True
                    env.drop_target = np.array([drop, 0.], np.float32)
                    envs.append(env)
                first, second = [e._get_obs()[0] for e in envs]
                np.testing.assert_array_equal(first[:-7], second[:-7])
                self.assertFalse(np.array_equal(first, second))
                for env, obs in zip(envs, (first, second)):
                    np.testing.assert_array_equal(obs[-7:-3], np.zeros(4))
                    np.testing.assert_allclose(obs[-3:], [env.drop_target[0] / env.R, 0,
                                                        env.drop_target[0] / env.R])
                results = [e.step(0) for e in envs]
                self.assertFalse(envs[0].with_passenger)
                self.assertTrue(envs[1].with_passenger)
                self.assertNotEqual(results[0][1], results[1][1])
                np.testing.assert_array_equal(results[0][0][-7:], np.zeros(7))

    def test_pickup_destination_lifecycle_and_four_zone_commitment(self):
        for filename in EXPERIMENTS:
            ns = experiment_definitions(filename)
            for deferred in (False, True):
                with self.subTest(notebook=filename, deferred=deferred):
                    request = ride(ns, (.1, 0), (.1, .1))
                    env = scripted_env(ns, [(0, 0), (4, 0)], initial=[(request, 10)], v=.19, dt=.5)
                    np.testing.assert_array_equal(env._get_obs()[0][-7:], np.zeros(7))
                    self.assertTrue(env._accept_ride(request, defer_pickup=deferred))
                    obs, mask = env._get_obs()
                    np.testing.assert_allclose(obs[-7:], [float(deferred), .01, 0, .01, .01, .01, .02])
                    self.assertEqual(len(env.ride_buffer), 0)
                    if deferred:
                        np.testing.assert_array_equal(mask, [1., 0., 0., 0., 0., 0.])
                        env.step(0)  # Delivery continues; known endpoints stay in the observation.
                        self.assertEqual(env._get_obs()[0][-7], 1.)
                        self.assertTrue(env._start_reserved_ride())
                        self.assertEqual(env._get_obs()[0][-7], 0.)
                    while env.to_pickup is not None:
                        obs = env.step(0)[0]
                        if env.to_pickup is not None:
                            self.assertAlmostEqual(obs[-4], np.abs(env.to_pickup - env.pos).sum() / env.R)
                    self.assertTrue(env.with_passenger)
                    np.testing.assert_array_equal(obs[-6:-3], np.zeros(3))
                    self.assertGreater(obs[-1], 0.)
                    while env.with_passenger:
                        obs = env.step(0)[0]
                    np.testing.assert_array_equal(obs[-7:], np.zeros(7))
                    obs, _ = env.reset(seed=1)
                    np.testing.assert_array_equal(obs[-7:], np.zeros(7))

    def test_pickup_coordinates_and_zero_distance_targets_have_explicit_state(self):
        for filename in EXPERIMENTS:
            ns = experiment_definitions(filename)
            env = scripted_env(ns, [(4, 0)])
            env.to_pickup = np.array([1., 0.], np.float32)
            env.drop_target = np.array([2., 0.], np.float32)
            first = env._get_obs()[0]
            env.to_pickup = np.array([0., 1.], np.float32)
            second = env._get_obs()[0]
            np.testing.assert_array_equal(first[:-7], second[:-7])
            self.assertFalse(np.array_equal(first[-6:-3], second[-6:-3]))
            self.assertEqual(first[-4], second[-4])  # Equal distance, different direction.
            env.to_pickup = np.zeros(2, np.float32)
            env.drop_target = np.zeros(2, np.float32)
            obs, _ = env._get_obs()
            flags_start = core_length(filename) - 4
            self.assertEqual(obs[flags_start], 1.)
            np.testing.assert_array_equal(obs[-7:], np.zeros(7))

    def test_ttl_tracks_sorted_request_identity_and_distinguishes_urgency(self):
        for filename in EXPERIMENTS:
            ns = experiment_definitions(filename)
            # Distinct requests at identical pickup/dropoff coordinates have different TTLs.
            farther = ride(ns, (.4, 0), (1, 0))
            first = ride(ns, (.1, 0), (1, 0))
            second = ride(ns, (.1, 0), (1, 0))
            expired = ride(ns, (0, 0), (1, 0))
            env = scripted_env(ns, [(0, 0), (4, 0)], ride_ttl_minutes=10,
                               initial=[(farther, 9), (first, 3), (second, 7), (expired, 0)])
            before_buffer = [(id(r), ttl) for r, ttl in env.ride_buffer]
            features = visible_features(filename, env)
            np.testing.assert_allclose(features[:3, -1], [.3, .7, .9])
            np.testing.assert_array_equal(features[0, :6], features[1, :6])
            np.testing.assert_array_equal(features[3:], np.zeros((2, 7)))
            self.assertEqual([(id(r), ttl) for r, ttl in env.ride_buffer], before_buffer)
            obs_before = env._get_obs()[0]
            env.ride_buffer[1] = (first, 2)
            obs_after = env._get_obs()[0]
            expected_index = core_length(filename) + 3 * env.k_pack + 6
            self.assertEqual(np.flatnonzero(obs_before != obs_after).tolist(), [expected_index])
            self.assertAlmostEqual(obs_after[expected_index], .2)
            env.max_visible = 2
            np.testing.assert_allclose(visible_features(filename, env)[:, -1], [.2, .7])

    def test_dimensions_padding_and_retained_partial_package_observation(self):
        for filename in EXPERIMENTS:
            ns = experiment_definitions(filename)
            for k, visible in ((10, 5), (3, 1), (4, 3)):
                with self.subTest(notebook=filename, packages=k, visible=visible):
                    env = ns["CoModalEnv"](seed=1, lam=0, max_visible=visible)
                    env.k_pack = k
                    obs, mask = env.reset(seed=1)
                    self.assertEqual(len(obs), core_length(filename) + 3 * k + 7 * visible + 7)
                    self.assertEqual(len(obs), env.obs_dim)
                    self.assertEqual(len(mask), env.act_dim)
                    self.assertTrue(np.isfinite(obs).all())
                    self.assertEqual(obs.dtype, np.float32)
                    np.testing.assert_array_equal(visible_features(filename, env), np.zeros((visible, 7)))
            env = ns["CoModalEnv"](seed=1, lam=0)
            nearest = [(i / 20, 0) for i in range(1, 11)]
            set_packages(env, nearest + [(4., 0.)])
            first = env._get_obs()[0]
            set_packages(env, nearest + [(-4., 0.)])
            # The chosen scope still hides distant package geometry; do not claim full Markov state.
            np.testing.assert_array_equal(first, env._get_obs()[0])


class PaperConventionTests(unittest.TestCase):
    def test_paper_conversion_preserves_physical_area_radius_and_ttl_cap(self):
        for R, alpha, v, dt, ttl in ((5.5, .2, .19, .5, 5), (8, .7, .4, .25, 7), (2, 0., 1., 3., 5)):
            with self.subTest(R=R, alpha=alpha):
                m = alpha_metadata(alpha, R=R, v=v, dt=dt, ttl_minutes=ttl)
                self.assertAlmostEqual(m["R_PAPER"] ** 2, 2 * R**2)
                self.assertAlmostEqual(m["R_PICK_ALPHA_PAPER"] ** 2, alpha**2 / 2)
                self.assertAlmostEqual(m["R_PICK_ALPHA_PAPER"] * m["R_PAPER"] / math.sqrt(2), m["PICKUP_RADIUS"])
                self.assertAlmostEqual(m["PICKUP_RADIUS"], alpha * R / math.sqrt(2))
                self.assertAlmostEqual(m["ALPHA_SATURATION"], v * max(1, round(ttl / dt)) * dt * math.sqrt(2) / R)
                self.assertAlmostEqual(m["ALPHA_EFFECTIVE_PAPER"], m["ALPHA_EFFECTIVE"] / math.sqrt(2))
                self.assertAlmostEqual(m["ALPHA_SATURATION_PAPER"], m["ALPHA_SATURATION"] / math.sqrt(2))

    def test_analysis_preserves_code_alpha_and_summarizes_paper_units_without_regrouping(self):
        frame = pd.DataFrame([dict(algo="DRL", R_PICK_ALPHA=a, rate=rate, R=5.5, V=.19,
                                   DT=.5, RIDE_TTL_MIN=5, OBS_DIM=79)
                              for a, rate in ((.1, 6), (.25, 2), (.4, 8))], index=[4, 4, 7])
        original = frame.copy(deep=True)
        converted = with_effective_alpha(frame)
        pd.testing.assert_frame_equal(frame, original)
        np.testing.assert_array_equal(converted.R_PICK_ALPHA, frame.R_PICK_ALPHA)
        np.testing.assert_allclose(converted.R_PICK_ALPHA_PAPER, frame.R_PICK_ALPHA / math.sqrt(2))
        summary = summarize_alpha_groups(frame)
        self.assertEqual(summary.rate_mean.tolist(), [6., 5.])
        self.assertEqual(summary.n_runs.tolist(), [1, 2])
        np.testing.assert_allclose(summary.ALPHA_EFFECTIVE_PAPER, summary.ALPHA_EFFECTIVE / math.sqrt(2))
        self.assertNotIn("R_PICK_ALPHA_PAPER", summary)  # Several nominal alphas may share a group.
        mixed = pd.concat([frame, frame.assign(OBS_DIM=67)])
        self.assertEqual(len(summarize_alpha_groups(mixed)), 4)
        empty = with_effective_alpha(frame.iloc[:0])
        self.assertTrue(empty.empty)
        self.assertIn("R_PICK_ALPHA_PAPER", empty)

    def test_sweep_and_log_save_both_conventions_and_actual_observation_dimension(self):
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename):
                ns = training_definitions(filename)
                ns["set_global_seeds"] = lambda seed: None

                def train(env, **kwargs):
                    meta = kwargs["combo_meta"]
                    self.assertAlmostEqual(meta["PICKUP_RADIUS"], env.r_pick)
                    self.assertEqual(meta["OBS_DIM"], env.obs_dim)
                    ns["append_training_log"](kwargs["log_path"], 0, "eval", meta=meta)
                    m = dict(avg_reward=1., avg_t=1., avg_rate=1., avg_ep_rate=1.,
                             avg_finish_time=1., finish_rate=1.)
                    return None, {a: m.copy() for a in ("drl", "heur", "heur_vor", "four_zone", "pure", "pure_or")}

                ns["train_policy_brief"] = train
                with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                    csv_path = Path(directory) / "runs.csv"
                    ns["TRAIN_LOG_PATH"] = Path(directory) / "train.csv"
                    ns["run_param_sweep"]([10], [.2, .25, .4], [.5], [5], [5], [6], [6],
                                          csv_path=csv_path, train_seeds=[42])
                    results = pd.read_csv(csv_path)
                    # C1 scopes each group's log to its independent artifact folder.
                    logged = pd.concat([pd.read_csv(p) for p in
                                        Path(directory).glob("runs_runs/*/combo_*/training_log.csv")])
                    summary = pd.read_csv(csv_path.with_name("runs_alpha_summary.csv"))
                for data in (results, logged):
                    np.testing.assert_allclose(data.R_PICK_ALPHA_PAPER, data.R_PICK_ALPHA / math.sqrt(2))
                    np.testing.assert_allclose(data.R_PAPER, data.R * math.sqrt(2))
                    self.assertTrue((data.OBS_DIM == (81 if filename.startswith("NonStationary/") else 79)).all())
                self.assertEqual(len(logged), 2)
                np.testing.assert_allclose(summary.ALPHA_EFFECTIVE_PAPER, summary.ALPHA_EFFECTIVE / math.sqrt(2))


if __name__ == "__main__":
    unittest.main()
