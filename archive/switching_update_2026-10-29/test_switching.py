"""Switching state-machine regression tests using the notebooks' real dynamics.

Run from any directory with Python and NumPy installed:
    python archive/switching_update_2026-10-29/test_switching.py -v

Only definitions are loaded; no notebook training, plots, CSV writes, or solvers run.
LOGISTICS_TEST_ROOT can select another project copy for the same checks.
"""
import ast
import dataclasses
import json
import math
import os
from pathlib import Path
import typing
import unittest

import numpy as np


ROOT = Path(os.environ["LOGISTICS_TEST_ROOT"]) if "LOGISTICS_TEST_ROOT" in os.environ else next(
    p for p in Path(__file__).resolve().parents if (p / "experiment_support.py").is_file()
)
NOTEBOOKS = ("experiment.ipynb", "NonStationary/experiment2.ipynb")
POLICIES = ("baseline_nearby_rule", "baseline_nearby_rule_voronoi")


def load_definitions(filename):
    nb = json.loads((ROOT / filename).read_text(encoding="utf-8"))
    ns = dict(np=np, math=math, dataclass=dataclasses.dataclass,
              rng=np.random.default_rng(0),
              **{name: getattr(typing, name) for name in ("List", "Tuple", "Optional", "Dict")})
    for node in ast.parse("".join(nb["cells"][1]["source"])).body:
        if isinstance(node, ast.Assign):
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), ns)
            except NameError:
                pass  # Torch/device-only globals are not needed for baseline execution.
    ns["HOURLY_MULTIPLIER"] = np.ones(24)
    tree = ast.parse("\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"))
    names = {"manhattan", "project_to_diamond", "step_towards", "RideReq", "CoModalEnv",
             "_baseline_switching", *POLICIES}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), ns)
    return ns


def request(ns, pickup, dropoff):
    pickup = np.asarray(pickup, dtype=np.float32)
    dropoff = np.asarray(dropoff, dtype=np.float32)
    return ns["RideReq"](pickup, dropoff, ns["manhattan"](pickup, dropoff))


def scripted_env(ns, points, *, initial=(), arrivals=None, start=(0, 0), radius=.2, **kwargs):
    """Fix inputs and trace public effects while retaining real environment movement."""
    class Scripted(ns["CoModalEnv"]):
        def reset(self, seed=None):
            # super().reset() calls _get_obs; initialize trace fields first.
            self.events, self.screens, self.steps = [], [], []
            super().reset(seed=seed)
            self.pkg_pos_all = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            self.pkg_delivered = np.zeros(len(points), dtype=bool)
            self.pkg_remaining_idx = list(range(len(points)))
            self.packages = self.pkg_pos_all.copy()
            self.pos = np.asarray(start, dtype=np.float32)
            self.ride_buffer = list(initial)
            return self._get_obs()

        def _sample_rides_this_step(self):
            return [] if arrivals is None else arrivals(self)

        def _visible_rides(self, reference_point=None, pickup_radius=None):
            if reference_point is not None:
                self.screens.append(dict(t=self.t, reference=np.asarray(reference_point).copy(),
                                         busy=self.to_pickup is not None or self.with_passenger,
                                         remaining=list(self.pkg_remaining_idx)))
            return super()._visible_rides(reference_point, pickup_radius)

        def _accept_ride(self, chosen, defer_pickup=False):
            accepted = super()._accept_ride(chosen, defer_pickup)
            if accepted:
                self.events.append(dict(kind="accept", t=self.t, request=chosen,
                                        remaining=list(self.pkg_remaining_idx),
                                        deferred=defer_pickup, pos=self.pos.copy()))
            return accepted

        def _start_reserved_ride(self):
            chosen = self.pending_ride
            started = super()._start_reserved_ride()
            if started:
                self.events.append(dict(kind="start_pickup", t=self.t, request=chosen,
                                        remaining=list(self.pkg_remaining_idx), pos=self.pos.copy()))
            return started

        def step(self, action):
            old_delivered = self.pkg_delivered.copy()
            old_passenger = self.with_passenger
            before = dict(t=self.t, pos=self.pos.copy(),
                          busy=self.to_pickup is not None or self.with_passenger,
                          pending=self.pending_ride is not None)
            result = super().step(action)
            before.update(pos_after=self.pos.copy(), done=result[2])
            self.steps.append(before)
            if old_passenger and not self.with_passenger:
                self.events.append(dict(kind="drop", t=self.t, pos=self.pos.copy()))
            delivered = np.flatnonzero(self.pkg_delivered & ~old_delivered).tolist()
            if delivered:
                self.events.append(dict(kind="deliver", t=self.t, ids=delivered))
            return result

    options = dict(R=10., v=.19, dt=.5, lam=0., gamma_pack=0., seed=1)
    options.update(kwargs)
    options["r_pick_alpha"] = radius * math.sqrt(2) / options["R"]
    return Scripted(**options)


def events(env, kind):
    return [event for event in env.events if event["kind"] == kind]


class SwitchingTests(unittest.TestCase):
    def configurations(self):
        for filename in NOTEBOOKS:
            for policy in POLICIES:
                yield filename, policy, load_definitions(filename)

    def assert_time_accounting(self, env, result):
        self.assertAlmostEqual(result[1], env.t)
        self.assertAlmostEqual(env.time_delivery_min + env.time_rides_min, env.t)
        self.assertAlmostEqual(result[2], result[0] / result[1] if result[1] else 0.)
        self.assertAlmostEqual(result[5], env.time_rides_min)
        self.assertAlmostEqual(result[6], env.time_delivery_min)

    def test_pre_grace_reserves_once_finishes_package_then_picks_up_despite_ttl_expiry(self):
        for filename, policy, ns in self.configurations():
            with self.subTest(notebook=filename, policy=policy):
                chosen = request(ns, (.09, 0), (1.5, 0))
                env = scripted_env(ns, [(.05, 0), (2, 0)], initial=[(chosen, 1)])
                result = ns[policy](env, seed=1)
                accepts = events(env, "accept")
                self.assertEqual(len(accepts), 1)
                self.assertTrue(accepts[0]["deferred"])
                self.assertIn(0, accepts[0]["remaining"])
                sequence = [event["kind"] for event in env.events]
                self.assertLess(sequence.index("accept"), sequence.index("deliver"))
                self.assertLess(sequence.index("deliver"), sequence.index("start_pickup"))
                first_pickup = events(env, "start_pickup")[0]
                self.assertEqual(first_pickup["t"], .5)
                self.assertNotIn(0, first_pickup["remaining"])
                np.testing.assert_allclose(first_pickup["pos"], [.05, 0])
                self.assertEqual(result[3:5], ("packages_done", 1))
                self.assertTrue(env.pkg_delivered.all())
                self.assert_time_accounting(env, result)

    def test_post_window_screens_completed_package_even_when_next_target_has_other_rides(self):
        for filename, policy, ns in self.configurations():
            with self.subTest(notebook=filename, policy=policy):
                near_a = request(ns, (.06, 0), (1.8, 0))
                near_b = request(ns, (1.95, 0), (2.1, 0))
                env = scripted_env(ns, [(.05, 0), (2, 0)], radius=.15,
                                   arrivals=lambda e: [near_b, near_a] if e.t == .5 else [])
                ns[policy](env, seed=1)
                first = events(env, "accept")[0]
                self.assertIs(first["request"], near_a)
                self.assertEqual(first["t"], .5)
                self.assertNotIn(0, first["remaining"])
                screens = [s for s in env.screens if s["t"] == .5]
                self.assertTrue(screens)
                np.testing.assert_allclose(screens[-1]["reference"], [.05, 0])

    def test_unmatched_post_window_keeps_moving_and_expires_without_accepting_late_ride(self):
        for filename, policy, ns in self.configurations():
            with self.subTest(notebook=filename, policy=policy):
                late = request(ns, (.05, 0), (3, 0))
                env = scripted_env(ns, [(.05, 0), (3, 0)], radius=.1,
                                   arrivals=lambda e: [late] if e.t == 2.0 else [])
                result = ns[policy](env, switch_grace_steps=3, seed=1)
                self.assertEqual(events(env, "accept"), [])
                self.assertGreater(next(s for s in env.steps if s["t"] == .5)["pos_after"][0], .05)
                a_times = [s["t"] for s in env.screens if np.allclose(s["reference"], [.05, 0])]
                self.assertEqual(a_times, [0., .5, 1., 1.5])
                self.assertEqual(result[3], "packages_done")
                self.assert_time_accounting(env, result)

    def test_delivery_inside_post_window_refreshes_reference_to_the_package_just_completed(self):
        for filename, policy, ns in self.configurations():
            with self.subTest(notebook=filename, policy=policy):
                near_b = request(ns, (.21, 0), (2.9, 0))
                env = scripted_env(ns, [(.05, 0), (.14, 0), (3, 0)], radius=.1,
                                   arrivals=lambda e: [near_b] if e.t == 1.0 else [])
                ns[policy](env, switch_grace_steps=3, seed=1)
                first = events(env, "accept")[0]
                self.assertIs(first["request"], near_b)
                self.assertEqual(first["t"], 1.)
                self.assertNotIn(1, first["remaining"])
                screens = [s for s in env.screens if s["t"] == 1.]
                np.testing.assert_allclose(screens[-1]["reference"], [.14, 0])

    def test_post_window_match_does_not_create_a_new_delivery_commitment_to_next_package(self):
        for filename, policy, ns in self.configurations():
            with self.subTest(notebook=filename, policy=policy):
                passenger = request(ns, (.05, 0), (2.8, 0))
                env = scripted_env(ns, [(.05, 0), (.23, 0), (3, 0)], radius=.2,
                                   arrivals=lambda e: [passenger] if e.t == 1. else [])
                ns[policy](env, seed=1)
                first = events(env, "accept")[0]
                self.assertEqual(first["t"], 1.)
                self.assertNotIn(0, first["remaining"])
                self.assertIn(1, first["remaining"])
                self.assertFalse(first["deferred"])
                # B is within one step, but this is A's still-active post window.
                self.assertLess(abs(float(first["pos"][0]) - .23), env.v * env.dt)
                pickup_step = next(s for s in env.steps if s["t"] == 1.)
                self.assertLess(pickup_step["pos_after"][0], first["pos"][0])

    def test_no_screening_before_last_delivery_step(self):
        for filename, policy, ns in self.configurations():
            with self.subTest(notebook=filename, policy=policy):
                early = request(ns, (2.5, 0), (5, 0))
                env = scripted_env(ns, [(2.5, 0), (5, 0)], initial=[(early, 1)], v=1., dt=1.)
                ns[policy](env, seed=1)
                self.assertEqual(events(env, "accept"), [])
                self.assertEqual(env.screens[0]["t"], 2.)

    def test_busy_trip_crossing_near_package_cannot_leave_grace_after_dropoff(self):
        for filename, policy, ns in self.configurations():
            with self.subTest(notebook=filename, policy=policy):
                first = request(ns, (.5, 0), (3, 0))
                late = request(ns, (1.5, .5), (5, 0))
                env = scripted_env(ns, [(.5, 0), (1.5, .5), (5, 0)],
                                   initial=[(first, 50)], v=1., dt=1., radius=.2,
                                   ride_ttl_minutes=1,
                                   arrivals=lambda e: [late] if e.t == 5. else [])
                ns[policy](env, switch_grace_steps=6, seed=1)
                self.assertEqual(events(env, "drop")[0]["t"], 5.)
                accepts = events(env, "accept")
                self.assertEqual(len(accepts), 1)
                self.assertIs(accepts[0]["request"], first)
                self.assertFalse(any(screen["busy"] for screen in env.screens))
                after_drop = next(s for s in env.steps if s["t"] == 5.)
                self.assertFalse(after_drop["busy"])
                self.assertLess(after_drop["pos_after"][0], 3.)

    def test_voronoi_excludes_current_package_before_it_is_completed(self):
        for filename in NOTEBOOKS:
            for policy in POLICIES:
                with self.subTest(notebook=filename, policy=policy):
                    ns = load_definitions(filename)
                    origin = request(ns, (.5, 0), (.5, 0))
                    other = request(ns, (.55, 0), (2, 0))
                    env = scripted_env(ns, [(.5, 0), (2, 0)], initial=[(origin, 1), (other, 1)],
                                       v=1., dt=1.)
                    ns[policy](env, seed=1)
                    first = events(env, "accept")[0]
                    self.assertIs(first["request"], other if policy.endswith("voronoi") else origin)
                    self.assertTrue(first["deferred"])

    def test_voronoi_rejects_already_completed_region_and_selects_next_eligible_ride(self):
        for filename in NOTEBOOKS:
            for policy in POLICIES:
                with self.subTest(notebook=filename, policy=policy):
                    ns = load_definitions(filename)
                    completed = request(ns, (.5, 0), (.5, 0))
                    remaining = request(ns, (.6, 0), (4, 0))
                    env = scripted_env(ns, [(.5, 0), (4, 0)], v=1., dt=1.,
                                       arrivals=lambda e: [completed, remaining] if e.t == 1 else [])
                    ns[policy](env, seed=1)
                    first = events(env, "accept")[0]
                    self.assertEqual(first["t"], 1.)
                    self.assertIs(first["request"], remaining if policy.endswith("voronoi") else completed)

    def test_last_package_reservation_finishes_ride_and_voronoi_has_no_unvisited_destination(self):
        for filename in NOTEBOOKS:
            for policy in POLICIES:
                with self.subTest(notebook=filename, policy=policy):
                    ns = load_definitions(filename)
                    passenger = request(ns, (.09, 0), (.5, 0))
                    env = scripted_env(ns, [(.05, 0)], initial=[(passenger, 1)])
                    result = ns[policy](env, seed=1)
                    self.assertEqual(result[3], "packages_done")
                    self.assertIsNone(env.pending_ride)
                    self.assertIsNone(env.to_pickup)
                    self.assertFalse(env.with_passenger)
                    if policy.endswith("voronoi"):
                        self.assertEqual(result[4], 0)
                        self.assertEqual(result[1], .5)
                    else:
                        self.assertEqual(result[4], 1)
                        self.assertGreater(result[1], .5)
                        self.assertFalse(env.steps[0]["done"])
                        self.assertEqual(len(events(env, "drop")), 1)
                    self.assert_time_accounting(env, result)

    def test_horizon_stops_reserved_trip_without_inventing_passenger_travel(self):
        for filename, policy, ns in self.configurations():
            with self.subTest(notebook=filename, policy=policy):
                ns["HORIZON_MIN"] = .5
                passenger = request(ns, (.09, 0), (1.5, 0))
                env = scripted_env(ns, [(.05, 0), (2, 0)], initial=[(passenger, 1)])
                result = ns[policy](env, seed=1)
                self.assertEqual(result[3:5], ("horizon_reached", 1))
                self.assertEqual(result[1], .5)
                self.assertEqual(env.time_rides_min, 0.)
                self.assertEqual(env.time_delivery_min, .5)
                self.assertTrue(env.pkg_delivered[0])
                self.assertFalse(env.pkg_delivered[1])
                self.assert_time_accounting(env, result)

    def test_seeded_demand_smoke_and_activity_accounting(self):
        for filename, policy, ns in self.configurations():
            for seed in (7, 19):
                with self.subTest(notebook=filename, policy=policy, seed=seed):
                    ns["HORIZON_MIN"] = 80.
                    ns["STEPS_PER_EP"] = 160
                    kwargs = dict(R=2., v=.5, dt=.5, lam=2., gamma_pack=.8,
                                  r_pick_alpha=.4, seed=seed)
                    if filename.startswith("NonStationary"):
                        kwargs["hourly_multiplier"] = np.linspace(.5, 1.5, 24)
                    env = ns["CoModalEnv"](**kwargs)
                    original_nearest = env._nearest_package
                    original_reference = env._visibility_reference_point
                    result = ns[policy](env, seed=seed)
                    self.assertEqual(env._nearest_package, original_nearest)
                    self.assertEqual(env._visibility_reference_point, original_reference)
                    self.assertIn(result[3], ("packages_done", "horizon_reached"))
                    self.assertTrue(np.isfinite(result[0:3]).all())
                    self.assert_time_accounting(env, result)
                    if result[3] == "packages_done":
                        self.assertTrue(env.pkg_delivered.all())
                        self.assertIsNone(env.pending_ride)
                        self.assertIsNone(env.to_pickup)
                        self.assertFalse(env.with_passenger)


if __name__ == "__main__":
    unittest.main()
