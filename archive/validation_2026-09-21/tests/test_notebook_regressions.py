"""Targeted tests of notebook definitions, without executing training/plot cells.

Run with Python and NumPy/pandas/OR-Tools installed:
    python -m unittest discover -s tests -v
"""
import ast
import dataclasses
import json
import itertools
import math
from pathlib import Path
import sys
import types
import typing
import unittest

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "experiment_support.py").is_file())
sys.path.insert(0, str(ROOT))
from experiment_support import (RolloutState, SWEEP_METADATA_COLUMNS, build_sweep_plan,
                                summarize_alpha_groups, with_effective_alpha,
                                best_equivalent_alpha_rate)
from project_paths import ProjectPaths
ANALYSES = ("analysis.ipynb", "NonStationary/analysis2.ipynb")
EXPERIMENTS = ("experiment.ipynb", "NonStationary/experiment2.ipynb")


def notebook_tree(filename):
    nb = json.loads((ROOT / filename).read_text(encoding="utf-8"))
    return nb, ast.parse("\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"))


def definitions(filename, names, ns=None):
    nb, tree = notebook_tree(filename)
    if ns is None:
        ns = dict(np=np, pd=pd, math=math, RATE_COL="rate")
    paths = ProjectPaths("nonstationary" if filename.startswith("NonStationary/") else "stationary")
    ns.update(Path=Path, paths=paths, results_path=paths.results, result_file=paths.result_file,
              output_file=paths.output_file)
    ns.update(RolloutState=RolloutState, SWEEP_METADATA_COLUMNS=SWEEP_METADATA_COLUMNS,
              build_sweep_plan=build_sweep_plan, summarize_alpha_groups=summarize_alpha_groups,
              with_effective_alpha=with_effective_alpha,
              best_equivalent_alpha_rate=best_equivalent_alpha_rate,
              LEGACY_GEOMETRY={"R": 5.5, "V": .19, "DT": .5})
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), ns)
    return ns


def experiment_definitions(filename):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    ns = dict(np=np, pd=pd, math=math, dataclass=dataclasses.dataclass,
              rng=np.random.default_rng(0), pywrapcp=pywrapcp, routing_enums_pb2=routing_enums_pb2,
              **{name: getattr(typing, name) for name in ("List", "Tuple", "Optional", "Dict")})
    nb, _ = notebook_tree(filename)
    for node in ast.parse("".join(nb["cells"][1]["source"])).body:
        if isinstance(node, ast.Assign):
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), ns)
            except NameError:
                pass  # E.g. training-only configuration dependent on torch.
    ns["HOURLY_MULTIPLIER"] = np.ones(24)
    return definitions(filename, {"manhattan", "l1_inside", "project_to_diamond", "step_towards",
                                  "RideReq", "CoModalEnv", "FixedPackageEnv", "l1_zone_id",
                                  "_l1_distance", "_build_distance_matrix_L1", "solve_zone_tsp_L1",
                                  "baseline_four_zone", "baseline_pure_ortools"}, ns)


def set_packages(env, points):
    env.pkg_pos_all = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    env.pkg_delivered = np.zeros(len(points), dtype=bool)
    env.pkg_remaining_idx = list(range(len(points)))
    env.packages = env.pkg_pos_all.copy()


def ride(ns, pickup, dropoff):
    pickup, dropoff = np.asarray(pickup, np.float32), np.asarray(dropoff, np.float32)
    return ns["RideReq"](pickup, dropoff, ns["manhattan"](pickup, dropoff))


def scripted_env(ns, points, *, initial=(), arrivals=None, start=(0, 0), **kwargs):
    """Real environment dynamics with deterministic demand and package locations."""
    class Scripted(ns["CoModalEnv"]):
        def reset(self, seed=None):
            super().reset(seed=seed)
            set_packages(self, points)
            self.pos = np.asarray(start, dtype=np.float32)
            self.ride_buffer = list(initial)
            self.events = []
            self.screens = []
            return self._get_obs()

        def _sample_rides_this_step(self):
            return [] if arrivals is None else arrivals(self)

        def _visible_rides(self, reference_point=None, pickup_radius=None):
            if reference_point is not None:
                self.screens.append((self.t, reference_point.copy(), self._nearest_package()))
            return super()._visible_rides(reference_point, pickup_radius)

        def _accept_ride(self, chosen, defer_pickup=False):
            accepted = super()._accept_ride(chosen, defer_pickup)
            if accepted:
                self.events.append(dict(kind="accept", t=self.t, req=chosen,
                                        remaining=list(self.pkg_remaining_idx), deferred=defer_pickup))
            return accepted

        def _start_reserved_ride(self):
            request = self.pending_ride
            started = super()._start_reserved_ride()
            if started:
                self.events.append(dict(kind="start_pickup", t=self.t, req=request,
                                        remaining=list(self.pkg_remaining_idx)))
            return started

        def step(self, action):
            old_delivered = self.pkg_delivered.copy()
            old_passenger = self.with_passenger
            result = super().step(action)
            if old_passenger and not self.with_passenger:
                self.events.append(dict(kind="drop", t=self.t, pos=self.pos.copy()))
            delivered = np.flatnonzero(self.pkg_delivered & ~old_delivered).tolist()
            if delivered:
                self.events.append(dict(kind="deliver", t=self.t, ids=delivered))
            return result

    options = dict(R=10., v=1., dt=1., lam=0., gamma_pack=0., seed=1)
    options.update(kwargs)
    return Scripted(**options)


def known_order(ns):
    # Policy tests isolate transitions/timing. Real OR-Tools is tested separately.
    ns["solve_zone_tsp_L1"] = lambda points, start: list(range(len(points)))


class FourZoneTests(unittest.TestCase):
    def test_window_final_reference_reservation_ttl_and_destination(self):
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename):
                ns = experiment_definitions(filename)
                known_order(ns)
                early = ride(ns, (4, 0), (0, 2))
                same_zone = ride(ns, (4, 0), (4.5, 0))
                accepted = ride(ns, (4, 0), (-.1, .2))
                # Initial request expires before t0 opens. At t=1, the first of
                # two new requests has an ineligible current-zone destination.
                env = scripted_env(ns, [(3, 0), (4, 0), (-3.5, 0), (0, 6)],
                                   initial=[(early, 1)], ride_ttl_minutes=2,
                                   r_pick_alpha=.25 * math.sqrt(2) / 10,
                                   arrivals=lambda e: [same_zone, accepted] if e.t == 1 else [])
                result = ns["baseline_four_zone"](env, screen_window_min=3, seed=1)
                accepts = [e for e in env.events if e["kind"] == "accept"]
                self.assertEqual(len(accepts), 1)
                self.assertIs(accepts[0]["req"], accepted)
                self.assertEqual(accepts[0]["t"], 1)
                self.assertIn(0, accepts[0]["remaining"])
                self.assertIn(1, accepts[0]["remaining"])
                self.assertEqual(env.screens[0][0], 1)
                np.testing.assert_array_equal(env.screens[0][1], [4, 0])
                np.testing.assert_array_equal(env.screens[0][2], [3, 0])
                pickup = next(e for e in env.events if e["kind"] == "start_pickup")
                self.assertEqual(pickup["t"], 4)  # After the unaccepted TTL would expire at t=3.
                self.assertNotIn(0, pickup["remaining"])
                self.assertNotIn(1, pickup["remaining"])
                drop_index = next(i for i, e in enumerate(env.events) if e["kind"] == "drop")
                next_delivery = next(e for e in env.events[drop_index + 1:] if e["kind"] == "deliver")
                self.assertEqual(next_delivery["ids"], [3])  # Top zone, although left is closer.
                self.assertEqual(result[3:5], ("packages_done", 1))
                self.assertTrue(env.pkg_delivered.all())

    def test_dense_demand_serves_each_destination_before_next_trip(self):
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename):
                ns = experiment_definitions(filename)
                known_order(ns)
                points = [(1, 0), (-2, 0), (0, 3), (0, -4)]
                drops = [(.25, 0), (-.25, 0), (0, .25), (0, -.25)]
                requests = [ride(ns, p, d) for p in points for d in drops]
                env = scripted_env(ns, points, initial=[(r, 100) for r in requests],
                                   arrivals=lambda e: requests, ride_ttl_minutes=100, max_visible=100,
                                   r_pick_alpha=.05)
                ns["baseline_four_zone"](env, screen_window_min=100, seed=1)
                starts = [e for e in env.events if e["kind"] == "start_pickup"]
                self.assertEqual(len(starts), 3)
                self.assertEqual(env.accepted_rides, 3)
                drops_seen = 0
                for i, event in enumerate(env.events):
                    if event["kind"] == "drop":
                        drops_seen += 1
                        dest = ns["l1_zone_id"](event["pos"])
                        delivery = next(e for e in env.events[i + 1:] if e["kind"] == "deliver")
                        self.assertEqual(ns["l1_zone_id"](env.pkg_pos_all[delivery["ids"][0]]), dest)
                        later_starts = [e for e in env.events[i + 1:] if e["kind"] == "start_pickup"]
                        if later_starts:
                            self.assertLessEqual(delivery["t"], later_starts[0]["t"])
                self.assertEqual(drops_seen, 3)
                self.assertTrue(env.pkg_delivered.all())

    def test_empty_zones_duplicates_and_no_demand(self):
        for filename in EXPERIMENTS:
            ns = experiment_definitions(filename)
            known_order(ns)
            for points in ([(1, 0), (1, 0), (-2, 0)], [(1, 0)], []):
                with self.subTest(notebook=filename, points=points):
                    env = scripted_env(ns, points)
                    original = env._nearest_package
                    result = ns["baseline_four_zone"](env, seed=1)
                    self.assertEqual(result[3:5], ("packages_done", 0))
                    self.assertAlmostEqual(result[0], len(points) * env.rp)
                    self.assertTrue(env.pkg_delivered.all())
                    self.assertEqual(env._nearest_package, original)

    def test_completion_instant_and_no_wait_for_later_requests(self):
        for filename in EXPERIMENTS:
            ns = experiment_definitions(filename)
            known_order(ns)
            request = ride(ns, (1, 0), (-.25, 0))
            for arrival_time, expected in ((1, 1), (2, 0)):
                with self.subTest(notebook=filename, arrival_time=arrival_time):
                    env = scripted_env(ns, [(1, 0), (-2, 0)],
                                       arrivals=lambda e: [request] if e.t == arrival_time else [])
                    result = ns["baseline_four_zone"](env, screen_window_min=0, seed=1)
                    self.assertEqual(result[4], expected)
                    if expected == 0:
                        self.assertEqual(result[1], 4)  # No idle screening step after clearing right.

    def test_window_uses_actual_projected_and_discrete_travel_time(self):
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename):
                ns = experiment_definitions(filename)
                known_order(ns)
                env = scripted_env(ns, [(.4, .5), (-.8, 0)], start=(.1, .9), R=1, v=.19, dt=.5)
                ns["baseline_four_zone"](env, screen_window_min=1, seed=1)
                finished = next(e["t"] for e in env.events if e["kind"] == "deliver" and 0 in e["ids"])
                self.assertEqual(env.screens[0][0], finished - 1)

    def test_final_reference_skips_packages_delivered_on_an_earlier_leg(self):
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename):
                ns = experiment_definitions(filename)
                known_order(ns)
                request = ride(ns, (2, 0), (-.25, 0))
                # The nominal final route entry duplicates the first stop. All
                # packages at that coordinate are delivered together at t=1.
                env = scripted_env(ns, [(1, 0), (2, 0), (1, 0), (-3, 0)],
                                   initial=[(request, 20)], r_pick_alpha=.05)
                result = ns["baseline_four_zone"](env, screen_window_min=1, seed=1)
                self.assertEqual(result[4], 1)
                np.testing.assert_array_equal(env.screens[0][1], [2, 0])
                self.assertEqual(env.screens[0][0], 1)

    def test_horizon_preserves_accepted_request_and_restores_routing(self):
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename):
                ns = experiment_definitions(filename)
                known_order(ns)
                ns["HORIZON_MIN"] = 2
                request = ride(ns, (4, 0), (0, 2))
                env = scripted_env(ns, [(3, 0), (4, 0), (0, 6)],
                                   initial=[(request, 10)], r_pick_alpha=.1)
                original = env._nearest_package
                result = ns["baseline_four_zone"](env, screen_window_min=3, seed=1)
                self.assertEqual(result[1], 2)
                self.assertEqual(result[3:5], ("horizon_reached", 1))
                self.assertIs(env.pending_ride, request)
                self.assertIsNone(env.to_pickup)
                self.assertEqual(env._nearest_package, original)
                env.reset(seed=1)
                self.assertIsNone(env.pending_ride)

    def test_restores_environment_on_failure(self):
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename):
                ns = experiment_definitions(filename)
                known_order(ns)
                env = scripted_env(ns, [(1, 0), (-2, 0)])
                original = env._nearest_package
                def fail_step(action):
                    raise RuntimeError("test interruption")
                env.step = fail_step
                with self.assertRaisesRegex(RuntimeError, "test interruption"):
                    ns["baseline_four_zone"](env, seed=1)
                self.assertEqual(env._nearest_package, original)

    def test_environment_radius_default_and_explicit_override(self):
        for filename in EXPERIMENTS:
            ns = experiment_definitions(filename)
            known_order(ns)
            request = ride(ns, (1.2, 0), (-.25, 0))
            for override, expected in ((None, 0), (.3 * math.sqrt(2) / 10, 1)):
                with self.subTest(notebook=filename, override=override):
                    env = scripted_env(ns, [(1, 0), (-2, 0)], initial=[(request, 20)],
                                       r_pick_alpha=.1 * math.sqrt(2) / 10)
                    original_radius = env.r_pick
                    result = ns["baseline_four_zone"](env, r_pick_alpha=override, seed=1)
                    self.assertEqual(result[4], expected)
                    self.assertEqual(env.r_pick, original_radius)


class EnvironmentAcceptanceTests(unittest.TestCase):
    def test_reserved_request_survives_ttl_and_blocks_other_acceptance(self):
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename):
                ns = experiment_definitions(filename)
                first = ride(ns, (1, 0), (0, 1))
                second = ride(ns, (1, 0), (0, 2))
                env = scripted_env(ns, [(1, 0)], initial=[(first, 1), (second, 10)])
                self.assertTrue(env._accept_ride(first, defer_pickup=True))
                self.assertFalse(env._accept_ride(second))
                self.assertFalse(env._accept_ride(first, defer_pickup=True))
                result = env.step(1)  # Still deliver; a reservation prevents accepting a second ride.
                self.assertFalse(result[2])  # All packages done, but the accepted ride is still owed.
                self.assertEqual(len(env.packages), 0)
                self.assertEqual(env.accepted_rides, 1)
                self.assertIs(env.pending_ride, first)
                self.assertEqual(result[4].sum(), 1)
                self.assertTrue(env._start_reserved_ride())
                self.assertFalse(env._start_reserved_ride())
                while not result[2]:
                    result = env.step(0)
                self.assertEqual(env.accepted_rides, 1)
                self.assertIsNone(env.pending_ride)

    def test_immediate_actions_still_use_preexisting_visible_indices(self):
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename):
                ns = experiment_definitions(filename)
                observed = ride(ns, (1, 0), (0, 1))
                new = ride(ns, (.9, 0), (0, 2))
                env = scripted_env(ns, [(1, 0), (4, 0)], initial=[(observed, 10)],
                                   arrivals=lambda e: [new])
                env.step(1)
                np.testing.assert_array_equal(env.drop_target, observed.dropoff)
                self.assertTrue(env.with_passenger)
                self.assertEqual(env.accepted_rides, 1)
                self.assertIsNone(env.pending_ride)
                fixed = ns["FixedPackageEnv"](2, seed=1, lam=0)
                self.assertIsNone(fixed.pending_ride)
                fixed.step(0)


class RoutingSolverTests(unittest.TestCase):
    def test_four_zone_integration_with_real_solver_and_random_demand(self):
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename):
                ns = experiment_definitions(filename)
                options = {}
                if filename.startswith("NonStationary/"):
                    options["hourly_multiplier"] = np.linspace(.5, 1.5, 24)
                env = ns["CoModalEnv"](R=2, lam=8, gamma_pack=1.25, v=.4, dt=.5,
                                       r_pick_alpha=1., max_visible=20, seed=6, **options)
                original = env._nearest_package
                result = ns["baseline_four_zone"](env, seed=6, screen_window_min=3)
                occupied_zones = len({ns["l1_zone_id"](p) for p in env.pkg_pos_all})
                self.assertEqual(result[3], "packages_done")
                self.assertTrue(np.isfinite(result[:3]).all())
                self.assertTrue(env.pkg_delivered.all())
                self.assertGreater(result[4], 0)
                self.assertLessEqual(result[4], occupied_zones - 1)
                self.assertAlmostEqual(result[0], env.revenue_cum)
                self.assertAlmostEqual(result[1], result[5] + result[6])
                self.assertIsNone(env.pending_ride)
                self.assertEqual(env._nearest_package, original)

    def test_distance_precision_and_real_open_route_objectives(self):
        from ortools.constraint_solver import pywrapcp
        points = [(1., 0.), (-1., 0.), (0., 2.)]
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename):
                ns = experiment_definitions(filename)
                self.assertEqual(ns["_l1_distance"]((0, 0), (.49, 0)), 490)
                self.assertEqual(ns["_l1_distance"]((0, 0), (.51, 0)), 510)
                objectives = []
                class ObservedRoutingModel:
                    def __init__(self, *args):
                        self.model = pywrapcp.RoutingModel(*args)

                    def __getattr__(self, name):
                        return getattr(self.model, name)

                    def SolveWithParameters(self, parameters):
                        solution = self.model.SolveWithParameters(parameters)
                        if solution is not None:
                            objectives.append(solution.ObjectiveValue())
                        return solution

                ns["pywrapcp"] = types.SimpleNamespace(
                    RoutingIndexManager=pywrapcp.RoutingIndexManager,
                    RoutingModel=ObservedRoutingModel,
                    DefaultRoutingSearchParameters=pywrapcp.DefaultRoutingSearchParameters)

                def open_cost(order):
                    sequence = [(0., 0.)] + [points[i] for i in order]
                    return sum(ns["_l1_distance"](a, b) for a, b in zip(sequence, sequence[1:]))

                optimum = min(open_cost(p) for p in itertools.permutations(range(len(points))))
                order = ns["solve_zone_tsp_L1"](points, (0., 0.))
                self.assertEqual(sorted(order), list(range(len(points))))
                self.assertEqual(objectives[-1], open_cost(order))
                self.assertEqual(objectives[-1], optimum)
                self.assertEqual(ns["solve_zone_tsp_L1"]([], (0., 0.)), [])
                env = scripted_env(ns, points, v=100, dt=.1)
                result = ns["baseline_pure_ortools"](env, seed=1)
                actual_order = [idx for e in env.events if e["kind"] == "deliver" for idx in e["ids"]]
                self.assertEqual(objectives[-1], open_cost(actual_order))
                self.assertEqual(objectives[-1], optimum)
                self.assertEqual(result[3:5], ("packages_done", 0))


class AnalysisAggregationTests(unittest.TestCase):
    def test_pure_delivery_averages_runs_and_ride_policies_optimize_alpha(self):
        for filename in ANALYSES:
            ns = definitions(filename, {"best_alpha_series", "best_alpha_by_lambda_gamma"})
            rows = []
            for algo in ("PURE", "PURE_OR", "HEUR_VOR", "DRL"):
                for lam in (20, 10):
                    for alpha, rate in ((.1, 1.), (.1, 3.), (.2, 8.)):
                        rows.append(dict(algo=algo, LAMBDA=lam, GAMMA_PACK=.5,
                                         R_PICK_ALPHA=alpha, RIDE_TTL_MIN=5, R=5.5, V=.19, DT=.5,
                                         rate=rate + (lam - 10)))
            frame = pd.DataFrame(rows)
            for algo in ("PURE", "PURE_OR", "HEUR_VOR", "DRL"):
                with self.subTest(notebook=filename, algo=algo):
                    expected = 4. if algo in {"PURE", "PURE_OR"} else 8.
                    xs, ys = ns["best_alpha_series"](frame, .5, algo)
                    np.testing.assert_array_equal(xs, [10, 20])
                    np.testing.assert_allclose(ys, [expected, expected + 10])
                    self.assertEqual(ns["best_alpha_by_lambda_gamma"](frame, 10, .5, algo), expected)
                    self.assertEqual(len(ns["best_alpha_series"](frame, .9, algo)[0]), 0)
                    self.assertTrue(np.isnan(ns["best_alpha_by_lambda_gamma"](frame, 10, .9, algo)))


if __name__ == "__main__":
    unittest.main()
