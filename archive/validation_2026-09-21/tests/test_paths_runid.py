"""Directory isolation, identified runs and actual convergence-cell integration."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_notebook_regressions import ROOT, EXPERIMENTS, ANALYSES, notebook_tree
from test_training_persistence import small_training, load_file
from project_paths import ProjectPaths, select_training_log, read_run_training_log
from training_persistence import SweepStore
from experiment_support import summarize_alpha_groups


@contextlib.contextmanager
def cwd(path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def write_log(store, combo, rates=(1., 2.)):
    path = store.combo_dir(combo) / "training_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(run_id=store.run_id, combo_id=combo, train_seed=42, update=i+1,
                       kind="train", step_avg_rate=rate, rate=rate, REPORT_UNSCALED=True)
                  for i, rate in enumerate(rates)]).to_csv(path, index=False)
    return path


class DirectoryTests(unittest.TestCase):
    def test_all_four_notebook_bootstraps_and_result_literals_from_both_working_directories(self):
        for filename in EXPERIMENTS + ANALYSES:
            nb, _ = notebook_tree(filename)
            initial = "".join(nb["cells"][0]["source"])
            bootstrap = initial[initial.index("# Shared training state"):initial.index("# Only for historical CSVs")]
            expected = ROOT / "NonStationary" / "Results" if filename.startswith("NonStationary/") else ROOT / "Results"
            for directory in (ROOT, ROOT / "NonStationary"):
                with self.subTest(notebook=filename, cwd=directory), cwd(directory), patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("LAST_MILE_PROJECT_ROOT", None)
                    ns = {}
                    exec(bootstrap, ns)
                    self.assertEqual(ns["RESULTS_DIR"], expected)
                    self.assertEqual(ns["result_file"]("Results/example.csv"), expected / "example.csv")
                    self.assertEqual(ns["result_file"]("example.csv"), expected / "example.csv")
                    # The global CSV setting is executable without loading data or training.
                    for cell in nb["cells"]:
                        if cell["cell_type"] != "code":
                            continue
                        for node in ast.parse("".join(cell["source"])).body:
                            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CSV_PATH" for t in node.targets):
                                exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), ns)
                                self.assertEqual(ns["CSV_PATH"], expected / "param_sweep_results_2.csv")

    def test_uber_discovery_skips_empty_directories_and_relative_override_is_rooted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            root.mkdir()
            empty = root.parent / "Uber_NYC"
            empty.mkdir()
            populated = root / "NonStationary/Uber_NYC"
            populated.mkdir(parents=True)
            (populated / "fhvhv_tripdata_2021-01.parquet").touch()
            paths = ProjectPaths("nonstationary", root)
            with patch.dict(os.environ, {}, clear=False), cwd(root.parent):
                os.environ.pop("UBER_PARQUET_DIR", None)
                self.assertEqual(paths.resolve_uber_parquet_dir(), populated)
                self.assertEqual(paths.resolve_uber_parquet_dir("data/uber"), root / "data/uber")
                with patch.dict(os.environ, {"UBER_PARQUET_DIR": "another/data"}):
                    self.assertEqual(paths.resolve_uber_parquet_dir(), root / "another/data")

    def test_nonstationary_demand_cell_uses_cache_without_raw_data_or_pyarrow_import(self):
        nb, _ = notebook_tree("NonStationary/experiment2.ipynb")
        source = "".join(nb["cells"][2]["source"])
        self.assertNotIn("import pyarrow", source)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            paths = ProjectPaths("nonstationary", directory)
            paths.results_dir.mkdir(parents=True)
            pd.DataFrame({"hour": np.arange(24), "alpha": np.ones(24)/24}).to_csv(
                paths.results("hourly_alpha_2021.csv"), index=False)
            ns = dict(paths=paths, results_path=paths.results)
            exec(source, ns)
            np.testing.assert_allclose(ns["HOURLY_MULTIPLIER"], np.ones(24))
            self.assertEqual(ns["HOURLY_ALPHA_CACHE"], paths.results("hourly_alpha_2021.csv"))


class RunIdentityTests(unittest.TestCase):
    def test_actual_sweep_defaults_identify_all_artifacts_and_resume_preserves_identity(self):
        for filename in EXPERIMENTS:
            with self.subTest(notebook=filename), tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                ns, _ = small_training(filename)
                paths = ProjectPaths("nonstationary" if filename.startswith("NonStationary/") else "stationary", directory)
                ns.update(paths=paths, results_path=paths.results, result_file=paths.result_file, output_file=paths.output_file)
                args = ([0], [.1], [.2], [5], [3], [6], [5.5])
                result = ns["run_param_sweep"](*args, train_seeds=[42, 43], show_progress=False)
                folder = Path(result.attrs["run_directory"])
                run_id = result.attrs["run_id"]
                csv_path = paths.results("param_sweep_results_2.csv")
                self.assertTrue(csv_path.is_file())
                self.assertEqual(result.run_id.unique().tolist(), [run_id])
                for name in ("manifest.json", "progress.json"):
                    self.assertEqual(json.loads((folder / name).read_text())["run_id"], run_id)
                for name in ("results.csv", "alpha_summary.csv"):
                    self.assertEqual(pd.read_csv(folder / name).run_id.unique().tolist(), [run_id])
                for group in (1, 2):
                    target = folder / f"combo_{group:04d}"
                    self.assertEqual(load_file(target / "latest.pt")["run_id"], run_id)
                    self.assertEqual(pd.read_csv(target / "training_log.csv").run_id.unique().tolist(), [run_id])
                    self.assertEqual(json.loads((target / "progress.json").read_text())["run_id"], run_id)
                selected, identity, group = select_training_log(csv_path)
                self.assertEqual((identity, group), (run_id, 2))
                self.assertEqual(read_run_training_log(selected).run_id.unique().tolist(), [run_id])
                resumed = ns["run_param_sweep"](*args, train_seeds=[42, 43], resume_dir=folder, show_progress=False)
                self.assertEqual(resumed.attrs["run_id"], run_id)
                pd.testing.assert_frame_equal(result, resumed)
                again = ns["run_param_sweep"](*args, train_seeds=[42, 43], show_progress=False)
                self.assertNotEqual(again.attrs["run_id"], run_id)

    def test_standalone_repeated_training_same_seed_has_distinct_ids_and_cannot_be_blended(self):
        ns, _ = small_training(EXPERIMENTS[0])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            log = Path(directory) / "training_log.csv"
            for _ in range(2):
                ns["train_policy_brief"](ns["CoModalEnv"](seed=42, lam=0), 0, log_path=log,
                    combo_meta={"combo_id": 1, "train_seed": 42}, show_progress=False)
            rows = pd.read_csv(log)
            self.assertEqual(rows.run_id.nunique(), 2)
            with self.assertRaisesRegex(ValueError, "Select one"):
                read_run_training_log(log)
            chosen = read_run_training_log(log, run_id=rows.run_id.iloc[0], combo_id=1)
            self.assertEqual(len(chosen), 1)

    def test_log_selection_is_explicit_and_never_falls_back_to_historical_common_log(self):
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / "results.csv"
            pd.DataFrame({"update": [1]}).to_csv(Path(directory) / "training_log.csv", index=False)
            with self.assertRaisesRegex(FileNotFoundError, "identified training log"):
                select_training_log(csv)
            older = SweepStore(csv, [{"combo_id": 1}], {})
            old_log = write_log(older, 1)
            newer = SweepStore(csv, [{"combo_id": 1}, {"combo_id": 2}], {})
            new_log = write_log(newer, 2, (8., 10.))
            empty = SweepStore(csv, [{"combo_id": 1}], {})
            self.assertEqual(select_training_log(csv)[0], new_log)
            self.assertEqual(select_training_log(csv, run_id=older.run_id, combo_id=1)[0], old_log)
            with self.assertRaises(FileNotFoundError):
                select_training_log(csv, run_id=newer.run_id, combo_id=1)
            with self.assertRaises(FileNotFoundError):
                select_training_log(csv, run_id=empty.run_id)
            with self.assertRaises(ValueError):
                select_training_log(csv, run_id=older.run_id, run_dir=newer.directory)
            duplicate = pd.read_csv(new_log)
            pd.concat([duplicate, duplicate]).to_csv(new_log, index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_run_training_log(new_log)

    def test_both_convergence_cells_plot_only_the_selected_run_group(self):
        for filename in ANALYSES:
            with self.subTest(notebook=filename), tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                csv = Path(directory) / "results.csv"
                older = SweepStore(csv, [{"combo_id": 1}], {})
                write_log(older, 1, (1., 2.))
                newer = SweepStore(csv, [{"combo_id": 1}], {})
                chosen = write_log(newer, 1, (8., 10.))
                plots = []
                plotting = types.SimpleNamespace(plot=lambda x, y, **kw: plots.append(list(y)), **{
                    name: (lambda *args, **kwargs: None) for name in
                    ("figure", "xlabel", "ylabel", "title", "grid", "legend", "show")})
                nb, _ = notebook_tree(filename)
                source = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
                              and "# DRL convergence for one" in "".join(c["source"]))
                ns = dict(CSV_PATH=csv, select_training_log=select_training_log,
                          read_run_training_log=read_run_training_log, plt=plotting)
                exec(source, ns)
                self.assertEqual(ns["log_path"], chosen)
                self.assertEqual(plots, [[8., 10.], [8., 9.]])

    def test_alpha_summary_keeps_independent_runs_separate(self):
        frame = pd.DataFrame([dict(run_id=run, algo="DRL", R_PICK_ALPHA=.3, RIDE_TTL_MIN=5,
                                   R=5.5, V=.19, DT=.5, rate=rate)
                              for run, rate in (("first", 1.), ("first", 3.), ("second", 9.))])
        summary = summarize_alpha_groups(frame).set_index("run_id")
        self.assertEqual(summary.loc["first", "rate_mean"], 2.)
        self.assertEqual(summary.loc["first", "n_runs"], 2)
        self.assertEqual(summary.loc["second", "rate_mean"], 9.)


if __name__ == '__main__':
    unittest.main()
