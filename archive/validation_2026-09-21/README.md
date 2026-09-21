# 2026-09-21 validation archive

Archived after the directory/run-id changes passed all 53 regression tests.

- `tests/`: regression suite, relocated from the project root. It finds the active project through its ancestors.
- `fixes/`: September 21 edit scripts, before-edit snapshots, probes, verification reports and the isolated `.venv` used for validation. Historical edit scripts target their original code snapshots; these are evidence, not current application entry points.
- `review_2026-09-05/`: extracted source, paper excerpts and prior review probes, relocated from `tmp/`.
- `archive_manifest.json`: original/destination mapping, file counts/sizes and SHA-256 checks for source/evidence files. Third-party `.venv` files and bytecode are retained and checked by count/size rather than individually hashed.

From the project root, with NumPy, pandas, OR-Tools and PyTorch available:

```text
python -m unittest discover -s archive/validation_2026-09-21/tests -v
```

The saved complete report is `fixes/validation_53_tests.txt`. The isolated environment contains OR-Tools; the successful CPU PyTorch used for this session resides in `%TEMP%/ll-b1-torch-20260921` and must precede the archived venv's partial torch installation on `sys.path`. This is a validation setup, not the production CUDA environment.

Existing historical training logs are kept separately under `archive/training_logs/stationary/` and `archive/training_logs/nonstationary/`; they were already archived before this cleanup. Historical result CSVs/figures and saved notebook outputs were not regenerated or moved.

The active shared modules remain at the project root: `experiment_support.py`, `training_persistence.py`, and `project_paths.py`. None depends on this archive to run the notebooks.
