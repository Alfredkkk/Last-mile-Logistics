"""Source-only migration of four notebooks to scenario paths and identified logs."""
import ast
from pathlib import Path
import re
from edit_notebooks import edit_sources, replace_function

BOOTSTRAP_OLD = '''_support_root = next(p for p in (Path.cwd(), *Path.cwd().parents)
                     if (p / "experiment_support.py").is_file())'''
BOOTSTRAP_NEW = '''import os
_root_candidates = ([Path(os.environ["LAST_MILE_PROJECT_ROOT"]).expanduser()]
                    if os.environ.get("LAST_MILE_PROJECT_ROOT") else [Path.cwd(), *Path.cwd().parents])
_support_root = next((p.resolve() for p in _root_candidates
                      if (p / "project_paths.py").is_file() and (p / "experiment.ipynb").is_file()), None)
if _support_root is None:
    raise FileNotFoundError("Start from this project or set LAST_MILE_PROJECT_ROOT to its folder")'''

CONVERGENCE = '''# DRL convergence for one explicitly identified run/group.
# None selects the newest run with recorded data, then its latest logged group.
ANALYSIS_RUN_ID = None
ANALYSIS_COMBO_ID = None
ANALYSIS_RUN_DIR = None  # optional absolute path to a particular saved run

log_path, selected_run_id, selected_combo_id = select_training_log(
    CSV_PATH, run_id=ANALYSIS_RUN_ID, combo_id=ANALYSIS_COMBO_ID, run_dir=ANALYSIS_RUN_DIR)
df = read_run_training_log(log_path, run_id=selected_run_id, combo_id=selected_combo_id)
print(f"Convergence: run_id={selected_run_id}, combo_id={selected_combo_id}, "
      f"train_seed={df['train_seed'].iloc[0]}\\nLog: {log_path}")

# Identity is validated before smoothing; unrelated runs are never averaged.
units = 'Revenue rate (per minute)'
if 'REPORT_UNSCALED' in df and str(df['REPORT_UNSCALED'].iloc[0]).lower() in ('false', '0', '0.0'):
    units = 'Revenue rate (scaled, per minute)'
for kind, column, label in [('train', 'step_avg_rate', 'Training'), ('eval', 'rate', 'Evaluation')]:
    curve = df.loc[df['kind'] == kind, ['update', column]].sort_values('update').copy()
    if curve.empty:
        continue
    curve['smooth'] = curve[column].rolling(window=3, min_periods=1).mean()
    plt.figure(figsize=(6, 4))
    plt.plot(curve['update'], curve[column], marker='o', alpha=0.4, label=label)
    plt.plot(curve['update'], curve['smooth'], linewidth=2, label='Rolling mean')
    plt.xlabel('PPO update')
    plt.ylabel(units)
    plt.title(f'DRL {label.lower()}: {selected_run_id} / group {selected_combo_id}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
'''


def anchor_literals(code):
    """Anchor result CSV/figure literals, leaving path-join components unchanged."""
    tree = ast.parse(code)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    lines = code.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    changes = []
    for node in ast.walk(tree):
        parent = parents.get(node)
        if isinstance(parent, (ast.JoinedStr, ast.BinOp)):
            continue
        if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id in (
                'results_path', 'result_file', 'output_file'):
            continue
        source = ast.get_source_segment(code, node)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if not re.fullmatch(r'(?:Results/)?(?:param_sweep[\w.-]*\.csv|[\w.-]+\.png)', value):
                continue
            source = repr(value.removeprefix('Results/'))
        elif isinstance(node, ast.JoinedStr):
            if not re.match(r'''f["'](?:Results/)?(?:plot_|rev_|ratio_)''', source or '') or not source.endswith(('.png"', ".png'")):
                continue
            source = source.replace('Results/', '', 1)
        else:
            continue
        # AST column offsets are UTF-8 byte offsets.
        start = offsets[node.lineno-1] + len(lines[node.lineno-1].encode()[:node.col_offset].decode())
        end = offsets[node.end_lineno-1] + len(lines[node.end_lineno-1].encode()[:node.end_col_offset].decode())
        changes.append((start, end, 'results_path(' + source + ')'))
    for start, end, replacement in sorted(changes, reverse=True):
        code = code[:start] + replacement + code[end:]
    return code


def transform(code, scenario):
    if '# DRL convergence using current training_log.csv' in code:
        return CONVERGENCE
    if BOOTSTRAP_OLD in code:
        code = code.replace(BOOTSTRAP_OLD, BOOTSTRAP_NEW)
        if '# ---------- Path Helpers ----------' in code:
            start = code.index('# ---------- Path Helpers ----------')
            end = code.index('# ---------- Global Parameters ----------', start)
            code = code[:start] + code[end:]
        code += f'''\n# Scenario paths are independent of the Jupyter working directory.
from project_paths import ProjectPaths, select_training_log, read_run_training_log
paths = ProjectPaths("{scenario}")
PROJECT_ROOT = paths.root
RESULTS_DIR = paths.results_dir
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
results_path = paths.results
result_file = paths.result_file
output_file = paths.output_file
'''
        # Path aliases must precede the initialization cell's CSV_PATH assignment.
        marker = '# Scenario paths are independent'
        at = code.index(marker)
        setup, code = code[at:], code[:at]
        insert = code.index('# Only for historical CSVs')
        code = code[:insert] + setup + '\n' + code[insert:]
    if '# Non-stationary demand fit from Uber NYC data' in code:
        start = code.index('def find_project_root(')
        end = code.index('def resolve_uber_parquet_dir(', start)
        code = code[:start] + code[end:]
        code = code.replace('import pyarrow.parquet as pq\n', '')
        code = replace_function(code, 'resolve_uber_parquet_dir', '''def resolve_uber_parquet_dir(preferred: Optional[Path] = None) -> Path:
    """Use project-rooted overrides and search directories containing actual data."""
    return paths.resolve_uber_parquet_dir(preferred)
''')
    code = code.replace('ns_results_path(', 'results_path(').replace('ns_path(', 'results_path(')
    code = re.sub(r"TRAIN_LOG_PATH = .*\n", '# Sweep logs live in each run/group directory; no common append target.\n', code)
    if 'def append_training_log(' in code:
        code = code.replace('    meta = meta or {}\n', '''    from training_persistence import log_run_id
    log_path = output_file(log_path)
    meta = dict(meta or {})
    if not meta.get('run_id'):
        meta['run_id'] = log_run_id(log_path)
''', 1)
    if 'def train_policy_brief(' in code:
        code = code.replace('from training_persistence import TrainingCheckpoint, TrainingProgress, training_settings',
                            'from training_persistence import TrainingCheckpoint, TrainingProgress, training_settings, new_run_id')
        code = code.replace('    checkpoint = None\n', '''    combo_meta = dict(combo_meta or {})
    if checkpoint_dir is not None:
        checkpoint_dir = result_file(checkpoint_dir)
    if log_path is not None:
        log_path = output_file(log_path)
    checkpoint = None
''', 1)
        code = code.replace('        log_path = checkpoint.log_path\n', '''        log_path = checkpoint.log_path
        combo_meta = checkpoint.meta
    else:
        combo_meta.setdefault('run_id', new_run_id())
''', 1)
        code = code.replace('callback=progress_callback)', 'callback=progress_callback, run_id=combo_meta["run_id"])', 1)
    if 'def run_param_sweep(' in code:
        code = code.replace('csv_path = "param_sweep_results.csv"', 'csv_path = results_path("param_sweep_results_2.csv")')
        code = code.replace('    settings = dict(training_settings(globals()), RP=RP,', '''    csv_path = result_file(csv_path)
    if resume_dir is not None:
        resume_dir = result_file(resume_dir)
    settings = dict(training_settings(globals()), RP=RP,''', 1)
        code = code.replace('meta = dict(setting, RP=RP,', 'meta = dict(setting, run_id=store.run_id, RP=RP,', 1)
        code = code.replace('        df.attrs["run_directory"] = str(store.directory)',
                            '        df.attrs["run_directory"] = str(store.directory)\n        df.attrs["run_id"] = store.run_id')
    code = code.replace('pd.read_csv(csv_path)', 'pd.read_csv(result_file(csv_path))')
    code = code.replace('plt.savefig(savepath,', 'plt.savefig(output_file(savepath),')
    code = code.replace('savepath_prefix="plot_rate_vs_lambda_optimal_alpha"',
                        'savepath_prefix=results_path("plot_rate_vs_lambda_optimal_alpha")')
    return anchor_literals(code)


if __name__ == '__main__':
    for filename in ('experiment.ipynb', 'analysis.ipynb', 'NonStationary/experiment2.ipynb', 'NonStationary/analysis2.ipynb'):
        scenario = 'nonstationary' if filename.startswith('NonStationary/') else 'stationary'
        edit_sources(filename, lambda code: transform(code, scenario))
