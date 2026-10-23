"""Stage only the two analysis plots and their shared reader; never train."""
import ast
import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import types

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = Path(r'C:\Users\93541\Research_project\Last-mile Logistics')
FILES = ['project_paths.py', 'analysis.ipynb', 'NonStationary/analysis2.ipynb', 'progress_log.md']
originals = HERE / 'before'
staged = HERE / 'staged'
hashes = {}
for relative in FILES:
    raw = (ROOT / relative).read_bytes()
    hashes[relative] = hashlib.sha256(raw).hexdigest()
    for directory in (originals, staged):
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
(HERE / 'before_hashes.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')

helper = (ROOT / 'project_paths.py').read_text(encoding='utf-8')
helper = helper.replace('explicit single-run convergence-log selection.', 'identified training-curve selection and aggregation.')
helper += (HERE / 'grid_helper.txt').read_text(encoding='utf-8')
ast.parse(helper)
(staged / 'project_paths.py').write_text(helper, encoding='utf-8')
module = types.ModuleType('project_paths')
module.__file__ = str(staged / 'project_paths.py')
exec(compile(helper, module.__file__, 'exec'), module.__dict__)
# Rendering uses the actual project's data, not this staging directory.
module.PROJECT_ROOT = ROOT
sys.modules['project_paths'] = module
sys.path.insert(0, str(ROOT))

results = {}
new_source = (HERE / 'convergence_cell.txt').read_text(encoding='utf-8')
for relative, scenario in [('analysis.ipynb', 'stationary'),
                           ('NonStationary/analysis2.ipynb', 'nonstationary')]:
    notebook = json.loads((originals / relative).read_text(encoding='utf-8'))
    original = json.loads((originals / relative).read_text(encoding='utf-8'))
    initialization = ''.join(notebook['cells'][0]['source']).replace(
        'from project_paths import ProjectPaths, select_training_log, read_run_training_log',
        'from project_paths import (ProjectPaths, select_training_log, read_run_training_log,\n'
        '                           read_grid_training_curve)')
    assert 'read_grid_training_curve)' in initialization
    notebook['cells'][0]['source'] = initialization.splitlines(keepends=True)
    notebook['cells'][4]['source'] = new_source.splitlines(keepends=True)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    namespace = {'__name__': '__main__'}
    exec(compile(initialization, relative + ':initialization', 'exec'), namespace)
    image_outputs, plotted = [], []
    def capture_show():
        fig = plt.gcf()
        plotted.append([{'x': line.get_xdata().tolist(), 'y': line.get_ydata().tolist()}
                        for line in fig.axes[0].lines])
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        name = 'training' if len(image_outputs) == 0 else 'evaluation'
        (HERE / f'{scenario}_{name}.png').write_bytes(buf.getvalue())
        image_outputs.append({'output_type': 'display_data', 'metadata': {},
                              'data': {'image/png': base64.b64encode(buf.getvalue()).decode(),
                                       'text/plain': ['<Figure size 600x400 with 1 Axes>']}})
        plt.close(fig)
    plt.show = capture_show
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(new_source, relative + ':convergence', 'exec'), namespace)
    notebook['cells'][4]['outputs'] = [
        {'output_type': 'stream', 'name': 'stdout', 'text': output.getvalue().splitlines(keepends=True)},
        *image_outputs]
    notebook['cells'][4]['execution_count'] = None
    assert len(image_outputs) == 2
    for i in [1, 2, 3, 5, 6, 7]:
        assert notebook['cells'][i] == original['cells'][i]
    assert notebook['metadata'] == original['metadata']
    # Evaluation data and smoothing must exactly preserve the previous plotting logic.
    df = namespace['df']
    expected_eval = df.loc[df.kind == 'eval', ['update', 'rate']].sort_values('update')
    assert plotted[1][0]['x'] == expected_eval['update'].tolist()
    assert plotted[1][0]['y'] == expected_eval.rate.tolist()
    assert plotted[1][1]['y'] == expected_eval.rate.rolling(3, min_periods=1).mean().tolist()
    namespace['training_curve'].to_csv(HERE / f'{scenario}_grid_mean.csv', index=False)
    results[scenario] = dict(coverage=namespace['coverage'],
                             first=float(namespace['training_curve'].step_avg_rate.iloc[0]),
                             last=float(namespace['training_curve'].step_avg_rate.iloc[-1]),
                             evaluation_group=namespace['selected_combo_id'],
                             evaluation_points=len(expected_eval), evaluation_unchanged=True)
    (staged / relative).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(output.getvalue())

entry = '''

## 2026-10-22 Full-grid Training Learning Curves

- Both analysis notebooks now plot the mean training rollout revenue rate across the selected run's complete effective parameter grid. Seeds are averaged within each effective setting before settings receive equal weight; equivalent alpha aliases are not re-expanded. The raw mean and a trailing three-point mean are displayed.
- The manifest fixes the denominator. Logs are checked against run/group/seed/parameter/reporting metadata and duplicate records are rejected. Only updates present for every planned group are plotted; missing groups and update coverage are printed. A partial run is not silently presented as a full-grid average.
- ANALYSIS_RUN_ID / ANALYSIS_RUN_DIR select the run; ANALYSIS_COMBO_ID affects evaluation only. Evaluation retains the previous selected-group data and three-point smoothing, with clearer labels. Training and evaluation therefore have different aggregation scopes and their levels are not a generalization-gap comparison.
- Regenerated only the convergence-cell outputs from existing data: stationary 240 settings x 50 updates, nonstationary 180 settings x 40 updates, each with training seed 42. Evaluation values and all other notebook cells/outputs are unchanged. No training, simulation, reward, result CSV, or checkpoint changes.
- Validation and before snapshots are archived under archive/training_grid_2026-10-22. Current runs all use REPORT_UNSCALED=True. The previously identified logger/label issue for REPORT_UNSCALED=False is outside this plotting change.
'''
(staged / 'progress_log.md').write_text((originals / 'progress_log.md').read_text(encoding='utf-8') + entry,
                                        encoding='utf-8')
(HERE / 'verification.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))
