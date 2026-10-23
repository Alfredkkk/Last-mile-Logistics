import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(r'C:\Users\93541\Research_project\Last-mile Logistics')
active = '--active' in sys.argv
if active:
    sys.argv.remove('--active')
spec = importlib.util.spec_from_file_location('grid_paths', (ROOT if active else HERE / 'staged') / 'project_paths.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GridCurveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='grid_curve_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        base = dict(GAMMA_PACK=.5, ALPHA_EFFECTIVE=.2, RIDE_TTL_MIN=5,
                    MAX_VISIBLE_RIDES=5, SWITCH_GRACE_STEPS=6, RT=6., R=5.5, V=.19, DT=.5)
        # Two seeds in setting A, one in B: settings, not rows, have equal weight.
        self.plan = [dict(base, combo_id=1, LAMBDA=2, train_seed=42, ALPHA_MEMBERS='[.2]'),
                     dict(base, combo_id=2, LAMBDA=2, train_seed=43, ALPHA_MEMBERS='[.2]'),
                     dict(base, combo_id=3, LAMBDA=40, train_seed=42, ALPHA_MEMBERS='[.2,.3,.4]')]
        manifest = dict(run_id='run-test', plan=self.plan, settings={'REPORT_UNSCALED': True})
        (self.root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        self.logs = []
        for row, values in zip(self.plan, [(2, 4), (4, 6), (9, 11)]):
            path = self.root / f"combo_{row['combo_id']:04d}" / 'training_log.csv'
            path.parent.mkdir()
            self.logs.append(path)
            pd.DataFrame([dict(row, run_id='run-test', REPORT_UNSCALED=True, kind='train',
                               update=i, step_avg_rate=value) for i, value in enumerate(values, 1)]).to_csv(path, index=False)

    def test_equal_setting_weights_and_no_alias_expansion(self):
        curve, coverage = module.read_grid_training_curve(self.root)
        self.assertEqual(curve.step_avg_rate.tolist(), [6., 8.])
        self.assertEqual(coverage['effective_settings'], 2)
        self.assertEqual(coverage['train_seeds'], [42, 43])

    def test_missing_group_and_header_only_log(self):
        self.logs[2].unlink()
        curve, coverage = module.read_grid_training_curve(self.root)
        self.assertTrue(curve.empty)
        self.assertEqual(coverage['missing_groups'], [3])
        pd.read_csv(self.logs[1]).iloc[:0].to_csv(self.logs[2], index=False)
        self.assertTrue(module.read_grid_training_curve(self.root)[0].empty)

    def test_late_missing_update_uses_fixed_denominator(self):
        pd.read_csv(self.logs[2]).iloc[:1].to_csv(self.logs[2], index=False)
        curve, coverage = module.read_grid_training_curve(self.root)
        self.assertEqual(curve['update'].tolist(), [1])
        self.assertEqual(curve.step_avg_rate.tolist(), [6.])
        self.assertEqual(coverage['observed_updates'], 2)

    def test_mixed_identity_duplicate_and_bad_rate_rejected(self):
        original = pd.read_csv(self.logs[0])
        for invalid in [pd.concat([original, original.iloc[:1]]),
                        original.assign(run_id=['run-test', 'foreign']),
                        original.assign(step_avg_rate=[float('nan'), 4]),
                        original.assign(train_seed=99),
                        original.assign(LAMBDA=99),
                        original.assign(REPORT_UNSCALED=False)]:
            with self.subTest(invalid=invalid.to_dict()):
                invalid.to_csv(self.logs[0], index=False)
                with self.assertRaises(ValueError):
                    module.read_grid_training_curve(self.root)
        original.to_csv(self.logs[0], index=False)

    def test_current_runs_match_independent_direct_mean(self):
        for relative, groups, updates in [('Results', 240, 50), ('NonStationary/Results', 180, 40)]:
            log, identity, group = module.select_training_log(ROOT / relative / 'param_sweep_results_2.csv')
            curve, coverage = module.read_grid_training_curve(log.parent.parent)
            raw = pd.concat([pd.read_csv(p) for p in log.parent.parent.glob('combo_*/training_log.csv')])
            direct = raw.loc[raw.kind == 'train'].groupby('update').step_avg_rate.mean()
            np.testing.assert_allclose(curve.step_avg_rate, direct, rtol=1e-14)
            self.assertEqual(coverage['planned_groups'], groups)
            self.assertEqual(coverage['common_updates'], updates)
            self.assertEqual(coverage['run_id'], identity)


if __name__ == '__main__':
    unittest.main(verbosity=2)
