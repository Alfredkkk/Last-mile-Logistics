"""Apply the verified, fixed file list to the active project, preserving evidence."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = Path(r'C:\Users\93541\Research_project\Last-mile Logistics').resolve()
expected = json.loads((HERE / 'before_hashes.json').read_text(encoding='utf-8'))
assert set(expected) == {'project_paths.py', 'analysis.ipynb', 'NonStationary/analysis2.ipynb', 'progress_log.md'}
for relative, digest in expected.items():
    target = (ROOT / relative).resolve()
    assert target.is_relative_to(ROOT)
    if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f'File changed after staging; preserving it: {relative}')
    assert (HERE / 'staged' / relative).is_file()
archive = ROOT / 'archive' / HERE.name
if archive.exists():
    raise RuntimeError(f'Evidence destination already exists: {archive}')
shutil.copytree(HERE, archive)
for relative in expected:
    target = ROOT / relative
    raw = (HERE / 'staged' / relative).read_bytes()
    target.write_bytes(raw)
    assert target.read_bytes() == raw
print('Applied and verified:', ', '.join(expected))
print('Originals, tests, plots and verification:', archive)
