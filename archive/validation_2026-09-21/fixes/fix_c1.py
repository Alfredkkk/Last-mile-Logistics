"""Apply C1 persistence/progress changes without modifying environment behavior."""
import ast
from pathlib import Path
from edit_notebooks import edit_sources, replace_function

source = Path(__file__).with_name("c1_functions.py").read_text(encoding="utf-8")
lines = source.splitlines(keepends=True)
functions = {n.name: "".join(lines[n.lineno-1:n.end_lineno]) for n in ast.parse(source).body}

def transform(code):
    for name, replacement in functions.items():
        code = replace_function(code, name, replacement)
    if 'TRAIN_SEEDS = [SEED]' in code and '# Checkpoints: every 5 updates by default.' not in code:
        code = code.replace('TRAIN_SEEDS = [SEED]',
                            '# Checkpoints: every 5 updates by default. To resume, pass the printed run folder as resume_dir.\nTRAIN_SEEDS = [SEED]')
    return code

for filename in ("experiment.ipynb", "NonStationary/experiment2.ipynb"):
    edit_sources(filename, transform)
