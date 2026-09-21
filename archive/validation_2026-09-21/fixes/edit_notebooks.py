"""Apply reviewed source-only edits without rewriting notebook outputs/metadata."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def edit_sources(filename, transform):
    path = ROOT / filename
    raw = path.read_text(encoding="utf-8")
    nb = json.loads(raw)
    changed = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        old = "".join(cell["source"])
        new = transform(old)
        if old == new:
            continue
        ast.parse(new)
        before = json.dumps(cell["source"], ensure_ascii=False, indent=1).replace("\n", "\n   ")
        after = json.dumps(new.splitlines(keepends=True), ensure_ascii=False, indent=1).replace("\n", "\n   ")
        assert raw.count('"source": ' + before) == 1, filename
        raw = raw.replace('"source": ' + before, '"source": ' + after, 1)
        changed += 1
    assert changed, filename
    json.loads(raw)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(raw)
    print(f"{filename}: edited {changed} code cells; preserved outputs and metadata")


def replace_function(code, name, replacement):
    tree = ast.parse(code)
    node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if node is None:
        return code
    lines = code.splitlines(keepends=True)
    return "".join(lines[:node.lineno - 1]) + replacement.rstrip() + "\n" + "".join(lines[node.end_lineno:])


def fix_analysis(code):
    code = replace_function(code, "best_alpha_series", '''def best_alpha_series(df_in: pd.DataFrame, gamma_val, algo_name):
    """Average pure-delivery runs across alpha; optimize alpha only for ride policies."""
    sub = df_in[(df_in["GAMMA_PACK"] == gamma_val) & (df_in["algo"] == algo_name)]
    if sub.empty:
        return np.array([]), np.array([])
    if algo_name in {"PURE", "PURE_OR"}:
        rates = sub.groupby("LAMBDA")["rate"].mean().sort_index()
        return rates.index.to_numpy(), rates.to_numpy()
    grp = (sub.groupby(["LAMBDA", "R_PICK_ALPHA"])["rate"]
              .mean()
              .reset_index())
    best = grp.loc[grp.groupby("LAMBDA")["rate"].idxmax()].sort_values("LAMBDA")
    return best["LAMBDA"].to_numpy(), best["rate"].to_numpy()
''')
    return replace_function(code, "best_alpha_by_lambda_gamma", '''def best_alpha_by_lambda_gamma(df_in: pd.DataFrame, lam, gamma, algo):
    """Average pure-delivery runs; select the best alpha only for ride policies."""
    sub = df_in[(df_in["LAMBDA"] == lam) &
                (df_in["GAMMA_PACK"] == gamma) &
                (df_in["algo"] == algo)]
    if sub.empty:
        return np.nan
    if algo in {"PURE", "PURE_OR"}:
        return float(sub[RATE_COL].mean())
    # Average repeated runs at the same alpha before selecting the best alpha.
    alpha_mean = sub.groupby("R_PICK_ALPHA")[RATE_COL].mean()
    return float(alpha_mean.max())
''')


if __name__ == "__main__":
    for filename in ("analysis.ipynb", "NonStationary/analysis2.ipynb"):
        edit_sources(filename, fix_analysis)
