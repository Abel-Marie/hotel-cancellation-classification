"""Build the self-contained notebook from its jupytext source, then execute it.

Sources live in notebooks/_src/*.py (percent format). Three directives make the
notebook self-contained while running exactly the package's code:

    # @inline module: name, name    -> the source of those top-level definitions
    # @inline module: *             -> every top-level definition in the module
    # @inline imports               -> every third-party import used by the inlined code
    # @config                       -> configs/config.yaml as a string (CONFIG_YAML)
    # @versions                     -> the exact versions pinned in pyproject.toml (TESTED_VERSIONS)

In markdown cells, ``<<author_line>>`` becomes the author line from docs/author.json (nothing
while that file is empty).

The executed notebook writes metrics.json to OUTPUT_DIR; this script copies it
to reports/metrics/notebook/ for the notebook-vs-pipeline equivalence test.

A full execution takes about half an hour. After an edit that touches only the text
(markdown cells, the author line), ``--reuse-outputs`` rebuilds the notebook and keeps
the outputs of the last execution; it refuses if any code cell has changed.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import sys
import tomllib
from datetime import date
from pathlib import Path

import jupytext
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "hotel_cancellation"
FUTURE = "from __future__ import annotations\n\n"


def module_path(module: str) -> Path:
    parts = module.split(".")
    if parts[0] != "hotel_cancellation":
        raise ValueError(f"only package modules can be inlined: {module}")
    return PACKAGE.joinpath(*parts[1:]).with_suffix(".py")


def top_level_nodes(tree: ast.Module) -> dict[str, ast.stmt]:
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = node
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            for t in node.targets:
                out[t.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out[node.target.id] = node
    return out


def extract(module: str, names: list[str]) -> tuple[str, list[str]]:
    """Source of the requested definitions, in file order; plus the names defined."""
    src = module_path(module).read_text(encoding="utf-8")
    lines = src.splitlines()
    nodes = top_level_nodes(ast.parse(src))
    wanted = [n for n in nodes if n != "__all__"] if names == ["*"] else names
    missing = [n for n in wanted if n not in nodes]
    if missing:
        raise KeyError(f"{module} has no top-level {missing}")
    chosen = sorted({id(nodes[n]): nodes[n] for n in wanted}.values(), key=lambda n: n.lineno)
    blocks = []
    for node in chosen:
        start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
        blocks.append("\n".join(lines[start - 1:node.end_lineno]))
    return "\n\n\n".join(blocks) + "\n", wanted


def external_imports(module: str) -> list[str]:
    src = module_path(module).read_text(encoding="utf-8")
    out = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom) and node.level == 0 and node.module != "__future__":
            out.append(ast.unparse(node))
    return out


def parse_directive(line: str) -> tuple[str, list[str]]:
    body = line.split("@inline", 1)[1].strip()
    module, _, names = body.partition(":")
    return module.strip(), [n.strip() for n in names.split(",") if n.strip()]


def pinned_versions() -> dict[str, str]:
    """Exact pins from pyproject.toml (runtime dependencies), as {package: version}."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return dict(d.split("==") for d in project["dependencies"] if "==" in d)


def author_line() -> str:
    """The author lines and the date, from docs/author.json; empty if nothing is filled in."""
    path = ROOT / "docs" / "author.json"
    info = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    labels = [("name", "Author"), ("student_id", "Student ID"), ("course", "Course"), ("instructor", "Instructor"),
              ("institution", "Institution")]
    lines = [f"**{label}:** {info[key]}" for key, label in labels if info.get(key)]
    if lines:
        lines.append(f"**Date:** {date.today():%d %B %Y}".replace(" 0", " "))
    return "  \n".join(lines)


def build(source: Path, out: Path) -> None:
    nb = jupytext.read(source)
    modules, defined = [], {}
    for cell in nb.cells:
        first = cell.source.lstrip().splitlines()[0] if cell.source.strip() else ""
        if cell.cell_type == "code" and first.startswith("# @inline") and "imports" not in first:
            module, names = parse_directive(first)
            code, wanted = extract(module, names)
            for name in wanted:
                if name in defined and defined[name] != module:
                    raise ValueError(f"{name} defined in both {defined[name]} and {module}")
                defined[name] = module
            modules.append(module)
            cell.source = f"# Copied from {module} by scripts/build_notebooks.py\n" + FUTURE + code.rstrip()
    imports = sorted({imp for m in dict.fromkeys(modules) for imp in external_imports(m)},
                     key=lambda s: (not s.startswith("import "), s))
    config_text = (ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")
    for cell in nb.cells:
        first = cell.source.lstrip().splitlines()[0] if cell.source.strip() else ""
        if cell.cell_type == "code" and first.startswith("# @inline imports"):
            cell.source = "# Every third-party import used by the code in this notebook\n" + "\n".join(imports)
        elif cell.cell_type == "code" and first.startswith("# @config"):
            cell.source = ("# configs/config.yaml, embedded so this notebook runs on its own\n"
                           f'CONFIG_YAML = r"""\n{config_text}"""')
    for cell in nb.cells:
        if cell.cell_type == "code" and "# @versions" in cell.source:
            cell.source = cell.source.replace(
                "# @versions", "# Versions this notebook was tested with (from pyproject.toml)\n"
                               f"TESTED_VERSIONS = {json.dumps(pinned_versions())}")
        elif cell.cell_type == "markdown" and "<<author_line>>" in cell.source:
            cell.source = cell.source.replace("<<author_line>>", author_line())
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata.pop("jupytext", None)
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out)


def reuse_outputs(previous, path: Path) -> None:
    """Put the outputs of the last execution back into the freshly built notebook.

    Valid only when every code cell is unchanged, so the saved outputs are exactly what the
    cells would produce again. Anything else needs a real execution.
    """
    nb = nbformat.read(path, as_version=4)
    new = [c for c in nb.cells if c.cell_type == "code"]
    old = [c for c in previous.cells if c.cell_type == "code"] if previous is not None else []
    if not old or any(c.get("execution_count") is None for c in old):
        raise SystemExit(f"{path.name}: there is no executed copy to take outputs from; run without --reuse-outputs")
    changed = [i + 1 for i, (a, b) in enumerate(zip(new, old)) if a.source != b.source]
    if len(new) != len(old) or changed:
        where = f"first difference in code cell {changed[0]}" if changed else f"{len(old)} code cells before, {len(new)} now"
        raise SystemExit(f"{path.name}: the code changed ({where}); run without --reuse-outputs")
    for a, b in zip(new, old):
        a["outputs"], a["execution_count"] = b["outputs"], b["execution_count"]
    nb.metadata = previous.metadata
    nbformat.write(nb, path)


def execute(path: Path, workdir: Path, env: dict[str, str], timeout: int = 3600) -> None:
    nb = nbformat.read(path, as_version=4)
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        ExecutePreprocessor(timeout=timeout, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(workdir)}})
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    nbformat.write(nb, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-execute", action="store_true")
    parser.add_argument("--fast", action="store_true", help="execute with ML_FAST=1 (smoke test only)")
    parser.add_argument("--only", default=None, help="build only notebooks whose name contains this")
    parser.add_argument("--reuse-outputs", action="store_true",
                        help="keep the outputs of the last execution (only allowed when no code cell changed)")
    args = parser.parse_args()
    for source in sorted((ROOT / "notebooks" / "_src").glob("*.py")):
        if args.only and args.only not in source.stem:
            continue
        out = ROOT / "notebooks" / f"{source.stem}.ipynb"
        previous = nbformat.read(out, as_version=4) if args.reuse_outputs and out.exists() else None
        build(source, out)
        print(f"built {out.relative_to(ROOT)}")
        if args.reuse_outputs:
            try:
                reuse_outputs(previous, out)
            except SystemExit:
                if previous is not None:
                    nbformat.write(previous, out)  # leave the executed copy as it was
                raise
            print(f"kept the saved outputs of {out.relative_to(ROOT)}: no code cell changed")
            continue
        if args.no_execute:
            continue
        run_dir = ROOT / "build" / "notebook-run" / source.stem
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True)
        env = {"OUTPUT_DIR": str(run_dir / "outputs"), "ML_FAST": "1" if args.fast else "0"}
        execute(out, run_dir, env)
        print(f"executed {out.relative_to(ROOT)}")
        metrics = run_dir / "outputs" / "metrics.json"
        if metrics.exists() and not args.fast:
            dest = ROOT / "reports" / "metrics" / "notebook" / f"{source.stem}.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(metrics, dest)
            print(f"copied {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
