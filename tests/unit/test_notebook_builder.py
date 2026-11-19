import sys

import nbformat


def test_extract_copies_definitions_verbatim(root):
    sys.path.insert(0, str(root / "scripts"))
    from build_notebooks import extract

    code, names = extract("hotel_cancellation.rowlevel", ["is_missing", "as_number"])
    assert names == ["is_missing", "as_number"]
    assert code.startswith("def is_missing(") and "def as_number(" in code


def test_built_notebooks_import_nothing_from_the_repo(root):
    for path in (root / "notebooks").glob("*.ipynb"):
        nb = nbformat.read(path, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "code":
                for line in cell.source.splitlines():
                    stripped = line.strip()
                    assert not stripped.startswith(("from hotel_cancellation", "import hotel_cancellation")), path
                    assert not stripped.startswith("from ."), path
