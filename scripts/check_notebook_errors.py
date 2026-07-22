"""Exit non-zero if any code cell in the given notebook has an error output.

Used by scripts/run_notebooks.ps1 as a defense-in-depth check after each
`jupyter nbconvert --execute` call (nbconvert itself should already exit
non-zero on a cell exception, but this gives an independent, explicit check
of the resulting .ipynb file).

Usage:
    python check_notebook_errors.py <path-to-notebook.ipynb>
"""
from __future__ import annotations

import sys

import nbformat


def find_errors(path: str) -> list[tuple[int, str, str]]:
    nb = nbformat.read(path, as_version=4)
    errors = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                errors.append((i, out.get("ename", "?"), out.get("evalue", "?")))
    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python check_notebook_errors.py <notebook.ipynb>", file=sys.stderr)
        return 2

    path = sys.argv[1]
    errors = find_errors(path)
    for i, ename, evalue in errors:
        print(f"[{path}] cell {i}: {ename}: {evalue}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
