"""
Build notebook/reservoir_structure.ipynb from the files in src/ (one cell per file,
in notebook order). Run from the repository root:  python tools/build_notebook.py
The .py files in src/ are the reference; the notebook is regenerated from them.
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELLS = [  # (step, file, what it does)
    ("1", "complexity_vector", "complexity vector (8 axes)"),
    ("2a", "reservoir", "leaky ESN, two-copy protocol"),
    ("3", "substrates", "common substrate interface and technical check"),
    ("4", "vector_space", "fixed normalisation, parallel coordinates"),
    ("5", "sweep", "benchmark (memory capacity, NARMA-10), ESN sweep"),
    ("6", "coulomb", "charges in a harmonic well (Coulomb)"),
    ("7", "cross", "transfer test ESN -> Coulomb (about 10 min)"),
    ("8", "bench_check", "check of the corrected benchmark"),
    ("9", "massspring", "mass-spring networks"),
    ("10", "geometry", "geometries as hidden parameters (about 5 min)"),
    ("11", "forecast", "forecasting benchmark, linear solvability rule"),
    ("12", "structure", "structural filter Q1-Q4"),
    ("13", "confirm", "mass chains, first confirmation of the filter"),
    ("14", "linear_mc", "memory capacity from the linearisation"),
    ("15", "features", "graded structural profile, two new substrates"),
    ("16", "explore", "exploration on already-seen systems"),
    ("17", "confirm2", "confirmation on 90 unseen systems"),
    ("2b", "live_time", "live display (needs %matplotlib widget)"),
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def main():
    cells = [md(
        "# Can the equations of a physical reservoir predict its computing capacity?\n\n"
        "> **Work in progress, not peer reviewed.** See the README.\n\n"
        "Run the cells in order: each cell uses functions defined by the cells before it. "
        "Each cell prints its debugging milestones as `[OK]` / `[FAIL]` lines "
        "(comments and messages are in French).\n\n"
        "Generated from `src/` by `tools/build_notebook.py`.")]
    for step, name, what in CELLS:
        with open(os.path.join(ROOT, "src", name + ".py"), encoding="utf-8") as f:
            src = "\n".join(l for l in f.read().split("\n") if not l.startswith("from complexity_vector import"))
        cells += [md(f"## Step {step} — `{name}.py`\n\n{what}"), code(src.rstrip() + "\n")]
    cells += [md("## Re-analyse the saved runs without re-simulating\n\n"
                 "The exploration and confirmation runs (about 25 min each) are saved in `data/`."),
              code('explo = load_rows("../data/exploration.jsonl")\n'
                   'frozen = explore_analysis(explo)\n'
                   'conf = load_rows("../data/confirmation.jsonl")\n'
                   'criteria = confirm2_analysis(explo, conf)\n')]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out = os.path.join(ROOT, "notebook", "reservoir_structure.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("written:", out, f"({len(cells)} cells)")


if __name__ == "__main__":
    main()
