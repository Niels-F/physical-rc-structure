"""
Run the whole notebook from the command line.

The files in src/ are notebook cells: each one uses functions defined by the cells
before it and never imports from the others. This script reproduces that behaviour
by executing every file, in notebook order, in one shared namespace.

    python run_all.py                     # load every cell and run all milestones
    python run_all.py --only structure    # load every cell, run milestones of one cell
    python run_all.py --quiet             # load every cell, run no milestone
    python run_all.py --then "rows = load_rows('data/exploration.jsonl'); explore_analysis(rows)"

Each cell prints its milestones as [OK] / [FAIL] lines (messages are in French).
The interactive display (live_time.py) is loaded but never started from here.
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")

# Notebook order (step number in the report -> file)
ORDER = [
    "complexity_vector",  # 1
    "reservoir",          # 2a
    "substrates",         # 3
    "vector_space",       # 4
    "sweep",              # 5
    "coulomb",            # 6
    "cross",              # 7
    "bench_check",        # 8
    "massspring",         # 9
    "geometry",           # 10
    "forecast",           # 11
    "structure",          # 12
    "confirm",            # 13
    "linear_mc",          # 14
    "features",           # 15
    "explore",            # 16
    "confirm2",           # 17
    "live_time",          # 2b, always last
]
NO_MILESTONES = {"live_time"}  # needs an interactive matplotlib backend


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated cells whose milestones are run (default: all)")
    ap.add_argument("--quiet", action="store_true", help="load the cells without running any milestone")
    ap.add_argument("--then", default="", help="Python code executed after loading, in the same namespace")
    args = ap.parse_args()

    if args.quiet:
        run = set()
    elif args.only:
        run = set(args.only.split(","))
        unknown = run - set(ORDER)
        if unknown:
            sys.exit(f"unknown cell(s): {', '.join(sorted(unknown))}")
    else:
        run = set(ORDER) - NO_MILESTONES

    os.environ.setdefault("MPLBACKEND", "Agg")  # figures are saved, not shown
    ns = {}
    t0 = time.perf_counter()
    for name in ORDER:
        with open(os.path.join(SRC, name + ".py"), encoding="utf-8") as f:
            lines = f.read().split("\n")
        # In the notebook the cells share one namespace, so cross-file imports are dropped.
        src = "\n".join(l for l in lines if not l.startswith("from complexity_vector import"))
        ns["__name__"] = "__main__" if name in run else "cell"
        if name in run:
            print(f"\n########## {name}.py", flush=True)
        exec(compile(src, name + ".py", "exec"), ns)
    if args.then:
        print(f"\n########## {args.then}", flush=True)
        exec(args.then, ns)
    print(f"\nDone in {time.perf_counter() - t0:.0f} s.")


if __name__ == "__main__":
    main()
