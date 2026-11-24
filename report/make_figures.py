"""
Figures of the short report, drawn from the saved data (no simulation is re-run).

    python report/make_figures.py      # from the repository root

Reads data/linear_mc.csv, data/confirmation.jsonl; writes figures/short_*.png.
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, FIG = os.path.join(ROOT, "data"), os.path.join(ROOT, "figures")

# Family names are stored in French in the data files.
NAMES = {"ESN": "ESN", "Ressorts": "Spring networks", "Chaîne": "Mass chains",
         "Retard": "Delay node (new)", "Coulomb répulsif": "Repulsive Coulomb (new)"}
ORDER = ["ESN", "Ressorts", "Chaîne", "Retard", "Coulomb répulsif"]
# Okabe-Ito colours (colour-blind safe), plus a distinct marker per family
COLOR = {"ESN": "#0072B2", "Ressorts": "#E69F00", "Chaîne": "#009E73",
         "Retard": "#D55E00", "Coulomb répulsif": "#CC79A7"}
MARK = {"ESN": "o", "Ressorts": "^", "Chaîne": "s", "Retard": "D", "Coulomb répulsif": "X"}
LINEAR_BAND = (0.42, 0.45)   # NRMSE of the linear AR baseline at horizons 10 and 20

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight"})


def load_conf():
    with open(os.path.join(DATA, "confirmation.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def fig_linear_mc():
    fam_of = {"ESN": "ESN", "chain_a": "Chaîne", "springs": "Ressorts"}
    with open(os.path.join(DATA, "linear_mc.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for code, fam in fam_of.items():
        r = [x for x in rows if x["family"] == code]
        nl = np.array([float(x["static_nonlinearity"]) for x in r])
        ratio = np.array([float(x["mc_measured"]) / float(x["mc_theory"]) for x in r])
        ax.scatter(nl, ratio, s=26, marker=MARK[fam], color=COLOR[fam], edgecolor="white",
                   linewidth=0.6, label=NAMES[fam], zorder=3)
    ax.axvline(0.02, color="0.35", lw=0.9, ls="--")
    ax.text(0.019, 0.47, "nearly linear\n(criterion P1)", ha="right", va="bottom", fontsize=7.5, color="0.3")
    ax.axhline(1.0, color="0.35", lw=0.9)
    ax.set(xscale="log", ylim=(0.4, 1.04), xlabel="static non-linearity at the operating point",
           ylabel="measured / predicted\nmemory capacity")
    ax.legend(loc="lower left", fontsize=7.5)
    fig.savefig(os.path.join(FIG, "short_linear_mc.png"))
    plt.close(fig)


def fig_filter(conf):
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    rng = np.random.default_rng(0)
    ax.axhspan(*LINEAR_BAND, color="0.88", zorder=0)
    ax.text(4.45, np.mean(LINEAR_BAND), "linear\nbaseline", va="center", fontsize=7, color="0.35")
    for i, fam in enumerate(ORDER):
        r = [x for x in conf if x["family"] == fam]
        for adm, dx in ((True, -0.17), (False, 0.17)):
            y = np.array([x["perf"] for x in r if x["admis"] == adm])
            if not len(y):
                continue
            xs = i + dx + rng.uniform(-0.09, 0.09, len(y))
            if adm:
                ax.scatter(xs, y, s=26, marker=MARK[fam], color=COLOR[fam], edgecolor="white", linewidth=0.5, zorder=3)
            else:
                ax.scatter(xs, y, s=26, marker=MARK[fam], facecolor="none", edgecolor=COLOR[fam], linewidth=0.9, zorder=3)
    ax.set_xticks(range(len(ORDER)), [NAMES[f].replace(" (", "\n(").replace("Spring n", "Spring\nn").replace("Repulsive C", "Repulsive\nC") for f in ORDER])
    ax.set(ylabel="forecast error (NRMSE)", ylim=(0, 1.05), xlim=(-0.6, 4.9))
    ax.grid(axis="x", visible=False)
    ax.scatter([], [], s=18, marker="o", color="0.3", label="admitted by the filter (left)")
    ax.scatter([], [], s=18, marker="o", facecolor="none", edgecolor="0.3", label="rejected (right)")
    ax.legend(loc="upper left", fontsize=7.5, ncol=2)
    fig.savefig(os.path.join(FIG, "short_filter.png"))
    plt.close(fig)


def fig_quantitative(conf):
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.8))
    for fam in ORDER:
        r = [x for x in conf if x["family"] == fam and x["admis"]]
        if r:
            a.scatter([x["nl"] + 1e-4 for x in r], [x["perf"] for x in r], s=26, marker=MARK[fam],
                      color=COLOR[fam], edgecolor="white", linewidth=0.5, label=NAMES[fam], zorder=3)
    a.axhspan(*LINEAR_BAND, color="0.88", zorder=0)
    a.axvline(0.01, color="0.35", lw=0.9, ls="--")
    a.set(xscale="log", ylim=(0, 1.05), xlabel="static non-linearity", ylabel="forecast error (NRMSE)",
          title="(a) admitted systems", )
    a.legend(fontsize=6.5, loc="upper left")
    s = [x for x in conf if x["family"] == "Ressorts"]
    for adm in (True, False):
        r = [x for x in s if x["admis"] == adm]
        kw = dict(color=COLOR["Ressorts"], edgecolor="white", linewidth=0.5) if adm else \
             dict(facecolor="none", edgecolor=COLOR["Ressorts"], linewidth=0.9)
        b.scatter([x["mc_lin"] for x in r], [x["perf"] for x in r], s=30, marker="^", zorder=3,
                  label="admitted" if adm else "rejected", **kw)
    b.set(ylim=(0, 1.05), xlabel="memory capacity predicted from the equations",
          title="(b) 40 new spring networks")
    b.legend(fontsize=7, loc="upper right")
    for ax in (a, b):
        ax.title.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "short_quantitative.png"))
    plt.close(fig)


if __name__ == "__main__":
    conf = load_conf()
    fig_linear_mc()
    fig_filter(conf)
    fig_quantitative(conf)
    print("figures written to", FIG)
