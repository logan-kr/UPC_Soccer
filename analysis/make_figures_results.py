"""Figures for the Results section: the predicted crossing distribution at each sigma.

    cd ../monte-carlo-penalty-kick
    .venv/bin/python ../UPC_Soccer/analysis/make_figures_results.py

Reads the density grids written by `python -m penalty_mc.model_sweep`
(penalty_mc/outputs/model_sweep/kde_*.npz and model_sweep.csv). Writes PDFs to figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
SWEEP = HERE.parents[1] / "monte-carlo-penalty-kick" / "penalty_mc" / "outputs" / "model_sweep"
FIG = HERE.parent / "figures"
EXACT_BEST = (2.68, 1.92)                   # Section 4 best fit

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#e8f1fc", "#a9cbf3", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Charter", "Bitstream Charter", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#b9b8b2", "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")


def predicted_crossings(model="standard"):
    table = pd.read_csv(SWEEP / "model_sweep.csv")
    rows = table[table.model == model].sort_values("sigma_cm")
    fig, axes = plt.subplots(1, len(rows), figsize=(7.0, 2.6), sharey=True)
    for ax, r in zip(np.atleast_1d(axes), rows.itertuples()):
        g = np.load(SWEEP / f"kde_{model}_{r.sigma_cm / 100:g}.npz")
        dens = np.ma.masked_less(g["density"], 1e-3 * g["density"].max())
        ax.pcolormesh(g["gy"], g["gz"], dens, cmap=SEQ, shading="auto", rasterized=True)
        ax.contour(g["gy"], g["gz"], g["density"], levels=[float(g["t95"]), float(g["t68"])],
                   colors=[INK2, INK], linestyles=["--", "-"], linewidths=[0.9, 1.1])
        ax.plot([0.5, 3.66, 3.66], [2.44, 2.44, 0.0], color=INK, lw=2.2, solid_capstyle="butt")
        ax.plot(*EXACT_BEST, marker="o", ms=5, mfc="white", mec=INK, mew=0.9, ls="none", zorder=4)
        ax.plot(*g["mode"], marker="*", ms=11, mfc=ORANGE, mec="white", mew=0.8, ls="none", zorder=5)
        ax.set_xlim(4.4, 1.0)                      # kicker's view: +y on the left
        ax.set_ylim(1.0, 3.0)
        ax.set_aspect("equal")
        ax.set_title(f"$\\sigma$ = {r.sigma_cm:g} cm", pad=4)
        ax.set_xlabel("Lateral position, $y$ (m)")
    np.atleast_1d(axes)[0].set_ylabel("Height, $z$ (m)")
    a0 = np.atleast_1d(axes)[0]
    a0.plot([], [], color=INK, lw=1.1, label="68% region")
    a0.plot([], [], color=INK2, lw=0.9, ls="--", label="95% region")
    a0.plot([], [], marker="*", ms=9, mfc=ORANGE, mec="white", ls="none", label="Most likely point")
    a0.plot([], [], marker="o", ms=5, mfc="white", mec=INK, ls="none", label="Exact best fit (Sec. 4)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    save(fig, "predicted_crossings")


if __name__ == "__main__":
    predicted_crossings()
