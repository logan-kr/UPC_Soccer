"""Figures for the model section and the well-constrained (fixed-parameter) case.

    cd ../monte-carlo-penalty-kick
    .venv/bin/python ../UPC_Soccer/analysis/make_figures_model.py

Needs analysis/data/feasible_grid.csv from feasible_spins.py. Writes PDFs to figures/.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "monte-carlo-penalty-kick"))
from penalty_mc.physics import KLBall      # noqa: E402

FIG = HERE.parent / "figures"
DATA = HERE / "data" / "feasible_grid.csv"
TOL_MM = 5.0
PRIOR_RPS, PRIOR_DEG = 9.3, 110.0          # Monte Carlo spin prior (config.yaml), for reference

# palette: reference categorical slots 1-3, ink, a single-hue sequential ramp
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
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
    fig.savefig(FIG / f"{name}.png", dpi=200)   # for quick previews only
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")




def hcolorbar(fig, mappable, ax, label, pad=0.16, **kw):
    """Horizontal colorbar under the axes, with a horizontal label."""
    cb = fig.colorbar(mappable, ax=ax, orientation="horizontal", fraction=0.05, pad=pad, aspect=35, **kw)
    cb.set_label(label)
    cb.outline.set_visible(False)
    return cb


# ------------------------------------------------------------------ forces schematic
def forces():
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1.0, fc="#f3f2ee", ec=INK2, lw=1.2))

    def arrow(x0, y0, x1, y1, c, lw=1.8):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                                     color=c, lw=lw, shrinkA=0, shrinkB=0))

    d = np.array([math.cos(math.radians(20)), math.sin(math.radians(20))])   # velocity direction
    n = np.array([-d[1], d[0]])                                              # perpendicular (lift)
    arrow(*(1.05 * d), *(3.0 * d), BLUE)
    ax.text(*(3.1 * d + [0.05, 0.05]), r"$\mathbf{v}$", color=BLUE, fontsize=11)
    arrow(*(-1.05 * d), *(-2.6 * d), ORANGE)
    ax.text(*(-2.75 * d + [-0.55, -0.15]), r"$\mathbf{F}_D$", color=ORANGE, fontsize=11)
    arrow(*(1.05 * n), *(2.4 * n), AQUA)
    ax.text(*(2.5 * n + [-0.2, 0.05]), r"$\mathbf{F}_L$", color=AQUA, fontsize=11)
    arrow(0, -1.05, 0, -2.3, INK2)
    ax.text(0.12, -2.25, r"$m\mathbf{g}$", color=INK2, fontsize=11)
    ax.plot([0], [0], marker="o", ms=5, color=INK)          # spin axis out of the page
    ax.plot([0], [0], marker="o", ms=11, mfc="none", mec=INK, mew=1.0)
    ax.text(0.18, 0.18, r"$\boldsymbol{\omega}$", fontsize=11)
    ax.text(-3.3, -1.55, "Spin axis out of the page\n(backspin)", fontsize=7.5,
            color=INK2, va="top")
    ax.set_xlim(-3.4, 3.6)
    ax.set_ylim(-2.5, 2.6)
    save(fig, "forces")


# ------------------------------------------------------------------ coefficient curves
def coefficients():
    ball = KLBall()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.6))
    v = np.linspace(5, 40, 400)
    for S, c in zip((0.0, 0.1, 0.2), (BLUE, ORANGE, AQUA)):
        a.plot(v, ball.cd(v, S), color=c, lw=1.8, label=rf"$S_\perp = {S:g}$")
    a.axvspan(26, 33, color=GRID, alpha=0.6, lw=0)
    a.text(29.5, 0.115, "Penalty\nspeeds", ha="center", va="bottom", fontsize=7.5, color=INK2)
    a.set_xlabel(r"Speed, $|\mathbf{v}|$ (m/s)")
    a.set_ylabel(r"Drag coefficient, $C_D$")
    a.set_title("(a)", loc="left")
    a.set_ylim(0.1, 0.55)
    a.legend(loc="upper right")

    S = np.linspace(0, 0.4, 400)
    b.plot(S, ball.cl(30.0, S), color=BLUE, lw=1.8, label="Model at 30 m/s")
    b.plot([0.06, 0.19, 0.31], [0.14, 0.29, 0.33], ls="none", marker="o", ms=6, mfc=ORANGE, mec="white",
           mew=1.0, label="Calibration values\n(Kiratidis and Leinweber, 2018)")
    b.axvspan(0, 0.26, color=GRID, alpha=0.6, lw=0)
    b.text(0.13, 0.375, "Measured kicks", ha="center", fontsize=7.5, color=INK2)
    b.set_xlabel(r"Perpendicular spin parameter, $S_\perp$")
    b.set_ylabel(r"Lift coefficient, $C_L$")
    b.set_title("(b)", loc="left")
    b.set_ylim(0, 0.4)
    b.legend(loc="lower right")
    fig.tight_layout(w_pad=2.0)
    save(fig, "coefficients")


# ------------------------------------------------------------------ fixed-parameter case
def prior_outline(ax):
    th = np.radians(np.linspace(-PRIOR_DEG, PRIOR_DEG, 200))
    xs = np.r_[0, PRIOR_RPS * np.cos(th), 0]
    ys = np.r_[0, PRIOR_RPS * np.sin(th), 0]
    ax.plot(xs, ys, color=INK2, lw=1.0, ls="--")


def spin_map(df):
    g = df.pivot(index="side_rps", columns="back_rps", values="max_miss_mm")
    B, S = np.meshgrid(g.columns.to_numpy(), g.index.to_numpy())
    Z = g.to_numpy()
    fig, ax = plt.subplots(figsize=(4.4, 5.0))
    pc = ax.pcolormesh(B, S, np.clip(Z, 0, 20), cmap=SEQ.reversed(), vmin=0, vmax=20, shading="nearest",
                       rasterized=True)
    ax.contour(B, S, Z, levels=[TOL_MM], colors=[ORANGE], linewidths=1.8)
    prior_outline(ax)
    best = df.loc[df.max_miss_mm.idxmin()]
    ax.plot(best.back_rps, best.side_rps, marker="*", ms=13, mfc=ORANGE, mec="white", mew=1.0, ls="none")
    ax.plot(0, 0, marker="o", ms=6, mfc="white", mec=INK, mew=1.0, ls="none")
    ax.annotate("Best fit", (best.back_rps, best.side_rps), xytext=(12, -46), textcoords="offset points",
                fontsize=7.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.annotate("No spin", (0, 0), xytext=(-40, 14), textcoords="offset points", fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.annotate("Measured\nkicks", (6.5, 4.5), xytext=(14, 30), textcoords="offset points", fontsize=7.5,
                color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.text(-12, -31, "5 mm boundary", color=ORANGE, fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-30, 30)
    ax.set_ylim(-36, 24)
    ax.grid(False)
    ax.set_xlabel("Backspin, $\\omega_b$ (rev/s)")
    ax.set_ylabel("Sidespin, $\\omega_s$ (rev/s)")
    hcolorbar(fig, pc, ax, "Largest coordinate miss, $\\delta_{\\max}$ (mm)", extend="max")
    save(fig, "spin_map")


def goal_crossings(df):
    ok = df[df.max_miss_mm <= TOL_MM].copy()
    ok["rate"] = np.hypot(ok.back_rps, ok.side_rps)
    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    ax.plot([1.0, 3.66, 3.66], [2.44, 2.44, 0.0], color=INK, lw=2.5, solid_capstyle="butt")   # bar and post
    ax.axhline(0, color=INK2, lw=0.8)
    sc = ax.scatter(ok.y_goal, ok.z_goal, c=ok.rate, cmap=SEQ, vmin=5, vmax=30, s=6, lw=0)
    zero = df[(df.back_rps == 0) & (df.side_rps == 0)].iloc[0]
    best = df.loc[df.max_miss_mm.idxmin()]
    ax.plot(zero.y_goal, zero.z_goal, marker="o", ms=7, mfc="white", mec=INK, mew=1.0, ls="none")
    ax.annotate("No spin\n($\\delta_{\\max}$ = 11.5 mm)", (zero.y_goal, zero.z_goal), xytext=(10, -48),
                textcoords="offset points", fontsize=7.5, ha="left",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.plot(best.y_goal, best.z_goal, marker="*", ms=13, mfc=ORANGE, mec="white", mew=1.0, ls="none")
    ax.annotate("Best fit", (best.y_goal, best.z_goal), xytext=(20, -26), textcoords="offset points",
                fontsize=7.5, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.text(3.72, 1.3, "Post", fontsize=7.5, color=INK2, ha="right")
    ax.text(1.4, 2.5, "Crossbar", fontsize=7.5, color=INK2)
    ax.set_xlim(4.0, 1.0)                          # kicker's view: +y on the left
    ax.set_ylim(1.2, 2.8)
    ax.set_aspect("equal")
    ax.set_xlabel("Lateral position, $y$ (m)")
    ax.set_ylabel("Height, $z$ (m)")
    hcolorbar(fig, sc, ax, "Spin rate, $|\\boldsymbol{\\omega}_0|$ (rev/s)", pad=0.24)
    save(fig, "goal_crossings")


def residuals(df):
    cols = ["res_x1_mm", "res_y1_mm", "res_z1_mm", "res_x2_mm", "res_y2_mm", "res_z2_mm"]
    labels = [r"$x(t_1)$", r"$y(t_1)$", r"$z(t_1)$", r"$x(t_2)$", r"$y(t_2)$", r"$z(t_2)$"]
    zero = df[(df.back_rps == 0) & (df.side_rps == 0)].iloc[0]
    best = df.loc[df.max_miss_mm.idxmin()]
    x = np.arange(6)
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    ax.axhspan(-TOL_MM, TOL_MM, color=GRID, alpha=0.7, lw=0)
    ax.text(-0.45, 4.3, "Rounding bound, ±5 mm", fontsize=7.5, color=INK2, ha="left", va="top")
    ax.bar(x - w / 2 - 0.01, zero[cols].to_numpy(float), w, color=MUTED, label="No spin")
    ax.bar(x + w / 2 + 0.01, best[cols].to_numpy(float), w, color=BLUE,
           label=f"Best fit ({math.hypot(best.back_rps, best.side_rps):.1f} rev/s)")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Residual (mm)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right")
    save(fig, "residuals")


if __name__ == "__main__":
    df = pd.read_csv(DATA)
    forces()
    coefficients()
    spin_map(df)
    goal_crossings(df)
    residuals(df)
