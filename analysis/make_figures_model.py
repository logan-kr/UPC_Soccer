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

# Custom palette strictly defined from user specifications & param_sensitivity:
COLOR_MINT     = "#d7fff7"   # Swatch 1: light mint ice (ball body)
COLOR_SLATE    = "#82a0b9"   # Swatch 2: slate / steel blue (velocity vector v)
COLOR_NAVY     = "#2e457d"   # Swatch 3: classic deep navy (Magnus lift FL, ball outline)
COLOR_INDIGO   = "#272652"   # Swatch 4: midnight indigo (Drag force FD)
COLOR_OBSIDIAN = "#111227"   # Swatch 5: obsidian ink (Gravity mg, spin axis omega)


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
    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    ax.set_aspect("equal")
    ax.axis("off")

    # Ball: crisp mint fill with navy border and subtle inner white highlight rim
    ball = Circle((0, 0), 1.0, fc=COLOR_MINT, ec=COLOR_NAVY, lw=1.8, zorder=2)
    ax.add_patch(ball)
    rim = Circle((0, 0), 0.965, fc="none", ec="#ffffff", lw=1.2, alpha=0.85, zorder=3)
    ax.add_patch(rim)

    def arrow(x0, y0, x1, y1, c, lw=2.3, mut=15):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=mut,
                                     color=c, lw=lw, shrinkA=0, shrinkB=0,
                                     capstyle="round", joinstyle="miter", zorder=4))

    theta = math.radians(22)
    d = np.array([math.cos(theta), math.sin(theta)])   # velocity direction
    n = np.array([-d[1], d[0]])                        # perpendicular (lift direction)

    # 1. Velocity v (Slate)
    arrow(*(1.06 * d), *(3.05 * d), COLOR_SLATE, lw=2.3, mut=15)
    ax.text(3.18 * d[0] + 0.05, 3.18 * d[1] + 0.02, r"$\mathbf{v}$",
            color=COLOR_SLATE, fontsize=12, fontweight="bold", va="center", ha="left")

    # 2. Lift F_L (Navy)
    arrow(*(1.06 * n), *(2.55 * n), COLOR_NAVY, lw=2.3, mut=15)
    ax.text(2.65 * n[0] - 0.10, 2.65 * n[1] + 0.08, r"$\mathbf{F}_L$",
            color=COLOR_NAVY, fontsize=12, fontweight="bold", va="bottom", ha="right")

    # 3. Drag F_D (Indigo)
    arrow(*(-1.06 * d), *(-2.70 * d), COLOR_INDIGO, lw=2.3, mut=15)
    ax.text(-2.82 * d[0] - 0.22, -2.82 * d[1] - 0.08, r"$\mathbf{F}_D$",
            color=COLOR_INDIGO, fontsize=12, fontweight="bold", va="top", ha="right")

    # 4. Gravity m g (Obsidian)
    arrow(0, -1.06, 0, -2.40, COLOR_OBSIDIAN, lw=2.3, mut=15)
    ax.text(0.14, -2.38, r"$m\mathbf{g}$",
            color=COLOR_OBSIDIAN, fontsize=12, fontweight="bold", va="center", ha="left")

    # 5. Spin axis out of the page (circle with dot in Obsidian)
    ax.plot([0], [0], marker="o", ms=13.0, mfc="#ffffff", mec=COLOR_OBSIDIAN, mew=1.6, zorder=5)
    ax.plot([0], [0], marker="o", ms=4.5, mfc=COLOR_OBSIDIAN, mec=COLOR_OBSIDIAN, zorder=6)
    ax.text(0.22, 0.17, r"$\boldsymbol{\omega}$",
            color=COLOR_OBSIDIAN, fontsize=12, fontweight="bold", zorder=7)

    # 6. Explanatory text annotation
    ax.text(-3.35, -1.70, "Spin axis out of the page\n(backspin)",
            fontsize=8.0, color=COLOR_NAVY, linespacing=1.25, va="top", ha="left")

    ax.set_xlim(-3.6, 3.8)
    ax.set_ylim(-2.65, 2.65)
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
