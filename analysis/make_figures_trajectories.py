"""Example trajectories from the Monte Carlo: the draw crossing nearest the most likely point, two
draws at opposite ends of the 95% region, and the best trajectory without spin for reference.

    cd ../monte-carlo-penalty-kick
    .venv/bin/python ../UPC_Soccer/analysis/make_figures_trajectories.py

Runs the standard model (config.yaml) on the scenario with 10,000 draws the first time and caches
the draws in analysis/data/mc_draws.csv. Writes figures/example_trajectories.pdf.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "monte-carlo-penalty-kick"))
from penalty_mc import baseline, montecarlo as mc, physics  # noqa: E402
from penalty_mc.fit import fit_v0                           # noqa: E402
from penalty_mc.priors import load_config, theta_from_record  # noqa: E402
from penalty_mc.summarize import summarize                  # noqa: E402

FIG = HERE.parent / "figures"
CACHE = HERE / "data" / "mc_draws.csv"
N = 10_000

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Charter", "Bitstream Charter", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#b9b8b2", "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def draws(cfg):
    if CACHE.exists():
        return pd.read_csv(CACHE)
    df = mc.run(mc.job_from_config(cfg, "C"), N, n_workers=mc.default_workers(cfg))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def pick(df, kde, summary):
    """Rows: nearest the most likely point, and the two ends of the 95% region."""
    ok = df[df.outcome.isin(["on_target", "wide", "over"]) & (df.w > 0.05 * df.w.max())].copy()
    pts = ok[["y", "z"]].to_numpy()
    mode = np.asarray(summary["mode"])
    centre = ok.iloc[int(np.argmin(np.hypot(*(pts - mode).T)))]
    dens = kde.kde(pts.T)
    inner = ok[(dens >= kde.thresholds[0.95]) & (dens < kde.thresholds[0.68])]
    cov = np.asarray(summary["cov"])
    axis = np.linalg.eigh(cov)[1][:, -1]                       # long axis of the spread
    proj = (inner[["y", "z"]].to_numpy() - mode) @ axis
    return {"Most likely": centre,
            "Edge of 95% region (a)": inner.iloc[int(np.argmax(proj))],
            "Edge of 95% region (b)": inner.iloc[int(np.argmin(proj))]}


def label(rec):
    side, back = rec.spin_rps * rec.spin_side, rec.spin_rps * rec.spin_back
    turn = "right" if side < 0 else "left"
    spin = "backspin" if back >= 0 else "topspin"
    return (f"{rec.spin_rps:.1f} rev/s: sidespin {abs(side):.1f} (curls {turn}), {spin} {abs(back):.1f}; "
            f"$f_L$ = {rec.lift_scale:.2f}")


VIEW = dict(elev=20, azim=-25, focal=0.4)   # behind the goal, looking back at the kicker


def main():
    cfg = load_config()
    sc = cfg["scenario"]
    obs, times, r0 = np.array(sc["obs"]), np.array(sc["times"]), np.array(sc["r0"])
    df = draws(cfg)
    summary, kde = summarize(df, "C")
    picks = pick(df, kde, summary)

    paths = {}
    for name, rec in picks.items():
        theta = theta_from_record(rec, cfg)
        v0 = rec[["v0_x", "v0_y", "v0_z"]].to_numpy(float)
        paths[name] = (physics.trajectory(v0, theta, r0=r0, n=400), rec)
    base = theta_from_record(picks["Most likely"], cfg).with_(omega=(0.0, 0.0, 0.0), lift_scale=1.0,
                                                               drag_scale=1.0)
    f = fit_v0(obs, times, base, np.array(cfg["priors"]["sigma"]), r0=r0)
    nospin = physics.trajectory(f.v0, base, r0=r0, n=400)

    colors = {"Most likely": ORANGE, "Edge of 95% region (a)": BLUE, "Edge of 95% region (b)": AQUA}
    fig = plt.figure(figsize=(5.2, 3.0))
    ax = fig.add_axes((-0.08, 0.13, 1.16, 1.5), projection="3d", computed_zorder=False)
    W, H = 3.66, 2.44
    # goal line
    ax.plot([11, 11], [-4.5, 4.5], [0, 0], color=MUTED, lw=0.8)
    # goal frame
    ax.plot([11, 11, 11, 11], [-W, -W, W, W], [0, H, H, 0], color=INK, lw=2.2)
    for name, (p, rec) in paths.items():
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=colors[name], lw=1.8)
        ax.plot([p[-1, 0]], [p[-1, 1]], [p[-1, 2]], marker="o", ms=4, color=colors[name])
    # constant-acceleration curve through the kick point and the two observations (baseline)
    v, acc = baseline._fit_va(obs, times, r0)
    t_k = baseline.extrapolate(obs, times, r0)[0]
    tt = np.linspace(0.0, t_k, 200)
    K = r0 + np.outer(tt, v) + 0.5 * np.outer(tt ** 2, acc)
    ax.plot(K[:, 0], K[:, 1], K[:, 2], color=INK2, lw=1.2, ls="--")
    ax.plot([K[-1, 0]], [K[-1, 1]], [K[-1, 2]], marker="o", ms=3.5, mfc="white", mec=INK2)
    ax.plot([r0[0]], [r0[1]], [r0[2]], marker="o", ms=4, color=INK)
    ax.plot(obs[:, 0], obs[:, 1], obs[:, 2], marker="o", ms=4, mfc="white", mec=INK, ls="none")
    ax.set_xlim(0, 11.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_zlim(0, 2.6)
    ax.set_box_aspect((11.5, 9.0, 2.6), zoom=0.95)
    ax.view_init(**{k: v for k, v in VIEW.items() if k != "focal"})
    if VIEW.get("focal"):
        ax.set_proj_type("persp", focal_length=VIEW["focal"])
    ax.set_xlabel("$x$ (m)", labelpad=6)
    ax.set_ylabel("$y$ (m)", labelpad=2)
    ax.set_zlabel("$z$ (m)", labelpad=0)
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.set_zticks([0, 1, 2, 3])
    ax.tick_params(pad=0)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0))
        a.pane.set_edgecolor(GRID)
        a._axinfo["grid"]["color"] = GRID
        a._axinfo["grid"]["linewidth"] = 0.5

    handles = [plt.Line2D([], [], color=colors[n], lw=1.8) for n in paths]
    labels = [f"{n}: {label(rec)}" for n, (p, rec) in paths.items()]
    handles += [plt.Line2D([], [], color=INK2, lw=1.2, ls="--"),
                plt.Line2D([], [], marker="o", ms=4, mfc="white", mec=INK, ls="none")]
    labels += ["Constant acceleration through the observations (no physics)", "Observed positions"]
    fig.legend(handles, labels, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.02), fontsize=6.5)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "example_trajectories.pdf", bbox_inches=None)
    fig.savefig(FIG / "example_trajectories.png", dpi=300, bbox_inches=None)
    for name, (p, rec) in paths.items():
        print(f"{name}: cross ({rec.y:.2f}, {rec.z:.2f}) | {label(rec)} | w/max {rec.w / df.w.max():.2f}")
    print("no spin crossing:", nospin[-1].round(2), "| constant acceleration crossing:", K[-1].round(2))


if __name__ == "__main__":
    main()
