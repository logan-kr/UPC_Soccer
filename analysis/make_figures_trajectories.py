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
from penalty_mc import montecarlo as mc, physics            # noqa: E402
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
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.0), gridspec_kw=dict(height_ratios=[1.0, 0.8]))
    (top, side), (dy, dz) = ax

    def ref(p, i):
        return np.interp(p[:, 0], nospin[:, 0], nospin[:, i])

    for a, i in ((top, 1), (side, 2)):
        a.plot(nospin[:, 0], nospin[:, i], color=MUTED, lw=1.0, ls="--")
    for name, (p, rec) in paths.items():
        top.plot(p[:, 0], p[:, 1], color=colors[name], lw=1.4)
        side.plot(p[:, 0], p[:, 2], color=colors[name], lw=1.4)
        dy.plot(p[:, 0], 100 * (p[:, 1] - ref(p, 1)), color=colors[name], lw=1.4)
        dz.plot(p[:, 0], 100 * (p[:, 2] - ref(p, 2)), color=colors[name], lw=1.4)
    for a, i in ((top, 1), (side, 2)):
        a.plot([0.0], [r0[i]], marker="o", ms=4, color=INK, ls="none")
        a.plot(obs[:, 0], obs[:, i], marker="o", ms=4, mfc="white", mec=INK, ls="none")
    for a, i in ((dy, 1), (dz, 2)):
        a.plot(obs[:, 0], 100 * (obs[:, i] - ref(obs, i)), marker="o", ms=4, mfc="white", mec=INK, ls="none")
        a.axhline(0, color=MUTED, lw=1.0, ls="--")
    top.plot([11, 11], [-3.66, 3.66], color=INK, lw=2.2)          # goal mouth, seen from above
    side.plot([11, 11], [0, 2.44], color=INK, lw=2.2)              # post and crossbar height
    top.set_ylim(-0.5, 3.9)
    side.set_ylim(0, 2.8)
    top.set_ylabel("Lateral position, $y$ (m)")
    side.set_ylabel("Height, $z$ (m)")
    dy.set_ylabel("Sideways shift (cm)")
    dz.set_ylabel("Height shift (cm)")
    top.set_title("Top view")
    side.set_title("Side view")
    for a in ax.flat:
        a.set_xlim(0, 11.3)
    for a in (dy, dz):
        a.set_xlabel("Distance toward goal, $x$ (m)")
    top.text(10.85, 3.66, "post", fontsize=7.5, color=INK2, ha="right", va="center")
    side.text(10.85, 2.44, "crossbar", fontsize=7.5, color=INK2, ha="right", va="center")
    dy.text(0.3, dy.get_ylim()[0] * 0.8, "relative to the path without spin;\nnegative is toward the kicker's right",
            fontsize=7.5, color=INK2, va="bottom")
    dz.text(0.3, dz.get_ylim()[1] * 0.8, "relative to the path without spin", fontsize=7.5, color=INK2, va="top")

    handles = [plt.Line2D([], [], color=colors[n], lw=1.6) for n in paths]
    labels = [f"{n}: {label(rec)}" for n, (p, rec) in paths.items()]
    handles += [plt.Line2D([], [], color=MUTED, lw=1.0, ls="--"),
                plt.Line2D([], [], marker="o", ms=4, mfc="white", mec=INK, ls="none")]
    labels += ["No spin (best fit, does not match the data)", "Observed positions"]
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=1, bbox_to_anchor=(0.5, 0.0), fontsize=7.5)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "example_trajectories.pdf")
    fig.savefig(FIG / "example_trajectories.png", dpi=300)
    for name, (p, rec) in paths.items():
        print(f"{name}: cross ({rec.y:.2f}, {rec.z:.2f}) | {label(rec)} | w/max {rec.w / df.w.max():.2f}")
    print("no spin crossing:", nospin[-1].round(2))


if __name__ == "__main__":
    main()
