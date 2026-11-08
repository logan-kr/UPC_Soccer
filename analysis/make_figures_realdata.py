"""Figure for the real-kick test (Section 6.6): one kick's speed trace with the test's time marks,
and the misses of all 17 kicks against the predicted two-standard-deviation range.

    cd ../monte-carlo-penalty-kick
    .venv/bin/python -m penalty_mc.realdata all full-game.gz     # once: kicks from the tracking data
    .venv/bin/python -m penalty_mc.realtest --plane --sigma 0.008,0.008,0.025
    .venv/bin/python ../UPC_Soccer/analysis/make_figures_realdata.py

Writes figures/real_kicks.pdf.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1] / "monte-carlo-penalty-kick"
TRACKS = CODE / "runs" / "realdata" / "kick_tracks.csv.gz"
RESULTS = CODE / "penalty_mc" / "outputs" / "realtest" / "realtest_plane.csv"
FIG = HERE.parent / "figures"
EXAMPLE = 115
T_START, HORIZON = 0.02, 0.38

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


def main():
    tr = pd.read_csv(TRACKS)
    tr = tr[tr.kick == EXAMPLE].sort_values("t")
    r = pd.read_csv(RESULTS)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw=dict(width_ratios=[1.35, 1.0]))

    # speed from the positions (straight-line fit over +/-5 ms) and as reported by the tracker
    tt, P = tr.t.to_numpy(), tr[["x", "y", "z"]].to_numpy()
    tc = tt[(tt > -0.045) & (tt < 0.445)]
    vp = [np.linalg.norm([np.polyfit(tt[(tt >= c - 0.005) & (tt <= c + 0.005)],
                                     P[(tt >= c - 0.005) & (tt <= c + 0.005), i], 1)[0] for i in range(3)])
          for c in tc]
    a.plot(tr.t, tr.speed, color=MUTED, lw=1.0, label="reported by the tracker")
    a.plot(tc, vp, color=BLUE, lw=1.2, label="from the positions")
    a.legend(loc="lower right", fontsize=7.5)
    marks = [(0.0, "strike"), (T_START, "start"), (T_START + 0.05, "sightings"),
             (T_START + 0.10, None), (T_START + HORIZON, "plane")]
    top = tr.speed.max() * 1.12
    for t, lbl in marks:
        a.axvline(t, color=INK2 if lbl != "strike" else ORANGE, lw=0.8, ls="--" if lbl != "strike" else "-")
        if lbl:
            a.text(t + 0.006, top, lbl, fontsize=7.5, color=INK2, rotation=90, va="top", ha="left")
    a.set_xlim(-0.05, 0.45)
    a.set_ylim(0, top * 1.02)
    a.set_xlabel("Time after the peak acceleration (s)")
    a.set_ylabel("Ball speed (m/s)")
    a.set_title(f"(a) One kick in the tracking data", loc="left")

    dy = 100 * (r.pred_y - r.true_y)
    dz = 100 * (r.pred_z - r.true_z)
    b.add_patch(Ellipse((0, 0), 4 * 100 * r.sd_y.median(), 4 * 100 * r.sd_z.median(), fill=False,
                        ec=INK2, lw=0.9, ls="--"))
    b.scatter(dy, dz, s=16, color=ORANGE, edgecolor="white", lw=0.5, zorder=3)
    b.axhline(0, color=MUTED, lw=0.6)
    b.axvline(0, color=MUTED, lw=0.6)
    lim = 70
    b.set_xlim(-lim, lim)
    b.set_ylim(-lim, lim)
    b.set_aspect("equal")
    b.set_xlabel("Sideways miss (cm)")
    b.set_ylabel("Height miss (cm)")
    b.set_title("(b) Misses of all 17 kicks", loc="left")
    b.text(lim * 0.95, -lim * 0.9, "dashed: typical predicted\n2-standard-deviation range", fontsize=7,
           color=INK2, ha="right", va="bottom")
    fig.tight_layout(w_pad=2.0)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "real_kicks.pdf")
    fig.savefig(FIG / "real_kicks.png", dpi=300)
    print("wrote", FIG / "real_kicks.pdf")


if __name__ == "__main__":
    main()
