"""Well-constrained case: which spins let the model pass within the rounding of both sightings?

Air density, ball mass and spin-decay length are held fixed. For every (backspin, sidespin)
on a grid (screw spin = 0), the launch velocity v0 is fitted to the two sightings by least
squares, then adjusted to minimise the WORST of the six coordinate misses (a small linear
program on the fit's Jacobian, checked against a full re-integration). A spin is consistent
with the data if that worst miss is at most 5 mm, half the 1 cm rounding of the sightings.
Each grid point's trajectory is then flown to the goal plane.

Uses the model code in ../monte-carlo-penalty-kick (not modified). Run from that repo's venv:

    cd ../monte-carlo-penalty-kick
    .venv/bin/python ../UPC_Soccer/analysis/feasible_spins.py [--step 0.5] [--lim 40]

Writes analysis/data/feasible_grid.csv next to this script.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

MODEL = Path(__file__).resolve().parents[2] / "monte-carlo-penalty-kick"
sys.path.insert(0, str(MODEL))

from penalty_mc import physics                    # noqa: E402
from penalty_mc.best_fit import Problem           # noqa: E402
from penalty_mc.priors import load_config         # noqa: E402

TOL_MM = 5.0
OUT = Path(__file__).resolve().parent / "data" / "feasible_grid.csv"


def minimax_v0(pr: Problem, v0, back, side):
    """v0 minimising the largest |residual| (mm), from a least-squares start, via an LP on the
    linearised residuals, then re-evaluated with a full integration."""
    sigma_mm = pr.sigma[0] * 1000.0
    pos, jac = physics.positions_and_jacobian(v0, pr.params(back, side), pr.times, r0=pr.r0)
    r = ((pos - pr.obs) * 1000.0).ravel()          # mm
    J = (jac * 1000.0).reshape(-1, 3)               # mm per (m/s)
    # variables (d_vx, d_vy, d_vz, t): minimise t subject to -t <= r + J d <= t
    c = np.r_[0.0, 0.0, 0.0, 1.0]
    ones = np.ones((len(r), 1))
    A = np.vstack([np.hstack([J, -ones]), np.hstack([-J, -ones])])
    b = np.r_[-r, r]
    lp = linprog(c, A_ub=A, b_ub=b, bounds=[(-1, 1)] * 3 + [(0, None)], method="highs")
    v_mm = v0 + lp.x[:3] if lp.success else v0
    r_mm = pr.residuals(v_mm, back, side) * sigma_mm
    return v_mm, r_mm


def work(args):
    cfg, rows = args
    pr = Problem(cfg)
    out = []
    for back, side in rows:
        f = pr.fit_v0(back, side)
        if not f.success:
            continue
        r_ls = pr.residuals(f.v0, back, side) * pr.sigma[0] * 1000.0
        v_mm, r_mm = minimax_v0(pr, f.v0, back, side)
        if np.abs(r_mm).max() > np.abs(r_ls).max():   # LP step did not help: keep the LSQ fit
            v_mm, r_mm = f.v0, r_ls
        c = pr.crossing(v_mm, back, side)
        out.append((back, side, f.S_min, np.abs(r_ls).max(), np.abs(r_mm).max(),
                    *v_mm, float(np.linalg.norm(v_mm)), c.y, c.z, c.t, c.outcome, *r_mm))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lim", type=float, default=40.0, help="spin grid half-width (rev/s)")
    ap.add_argument("--step", type=float, default=0.5, help="spin grid spacing (rev/s)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    g = np.arange(-args.lim, args.lim + 1e-9, args.step)
    B, S = np.meshgrid(g, g, indexing="ij")
    pts = np.column_stack([B.ravel(), S.ravel()])
    chunks = [(cfg, c) for c in np.array_split(pts, max(1, len(pts) // 200))]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = [r for part in ex.map(work, chunks) for r in part]
    cols = ["back_rps", "side_rps", "chi2", "max_miss_lsq_mm", "max_miss_mm", "v0_x", "v0_y", "v0_z",
            "speed", "y_goal", "z_goal", "t_cross", "outcome",
            "res_x1_mm", "res_y1_mm", "res_z1_mm", "res_x2_mm", "res_y2_mm", "res_z2_mm"]
    df = pd.DataFrame(rows, columns=cols)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    ok = df[df.max_miss_mm <= TOL_MM]
    print(f"{len(df)} grid points in {time.perf_counter() - t0:.0f} s; {len(ok)} within {TOL_MM} mm -> {OUT}")


if __name__ == "__main__":
    main()
