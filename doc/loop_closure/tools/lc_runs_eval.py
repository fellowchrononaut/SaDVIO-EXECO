#!/usr/bin/env python3
"""Evaluate SaDVIO runs with loop closure (ground truth in evaluation only).

For each run under doc/vio_imu_fix/runs/<label>/<seq>/<mode>/run_NN: the odometry ATE (results.csv), the ATE of the
loop-closed trajectory (results_loop.csv: final, after all pose graph solves) and of the online output
(results_loop_online.csv: each KF as corrected when it was estimated), the loops accepted and how many are true by
the ground truth (the measured distance between the two KFs within 0.15 m of the true one), and the cost (loop
closure per KF, pose graph per solve, back end per KF) from slam_profiler.txt.

usage: lc_runs_eval.py <label> [<label> ...] [--md out.md]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

VIO = Path(__file__).resolve().parents[2] / 'vio_imu_fix'
sys.path.insert(0, str(VIO / 'tools'))
from eval_traj import associate, evaluate, load_gt  # noqa: E402

EDGE_TOL = 0.15


def profiler(rd):
    out = {}
    pf = rd / 'log_slam' / 'slam_profiler.txt'
    for ln in pf.read_text().splitlines() if pf.exists() else []:
        if ':' in ln:
            k, v = ln.split(':', 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
    return out


def loop_precision(rd, seq):
    """Loops with ground truth, and how many have a correct relative pose: the measured distance between the two
    KFs agrees with the ground truth's within EDGE_TOL m (frame conventions aside, a wrong loop is off by metres)"""
    f = rd / 'log_slam' / 'loops.csv'
    if not f.exists():
        return 0, 0
    ts = np.genfromtxt(f, delimiter=',', skip_header=1, usecols=(0, 1), dtype=np.int64, ndmin=2)
    if ts.size == 0:
        return 0, 0
    t_meas = np.genfromtxt(f, delimiter=',', skip_header=1, usecols=(6, 7, 8), ndmin=2)
    tg, pg, Rg = load_gt(seq)
    okq, pq, _ = associate(ts[:, 0] * 1e-9, tg, pg, Rg)
    okc, pc, _ = associate(ts[:, 1] * 1e-9, tg, pg, Rg)
    err = np.abs(np.linalg.norm(t_meas, axis=1) - np.linalg.norm(pq - pc, axis=1))
    ok = okq & okc
    return int(ok.sum()), int((ok & (err < EDGE_TOL)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('labels', nargs='+')
    ap.add_argument('--md', type=Path)
    a = ap.parse_args()
    lines = ['| label | seq | mode | ATE odometry | ATE loop final | ATE loop online | loops (correct / with GT) | '
             'loop ms/KF | pose graph ms/solve | back end ms/KF |', '|---|---|---|---|---|---|---|---|---|---|']
    for label in a.labels:
        for rd in sorted((VIO / 'runs' / label).glob('*/*/run_*')):
            seq, mode = rd.parent.parent.name, rd.parent.name
            try:
                m0 = evaluate(rd)
                m1 = evaluate(rd, 'results_loop.csv')
                m2 = evaluate(rd, 'results_loop_online.csv')
            except Exception as e:  # noqa: BLE001
                lines.append(f'| {label} | {seq} | {mode} | error: {e} |')
                continue
            # Segments joined by loops (column group of results_loop.csv): their ATE aligned as one trajectory
            merged = ''
            lf = rd / 'log_slam' / 'results_loop.csv'
            rows = [ln.split(',') for ln in lf.read_text().splitlines()[1:] if ln.strip()]
            if rows and len(rows[0]) > 15 and len({r[15] for r in rows}) < len({r[14] for r in rows}):
                (rd / 'log_slam' / 'results_loop_merged.csv').write_text(
                    lf.read_text().splitlines()[0] + '\n' +
                    ''.join(','.join(r[:14] + [r[15].strip()]) + '\n' for r in rows))
                mm = evaluate(rd, 'results_loop_merged.csv')
                merged = f" (merged {len({r[14] for r in rows})} -> {len({r[15] for r in rows})} segments: " \
                         f"{mm['ate_rmse']:.3f})"
            n_gt, n_true = loop_precision(rd, seq)
            p = profiler(rd)
            lines.append(f"| {label} | {seq} | {mode} | {m0['ate_rmse']:.3f} | {m1['ate_rmse']:.3f} | "
                         f"{m2['ate_rmse']:.3f}{merged} | {n_true} / {n_gt} | {p.get('Loop closure dt', float('nan')):.1f} | "
                         f"{p.get('Pose graph dt', float('nan')):.0f} | {p.get('Back end dt', float('nan')):.1f} |")
    text = '\n'.join(lines)
    print(text)
    if a.md:
        a.md.write_text(text + '\n')


if __name__ == '__main__':
    main()
