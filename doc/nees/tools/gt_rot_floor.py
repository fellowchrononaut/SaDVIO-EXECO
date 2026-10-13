#!/usr/bin/env python3
"""Rotation noise floor of the ground truth over the KF intervals of a run, without the estimator: the GT relative
rotation of each KF pair against the raw gyroscope integrated over the same interval (one constant gyro bias per
sequence, least squares). Gyro white noise over 0.1-0.25 s is ~0.003 deg, so the median disagreement is the GT's
relative-rotation noise (plus time-sync error). Usage: gt_rot_floor.py <label> [--mode bimonovio]"""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nees_eval as ne  # noqa: E402


def floor(seq, run):
    gt = ne.et.load_gt(seq)
    imu = ne.et.read_csv(ne.et.SEQ_DIRS[seq] / 'imu0/data.csv', ncols=7)
    ti, w = imu[:, 0] * 1e-9, imu[:, 1:4]
    wi, _ = ne.read_rows(run / 'log_slam/cov_window.csv', 3)
    seen, P, G = set(), [], []
    for ints in wi:
        if int(ints[0]) in seen:
            continue
        seen.add(int(ints[0]))
        gp, gn = ne.gt_interp(gt, int(ints[1])), ne.gt_interp(gt, int(ints[0]))
        if gp is not None and gn is not None:
            P.append((ints[1] * 1e-9, ints[0] * 1e-9))
            G.append(gp[:3, :3].T @ gn[:3, :3])

    def integ(t0, t1, bg):
        i, R, t = np.searchsorted(ti, t0), np.eye(3), t0
        while i < len(ti) and ti[i] < t1:
            if ti[i] > t:
                R, t = R @ ne.exp_so3((w[i - 1] - bg) * (ti[i] - t)), ti[i]
            i += 1
        return R @ ne.exp_so3((w[i - 1] - bg) * (t1 - t))

    bg = np.zeros(3)
    for _ in range(3):
        b = [ne.log_so3(integ(t0, t1, bg).T @ Rg) for (t0, t1), Rg in zip(P, G)]
        A = np.vstack([-(t1 - t0) * np.eye(3) for t0, t1 in P])
        bg = bg + np.linalg.lstsq(A, np.concatenate(b), rcond=None)[0]
    err = [np.degrees(np.linalg.norm(ne.log_so3(integ(t0, t1, bg).T @ Rg))) for (t0, t1), Rg in zip(P, G)]
    return np.median(err), np.median([t1 - t0 for t0, t1 in P])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--mode', default='bimonovio')
    a = ap.parse_args()
    print('| seq | median KF interval (s) | median gyro-vs-GT relative rotation (deg) |\n|---|---|---|')
    for q in ne.SEQS:
        run = ne.RUNS / a.label / q / a.mode / 'run_00'
        if (run / 'log_slam/cov_window.csv').exists():
            m, dt = floor(q, run)
            print(f'| {q} | {dt:.2f} | {m:.3f} |', flush=True)
