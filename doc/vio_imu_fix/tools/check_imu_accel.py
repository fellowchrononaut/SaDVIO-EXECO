#!/usr/bin/env python3
"""Accelerometer vs camera motion consistency (no calibration target needed).

From a VO trajectory (results_perframe.csv, body = configured IMU frame): world acceleration of the body by a local
quadratic fit of the positions over a sliding window; then least squares for the gravity vector g_w (VO world)
and the accelerometer bias b_a (body) in   p''_w = R_wb (a_imu - b_a) + g_w.
Reports |g_w| (should be ~9.81), the bias, the residual, and a scale fit s in  p''_w = s R_wb (a - b_a) + g_w.
Usage: check_imu_accel.py <run_dir> <seq_mav0_dir> [window_s]
"""
import sys, numpy as np
from pathlib import Path
run, mav0 = Path(sys.argv[1]), Path(sys.argv[2])
W = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
a = np.genfromtxt(run / 'log_slam/results_perframe.csv', delimiter=',', skip_header=1)
last = {}
for r in a: last[int(r[0])] = r
a = np.array([last[k] for k in sorted(last)])
t = a[:, 0] * 1e-9; p = a[:, [5, 9, 13]]
R = np.stack([a[:, [2, 3, 4]], a[:, [6, 7, 8]], a[:, [10, 11, 12]]], axis=1)
imu = np.genfromtxt(mav0 / 'imu0/data.csv', delimiter=',', skip_header=1)
ti, acc = imu[:, 0] * 1e-9, imu[:, 4:7]
rows_A, rows_b, keep = [], [], []
for i in range(len(t)):
    m = np.abs(t - t[i]) <= W / 2
    if m.sum() < 7 or t[m][-1] - t[m][0] < 0.8 * W: continue
    tt = t[m] - t[i]
    X = np.stack([np.ones_like(tt), tt, tt ** 2], 1)
    coef, *_ = np.linalg.lstsq(X, p[m], rcond=None)
    pdd = 2 * coef[2]
    # IMU accel averaged over the same window (the fit is a window average of the acceleration)
    mi = np.abs(ti - t[i]) <= W / 2
    if mi.sum() < 10: continue
    am = acc[mi].mean(0)
    # pdd = R am - R b + g  -> unknowns [g (3), b (3)]
    rows_A.append(np.hstack([np.eye(3), -R[i]])); rows_b.append(pdd - R[i] @ am); keep.append((i, am, pdd))
A, b = np.vstack(rows_A), np.hstack(rows_b)
x, *_ = np.linalg.lstsq(A, b, rcond=None)
g, ba = x[:3], x[3:]
res = (b - A @ x).reshape(-1, 3)
# scale fit: pdd = s R (am - ba) + g
rows_A2, rows_b2 = [], []
for (i, am, pdd) in keep:
    rows_A2.append(np.hstack([np.eye(3), (R[i] @ (am - ba))[:, None]])); rows_b2.append(pdd)
x2, *_ = np.linalg.lstsq(np.vstack(rows_A2), np.hstack(rows_b2), rcond=None)
dyn = np.array([np.linalg.norm(pdd) for (_, _, pdd) in keep])
print(f'{run.name} [{run.parent.parent.name}]: {len(keep)} windows of {W} s, VO accel rms {np.sqrt((dyn**2).mean()):.3f} m/s2')
print(f'  |g_w| = {np.linalg.norm(g):.3f} m/s2 (expect 9.81), accel bias b_a = {ba.round(3)} m/s2')
print(f'  residual rms {np.sqrt((res**2).sum(1).mean()):.3f} m/s2; with a free scale: s = {x2[3]:.3f}, |g| = {np.linalg.norm(x2[:3]):.3f}')
