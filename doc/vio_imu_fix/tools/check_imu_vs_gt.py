#!/usr/bin/env python3
"""IMU vs OptiTrack ground truth (gravity-aligned ENU world): independent check of the RealSense IMU.

1. R_body_imu (constant mocap-body -> IMU rotation) and time offset from gyro vs GT angular velocity (Kabsch).
2. Specific force predicted from GT: f = R_w_imu^T (p''_w - g_w), g_w = (0, 0, -9.81), p'' by a local quadratic fit
   of the GT positions over W seconds (lever arm mocap-body -> IMU neglected: a few cm).
3. Compared with the accelerometer averaged over the same windows: raw residual, then a per-axis scale + bias fit.
Usage: check_imu_vs_gt.py <mav0> [W]
"""
import sys, numpy as np
from pathlib import Path

def quat_R(qx, qy, qz, qw):
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw); qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([[1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                     [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                     [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]])
def log_so3(R):
    a = np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))
    return np.zeros(3) if a < 1e-9 else a/(2*np.sin(a))*np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

mav0 = Path(sys.argv[1]); W = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
gt = np.loadtxt(mav0/'gt/data.csv', delimiter=',', skiprows=1)
# drop near-duplicate GT stamps (bursts)
keep = np.hstack([True, np.diff(gt[:, 0]) > 2e-3]); gt = gt[keep]
tg, pg = gt[:, 0], gt[:, 1:4]; Rg = np.stack([quat_R(*q) for q in gt[:, 4:8]])
imu = np.genfromtxt(mav0/'imu0/data.csv', delimiter=',', skip_header=1)
ti, wi, ai = imu[:, 0]*1e-9, imu[:, 1:4], imu[:, 4:7]

# 1. angular velocity of the GT body over 0.1 s steps
tc = np.arange(tg[0]+0.1, tg[-1]-0.1, 0.05)
def R_at(t):
    i = np.clip(np.searchsorted(tg, t), 1, len(tg)-1); return Rg[i]
wg = np.array([log_so3(R_at(t-0.05).T @ R_at(t+0.05))/0.1 for t in tc])
def gyro_avg(t, h):
    m = (ti >= t-h) & (ti < t+h); return wi[m].mean(0) if m.sum() else np.full(3, np.nan)
best = None
for td in np.arange(-0.06, 0.0601, 0.002):
    gi = np.array([gyro_avg(t+td, 0.05) for t in tc]); ok = np.isfinite(gi).all(1)
    c = np.corrcoef(np.linalg.norm(gi[ok], axis=1), np.linalg.norm(wg[ok], axis=1))[0, 1]
    if best is None or c > best[1]: best = (td, c, gi, ok)
td, corr, gi, ok = best
A, B = wg[ok]-wg[ok].mean(0), gi[ok]-gi[ok].mean(0)
U, S, Vt = np.linalg.svd(B.T @ A); D = np.diag([1, 1, np.sign(np.linalg.det(U@Vt))])
R_imu_body = U @ D @ Vt            # w_imu = R_imu_body w_body
gres = gi[ok] - (wg[ok] @ R_imu_body.T); gres -= gres.mean(0)
print(f'{mav0.parent.name}: GT-IMU time offset {td*1e3:.0f} ms (|w| corr {corr:.3f}), gyro residual rms {np.sqrt((gres**2).sum(1).mean()):.3f} rad/s')

# 2.-3. accelerometer
rows = []
for t in np.arange(tg[0]+W, tg[-1]-W, W/2):
    m = np.abs(tg-t) <= W/2
    if m.sum() < 8: continue
    tt = tg[m]-t; X = np.stack([np.ones_like(tt), tt, tt**2], 1)
    coef, *_ = np.linalg.lstsq(X, pg[m], rcond=None)
    pdd = 2*coef[2]
    R_w_imu = R_at(t) @ R_imu_body.T
    f_pred = R_w_imu.T @ (pdd - np.array([0, 0, -9.81]))
    mi = np.abs(ti-(t+td)) <= W/2
    if mi.sum() < 10: continue
    rows.append((f_pred, ai[mi].mean(0), np.linalg.norm(pdd)))
fp = np.array([r[0] for r in rows]); fm = np.array([r[1] for r in rows]); dyn = np.array([r[2] for r in rows])
res = fm - fp
print(f'  accel: {len(rows)} windows of {W} s, motion accel rms {np.sqrt((dyn**2).mean()):.2f} m/s2')
print(f'  raw residual (measured - predicted): mean {res.mean(0).round(3)}, rms after removing the mean {np.sqrt(((res-res.mean(0))**2).sum(1).mean()):.3f} m/s2')
# per-axis scale + bias: fm = s * fp + b
for k, ax in enumerate('xyz'):
    X = np.stack([fp[:, k], np.ones(len(fp))], 1); c, *_ = np.linalg.lstsq(X, fm[:, k], rcond=None)
    r = fm[:, k] - X @ c
    print(f'    axis {ax}: scale {c[0]:.3f} bias {c[1]:+.3f} | residual rms {r.std():.3f} | predicted range {fp[:,k].min():.1f}..{fp[:,k].max():.1f}')
