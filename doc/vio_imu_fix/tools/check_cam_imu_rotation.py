#!/usr/bin/env python3
"""Check the camera-IMU rotation and time offset from data (no calibration target needed).

Body angular velocity from a VO trajectory (results_perframe.csv: per-frame T_w_b, body = the config's body
frame, so the cameras are placed through the configured extrinsic) is compared with the gyro:
  gyro(t + td) - b_g = R_err w_vo(t)
R_err should be identity if the configured camera-IMU rotation is right. td is found by cross-correlating
|w|, then R_err by Kabsch on the time-aligned angular velocities (bias removed by centring).
Usage: check_cam_imu_rotation.py <run_dir> <seq_mav0_dir>
"""
import sys, numpy as np
from pathlib import Path

def log_so3(R):
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if a < 1e-9: return np.zeros(3)
    return a / (2 * np.sin(a)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

run, mav0 = Path(sys.argv[1]), Path(sys.argv[2])
a = np.genfromtxt(run / 'log_slam/results_perframe.csv', delimiter=',', skip_header=1)
last = {}
for r in a: last[int(r[0])] = r
a = np.array([last[k] for k in sorted(last)])
t = a[:, 0] * 1e-9
R = np.stack([a[:, [2, 3, 4]], a[:, [6, 7, 8]], a[:, [10, 11, 12]]], axis=1)
w_vo = np.array([log_so3(R[i].T @ R[i + 1]) / (t[i + 1] - t[i]) for i in range(len(t) - 1)])
tm = 0.5 * (t[1:] + t[:-1])
ok = np.isfinite(w_vo).all(1) & (np.linalg.norm(w_vo, axis=1) < 10)
tm, w_vo = tm[ok], w_vo[ok]

imu = np.genfromtxt(mav0 / 'imu0/data.csv', delimiter=',', skip_header=1)
ti, g = imu[:, 0] * 1e-9, imu[:, 1:4]
# gyro averaged over each VO interval would be best; interpolate (200 Hz vs 30 Hz is fine for this check)
def gyro_at(times):
    return np.stack([np.interp(times, ti, g[:, k]) for k in range(3)], 1)

best = None
for td in np.arange(-0.05, 0.0501, 0.001):
    gi = gyro_at(tm + td)
    c = np.corrcoef(np.linalg.norm(gi, axis=1), np.linalg.norm(w_vo, axis=1))[0, 1]
    if best is None or c > best[1]: best = (td, c)
td = best[0]
gi = gyro_at(tm + td)
A, B = w_vo - w_vo.mean(0), gi - gi.mean(0)
U, S, Vt = np.linalg.svd(B.T @ A)
D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
R_err = U @ D @ Vt   # gyro ~ R_err w_vo
bias = gi.mean(0) - R_err @ w_vo.mean(0)
res = gi - (w_vo @ R_err.T + bias)
print(f'{run}: {len(tm)} intervals, |w_vo| rms {np.sqrt((w_vo**2).sum(1).mean()):.3f} rad/s')
print(f'  time offset (gyro later than camera by) td = {td*1e3:.1f} ms (|w| correlation {best[1]:.3f})')
print(f'  rotation error of the configured extrinsic: {np.degrees(np.linalg.norm(log_so3(R_err))):.2f} deg, axis-angle {np.degrees(log_so3(R_err)).round(2)} deg')
print(f'  gyro bias {bias.round(4)} rad/s, residual rms {np.sqrt((res**2).sum(1).mean()):.4f} rad/s')
print('  R_err =\n', R_err.round(4))
