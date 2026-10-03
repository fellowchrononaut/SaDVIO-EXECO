#!/usr/bin/env python3
"""Diagnostic: replace a RealSense trajectory's IMU by one synthesised from the OptiTrack ground truth.

The GT rigid-body pose is moved into the IMU frame with the rotation R_imu_body fitted from gyro vs GT angular
velocity (check_imu_vs_gt.py; lever arm neglected), smoothed with splines, and differentiated at the real IMU
timestamps (shifted by the fitted GT-gyro time offset):
  w = R_w_imu^T * d/dt(R_w_imu) (vee),   f = R_w_imu^T (p'' - g),  g = (0, 0, -9.81) in the ENU GT world.
No noise, no bias. If VIO works with this IMU but not with the real one, the real IMU data is the problem; if it
fails with both, the problem is in SaDVIO or the camera side of the setup.
Writes <out_mav0>/ (symlinked cam0/cam1/gt + synthetic imu0). Run with the execosim python (scipy).
Usage: synth_imu_from_gt.py <mav0> <out_mav0>
"""
import sys, io, contextlib, os, numpy as np
from pathlib import Path
from scipy.interpolate import make_smoothing_spline
from scipy.spatial.transform import Rotation, RotationSpline

mav0, out = Path(sys.argv[1]), Path(sys.argv[2])
pre = (Path(__file__).parent / 'check_imu_vs_gt.py').read_text().split('# 2.-3. accelerometer')[0]
g = {}
sys.argv = ['x', str(mav0)]
with contextlib.redirect_stdout(io.StringIO()):
    exec(pre, g)
R_imu_body, td, tg, pg, Rg = g['R_imu_body'], g['td'], g['tg'], g['pg'], g['Rg']
imu = np.genfromtxt(mav0 / 'imu0/data.csv', delimiter=',', skip_header=1)
ti_ns = imu[:, 0].astype(np.int64)

# IMU-frame trajectory in the GT world
R_w_imu = Rotation.from_matrix(np.einsum('nij,jk->nik', Rg, R_imu_body.T))
t_rel = tg - tg[0]
pos_spl = [make_smoothing_spline(t_rel, pg[:, k], lam=1e-4) for k in range(3)]
rot_spl = RotationSpline(t_rel, R_w_imu)

# sample at the real IMU stamps; GT time = IMU time - td (gyro(t + td) ~ GT(t))
tq = np.clip(ti_ns * 1e-9 - td - tg[0], t_rel[0], t_rel[-1])
acc_w = np.stack([s.derivative(2)(tq) for s in pos_spl], 1)
R_q = rot_spl(tq)
w_imu = rot_spl(tq, 1)                # scipy returns the angular rate in the body (here IMU) frame
f_imu = R_q.inv().apply(acc_w - np.array([0, 0, -9.81]))

(out / 'imu0').mkdir(parents=True, exist_ok=True)
with open(out / 'imu0/data.csv', 'w') as fo:
    fo.write('#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n')
    for k in range(len(ti_ns)):
        fo.write(f'{ti_ns[k]},' + ','.join(f'{x:.9f}' for x in w_imu[k]) + ',' + ','.join(f'{x:.9f}' for x in f_imu[k]) + '\n')
for d in ('cam0', 'cam1', 'gt'):
    if not (out / d).exists():
        (out / d).symlink_to(os.path.relpath(mav0 / d, out))
real = imu[:, 1:4]
print(f'{out}: {len(ti_ns)} samples; synthetic vs real gyro rms diff {np.sqrt(((w_imu - real) ** 2).sum(1).mean()):.3f} rad/s, '
      f'accel rms diff {np.sqrt(((f_imu - imu[:, 4:7]) ** 2).sum(1).mean()):.3f} m/s2')
