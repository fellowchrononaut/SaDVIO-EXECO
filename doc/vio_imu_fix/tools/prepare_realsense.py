#!/usr/bin/env python3
"""Build EuRoC-format VIO trees for the 9 real RealSense D455 trajectories (2026-05-06).

Source: the existing bag extraction in RealSense_Captures_060526/dsol_calibrated_synced_0726/png/<traj>/realsense_real/
(infra1/infra2 PNGs + timestamps_ns.txt + imu.csv, i.e. /camera/camera/imu as recorded) and the OptiTrack CSV
rosbag2_2026_05_06-<traj>.csv. The existing photoslam mav0 trees carry an all-zero IMU (built for VO), hence these.

Output (container-visible data folder): SaD_VIO_data/realsense/<traj>/mav0/
  cam0/data/<ts>.png, cam1/data/<ts>.png   relative symlinks to the PNGs (no pixel data duplicated)
  cam0/data.csv, cam1/data.csv             EuRoC image lists
  imu0/data.csv                            EuRoC IMU: timestamp [ns], w_x, w_y, w_z, a_x, a_y, a_z (IMU optical frame)
  gt/data.csv                              OptiTrack pose of the rigid body (time [s], px, py, pz, qx, qy, qz, qw; ENU)
"""
import argparse, csv, os
from pathlib import Path

SRC = Path('/home/deos/s.jois/EXECO/Simulator_Validation/RealSense_Captures_060526')
DATA = Path('/home/deos/s.jois/EXECO/Simulator_Validation/SaD_VIO_data/realsense')
TRAJS = ['13_18_29', '13_21_59', '13_30_22', '13_31_26', '13_34_21', '13_36_44', '13_41_28', '13_42_11', '13_45_18']


def build(traj: str):
    png = SRC / 'dsol_calibrated_synced_0726/png' / traj / 'realsense_real'
    mav0 = DATA / traj / 'mav0'
    stamps = [int(x) for x in (png / 'timestamps_ns.txt').read_text().split()]
    for cam, src in (('cam0', 'infra1'), ('cam1', 'infra2')):
        d = mav0 / cam / 'data'
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, ts in enumerate(stamps):
            target = png / src / f'{i:08d}.png'
            assert target.exists(), target
            link = d / f'{ts}.png'
            if not link.is_symlink():
                link.symlink_to(os.path.relpath(target, d))
            rows.append(f'{ts},{ts}.png')
        (mav0 / cam / 'data.csv').write_text('#timestamp [ns],filename\n' + '\n'.join(rows) + '\n')

    (mav0 / 'imu0').mkdir(parents=True, exist_ok=True)
    with open(png / 'imu.csv') as fin, open(mav0 / 'imu0/data.csv', 'w') as fout:
        fout.write('#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],'
                   'a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n')
        n = 0
        for r in csv.DictReader(fin):
            fout.write(f"{r['ts_ns']},{r['gx']},{r['gy']},{r['gz']},{r['ax']},{r['ay']},{r['az']}\n")
            n += 1

    (mav0 / 'gt').mkdir(parents=True, exist_ok=True)
    gt = (SRC / traj / f'rosbag2_2026_05_06-{traj}.csv').read_text()
    (mav0 / 'gt/data.csv').write_text(gt)
    print(f'{traj}: {len(stamps)} stereo pairs, {n} IMU samples, {gt.count(chr(10)) - 1} GT poses')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('trajs', nargs='*', default=TRAJS)
    for t in ap.parse_args().trajs:
        build(t)
