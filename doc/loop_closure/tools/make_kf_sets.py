#!/usr/bin/env python3
"""Keyframe image sets for the offline loop-detection benchmark (phase 0).

For each sequence, the keyframes of a reference SaDVIO run (results.csv timestamps) give the images to describe:
  SaD_VIO_data/scratch/lc_kfsets/<seq>.txt   one container image path per line (cam0), read by the describe tools
The ground truth at those keyframes stays in the repo (evaluation only, never seen by the describe tools):
  doc/loop_closure/results/kfsets/<seq>_gt.npz   t [s], p (N x 3), R (N x 3 x 3) of the body in the GT frame

usage: make_kf_sets.py [seq ...]   (default: TUM-VI rooms 1-6 and the 11 EuRoC sequences)
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
VIO = HERE.parent.parent / 'vio_imu_fix'
sys.path.insert(0, str(VIO / 'tools'))
from eval_traj import load_gt, associate, SEQ_DIRS, EUROC_SEQS  # noqa: E402

DATA = Path('/home/deos/s.jois/EXECO/Simulator_Validation/SaD_VIO_data')
CROOT = '/Simulator_Validation/SaD_VIO_data'
OUT_LIST = DATA / 'scratch' / 'lc_kfsets'
OUT_GT = HERE.parent / 'results' / 'kfsets'
ROOMS = [f'room{i}' for i in range(1, 7)]


def reference_run(seq):
    if seq in EUROC_SEQS:
        return VIO / 'runs' / 'sp2_fix_sadvio' / seq / 'bimonovio' / 'run_00'
    return VIO / 'runs' / 'viewer8_vio' / seq / 'bimonovio' / 'run_00'


def main():
    seqs = sys.argv[1:] or ROOMS + list(EUROC_SEQS)
    OUT_LIST.mkdir(parents=True, exist_ok=True)
    OUT_GT.mkdir(parents=True, exist_ok=True)
    for seq in seqs:
        rd = reference_run(seq)
        t_ns = np.loadtxt(rd / 'log_slam' / 'results.csv', delimiter=',', skiprows=1, usecols=0, dtype=np.int64)
        t_ns = np.unique(t_ns)
        cam = SEQ_DIRS[seq] / 'cam0' / 'data'
        keep = np.array([(cam / f'{t}.png').exists() for t in t_ns])
        t_ns = t_ns[keep]
        tg, pg, Rg = load_gt(seq)
        ok, p, R = associate(t_ns * 1e-9, tg, pg, Rg)
        t_ns, p, R = t_ns[ok], p[ok], R[ok]
        rel = cam.relative_to(DATA)
        (OUT_LIST / f'{seq}.txt').write_text(''.join(f'{CROOT}/{rel}/{t}.png\n' for t in t_ns))
        np.savez(OUT_GT / f'{seq}_gt.npz', t_ns=t_ns, p=p, R=R)
        print(f'{seq}: {len(t_ns)} keyframes ({(~keep).sum()} without image, {(~ok).sum()} without GT) from {rd.parent.parent.parent.name}')


if __name__ == '__main__':
    main()
