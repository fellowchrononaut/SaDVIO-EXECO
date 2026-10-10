#!/usr/bin/env python3
"""Dense submaps placed by the loop closure (dense_submap_kfs): how far the submap anchors are from the ground truth
before (odometry pose the submap was fused at) and after the loop closure's correction (log_slam/dense_submaps/
submaps.csv), each set aligned to the ground truth by one SE3 per segment (ground truth in evaluation only). The anchor
is the rectified left camera; the ground truth is the body, a fixed lever arm of a few centimetres that both sets share.

usage: dsm_eval.py <label> [<label> ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

VIO = Path(__file__).resolve().parents[2] / 'vio_imu_fix'
sys.path.insert(0, str(VIO / 'tools'))
from eval_traj import associate, evaluate, load_gt, umeyama  # noqa: E402


def anchor_ate(ts, seg, p, tg, pg, Rg):
    ok, p_gt, _ = associate(ts, tg, pg, Rg)
    err = []
    for k in np.unique(seg[ok]):
        sel = ok & (seg == k)
        if sel.sum() < 3:
            continue
        s, R, t = umeyama(p[sel], p_gt[sel], with_scale=False)
        err.append(np.linalg.norm((R @ p[sel].T).T + t - p_gt[sel], axis=1))
    e = np.concatenate(err) if err else np.zeros(0)
    return (float(np.sqrt(np.mean(e ** 2))) if len(e) else float('nan')), int(len(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('labels', nargs='+')
    a = ap.parse_args()
    print('| label | seq | submaps | anchor ATE odometry | anchor ATE corrected | KF ATE odometry | KF ATE loop | '
          'mesh faces |')
    print('|---|---|---|---|---|---|---|---|')
    for label in a.labels:
        for rd in sorted((VIO / 'runs' / label).glob('*/*/run_*')):
            seq = json.loads((rd / 'run.json').read_text())['seq']
            f = rd / 'log_slam' / 'dense_submaps' / 'submaps.csv'
            if not f.exists():
                print(f'| {label} | {seq} | no submaps.csv |')
                continue
            d = np.genfromtxt(f, delimiter=',', skip_header=1, ndmin=2)
            ts, seg = d[:, 1] * 1e-9, d[:, 2].astype(int)
            tg, pg, Rg = load_gt(seq)
            e0, n = anchor_ate(ts, seg, d[:, 5:8], tg, pg, Rg)
            e1, _ = anchor_ate(ts, seg, d[:, 12:15], tg, pg, Rg)
            m0, m1 = evaluate(rd), evaluate(rd, 'results_loop.csv')
            faces = '?'
            ply = rd / 'log_slam' / 'dense_vdbgpdf_mesh.ply'
            if ply.exists():
                with open(ply, 'rb') as fh:
                    for ln in fh:
                        if ln.startswith(b'element face'):
                            faces = int(ln.split()[2])
                        if ln.startswith(b'end_header'):
                            break
            print(f"| {label} | {seq} | {len(d)} ({n} with GT) | {e0:.3f} | {e1:.3f} | {m0['ate_rmse']:.3f} | "
                  f"{m1['ate_rmse']:.3f} | {faces} |")


if __name__ == '__main__':
    main()
