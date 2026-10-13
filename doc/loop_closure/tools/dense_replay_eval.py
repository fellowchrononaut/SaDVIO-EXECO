#!/usr/bin/env python3
"""Score dense meshes against the EuRoC Vicon-room laser scan, all of each mesh's surface counted once (one vertex per
5 cm cell): median distance to the scan (capped at 20 cm), share within 5 / 10 cm, and completeness (share of the scan
points within --region m of the ground-truth camera path that lie within 10 cm of the mesh; the same points for every
mesh of a sequence). For dense_replay meshes and for whole meshes written by a run (anchored union, fused view,
global map). dense_anchor_eval.py samples the same number of vertices per anchor, which under-weights large anchors
(the far, noisy surface) and scored anchored layers about 5 cm too well (2026-10-13).
A mesh built with ground-truth poses is already in the scan's frame. One built with SLAM poses is aligned first by an
SE3 fit of a run's keyframe trajectory (--align <run dir>:<results csv>; ground truth in evaluation only).

usage: dense_replay_eval.py --seq V2_02 [--region 4.0] mesh.ply[@run_dir:results_loop.csv] [...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense_anchor_eval import DATA, SEQ, scan, stats, traj_align  # noqa: E402
from dsm_mesh_eval import GridNN, read_ply_vertices  # noqa: E402


def thin(v, c=0.05):
    return v[np.unique(np.floor(v / c).astype(np.int64), axis=0, return_index=True)[1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True)
    ap.add_argument('--region', type=float, default=4.0, help='m around the ground-truth camera path: scored scan')
    ap.add_argument('meshes', nargs='+', help='PLY, optionally PLY@run_dir:results.csv to align it by that trajectory')
    a = ap.parse_args()
    cloud, nn = scan(a.seq)
    gt = np.loadtxt(DATA / SEQ[a.seq] / 'mav0/state_groundtruth_estimate0/data.csv', delimiter=',', comments='#')
    thin_scan = cloud[np.unique(np.floor(cloud / 0.02).astype(np.int64), axis=0, return_index=True)[1]]
    path = thin(gt[:, 1:4], 0.1)
    seen = thin_scan[GridNN(path, a.region).query(thin_scan) < a.region]
    seen = seen[np.random.default_rng(0).choice(len(seen), min(len(seen), 40000), replace=False)]
    seen_d = GridNN(path, 4.0).query(seen)  # each scored scan point's distance to the camera path
    for spec in a.meshes:
        ply, _, align = spec.partition('@')
        m = thin(read_ply_vertices(ply))
        if align:
            rd, _, res = align.partition(':')
            R, t = traj_align(Path(rd), res, gt)
            m = (R @ m.T).T + t
        hit = GridNN(m, 0.2).query(seen) < 0.10
        comp = np.mean(hit) * 100
        # by distance from the ground-truth camera path: near surfaces, and the far ones only seen from afar
        bands = []
        dm = GridNN(path, 4.0).query(m)
        err = nn.query(m)
        for lo, hi in ((0, 1.5), (1.5, 3.0), (3.0, 99)):
            s_scan = (seen_d >= lo) & (seen_d < hi)
            s_mesh = (dm >= lo) & (dm < hi)
            if s_scan.sum() > 50 and s_mesh.sum() > 50:
                bands.append(f'{lo:g}-{hi:g} m: {np.median(err[s_mesh]) * 100:.1f} cm, '
                             f'{np.mean(hit[s_scan]) * 100:.0f}% complete')
        print(f'{Path(ply).name}: {stats(err)}, complete {comp:.0f}% ({len(m)} cells) | ' + ' | '.join(bands))


if __name__ == '__main__':
    main()
