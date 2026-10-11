#!/usr/bin/env python3
"""Anchored dense maps (dense_anchored, doc/dense_anchored_maps.md) against the EuRoC Vicon-room laser scan
(mav0/pointcloud0, Leica MS50, world frame). For each run, three meshes:
  odometry   the anchors (log_slam/dense_anchors/anchor_NNNNN.ply) at the odometry poses they were built at
  corrected  the anchors at their loop-corrected poses (anchors.csv)
  fused      the fused view (log_slam/dense_fused_mesh.ply, written at the end by the full rebuild)
Each is aligned to the ground truth by an SE3 fit of the keyframe positions (results.csv for odometry and global,
results_loop.csv for corrected and fused; ground truth in evaluation only). With --global, runs without anchors
(dense_anchored off) are evaluated on their global map instead (dense_vdbgpdf_mesh.ply or dense_gp_global_mesh.ply).
With --area, accuracy is per surface area: vertices thinned to one per 5 cm cell first (an anchored union repeats
every surface its overlapping anchors share, which otherwise weighs the well-observed surfaces several times). Accuracy: distance of mesh vertices to the scan (capped at 20 cm); completeness: share of scan points (in the region
the run saw) within 10 cm of the mesh. usage: dense_anchor_eval.py <label> [<label> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsm_mesh_eval import GridNN, quat_R, read_ply_vertices, umeyama_se3  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'vio_imu_fix' / 'tools'))
from eval_traj import load_estimate  # noqa: E402

VIO = Path(__file__).resolve().parents[2] / 'vio_imu_fix'
DATA = Path('/home/deos/s.jois/EXECO/Simulator_Validation/SaD_VIO_data/euroc')
SEQ = {'V1_01': 'V1_01_easy', 'V1_02': 'V1_02_medium', 'V1_03': 'V1_03_difficult', 'V2_01': 'V2_01_easy',
       'V2_02': 'V2_02_medium', 'V2_03': 'V2_03_difficult'}
_scans = {}


def scan(seq):
    if seq not in _scans:
        c = read_ply_vertices(DATA / SEQ[seq] / 'mav0/pointcloud0/data.ply')
        c = c[np.unique(np.floor(c / 0.01).astype(np.int64), axis=0, return_index=True)[1]]
        _scans[seq] = (c, GridNN(c, 0.2))
    return _scans[seq]


def stats(d):
    return f'{np.median(d) * 100:.1f} cm, <5 cm {np.mean(d < 0.05) * 100:.0f}%, <10 cm {np.mean(d < 0.10) * 100:.0f}%'


def traj_align(rd, results, gt):
    """SE3 fit of the KF positions of log_slam/<results> to the ground truth (first segment)"""
    t, p, _, _, seg = load_estimate(rd, with_segments=True, results=results)
    sel = seg == seg[0]
    t, p = t[sel], p[sel]
    i = np.clip(np.searchsorted(gt[:, 0] * 1e-9, t), 0, len(gt) - 1)
    ok = np.abs(gt[i, 0] * 1e-9 - t) < 0.02
    return umeyama_se3(p[ok], gt[i[ok], 1:4])


def main():
    rng = np.random.default_rng(0)
    args = sys.argv[1:]
    use_global = '--global' in args
    area = '--area' in args
    for label in [a for a in args if a not in ('--global', '--area')]:
        for rd in sorted((VIO / 'runs' / label).glob('*/*/run_*')):
            seq = json.loads((rd / 'run.json').read_text())['seq']
            log = rd / 'log_slam'
            f = log / 'dense_anchors/anchors.csv'
            gt = np.loadtxt(DATA / SEQ[seq] / 'mav0/state_groundtruth_estimate0/data.csv', delimiter=',', comments='#')
            if use_global:
                ply = next((log / n for n in ('dense_vdbgpdf_mesh.ply', 'dense_gp_global_mesh.ply')
                            if (log / n).exists()), None)
                if ply is None:
                    print(f'{label} {seq}: no global mesh')
                    continue
                cloud, nn = scan(seq)
                R, t = traj_align(rd, 'results.csv', gt)
                v = read_ply_vertices(ply)
                m = (R @ v[rng.choice(len(v), min(len(v), 150000), replace=False)].T).T + t
                if area:
                    m = m[np.unique(np.floor(m / 0.05).astype(np.int64), axis=0, return_index=True)[1]]
                thin = cloud[np.unique(np.floor(cloud / 0.02).astype(np.int64), axis=0, return_index=True)[1]]
                seen = thin[GridNN(m, 0.3).query(thin) < 0.3]
                seen = seen[rng.choice(len(seen), min(len(seen), 40000), replace=False)]
                comp = np.mean(GridNN(m, 0.2).query(seen) < 0.10) * 100
                print(f'{label} {seq} global ({ply.name}): {stats(nn.query(m))}, complete {comp:.0f}% (own region)')
                continue
            if seq not in SEQ or not f.exists():
                print(f'{label} {seq}: no scan or no anchors.csv')
                continue
            d = np.genfromtxt(f, delimiter=',', skip_header=1, ndmin=2)
            cloud, nn = scan(seq)
            per = max(200, 150000 // max(1, len(d)))
            meshes, aligns = {}, {}
            for name, cols, res in (('odometry', slice(5, 12), 'results.csv'),
                                    ('corrected', slice(12, 19), 'results_loop.csv')):
                poses = d[:, cols]
                R, t = traj_align(rd, res, gt)
                aligns[name] = (R, t)
                pts = []
                for k, row in enumerate(d):
                    ply = log / f'dense_anchors/anchor_{int(row[0]):05d}.ply'
                    if not ply.exists():
                        continue
                    v = read_ply_vertices(ply)
                    if len(v) == 0:
                        continue
                    v = v[rng.choice(len(v), min(len(v), per), replace=False)]
                    pts.append((R @ (quat_R(poses[k, 3:7]) @ v.T + poses[k, :3, None])).T + t)
                meshes[name] = np.concatenate(pts) if pts else np.zeros((0, 3))
            fused_ply = log / 'dense_fused_mesh.ply'
            if fused_ply.exists():
                v = read_ply_vertices(fused_ply)
                v = v[rng.choice(len(v), min(len(v), 150000), replace=False)]
                R, t = aligns['corrected']
                meshes['fused'] = (R @ v.T).T + t
            # completeness over the scan region the run saw: scan points (2 cm thinned) within 0.3 m of the corrected
            # anchors' mesh
            thin = cloud[np.unique(np.floor(cloud / 0.02).astype(np.int64), axis=0, return_index=True)[1]]
            seen = thin[GridNN(meshes['corrected'], 0.3).query(thin) < 0.3]
            seen = seen[rng.choice(len(seen), min(len(seen), 40000), replace=False)]
            out = []
            for name, m in meshes.items():
                if len(m) == 0:
                    continue
                comp = np.mean(GridNN(m, 0.2).query(seen) < 0.10) * 100
                # Per surface area: one vertex per 5 cm cell, so a surface repeated by overlapping anchors (the
                # anchored union keeps one layer per anchor) counts once, as in a fused mesh; layers more than a cell
                # apart (a real misalignment) still count each
                if area:
                    m = m[np.unique(np.floor(m / 0.05).astype(np.int64), axis=0, return_index=True)[1]]
                out.append(f'{name}: {stats(nn.query(m))}, complete {comp:.0f}%')
            rebuilt = 'full rebuild' in (rd / 'stdout.log').read_text(errors='replace')
            print(f'{label} {seq} ({len(d)} anchors{", final rebuild" if rebuilt else ", NO final rebuild"}): '
                  + ' | '.join(out))


if __name__ == '__main__':
    main()
