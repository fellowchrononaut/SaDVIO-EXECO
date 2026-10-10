#!/usr/bin/env python3
"""Dense submaps against the EuRoC Vicon-room laser scan (mav0/pointcloud0/data.ply, Leica MS50, world frame):
the closed submaps' vertices placed at their anchors' odometry poses and at their loop-corrected poses
(dense_submaps/submaps.csv), each set aligned to the ground truth by an SE3 fit of its anchor positions (ground truth
in evaluation only). Reports the distance of the vertices to the nearest scan point (up to 0.5 m; a grid search,
NumPy only). usage: dsm_mesh_eval.py <log_slam dir> <euroc sequence dir> [<log_slam dir> <euroc sequence dir> ...]
"""
import sys
from pathlib import Path

import numpy as np


class GridNN:
    """Nearest scan point within `cell` (m): the scan bucketed in cells, each query cell searched with its 26
    neighbours; farther than `cell` reads as `cell`"""

    def __init__(self, pts, cell=0.5):
        self.cell = cell
        key = np.floor(pts / cell).astype(np.int64)
        order = np.lexsort(key.T[::-1])
        self.pts, key = pts[order], key[order]
        uniq, start, count = np.unique(key, axis=0, return_index=True, return_counts=True)
        self.index = {tuple(k): (s, s + c) for k, s, c in zip(uniq, start, count)}

    def query(self, q):
        out = np.full(len(q), self.cell)
        key = np.floor(q / self.cell).astype(np.int64)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
        for u, k in enumerate(uniq):
            sel = np.nonzero(inv == u)[0]
            cand = [self.pts[slice(*self.index[t])] for t in
                    ((k[0] + a, k[1] + b, k[2] + c) for a, b, c in offs) if t in self.index]
            if not cand:
                continue
            cand = np.concatenate(cand)
            for c0 in range(0, len(sel), 256):
                s = sel[c0:c0 + 256]
                d2 = ((q[s, None, :] - cand[None, :, :]) ** 2).sum(-1).min(1)
                out[s] = np.minimum(np.sqrt(d2), self.cell)
        return out


def read_ply_vertices(path):
    with open(path, 'rb') as f:
        n = 0
        ascii_fmt = False
        while True:
            ln = f.readline()
            if ln.startswith(b'format ascii'):
                ascii_fmt = True
            if ln.startswith(b'element vertex'):
                n = int(ln.split()[2])
            if ln.startswith(b'end_header'):
                break
        if ascii_fmt:
            return np.loadtxt(f, max_rows=n, usecols=(0, 1, 2))
        return np.frombuffer(f.read(n * 16), dtype='<f4').reshape(n, 4)[:, :3].astype(float)


def quat_R(q):  # x y z w
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def umeyama_se3(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    return R, md - R @ ms


def main():
    rng = np.random.default_rng(0)
    for log_dir, seq_dir in zip(sys.argv[1::2], sys.argv[2::2]):
        log_dir, seq_dir = Path(log_dir), Path(seq_dir)
        cloud = read_ply_vertices(seq_dir / 'mav0/pointcloud0/data.ply')
        cloud = cloud[np.unique(np.floor(cloud / 0.01).astype(np.int64), axis=0, return_index=True)[1]]  # 1 cm
        scan = GridNN(cloud, 0.5)
        gt = np.loadtxt(seq_dir / 'mav0/state_groundtruth_estimate0/data.csv', delimiter=',', comments='#')
        d = np.genfromtxt(log_dir / 'dense_submaps/submaps.csv', delimiter=',', skip_header=1, ndmin=2)
        d = d[d[:, 4] == 1]  # closed submaps (their PLY is written)
        near = np.searchsorted(gt[:, 0], d[:, 1])
        near = np.clip(near, 0, len(gt) - 1)
        p_gt = gt[near, 1:4]
        out = []
        for name, cols in (('odometry', slice(5, 12)), ('corrected', slice(12, 19))):
            poses = d[:, cols]
            R, t = umeyama_se3(poses[:, :3], p_gt)
            pts = []
            for k, row in enumerate(d):
                v = read_ply_vertices(log_dir / f'dense_submaps/submap_{int(row[0]):04d}.ply')
                if len(v) == 0:
                    continue
                v = v[rng.choice(len(v), min(len(v), 1500), replace=False)]
                Ra, ta = quat_R(poses[k, 3:7]), poses[k, :3]
                pts.append((R @ (Ra @ v.T + ta[:, None])).T + t)
            pts = np.concatenate(pts)
            dist = scan.query(pts)
            out.append(f'{name}: median {np.median(dist) * 100:.1f} cm, mean {np.mean(dist) * 100:.1f} cm, '
                       f'< 5 cm {np.mean(dist < 0.05) * 100:.0f} %, < 10 cm {np.mean(dist < 0.10) * 100:.0f} %')
        print(f'{log_dir} ({len(d)} submaps): ' + ' | '.join(out))


if __name__ == '__main__':
    main()
