#!/usr/bin/env python3
"""Stereo depth error against the EuRoC laser scan, by range (ground truth in evaluation only). The keyframes of a
dense dump (EXECO_DENSE_DUMP) are placed at their ground-truth camera poses; for each point, the nearest scan point
gives the error, signed along the camera ray (positive: the measured point lies beyond the surface, i.e. too far).
A range-dependent bias (rectification, baseline) makes views from different distances disagree, which a long TSDF
volume blurs and a short submap does not.

usage: depth_error_by_range.py <dump dir> <seq> [--every 5] [--points 4000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense_anchor_eval import DATA, SEQ, scan  # noqa: E402


def read_kf(path):
    b = path.read_bytes()
    ts = np.frombuffer(b, '<u8', 1, 0)[0]
    A = np.frombuffer(b, '<f8', 16, 16).reshape(4, 4).T
    B = np.frombuffer(b, '<f8', 16, 144).reshape(4, 4).T
    n = np.frombuffer(b, '<u4', 1, 272)[0]
    pts = np.frombuffer(b, '<f4', 3 * n, 276).reshape(n, 3).astype(float)
    return ts, A, B, pts


def quat_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class NearestPoint:
    """Nearest scan point within `cell` (bucketed), returning the point itself"""

    def __init__(self, pts, cell=0.3):
        self.cell = cell
        key = np.floor(pts / cell).astype(np.int64)
        order = np.lexsort(key.T[::-1])
        self.pts, key = pts[order], key[order]
        u, s, c = np.unique(key, axis=0, return_index=True, return_counts=True)
        self.index = {tuple(k): (a, a + b) for k, a, b in zip(u, s, c)}

    def query(self, q):
        best = np.full(len(q), np.inf)
        near = np.zeros_like(q)
        key = np.floor(q / self.cell).astype(np.int64)
        uk, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
        for u, k in enumerate(uk):
            sel = np.nonzero(inv == u)[0]
            cand = [self.pts[slice(*self.index[t])] for t in
                    ((k[0] + a, k[1] + b, k[2] + c) for a, b, c in offs) if t in self.index]
            if not cand:
                continue
            cand = np.concatenate(cand)
            d2 = ((q[sel, None, :] - cand[None, :, :]) ** 2).sum(-1)
            j = d2.argmin(1)
            best[sel] = np.sqrt(d2[np.arange(len(sel)), j])
            near[sel] = cand[j]
        return best, near


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump')
    ap.add_argument('seq')
    ap.add_argument('--every', type=int, default=5)
    ap.add_argument('--points', type=int, default=4000)
    a = ap.parse_args()
    cloud, _ = scan(a.seq)
    nn = NearestPoint(cloud, 0.3)
    gt = np.loadtxt(DATA / SEQ[a.seq] / 'mav0/state_groundtruth_estimate0/data.csv', delimiter=',', comments='#')
    rng = np.random.default_rng(0)
    rows = []
    files = sorted(Path(a.dump).glob('kf_*.bin'))[::a.every]
    for f in files:
        ts, T_w_rc, T_w_f, pts = read_kf(f)
        i = np.searchsorted(gt[:, 0], ts)
        if i <= 0 or i >= len(gt) or abs(gt[i, 0] - ts) > 5e6:
            continue
        T_gt = np.eye(4)
        T_gt[:3, :3] = quat_R(*gt[i, 4:8])
        T_gt[:3, 3] = gt[i, 1:4]
        T = T_gt @ np.linalg.inv(T_w_f) @ T_w_rc  # ground-truth pose of the rectified camera
        sel = pts[rng.choice(len(pts), min(len(pts), a.points), replace=False)]
        rng_m = np.linalg.norm(sel, axis=1)
        pw = (T[:3, :3] @ sel.T).T + T[:3, 3]
        d, q = nn.query(pw)
        ray = pw - T[:3, 3]
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        signed = np.einsum('ij,ij->i', pw - q, ray)
        ok = d < 0.3
        rows.append(np.c_[rng_m[ok], signed[ok], d[ok]])
    r = np.concatenate(rows)
    print(f'{len(files)} keyframes, {len(r)} points within 30 cm of the scan')
    print('range (m) | points | median signed along ray (cm) | median |dist| (cm) | p90 |dist| (cm)')
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8)]:
        s = (r[:, 0] >= lo) & (r[:, 0] < hi)
        if s.sum() < 50:
            continue
        print(f'{lo}-{hi} | {s.sum():6d} | {np.median(r[s, 1]) * 100:+6.1f} | {np.median(r[s, 2]) * 100:5.1f} | '
              f'{np.percentile(r[s, 2], 90) * 100:5.1f}')


if __name__ == '__main__':
    main()
