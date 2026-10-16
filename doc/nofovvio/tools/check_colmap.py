#!/usr/bin/env python3
"""Check a SaDVIO COLMAP export (config export_colmap; log_slam/colmap/): reproject every 3D point into the images that
observe it with COLMAP's camera models (PINHOLE, OPENCV, OPENCV_FISHEYE, as COLMAP computes them) and report the
reprojection errors per camera, and the observations behind their camera (which COLMAP cannot project).
usage: check_colmap.py <colmap dir>
"""
import sys
from pathlib import Path

import numpy as np


def quat_to_R(w, x, y, z):
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def project(model, p, X):
    a, b = X[0] / X[2], X[1] / X[2]
    fx, fy, cx, cy = p[:4]
    if model == 'PINHOLE':
        return np.array([fx * a + cx, fy * b + cy])
    if model == 'OPENCV':
        k1, k2, p1, p2 = p[4:8]
        r2 = a * a + b * b
        rad = 1 + k1 * r2 + k2 * r2 * r2
        xd = a * rad + 2 * p1 * a * b + p2 * (r2 + 2 * a * a)
        yd = b * rad + p1 * (r2 + 2 * b * b) + 2 * p2 * a * b
        return np.array([fx * xd + cx, fy * yd + cy])
    if model == 'OPENCV_FISHEYE':
        k1, k2, k3, k4 = p[4:8]
        r = np.hypot(a, b)
        th = np.arctan(r)
        thd = th * (1 + k1 * th ** 2 + k2 * th ** 4 + k3 * th ** 6 + k4 * th ** 8)
        s = thd / r if r > 1e-12 else 1.0
        return np.array([fx * s * a + cx, fy * s * b + cy])
    raise ValueError(model)


def lines(path):
    return [ln for ln in path.read_text().splitlines() if ln and not ln.startswith('#')]


def main():
    d = Path(sys.argv[1])
    cams = {}
    for ln in lines(d / 'cameras.txt'):
        v = ln.split()
        cams[int(v[0])] = (v[1], [float(x) for x in v[4:]])
    pts = {int(ln.split()[0]): np.array([float(x) for x in ln.split()[1:4]]) for ln in lines(d / 'points3D.txt')}
    L = [ln for ln in (d / 'images.txt').read_text().splitlines() if not ln.startswith('#')]
    errs, behind = {c: [] for c in cams}, {c: 0 for c in cams}
    n_img = 0
    for i in range(0, len(L) - 1, 2):
        v = L[i].split()
        if len(v) < 10:
            continue
        n_img += 1
        R = quat_to_R(*map(float, v[1:5]))
        t = np.array([float(x) for x in v[5:8]])
        cid = int(v[8])
        p = L[i + 1].split()
        for j in range(0, len(p), 3):
            pid = int(p[j + 2])
            if pid < 0:
                continue
            Xc = R @ pts[pid] + t
            if Xc[2] <= 0:
                behind[cid] += 1
                continue
            errs[cid].append(np.linalg.norm(project(cams[cid][0], cams[cid][1], Xc) - [float(p[j]), float(p[j + 1])]))
    print(f'{d}: {len(cams)} camera(s), {n_img} images, {len(pts)} points')
    for c, e in errs.items():
        e = np.array(e)
        if not len(e):
            print(f'  camera {c} ({cams[c][0]}): no observation')
            continue
        print(f'  camera {c} ({cams[c][0]}): {len(e)} observations, reprojection error median {np.median(e):.2f} px, '
              f'mean {e.mean():.2f}, p90 {np.percentile(e, 90):.2f}, p99 {np.percentile(e, 99):.2f}; behind the camera '
              f'{behind[c]}')


if __name__ == '__main__':
    main()
