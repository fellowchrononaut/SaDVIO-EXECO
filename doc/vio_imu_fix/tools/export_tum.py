#!/usr/bin/env python3
"""Export a run's keyframe trajectory and its sequence's ground truth to TUM format (for evo).

Estimate: log_slam/results.csv, last write per timestamp (as eval_traj.py). GT: TUM-VI mocap0 or the RealSense
OptiTrack CSV. Writes <run_dir>/est.tum and <run_dir>/gt.tum (timestamp[s] tx ty tz qx qy qz qw).
Usage: export_tum.py <run_dir> [<run_dir> ...]
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_traj import SEQ_DIRS, load_estimate, load_gt  # noqa: E402


def rot_to_quat(R):
    """Rotation matrix -> (qx, qy, qz, qw)."""
    q = np.empty(4)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q[3] = 0.25 * s
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = (R[0, 2] - R[2, 0]) / s
        q[2] = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax(np.diag(R))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q[i] = 0.25 * s
        q[3] = (R[k, j] - R[j, k]) / s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def write_tum(path, t, p, R):
    with open(path, 'w') as f:
        for ti, pi, Ri in zip(t, p, R):
            q = rot_to_quat(Ri)
            f.write(f'{ti:.9f} {pi[0]:.6f} {pi[1]:.6f} {pi[2]:.6f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n')


for rd in map(Path, sys.argv[1:]):
    seq = json.loads((rd / 'run.json').read_text())['seq']
    t, p, R, _ = load_estimate(rd)
    write_tum(rd / 'est.tum', t, p, R)
    tg, pg, Rg = load_gt(seq)
    write_tum(rd / 'gt.tum', tg, pg, Rg)
    print(f'{rd}: est {len(t)} poses, gt {len(tg)} poses')
