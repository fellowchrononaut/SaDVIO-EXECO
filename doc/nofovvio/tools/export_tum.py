#!/usr/bin/env python3
"""Export a SaDVIO run's trajectory in TUM format, one pose per input frame, plus a summary of the estimated
calibration (doc/nofovvio/NOFOVVIO_LEDGER.md, item 1.2).

Per-frame poses come from log_slam/results_perframe.csv: every frame's front-end pose, written when the run sets
EXECO_PERFRAME_LOG, which run_sadvio.py does. The keyframes' final poses come from log_slam/results_loop.csv (loop
closure on) or log_slam/results.csv (window refinement). Each frame takes the correction of the latest keyframe at or
before it, T = T_final(kf) T_front(kf)^-1 T_front(frame), so the per-frame trajectory follows the window refinement
and the loop closure. Frames before the first keyframe keep their front-end pose.

Writes in <run>/log_slam/:
  trajectory_tum.txt   per frame: "t tx ty tz qx qy qz qw" (t in s), pose of --frame (body: the SLAM frame, the IMU in
                       VIO modes; cam0: camera 0, through the dataset's T_BS)
  keyframes_tum.txt    the keyframes' final poses, same format
  run_summary.json     final dt_imu_cam (estimate_td; else the dataset's value, dt_imu_cam_estimated false), last
                       IMU biases (vio_diag.csv), the T_BS used, counts

usage: export_tum.py <run dir> [--frame body|cam0]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def read_poses(path):
    """{ts: 4x4 T_w_f} from a SaDVIO pose CSV (ts, nframes, 3x4 row-major T_wf, ...); the last write per ts wins"""
    out = {}
    for ln in path.read_text().splitlines()[1:]:
        v = ln.split(',')
        if len(v) < 14:
            continue
        T = np.eye(4)
        T[:3, :4] = np.array([float(x) for x in v[2:14]]).reshape(3, 4)
        if np.isfinite(T).all():
            out[int(v[0])] = T
    return out


def quat_xyzw(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def tum_line(ts, T):
    q = quat_xyzw(T[:3, :3])
    t = T[:3, 3]
    return f'{ts * 1e-9:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}'


def dataset_text(run):
    """The run's dataset YAML (config/dataset/<dataset_id>.yaml), or '' if missing"""
    cfg = (run / 'config/config.yaml').read_text()
    m = re.search(r'^dataset_id\s*:\s*"?([^"\s]+)"?', cfg, re.M)
    ds = run / 'config/dataset' / f'{m.group(1)}.yaml' if m else None
    return ds.read_text() if ds and ds.exists() else ''


def dataset_T_BS(run):
    """camera_0's T_BS (camera -> body) and every camera's, from the run's dataset YAML (plain parsing)"""
    cfg = (run / 'config/config.yaml').read_text()
    m = re.search(r'^dataset_id\s*:\s*"?([^"\s]+)"?', cfg, re.M)
    ds = run / 'config/dataset' / f'{m.group(1)}.yaml' if m else None
    out = {}
    if ds and ds.exists():
        text = ds.read_text()
        for cam in re.finditer(r'^(camera_\d+):', text, re.M):
            block = text[cam.end():]
            d = re.search(r'data:\s*\[([^\]]*)\]', block)
            if d:
                vals = [float(x) for x in d.group(1).replace('\n', ' ').split(',') if x.strip()]
                out[cam.group(1)] = np.array(vals).reshape(4, 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('--frame', choices=['body', 'cam0'], default='body')
    a = ap.parse_args()
    log = a.run / 'log_slam'
    front = read_poses(log / 'results_perframe.csv')
    final_file = log / 'results_loop.csv' if (log / 'results_loop.csv').exists() else log / 'results.csv'
    final = read_poses(final_file)
    T_BS = dataset_T_BS(a.run)
    T_f_out = T_BS.get('camera_0', np.eye(4)) if a.frame == 'cam0' else np.eye(4)

    kf_ts = sorted(t for t in final if t in front)
    lines, n_corr = [], 0
    j = -1
    for ts in sorted(front):
        while j + 1 < len(kf_ts) and kf_ts[j + 1] <= ts:
            j += 1
        T = front[ts]
        if j >= 0:
            k = kf_ts[j]
            T = final[k] @ np.linalg.inv(front[k]) @ T
            n_corr += 1
        lines.append(tum_line(ts, T @ T_f_out))
    (log / 'trajectory_tum.txt').write_text('\n'.join(lines) + '\n')
    (log / 'keyframes_tum.txt').write_text('\n'.join(tum_line(t, final[t] @ T_f_out) for t in sorted(final)) + '\n')

    summary = dict(frames=len(front), keyframes=len(final), frames_corrected_by_a_keyframe=n_corr,
                   keyframe_poses_from=final_file.name, pose_frame=a.frame)
    stdout = (a.run / 'stdout.log').read_text(errors='replace') if (a.run / 'stdout.log').exists() else ''
    td = re.findall(r'Global time offset : ([-0-9.eE+]+)', stdout)
    if td:
        summary['dt_imu_cam_final_s'] = float(td[-1])
        summary['dt_imu_cam_estimated'] = True
    else:  # estimate_td off: the dataset's value was used throughout
        m = re.search(r'^\s*dt_imu_cam\s*:\s*([-0-9.eE+]+)', dataset_text(a.run), re.M)
        if m:
            summary['dt_imu_cam_final_s'] = float(m.group(1))
            summary['dt_imu_cam_estimated'] = False
    diag = log / 'vio_diag.csv'
    if diag.exists():
        rows = diag.read_text().splitlines()
        h = [c.strip() for c in rows[0].split(',')]
        last = rows[-1].split(',')
        summary['bias_acc'] = [float(last[h.index(c)]) for c in ('ba_x', 'ba_y', 'ba_z')]
        summary['bias_gyr'] = [float(last[h.index(c)]) for c in ('bg_x', 'bg_y', 'bg_z')]
    summary['T_BS_used'] = {k: v.tolist() for k, v in T_BS.items()}
    (log / 'run_summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'T_BS_used'}, indent=1))


if __name__ == '__main__':
    main()
