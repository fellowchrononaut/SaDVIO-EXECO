#!/usr/bin/env python3
"""Loop-closure playback data for the presentation's loop-closure viewer (sources/sadvio-lc, no mesh) from an offline
SaDVIO run on an Insta360 X5 capture (doc/nofovvio/NOFOVVIO_LEDGER.md, item 4.1).

The viewer's live-capture views come from ROS recordings; here the same data comes from the run's logs:
  raw      per-frame front-end odometry (log_slam/results_perframe.csv), the SLAM world, metres
  online   the same frames moved by the loop correction known at their time (the latest keyframe processed by the
           loop closure, or the latest solve)
  events   'path': a keyframe reaching the loop closure (it leaves the sliding window max_kf_number keyframes later, or
           at the end of the data), at its online pose (results_loop_online.csv); at every pose-graph solve the whole
           corrected keyframe path (results_loop_solves.csv) with 'edges' (accepted loops, at the solve's positions)
           and 'loops' (count); at the end the final path (results_loop.csv)
  gt       GUTWrap's reference front-camera positions, z up, in metres (measured scale, eval_x5.M_PER_UNIT) or scaled
           by the run's Sim(3) scale; align: rigid fits of the raw odometry to it (common start: first 20 s; whole run)
Writes <out>/meta.json, events.json, and empty vertices.b64.txt / indices.b64.txt (no mesh).
usage: build_x5_lc_view.py <run dir> --out <assets/display/sadvio-x5lc-012>
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_x5 import DATA, M_PER_UNIT, read_reference, umeyama  # noqa: E402
from export_tum import read_poses  # noqa: E402

R_ZUP = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])  # reference world (down = +y) -> z up


def se3(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    return R, mu_d - R @ mu_s


def rows_with_pose(path, first=0):
    """[(key ts, row ts, 4x4)] from a CSV whose pose row starts at column `first` (ts, nframes, 3x4, ...)"""
    out = []
    for ln in path.read_text().splitlines()[1:]:
        v = ln.split(',')
        if len(v) < first + 14:
            continue
        T = np.eye(4)
        T[:3, :4] = np.array([float(x) for x in v[first + 2:first + 14]]).reshape(3, 4)
        out.append((int(v[0]), int(v[first]), T))
    return out


def r(x, n=4):
    return [round(float(a), n) for a in x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    log = a.run / 'log_slam'
    seq = json.loads((a.run / 'run.json').read_text())['seq']
    cap = 'VID_' + seq.split('_')[1]
    cfg = (a.run / 'config/config.yaml').read_text()
    window = int(re.search(r'^max_kf_number\s*:\s*(\d+)', cfg, re.M).group(1))

    front = read_poses(log / 'results_perframe.csv')                 # frame ts -> T_w_f (front end)
    kf_handover = read_poses(log / 'results.csv')                     # KF ts -> window pose as it left the window
    kf_ts = sorted(kf_handover)
    online = rows_with_pose(log / 'results_loop_online.csv')          # in processing order
    solves = rows_with_pose(log / 'results_loop_solves.csv', first=1)  # (solve ts, KF ts, T)
    final = read_poses(log / 'results_loop.csv')
    loops = [ln.split(',') for ln in (log / 'loops.csv').read_text().splitlines()[1:]]
    loops = [(int(v[0]), int(v[1])) for v in loops if v[5].strip() == '1']
    frame_ts = sorted(front)
    duration = frame_ts[-1] * 1e-9

    # When each KF reaches the loop closure: as the KF `window` places later enters the window, else at the end
    reach = {}
    for i, t in enumerate(kf_ts):
        reach[t] = kf_ts[i + window] * 1e-9 if i + window < len(kf_ts) else duration
    pos = lambda T: T[:3, 3]

    events, corrections = [], []  # corrections: (time, C) known from then on
    for ts, _, T in online:
        tau = reach.get(ts, duration)
        events.append((tau, 0, ['path', [[round(ts * 1e-9, 3)] + r(pos(T))]]))
        if ts in kf_handover:
            corrections.append((tau, T @ np.linalg.inv(kf_handover[ts])))
    by_solve = {}
    for solve_ts, ts, T in solves:
        by_solve.setdefault(solve_ts, {})[ts] = T
    n_loops = 0
    for solve_ts in sorted(by_solve):
        snap = by_solve[solve_ts]
        tau = reach.get(solve_ts, duration)
        n_loops = sum(1 for q, m in loops if q <= solve_ts)
        edges = []
        for q, m in loops:
            if q <= solve_ts and q in snap and m in snap:
                edges += [r(pos(snap[q])), r(pos(snap[m]))]
        events.append((tau, 1, ['path', [[round(t * 1e-9, 3)] + r(pos(T)) for t, T in sorted(snap.items())]]))
        events.append((tau, 2, ['edges', edges]))
        events.append((tau, 3, ['loops', n_loops]))
        if solve_ts in kf_handover:
            corrections.append((tau, snap[solve_ts] @ np.linalg.inv(kf_handover[solve_ts])))
    final_edges = []
    for q, m in loops:
        if q in final and m in final:
            final_edges += [r(pos(final[q])), r(pos(final[m]))]
    events.append((duration, 4, ['path', [[round(t * 1e-9, 3)] + r(pos(T)) for t, T in sorted(final.items())]]))
    events.append((duration, 5, ['edges', final_edges]))
    events.append((duration, 6, ['loops', len(loops)]))
    events.sort(key=lambda e: (e[0], e[1]))
    events = [[round(t, 3)] + e for t, _, e in events]

    corrections.sort(key=lambda c: c[0])
    raw, onl = [], []
    j = -1
    for ts in frame_ts:
        t = ts * 1e-9
        while j + 1 < len(corrections) and corrections[j + 1][0] <= t:
            j += 1
        T = front[ts]
        raw.append([round(t, 3)] + r(pos(T)))
        onl.append([round(t, 3)] + r(pos(corrections[j][1] @ T if j >= 0 else T)))

    # Reference in metres, z up; rigid fits of the raw odometry onto it
    ref = read_reference(DATA / cap / 'reference_poses_cam0.csv')
    t_ref = np.array(sorted(ref)) * 1e-9
    P_ref = np.array([ref[k][1] for k in sorted(ref)]) @ R_ZUP.T
    ft = np.array(frame_ts) * 1e-9
    P_raw = np.array([p[1:] for p in raw])
    sel = [i for i, t in enumerate(t_ref) if np.min(np.abs(ft - t)) < 0.02]
    idx = [int(np.argmin(np.abs(ft - t_ref[i]))) for i in sel]
    mpu = M_PER_UNIT.get(cap)
    if mpu:
        to_m, gt_label = mpu[0], f'Reference (GUTWrap SfM, {mpu[0]:.3f} m per unit from the boxes)'
    else:
        s, _, _ = umeyama(P_raw[idx], P_ref[sel])
        to_m, gt_label = 1 / s, 'Reference (GUTWrap SfM, scaled to the VIO: not metric yet)'
    G = P_ref * to_m
    gt, last = [], None
    for t, p in zip(t_ref, G):
        if last is not None and t - last > 0.6:
            gt.append(None)
        gt.append([round(float(t), 3)] + r(p))
        last = t
    align = {}
    R, t = se3(P_raw[idx], G[sel])
    align['ate'] = {'R': [r(x, 6) for x in R], 't': r(t), 'matched_kf': len(sel)}
    early = [k for k, i in enumerate(sel) if t_ref[i] < t_ref[sel[0]] + 20]
    R, _ = se3(P_raw[[idx[k] for k in early]], G[[sel[k] for k in early]])
    t = G[sel[0]] - R @ P_raw[idx[0]]
    align['start'] = {'R': [r(x, 6) for x in R], 't': r(t), 'matched_kf': len(early)}

    loop_times = sorted(reach.get(q, duration) for q, _ in loops)
    meta = {
        'seq': cap, 'dataset_label': f'Insta360 X5 {cap}', 'method': 'Two-lens VIO (nofovvio)', 'mesh': False,
        'subtitle': 'Two back-to-back fisheye lenses + IMU · cross-lens DBoW3 loop closure · offline run',
        'camera_note': 'Front lens; the two-lens VIO tracks both lenses, and loops are closed across them.',
        'gt_label': gt_label, 'status_note': 'Offline run, x5_config.yaml',
        'run_id': '/'.join(a.run.parts[-4:]), 'anchors': [], 'loops': [round(x, 3) for x in loop_times],
        'summary': {'keyframes': len(kf_ts), 'loops_accepted': len(loops), 'solves': len(by_solve)},
        'gt': gt, 'align': align,
        'alignment_note': ('One rigid alignment of the raw odometry to the reference, shared by the odometry and the '
                           'loop-corrected keyframes; common start fits the first 20 s and pins the start. The path '
                           'updates when a keyframe reaches the loop closure (it leaves the sliding window) and at '
                           'every pose-graph solve.'),
        'display_mesh': {},
    }
    history = {'raw': raw, 'online': onl, 'events': events, 'loop_times': [round(x, 3) for x in loop_times],
               'duration': round(duration, 3), 'final_snapshot_at': round(duration, 3)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'meta.json').write_text(json.dumps(meta, separators=(',', ':')))
    (a.out / 'events.json').write_text(json.dumps(history, separators=(',', ':')))
    (a.out / 'vertices.b64.txt').write_text('')
    (a.out / 'indices.b64.txt').write_text('')
    print(f'{cap}: {len(raw)} frames, {len(kf_ts)} KFs, {len(loops)} loops, {len(by_solve)} solves, '
          f'{len(events)} events -> {a.out}')


if __name__ == '__main__':
    main()
