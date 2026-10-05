#!/usr/bin/env python3
"""Evaluate SaDVIO runs against TUM-VI mocap ground truth (evaluation only; GT never enters SaDVIO).

Estimate: log_slam/results.csv (keyframe poses T_w_f, frame = IMU body for the TUM-VI config).
profiling() rewrites the oldest window keyframe at every back-end step, so the last row per
timestamp is kept. GT: mav0/mocap0/data.csv (pose of the IMU in the mocap frame).

Alignment: SE3 for stereo modes, Sim3 for mono modes (scale reported).
  room1        ATE over the whole run.
  magistrale2  ATE over the start+end GT segments (TUM-VI protocol) and end drift after aligning
               on the start segment only.
Also: RPE (translation over 1 s and 5 s pairs, rotation over 1 s), coverage (estimated time span /
sequence span), resets (SaDVIO "Reinitializing" messages), NaN rows.

Usage: eval_traj.py <run_dir> [<run_dir> ...]      (writes metrics.json into each run dir)
       eval_traj.py --summary <label_dir>           (aggregates metrics.json, prints a table)
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np

DATA = Path('/home/deos/s.jois/EXECO/Simulator_Validation/SaD_VIO_data')  # datasets (ground truth)
SEQ_DIRS = {'room1': DATA / 'tumvi/dataset-room1_512_16/mav0',
            'magistrale2': DATA / 'tumvi/dataset-magistrale2_512_16/mav0'}
SEQ_DIRS.update({f'room{i}': DATA / f'tumvi/dataset-room{i}_512_16/mav0' for i in range(2, 7)})
# EuRoC Vicon rooms: ground truth = state_groundtruth_estimate0 (the IMU frame, SaDVIO's body frame)
EUROC_SEQS = {'V1_01': 'V1_01_easy', 'V1_02': 'V1_02_medium', 'V1_03': 'V1_03_difficult',
              'V2_01': 'V2_01_easy', 'V2_02': 'V2_02_medium', 'V2_03': 'V2_03_difficult',  # EuRoC Vicon rooms
              'MH_01': 'MH_01_easy', 'MH_02': 'MH_02_easy', 'MH_03': 'MH_03_medium',      # EuRoC Machine Hall
              'MH_04': 'MH_04_difficult', 'MH_05': 'MH_05_difficult'}
SEQ_DIRS.update({k: DATA / f'euroc/{v}/mav0' for k, v in EUROC_SEQS.items()})
# Real RealSense D455 trajectories: OptiTrack GT of the rigid body, whose transform to the IMU (SaDVIO's body) is
# not calibrated -> only metrics independent of it (positions; relative errors in the aligned world frame)
SEQ_DIRS.update({f'rs_{t}': DATA / f'realsense/{t}/mav0' for t in ['13_18_29', '13_21_59', '13_30_22', '13_31_26', '13_34_21', '13_36_44', '13_41_28', '13_42_11', '13_45_18']})
SEQ_DIRS['rs_13_18_29_synthimu'] = DATA / 'realsense/13_18_29_synthimu/mav0'
SEQ_DIRS['rs_hyb_realgyro'] = DATA / 'realsense/13_18_29_hyb_realgyro/mav0'
SEQ_DIRS['rs_hyb_realacc'] = DATA / 'realsense/13_18_29_hyb_realacc/mav0'
SEQ_DIRS['rs_accinterp'] = DATA / 'realsense/13_18_29_accinterp/mav0'
SEQ_DIRS['rs_synth_sh-30'] = DATA / 'realsense/13_18_29_synth_sh-30/mav0'
SEQ_DIRS['rs_synth_sh-15'] = DATA / 'realsense/13_18_29_synth_sh-15/mav0'
SEQ_DIRS['rs_synth_sh15'] = DATA / 'realsense/13_18_29_synth_sh15/mav0'
SEQ_DIRS['rs_synth_sh30'] = DATA / 'realsense/13_18_29_synth_sh30/mav0'
# Forced visual dropouts (make_dropout_seq.py): ground truth of the original sequence
DROP_SEQS = {'MH_01_drop': 'MH_01', 'room1_drop': 'room1'}
SEQ_DIRS['MH_01_drop'] = DATA / 'euroc/MH_01_easy_drop/mav0'
SEQ_DIRS['room1_drop'] = DATA / 'tumvi/dataset-room1_512_16_drop/mav0'
GT_MAX_GAP = 0.1   # s; no GT association across larger mocap gaps


def read_csv(path, ncols=None):
    rows = []
    for ln in Path(path).read_text().splitlines():
        if not ln.strip() or ln.lstrip().startswith(('#', 't')):
            continue
        rows.append([float(x) for x in ln.split(',')[:ncols]])
    return np.array(rows) if rows else np.zeros((0, 14))


def quat_to_R(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z); w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_estimate(run_dir, with_segments=False):
    """KF poses (last write per timestamp). With with_segments, also the segment of each KF: results.csv rows carry
    the number of re-initializations before them (column 15; older logs, truncated at every re-initialization, have
    a single segment). Each segment has its own world frame."""
    a = read_csv(run_dir / 'log_slam/results.csv')
    if len(a) == 0:
        empty = (np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3, 3)), 0)
        return empty + (np.zeros(0, int),) if with_segments else empty
    nan_rows = int(np.isnan(a).any(axis=1).sum())
    a = a[~np.isnan(a).any(axis=1)]
    last = {}
    for r in a:                                   # keep last write per timestamp
        last[int(r[0])] = r
    a = np.array([last[k] for k in sorted(last)])
    t = a[:, 0] * 1e-9
    R = np.stack([a[:, [2, 3, 4]], a[:, [6, 7, 8]], a[:, [10, 11, 12]]], axis=1)
    p = a[:, [5, 9, 13]]
    if with_segments:
        seg = a[:, 14].astype(int) if a.shape[1] > 14 else np.zeros(len(a), int)
        return t, p, R, nan_rows, seg
    return t, p, R, nan_rows


def load_gt(seq):
    seq = DROP_SEQS.get(seq, seq)
    if seq.startswith('rs_'):  # time [s], px, py, pz, qx, qy, qz, qw
        a = np.loadtxt(SEQ_DIRS[seq] / 'gt/data.csv', delimiter=',', skiprows=1)
        return a[:, 0], a[:, 1:4], np.stack([quat_to_R(q[3], q[0], q[1], q[2]) for q in a[:, 4:8]])
    gt_file = 'state_groundtruth_estimate0/data.csv' if seq in EUROC_SEQS else 'mocap0/data.csv'
    a = read_csv(SEQ_DIRS[seq] / gt_file, ncols=8)  # time [ns], px, py, pz, qw, qx, qy, qz
    t = a[:, 0] * 1e-9
    R = np.stack([quat_to_R(*q) for q in a[:, 4:8]])
    return t, a[:, 1:4], R


def associate(t_est, t_gt, p_gt, R_gt):
    """Linear interpolation of GT position at estimate times (rotation: nearest)."""
    idx = np.searchsorted(t_gt, t_est)
    ok = (idx > 0) & (idx < len(t_gt))
    i1 = np.clip(idx, 1, len(t_gt) - 1); i0 = i1 - 1
    ok &= (t_gt[i1] - t_gt[i0]) <= GT_MAX_GAP
    w = ((t_est - t_gt[i0]) / np.maximum(t_gt[i1] - t_gt[i0], 1e-9))[:, None]
    p = (1 - w) * p_gt[i0] + w * p_gt[i1]
    near = np.where(w[:, 0] < 0.5, i0, i1)
    return ok, p, R_gt[near]


def umeyama(src, dst, with_scale):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (np.trace(np.diag(D) @ S) / xs.var(0).sum()) if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def rot_angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)))


def rpe(t, p_est, p_gt, R_est, R_gt, delta, seg=None):
    errs_t, errs_r, j = [], [], 0
    for i in range(len(t)):
        while j < len(t) and t[j] - t[i] < delta:
            j += 1
        if j >= len(t):
            break
        if t[j] - t[i] > 1.5 * delta or (seg is not None and seg[i] != seg[j]):  # pairs within a segment only
            continue
        dp_e = R_est[i].T @ (p_est[j] - p_est[i]); dp_g = R_gt[i].T @ (p_gt[j] - p_gt[i])
        errs_t.append(np.linalg.norm(dp_e - dp_g))
        dR = (R_gt[i].T @ R_gt[j]).T @ (R_est[i].T @ R_est[j])
        errs_r.append(rot_angle_deg(dR[None])[0])
    if not errs_t:
        return None, None
    return float(np.sqrt(np.mean(np.square(errs_t)))), float(np.sqrt(np.mean(np.square(errs_r))))


def evaluate(run_dir: Path):
    meta = json.loads((run_dir / 'run.json').read_text())
    seq, mode = meta['seq'], meta['mode']
    mono = mode.startswith('mono')
    stdout = (run_dir / 'stdout.log').read_text(errors='replace') if (run_dir / 'stdout.log').exists() else ''
    crashed = meta['end_reason'].startswith('exited(') and meta['end_reason'] != 'exited(0)'
    m = dict(seq=seq, mode=mode, end_reason=meta['end_reason'], crashed=int(crashed), wall_s=meta['wall_s'],
             resets=stdout.count('Reinitializing'), stdout_nan=stdout.lower().count('nan'))
    t, p, R, nan_rows, seg = load_estimate(run_dir, with_segments=True)
    m['nan_rows'] = nan_rows
    m['n_kf'] = int(len(t))
    m['segments'] = int(len(np.unique(seg))) if len(seg) else 0
    cam_t = read_csv(SEQ_DIRS[seq] / 'cam0/data.csv', ncols=1)[:, 0] * 1e-9
    # coverage: time spanned by the segments (a re-initialization gap is not covered)
    span = sum(t[seg == k][-1] - t[seg == k][0] for k in np.unique(seg)) if len(t) > 1 else 0.0
    m['coverage'] = float(span / (cam_t[-1] - cam_t[0]))
    if len(t) < 3:
        m['ate_rmse'] = None
        (run_dir / 'metrics.json').write_text(json.dumps(m, indent=1))
        return m
    tg, pg, Rg = load_gt(seq)
    ok, p_gt, R_gt = associate(t, tg, pg, Rg)
    t, p, R, p_gt, R_gt, seg = t[ok], p[ok], R[ok], p_gt[ok], R_gt[ok], seg[ok]
    # whole run, one alignment for all segments (Sim3 in mono): what a user of the trajectory gets, including the jump
    # (or the continuity) at every re-initialization; equal to ate_rmse for an unbroken run
    if len(t) >= 3:
        sw, Rw, tw = umeyama(p, p_gt, with_scale=mono)
        ew = np.linalg.norm(sw * (Rw @ p.T).T + tw - p_gt, axis=1)
        m.update(ate_whole=float(np.sqrt(np.mean(ew ** 2))), ate_whole_max=float(ew.max()), scale_whole=float(sw))
        # jump at each re-initialization: the first KF of a segment placed with the alignment of the segment before it
        # (its own scale in mono); 0 for a perfectly continuous restart, the dropout drift when the state is carried
        jumps, ks = [], np.unique(seg)
        for k_prev, k in zip(ks[:-1], ks[1:]):
            sel = seg == k_prev
            if sel.sum() >= 3:
                sk, Rk, tk = umeyama(p[sel], p_gt[sel], with_scale=mono)
                i = np.argmax(seg == k)
                jumps.append(float(np.linalg.norm(sk * Rk @ p[i] + tk - p_gt[i])))
        m['junction_err'] = [round(j, 3) for j in jumps]
        m['junction_err_mean'] = float(np.mean(jumps)) if jumps else None
    # segments too short to align are left out
    keep = np.isin(seg, [k for k in np.unique(seg) if (seg == k).sum() >= 3])
    t, p, R, p_gt, R_gt, seg = t[keep], p[keep], R[keep], p_gt[keep], R_gt[keep], seg[keep]
    m['n_associated'] = int(len(t))
    if len(t) < 3:
        m['ate_rmse'] = None
        (run_dir / 'metrics.json').write_text(json.dumps(m, indent=1))
        return m
    # each segment aligned on its own (its own world frame, and scale in mono); the largest segment's scale is reported
    p_al, R_al = np.zeros_like(p), np.zeros_like(R)
    s, n_best = 1.0, 0
    for k in np.unique(seg):
        sel = seg == k
        sk, Ra, ta = umeyama(p[sel], p_gt[sel], with_scale=mono)
        p_al[sel] = sk * (Ra @ p[sel].T).T + ta
        R_al[sel] = np.einsum('ij,njk->nik', Ra, R[sel])
        if sel.sum() > n_best:
            s, n_best = sk, int(sel.sum())
    e = np.linalg.norm(p_al - p_gt, axis=1)
    m.update(ate_rmse=float(np.sqrt(np.mean(e ** 2))), ate_max=float(e.max()), scale=float(s),
             ate_rot_rmse_deg=float(np.sqrt(np.mean(rot_angle_deg(np.einsum('nji,njk->nik', R_gt, R_al)) ** 2))))
    if seq.startswith('rs_'):
        # GT body != estimate body: rotation metrics are meaningless; relative translation error in the aligned
        # world frame; Sim3 scale reported in all modes (metric-scale check)
        m['ate_rot_rmse_deg'] = None
        big = seg == np.bincount(seg).argmax()
        m['scale'] = float(umeyama(p[big], p_gt[big], with_scale=True)[0])
        for d in (1.0, 5.0):
            errs, j = [], 0
            for i in range(len(t)):
                while j < len(t) and t[j] - t[i] < d:
                    j += 1
                if j >= len(t):
                    break
                if t[j] - t[i] <= 1.5 * d and seg[i] == seg[j]:
                    errs.append(np.linalg.norm((p_al[j] - p_al[i]) - (p_gt[j] - p_gt[i])))
            m[f'rpe_t_{d:.0f}s'] = float(np.sqrt(np.mean(np.square(errs)))) if errs else None
        m['rpe_r_1s_deg'] = None
        same = seg[1:] == seg[:-1]  # steps within a segment
        m['path_ratio'] = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)[same]) /
                                np.sum(np.linalg.norm(np.diff(p_gt, axis=0), axis=1)[same]))
        (run_dir / 'metrics.json').write_text(json.dumps(m, indent=1))
        return m
    for d in (1.0, 5.0):
        rt, rr = rpe(t, p_al, p_gt, R_al, R_gt, d, seg)
        m[f'rpe_t_{d:.0f}s'] = rt
        if d == 1.0:
            m['rpe_r_1s_deg'] = rr
    if seq == 'magistrale2' and m['segments'] == 1:  # end drift after aligning on the start (one unbroken segment)
        start = t < t[0] + 60
        if start.sum() >= 3 and (~start).sum() >= 1:
            s0, R0, t0 = umeyama(p[start], p_gt[start], with_scale=mono)
            pe = s0 * (R0 @ p[~start].T).T + t0
            m['end_drift'] = float(np.linalg.norm(pe - p_gt[~start], axis=1).mean())
            m['end_segment_kf'] = int((~start).sum())
    (run_dir / 'metrics.json').write_text(json.dumps(m, indent=1))
    return m


def fmt(vals, digits=3):
    vals = [v for v in vals if v is not None]
    if not vals:
        return '—'
    if len(vals) == 1:
        return f'{vals[0]:.{digits}f}'
    return f'{np.mean(vals):.{digits}f} ± {np.std(vals):.{digits}f}'


def summary(label_dir: Path):
    rows = {}
    for mf in sorted(label_dir.glob('*/*/run_*/metrics.json')):
        m = json.loads(mf.read_text())
        rows.setdefault((m['seq'], m['mode']), []).append(m)
    cols = ['ate_rmse', 'ate_whole', 'rpe_t_1s', 'rpe_r_1s_deg', 'scale', 'end_drift', 'coverage', 'resets', 'crashed',
            'wall_s']
    print('| seq | mode | runs | ' + ' | '.join(cols) + ' |')
    print('|' + '---|' * (3 + len(cols)))
    for (seq, mode), ms in sorted(rows.items()):
        failed = sum(1 for m in ms if m.get('ate_rmse') is None)
        cells = [(str(sum(m.get(c, 0) for m in ms)) if c == 'crashed' else
                  fmt([m.get(c) for m in ms], 1 if c in ('resets', 'wall_s') else 3)) for c in cols]
        runs = f'{len(ms)}' + (f' ({failed} no traj)' if failed else '')
        print(f'| {seq} | {mode} | {runs} | ' + ' | '.join(cells) + ' |')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='*', type=Path)
    ap.add_argument('--summary', type=Path)
    a = ap.parse_args()
    for rd in a.paths:
        m = evaluate(rd)
        print(rd, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}))
    if a.summary:
        summary(a.summary)


if __name__ == '__main__':
    main()
