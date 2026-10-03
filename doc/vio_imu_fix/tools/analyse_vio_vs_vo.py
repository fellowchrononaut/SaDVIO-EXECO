import sys, numpy as np
"""VO vs VIO error structure on room1: drift growth, yaw/tilt split, Sim3 scale, VIO velocity and bias vs GT.
Usage: analyse_vio_vs_vo.py <label> [<label> ...]   (labels under ../runs)"""
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_traj import *

def rotvec(R):
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if a < 1e-9: return np.zeros(3)
    return a / (2 * np.sin(a)) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

def analyse(run_dir, seq='room1'):
    t, p, R, _ = load_estimate(run_dir)
    tg, pg, Rg = load_gt(seq)
    ok, p_gt, R_gt = associate(t, tg, pg, Rg)
    t, p, R, p_gt, R_gt = t[ok], p[ok], R[ok], p_gt[ok], R_gt[ok]
    out = {}
    s3, _, _ = umeyama(p, p_gt, True); out['sim3_scale'] = s3
    s, Ra, ta = umeyama(p, p_gt, False)
    p_al = (Ra @ p.T).T + ta; R_al = np.einsum('ij,njk->nik', Ra, R)
    e = np.linalg.norm(p_al - p_gt, axis=1)
    out['ate'] = np.sqrt(np.mean(e**2))
    # error growth: align on the first 10 s only, error at later times (drift from the start)
    first = t < t[0] + 10
    _, R0, t0 = umeyama(p[first], p_gt[first], False)
    pe = (R0 @ p.T).T + t0
    d = np.linalg.norm(pe - p_gt, axis=1)
    for T in (20, 40, 80, 120):
        m = (t - t[0] > T - 2) & (t - t[0] < T + 2)
        out[f'drift@{T}s'] = d[m].mean() if m.any() else np.nan
    # rotation error split in the GT (gravity-aligned mocap) world: yaw (about z) and tilt
    R0al = np.einsum('ij,njk->nik', R0, R)
    yaw, tilt = [], []
    for Re, Rgi in zip(R0al, R_gt):
        dR = Re @ Rgi.T                       # world-frame error
        w = rotvec(dR); yaw.append(abs(w[2])); tilt.append(np.linalg.norm(w[:2]))
    yaw, tilt = np.degrees(yaw), np.degrees(tilt)
    out['yaw_end_deg'] = yaw[-50:].mean(); out['tilt_end_deg'] = tilt[-50:].mean(); out['tilt_max_deg'] = tilt.max()
    # path length ratio (local scale) over 2 s steps
    step = []
    for i in range(0, len(t) - 1):
        j = np.searchsorted(t, t[i] + 2.0)
        if j >= len(t): break
        lg = np.linalg.norm(p_gt[j] - p_gt[i])
        if lg > 0.3: step.append(np.linalg.norm(p[j] - p[i]) / lg)
    out['local_scale_median'] = np.median(step); out['local_scale_iqr'] = np.subtract(*np.percentile(step, [75, 25]))
    return out, (t, p_gt, R_gt, R0, t0)

def vio_states(run_dir, aux, seq='room1'):
    a = np.genfromtxt(run_dir / 'log_slam/vio_diag.csv', delimiter=',', skip_header=1, dtype=float)
    tk = a[:, 0] * 1e-9; v = a[:, 13:16]; ba = a[:, 16:19]; bg = a[:, 19:22]
    tg, pg, Rg = load_gt(seq)
    # GT world velocity by central differences, rotated into the estimate world (R0^T)
    t, p_gt, R_gt, R0, t0 = aux
    vg = np.gradient(pg, tg, axis=0)
    idx = np.clip(np.searchsorted(tg, tk), 1, len(tg) - 1)
    good = np.abs(tg[idx] - tk) < 0.02
    v_gt_est = (R0.T @ vg[idx].T).T
    ev = np.linalg.norm(v[good] - v_gt_est[good], axis=1)
    return dict(n_kf_opt=len(tk), factor_dt_median=np.median(a[:, 2]), n_imu_factors_median=np.median(a[:, 6]),
                v_err_rms=np.sqrt(np.mean(ev**2)), v_gt_rms=np.sqrt(np.mean(np.sum(v_gt_est[good]**2, 1))),
                ba_norm_median=np.median(np.linalg.norm(ba, axis=1)), ba_norm_p90=np.percentile(np.linalg.norm(ba, axis=1), 90),
                ba_std=np.linalg.norm(ba.std(0)), bg_norm_median=np.median(np.linalg.norm(bg, axis=1)),
                ba_first=np.linalg.norm(ba[:20].mean(0)), ba_last=np.linalg.norm(ba[-50:].mean(0)))

root = Path(__file__).resolve().parent.parent / 'runs'
keys = ['ate', 'drift@40s', 'drift@120s', 'yaw_end_deg', 'tilt_end_deg', 'sim3_scale']
skeys = ['v_err_rms', 'ba_norm_median', 'ba_std', 'ba_last']
for label in sys.argv[1:]:
    for mode in ('bimono', 'bimonovio'):
        rds = sorted((root / label / 'room1' / mode).glob('run_*'))
        if not rds: continue
        O, S = [], []
        for rd in rds:
            o, aux = analyse(rd); O.append(o)
            if mode == 'bimonovio': S.append(vio_states(rd, aux))
        line = f'{label:12s} {mode:10s} ' + ' '.join(f'{k}={np.mean([o[k] for o in O]):.3f}±{np.std([o[k] for o in O]):.3f}' for k in keys)
        if S: line += ' | ' + ' '.join(f'{k}={np.mean([x[k] for x in S]):.3f}' for k in skeys)
        print(line)
