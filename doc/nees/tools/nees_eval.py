#!/usr/bin/env python3
"""NEES of SaDVIO's pose covariances against ground truth (evaluation only; GT never enters SaDVIO).

Inputs (written with log_pose_covariance: 1):
  A  front end  log_slam/cov_front.csv   ESKF estimate T_ref_f of each frame w.r.t. the last KF and its 6x6
                                          covariance of the right perturbation T E(theta, rho), E = [Exp(theta), rho]
                                          (order rotation, translation)
  B  back end   log_slam/cov_window.csv  after each VIO window solve: poses T_f_w of the newest, previous and oldest
                                          window frames and the 18x18 covariance of the pose parameter errors
                                          T_f_w <- T_f_w [Exp(w), t] (landmarks Schur-eliminated, gauge by
                                          pseudo-inverse)

Both are evaluated on the same quantity: the pose of each new KF relative to the previous KF, in A's
right-perturbation error, against GT interpolated at the frame timestamps (linear position, slerp rotation).
B's covariance is propagated to that relative pose with a numerical Jacobian. B is also evaluated relative to the
oldest window frame ("B_win").

NEES per sample: translation (3 dof), rotation (3 dof), full pose (6 dof). The figure replicates the colleagues'
plot: ln(sum_N NEES / (dof N)) over a sliding window of N keyframes, two-sided 95 % chi-square bounds, sequences
concatenated in a fixed order, with a strip of the sequence difficulty.

Usage: nees_eval.py <label> [--out doc/nees/results]
"""
import argparse, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'vio_imu_fix/tools'))
import eval_traj as et  # noqa: E402  (GT loading)

RUNS = Path(__file__).resolve().parents[2] / 'vio_imu_fix/runs'
SEQS = ['MH_01', 'MH_02', 'MH_03', 'MH_04', 'MH_05', 'V1_01', 'V1_02', 'V1_03', 'V2_01', 'V2_02', 'V2_03',
        'room1', 'room2', 'room3', 'room4', 'room5', 'room6']
CLASS = {'MH_01': 'EuRoC easy', 'MH_02': 'EuRoC easy', 'V1_01': 'EuRoC easy', 'V2_01': 'EuRoC easy',
         'MH_03': 'EuRoC medium', 'V1_02': 'EuRoC medium', 'V2_02': 'EuRoC medium',
         'MH_04': 'EuRoC difficult', 'MH_05': 'EuRoC difficult', 'V1_03': 'EuRoC difficult',
         'V2_03': 'EuRoC difficult'}
CLASSES = ['TUM-VI room', 'EuRoC difficult', 'EuRoC medium', 'EuRoC easy']  # strip rows, bottom to top
GT_GAP = 0.05  # s; no GT interpolation across larger gaps


# ---------------------------------------------------------------- SO(3) / SE(3)
def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def exp_so3(w):
    th = np.linalg.norm(w)
    K = skew(w)
    if th < 1e-10:
        return np.eye(3) + K
    return np.eye(3) + math.sin(th) / th * K + (1 - math.cos(th)) / th ** 2 * K @ K


def log_so3(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    th = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-8:
        return 0.5 * v
    if th > math.pi - 1e-6:  # near pi: from the symmetric part
        A = (R + np.eye(3)) / 2
        k = np.argmax(np.diag(A))
        a = A[:, k] / math.sqrt(A[k, k])
        return th * a * (1 if a @ v >= 0 else -1)
    return th / (2 * math.sin(th)) * v


def T_of(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def E(d):  # [Exp(w), t] of a 6-vector (w, t)
    return T_of(exp_so3(d[:3]), d[3:])


def right_err(T0, T):
    """xi with T = T0 [Exp(theta), rho]: theta = Log(R0^T R), rho = R0^T (t - t0)."""
    R0 = T0[:3, :3]
    return np.concatenate([log_so3(R0.T @ T[:3, :3]), R0.T @ (T[:3, 3] - T0[:3, 3])])


# ---------------------------------------------------------------- inputs
def read_rows(path, n_int):
    """Rows of a log: the first n_int columns as int (ns timestamps exceed float precision), the rest as float."""
    ints, vals = [], []
    for ln in Path(path).read_text().splitlines()[1:]:
        if not ln.strip():
            continue
        f = ln.split(',')
        ints.append([int(float(x)) if 'e' in x or '.' in x else int(x) for x in f[:n_int]])
        vals.append([float(x) for x in f[n_int:]])
    return np.array(ints, dtype=np.int64), np.array(vals)


def gt_interp(gt, ts_ns):
    t_gt, p_gt, R_gt = gt
    t = ts_ns * 1e-9
    i1 = np.searchsorted(t_gt, t)
    if i1 <= 0 or i1 >= len(t_gt) or t_gt[i1] - t_gt[i1 - 1] > GT_GAP:
        return None
    i0 = i1 - 1
    w = (t - t_gt[i0]) / (t_gt[i1] - t_gt[i0])
    p = (1 - w) * p_gt[i0] + w * p_gt[i1]
    R = R_gt[i0] @ exp_so3(w * log_so3(R_gt[i0].T @ R_gt[i1]))
    return T_of(R, p)


def nees_parts(e, P):
    """Translation, rotation and full-pose NEES of a (rotation, translation) error with covariance P."""
    try:
        return (float(e[3:] @ np.linalg.solve(P[3:, 3:], e[3:])), float(e[:3] @ np.linalg.solve(P[:3, :3], e[:3])),
                float(e @ np.linalg.solve(P, e)))
    except np.linalg.LinAlgError:
        return (np.nan, np.nan, np.nan)


def rel_cov(Ts, C, i_ref, i_cur, h=1e-6):
    """Covariance of the right error of T_ref_cur = T_ref_w T_cur_w^-1 from the joint covariance C of the
    T_f_w <- T_f_w E(d) errors of the logged frames (numerical Jacobian)."""
    def rel(d):
        Tr = Ts[i_ref] @ E(d[6 * i_ref:6 * i_ref + 6])
        Tc = Ts[i_cur] @ E(d[6 * i_cur:6 * i_cur + 6])
        return Tr @ np.linalg.inv(Tc)
    T0 = rel(np.zeros(18))
    J = np.zeros((6, 18))
    for k in list(range(6 * i_ref, 6 * i_ref + 6)) + list(range(6 * i_cur, 6 * i_cur + 6)):
        d = np.zeros(18)
        d[k] = h
        J[:, k] = (right_err(T0, rel(d)) - right_err(T0, rel(-d))) / (2 * h)
    return T0, J @ C @ J.T


def eval_run(run_dir, seq):
    gt = et.load_gt(seq)
    ls = run_dir / 'log_slam'
    wi, wv = read_rows(ls / 'cov_window.csv', 3)
    fi, fv = read_rows(ls / 'cov_front.csv', 3)
    front = {}
    for ints, v in zip(fi, fv):  # last row per frame
        front[int(ints[0])] = (int(ints[1]), int(ints[2]), v)
    seen, rows = set(), []
    for ints, v in zip(wi, wv):
        ts_new, ts_prev, ts_old = (int(x) for x in ints)
        if ts_new in seen:  # first window solve of each KF
            continue
        seen.add(ts_new)
        n_null = v[2]
        Ts = [v[5 + 12 * i:17 + 12 * i].reshape(3, 4) for i in range(3)]
        Ts = [np.vstack([T, [0, 0, 0, 1]]) for T in Ts]
        C = v[41:].reshape(18, 18)
        g_new, g_prev, g_old = gt_interp(gt, ts_new), gt_interp(gt, ts_prev), gt_interp(gt, ts_old)
        if g_new is None or g_prev is None:
            continue
        row = {'seq': seq, 'ts': ts_new, 'n_null': n_null}
        # B: relative to the previous KF
        T_pn, P = rel_cov(Ts, C, 1, 0)
        e = right_err(T_pn, np.linalg.inv(g_prev) @ g_new)
        row['B'] = nees_parts(e, P)
        row['B_err'] = e
        row['B_sd'] = np.sqrt(np.diag(P))
        # B_win: relative to the oldest window frame
        if g_old is not None and ts_old != ts_prev:
            T_on, P = rel_cov(Ts, C, 2, 0)
            row['Bw'] = nees_parts(right_err(T_on, np.linalg.inv(g_old) @ g_new), P)
        # A: the ESKF estimate of the same KF
        if ts_new in front:
            ref, ok, fvals = front[ts_new]
            g_ref = gt_interp(gt, ref)
            if ok and g_ref is not None:
                T_rf = np.vstack([fvals[:12].reshape(3, 4), [0, 0, 0, 1]])
                Pf = fvals[12:48].reshape(6, 6)
                e = right_err(T_rf, np.linalg.inv(g_ref) @ g_new)
                row['A'] = nees_parts(e, Pf)
                row['A_err'] = e
                row['A_sd'] = np.sqrt(np.diag(Pf))
                row['A_same_ref'] = ref == ts_prev
        rows.append(row)
    return rows


# ---------------------------------------------------------------- figure
INK, MUTED, GRID, SERIES = '#2b2a27', '#898781', '#e1e0d9', '#2a78d6'


def windowed(x, N):
    """ln(sum of N consecutive NEES / (dof N)) at the window centres (x already divided by dof)."""
    c = np.concatenate([[0], np.cumsum(x)])
    s = (c[N:] - c[:-N]) / N
    centre = np.arange(len(s)) + (N - 1) / 2
    return centre, np.log(s)


def figure(samples, key, title, path, Ns=(1, 1000, 5000), dof=3):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'text.color': INK, 'axes.titlesize': 10})
    S = [s for s in samples if key in s and np.all(np.isfinite(s[key]))]
    seqs = [s['seq'] for s in S]
    comp = {'Translation (3 dof)': np.array([s[key][0] for s in S]) / dof,
            'Rotation (3 dof)': np.array([s[key][1] for s in S]) / dof}
    n = len(S)
    fig, axes = plt.subplots(len(Ns) + 1, 2, figsize=(15, 9.5), sharex=True,
                             gridspec_kw={'height_ratios': [1] * len(Ns) + [0.8], 'hspace': 0.35, 'wspace': 0.12})
    fig.patch.set_facecolor('#fcfcfb')
    for c, (cname, x) in enumerate(comp.items()):
        for r, N in enumerate(Ns):
            ax = axes[r, c]
            ax.set_facecolor('#fcfcfb')
            ax.grid(True, color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            if N > n:
                ax.text(0.5, 0.5, f'N = {N} > {n} samples', transform=ax.transAxes, ha='center', color=MUTED)
                continue
            xc, y = windowed(x, N)
            ax.plot(xc, y, color=SERIES, lw=0.6 if N == 1 else 1.6, label='NEES')
            lo, hi = (math.log(chi2.ppf(q, dof * N) / (dof * N)) for q in (0.025, 0.975))
            for b in (lo, hi):
                ax.plot([xc[0], xc[-1]], [b, b], color=MUTED, ls='--', lw=1, label='95 % bounds' if b == lo else None)
            ax.axhline(0, color=MUTED, lw=0.6)
            inside = np.mean((y >= lo) & (y <= hi)) * 100
            above = np.mean(y > hi) * 100
            ax.set_title(f'{cname}, N = {N}', loc='left', fontweight='bold')
            ax.text(1.0, 1.02, f'{inside:.0f} % inside, {above:.0f} % above', transform=ax.transAxes, ha='right',
                    va='bottom', color=MUTED, fontsize=8)
            if N == 1:
                ax.set_ylim(min(-12, np.nanpercentile(y, 0.5) - 1), max(8, np.nanpercentile(y, 99.9) + 1))
        # sequence strip
        ax = axes[-1, c]
        ax.set_facecolor('#fcfcfb')
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        starts = [0] + [i for i in range(1, n) if seqs[i] != seqs[i - 1]] + [n]
        gap = 0.004 * n
        last_row, flip = None, False
        for a, b in zip(starts[:-1], starts[1:]):
            q = seqs[a]
            row = CLASSES.index(CLASS.get(q, 'TUM-VI room'))
            ax.add_patch(plt.Rectangle((a + gap / 2, row - 0.3), b - a - gap, 0.6, color=SERIES, lw=0))
            flip = (not flip) if row == last_row and b - a < 0.05 * n else False  # stagger crowded labels
            last_row = row
            short = q.replace('room', 'r').replace('MH_0', 'MH').replace('_0', '.')
            ax.text((a + b) / 2, row + (0.36 if not flip else -0.36), short, ha='center',
                    va='bottom' if not flip else 'top', fontsize=7, color=INK)
            for rr in range(len(Ns)):
                axes[rr, c].axvline(a, color=GRID, lw=0.8, zorder=0)
        ax.set_yticks(range(len(CLASSES)))
        ax.set_yticklabels(CLASSES)
        ax.set_ylim(-0.6, len(CLASSES) - 0.1)
        ax.set_xlim(0, n)
        ax.set_xlabel('Keyframe no. (sequences concatenated)')
        if c == 1:
            ax.set_yticklabels([])
    fig.text(0.075, 0.6, r'$\ln\left(\frac{\mathrm{NEES}}{\mathrm{dof}\cdot N}\right)$', rotation=90, va='center',
             fontsize=12)
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.985)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, ['NEES', 'Upper/lower bound (p = 0.05)'], loc='upper center', bbox_to_anchor=(0.5, 0.955),
               ncol=2, frameon=False, fontsize=9)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- summary
def bounds(dof, N=1):
    return chi2.ppf(0.025, dof * N) / (dof * N), chi2.ppf(0.975, dof * N) / (dof * N)


def summary_table(samples, keys):
    lines = ['| seq | KFs | ' + ' | '.join(f'{k} ANEES t / r / pose' for k in keys) + ' | ' +
             ' | '.join(f'{k} median t / r' for k in keys) + ' |',
             '|---|---|' + '---|' * (2 * len(keys))]
    groups = [(q, [s for s in samples if s['seq'] == q]) for q in SEQS] + [('all', samples)]
    for q, S in groups:
        if not S:
            continue
        cells, med = [], []
        for k in keys:
            a = np.array([s[k] for s in S if k in s])
            if len(a) == 0:
                cells.append('-'); med.append('-'); continue
            an = np.nanmean(a, axis=0) / np.array([3, 3, 6])
            md = np.nanmedian(a, axis=0) / np.array([3, 3, 6])
            cells.append(f'{an[0]:.2f} / {an[1]:.2f} / {an[2]:.2f}')
            med.append(f'{md[0]:.2f} / {md[1]:.2f}')
        lines.append(f'| {q} | {len(S)} | ' + ' | '.join(cells) + ' | ' + ' | '.join(med) + ' |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--mode', default='bimonovio')
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'results'))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = []
    for q in SEQS:
        runs = sorted((RUNS / a.label / q / a.mode).glob('run_*'))
        if not runs or not (runs[0] / 'log_slam/cov_window.csv').exists():
            print(f'{q}: no run'); continue
        rows = eval_run(runs[0], q)
        print(f'{q}: {len(rows)} KFs')
        samples += rows
    np.save(out / f'{a.label}_samples.npy', np.array(samples, dtype=object), allow_pickle=True)

    keys = ['A', 'B', 'Bw']
    md = [f'# NEES, {a.label} ({a.mode})', '',
          'ANEES = mean NEES / dof (1 = consistent; > 1 overconfident, < 1 underconfident). '
          'Median NEES / dof of a consistent 3-dof estimator: 0.79. '
          f'95 % bounds of a single sample (3 dof): {bounds(3)[0]:.3f} .. {bounds(3)[1]:.2f}.', '',
          'A = front-end ESKF relative pose; B = window covariance of the relative pose to the previous KF; '
          'Bw = window covariance of the pose relative to the oldest window frame.', '',
          summary_table(samples, keys), '']
    for k in ('A', 'B'):
        S = [s for s in samples if k + '_err' in s]
        err = np.array([s[k + '_err'] for s in S]); sd = np.array([s[k + '_sd'] for s in S])
        md.append(f'{k}: median |error| rot {np.degrees(np.median(np.linalg.norm(err[:, :3], axis=1))):.3f} deg, '
                  f'trans {1000 * np.median(np.linalg.norm(err[:, 3:], axis=1)):.2f} mm; median sigma per axis rot '
                  f'{np.degrees(np.median(sd[:, :3])):.3f} deg, trans {1000 * np.median(sd[:, 3:]):.2f} mm')
    same = [s['A_same_ref'] for s in samples if 'A_same_ref' in s]
    md.append(f'\nA reference KF = B previous KF for {100 * np.mean(same):.1f} % of the samples.')
    nn = np.array([s['n_null'] for s in samples])
    md.append(f'Window null space size: ' + ', '.join(f'{int(v)}: {c}' for v, c in zip(*np.unique(nn, return_counts=True))))
    (out / f'{a.label}_summary.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))

    figure(samples, 'A', 'Normalized NEES, front-end ESKF covariance (relative KF pose), stereo VIO dense prior',
           out / f'{a.label}_nees_A_front.png')
    figure(samples, 'B', 'Normalized NEES, sliding-window covariance (relative KF pose), stereo VIO dense prior',
           out / f'{a.label}_nees_B_window.png')


if __name__ == '__main__':
    main()
