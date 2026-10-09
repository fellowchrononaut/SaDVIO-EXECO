#!/usr/bin/env python3
"""Evaluate loop detection (phase 0) from keyframe similarity matrices, with ground truth (evaluation only).

Inputs, per method and sequence: <desc>/<method>/<seq>_sim.npy (from lc_describe.py or bow_bench), the ground truth at
the keyframes (results/kfsets/<seq>_gt.npz) and the reference run's estimated keyframe poses (for the proximity gate).

Labels for a query keyframe q and an earlier keyframe d (at least MIN_GAP s before q):
  positive    GT positions within POS_DIST m and body orientations within POS_ANGLE deg
  negative    farther than NEG_DIST m or more than NEG_ANGLE deg apart
  otherwise   ambiguous: neither a hit nor a false positive
Metrics (queries with at least one positive candidate count for recall):
  R@1, R@5    the top-1 / any of the top 5 candidates by similarity is positive
  R@100P      share of those queries found when the top-1 is accepted above a score threshold, at the highest
              threshold-free recall with no negative accepted over all queries (also the ones without a loop)
  R@95P       the same at 95 % precision
Proximity gate (option A, no descriptor): candidates are earlier keyframes whose *estimated* position (drifting
odometry of the reference run) is within r m of the query's; reports how often a positive is inside the gate and
the mean number of candidates in it, and the R@1 of each method restricted to the gate.

usage: lc_eval.py --desc DIR [--methods a,b] [--seqs ...] [--md out.md]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results'
VIO = HERE.parent.parent / 'vio_imu_fix'
sys.path.insert(0, str(HERE))
from make_kf_sets import reference_run  # noqa: E402

MIN_GAP, POS_DIST, POS_ANGLE, NEG_DIST, NEG_ANGLE = 20.0, 1.0, 30.0, 3.0, 60.0
GATES = (1.0, 2.0, 4.0)


def rot_angle_deg(Ra, Rb):
    c = (np.einsum('nij,mij->nm', Ra, Rb) - 1) / 2  # trace(Ra^T Rb) for all pairs
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def labels(gt):
    t = gt['t_ns'] * 1e-9
    p, R = gt['p'], gt['R']
    d = np.linalg.norm(p[:, None] - p[None], axis=2)
    a = rot_angle_deg(R, R)
    earlier = (t[:, None] - t[None]) > MIN_GAP
    pos = earlier & (d < POS_DIST) & (a < POS_ANGLE)
    neg = earlier & ((d > NEG_DIST) | (a > NEG_ANGLE))
    return earlier, pos, neg


def estimated_positions(seq, t_ns):
    rd = reference_run(seq)
    a = np.loadtxt(rd / 'log_slam' / 'results.csv', delimiter=',', skiprows=1)
    tt = a[:, 0].astype(np.int64)
    idx = np.searchsorted(tt, t_ns)
    idx = np.clip(idx, 0, len(tt) - 1)
    return a[idx][:, [5, 9, 13]]  # T_wf(03), T_wf(13), T_wf(23)


def recall_at_precision(top_score, top_pos, top_neg, n_with_loop, prec):
    order = np.argsort(-top_score)
    tp = np.cumsum(top_pos[order])
    fp = np.cumsum(top_neg[order])
    ok = tp / np.maximum(tp + fp, 1) >= prec
    ok &= (tp + fp) > 0
    return float(tp[ok].max() / n_with_loop) if ok.any() and n_with_loop else 0.0


def evaluate(sim, earlier, pos, neg, gate=None):
    S = np.where(earlier if gate is None else earlier & gate, sim.astype(np.float32), -np.inf)
    has_cand = np.isfinite(S).any(axis=1)
    with_loop = pos.any(axis=1)
    top5 = np.argsort(-S, axis=1)[:, :5]
    top1 = top5[:, 0]
    rows = np.arange(len(S))
    r1 = pos[rows, top1] & has_cand
    r5 = np.array([pos[q, top5[q][np.isfinite(S[q, top5[q]])]].any() for q in rows])
    n = with_loop.sum()
    top_score = np.where(has_cand, S[rows, top1], -np.inf)
    out = {'queries_with_loop': int(n), 'R@1': float(r1[with_loop].mean()) if n else np.nan,
           'R@5': float(r5[with_loop].mean()) if n else np.nan}
    valid = has_cand
    out['R@100P'] = recall_at_precision(top_score[valid], r1[valid], neg[rows, top1][valid], n, 1.0)
    out['R@95P'] = recall_at_precision(top_score[valid], r1[valid], neg[rows, top1][valid], n, 0.95)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--desc', required=True, type=Path)
    ap.add_argument('--methods', default='')
    ap.add_argument('--seqs', default='')
    ap.add_argument('--md', type=Path)
    a = ap.parse_args()
    methods = a.methods.split(',') if a.methods else sorted(
        str(p.relative_to(a.desc)) for p in a.desc.glob('**/') if list(p.glob('*_sim.npy')))
    seqs = a.seqs.split(',') if a.seqs else sorted(p.name[:-7] for p in (RES / 'kfsets').glob('*_gt.npz'))
    lines = ['| seq | queries with a loop | gate r (m): positive inside / candidates | ' +
             ' | '.join(f'{m} R@1 / R@5 / R@100P / R@95P / R@1 in 2 m gate' for m in methods) + ' |',
             '|---|---|---|' + '---|' * len(methods)]
    agg = {m: [] for m in methods}
    for s in seqs:
        gt = np.load(RES / 'kfsets' / f'{s}_gt.npz')
        earlier, pos, neg = labels(gt)
        est = estimated_positions(s, gt['t_ns'])
        de = np.linalg.norm(est[:, None] - est[None], axis=2)
        with_loop = pos.any(axis=1)
        gate_txt = []
        for r in GATES:
            g = earlier & (de < r)
            inside = (pos & g).any(axis=1)[with_loop].mean() if with_loop.any() else np.nan
            gate_txt.append(f'{r:g}: {inside:.2f} / {g.sum(axis=1)[with_loop].mean():.0f}')
        cells = []
        for m in methods:
            f = a.desc / m / f'{s}_sim.npy'
            if not f.exists():
                cells.append('–')
                continue
            sim = np.load(f)
            e = evaluate(sim, earlier, pos, neg)
            eg = evaluate(sim, earlier, pos, neg, gate=de < 2.0)
            agg[m].append((with_loop.sum(), e, eg))
            cells.append(f"{e['R@1']:.2f} / {e['R@5']:.2f} / {e['R@100P']:.2f} / {e['R@95P']:.2f} / {eg['R@1']:.2f}")
        lines.append(f'| {s} | {with_loop.sum()} | ' + ', '.join(gate_txt) + ' | ' + ' | '.join(cells) + ' |')
    # Query-weighted means over the sequences
    cells = []
    for m in methods:
        if not agg[m]:
            cells.append('–')
            continue
        w = np.array([x[0] for x in agg[m]], float)
        mean = lambda k, i=1: float(np.nansum(w * np.array([x[i][k] for x in agg[m]])) / w.sum())  # noqa: E731
        cells.append(f"{mean('R@1'):.2f} / {mean('R@5'):.2f} / {mean('R@100P'):.2f} / {mean('R@95P'):.2f} / "
                     f"{mean('R@1', 2):.2f}")
    lines.append('| **all (query-weighted)** | | | ' + ' | '.join(cells) + ' |')
    text = '\n'.join(lines)
    print(text)
    if a.md:
        a.md.write_text(text + '\n')


if __name__ == '__main__':
    main()
