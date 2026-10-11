#!/usr/bin/env python3
"""Summary of the final regression matrix (tools/run_final_matrix.sh, labels fin_<config>), ground truth in evaluation
only. Per configuration and sequence, means over the runs: ATE of the window (results.csv) and of the loop-closed
trajectory (results_loop.csv), re-initializations, coverage, runs that crashed. Medians over the sequences.
Writes results/final_matrix.md. usage: final_summary.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VIO = HERE.parents[1] / 'vio_imu_fix'
sys.path.insert(0, str(VIO / 'tools'))
from eval_traj import evaluate  # noqa: E402

CONFIGS = ['vio_noprior', 'vio_dense', 'vio_sparse', 'vo_stereo', 'vio_mono', 'vo_mono']
SEQS = ['room1', 'room2', 'room3', 'room4', 'room5', 'room6', 'magistrale2', 'MH_01', 'MH_02', 'MH_03', 'MH_04',
        'MH_05', 'V1_01', 'V1_02', 'V1_03', 'V2_01', 'V2_02', 'V2_03']


def cell(label, seq):
    runs = sorted((VIO / 'runs' / label / seq).glob('*/run_*'))
    rows = []
    for rd in runs:
        try:
            m0 = evaluate(rd)
            m1 = evaluate(rd, 'results_loop.csv') if (rd / 'log_slam/results_loop.csv').exists() else {}
        except Exception:  # noqa: BLE001
            rows.append(None)
            continue
        rows.append((m0.get('ate_rmse'), m1.get('ate_rmse'), m0['resets'], m0['coverage'], m0['crashed']))
    return rows


def fmt(v):
    return '–' if v is None or not np.isfinite(v) else f'{v:.3f}'


def main():
    out = ['# Final regression matrix (every shipped default, loop closure on)', '',
           'Means over 2 runs. ATE window = results.csv; loop = results_loop.csv (final loop-closed trajectory). '
           'Ground truth in evaluation only.', '']
    for c in CONFIGS:
        out += [f'## {c}', '', '| seq | ATE window | ATE loop | resets | coverage | crashed |', '|---|---|---|---|---|---|']
        med_w, med_l = [], []
        for q in SEQS:
            rows = [r for r in cell(f'fin_{c}', q) if r]
            if not rows:
                out.append(f'| {q} | – | – | – | – | – |')
                continue
            w = [r[0] for r in rows if r[0] is not None]
            lo = [r[1] for r in rows if r[1] is not None]
            mw = float(np.mean(w)) if w else None
            ml = float(np.mean(lo)) if lo else None
            if mw is not None:
                med_w.append(mw)
            if ml is not None:
                med_l.append(ml)
            out.append(f'| {q} | {fmt(mw)} | {fmt(ml)} | {sum(r[2] for r in rows)} | '
                       f'{np.mean([r[3] for r in rows]):.2f} | {sum(r[4] for r in rows)} |')
        out += ['', f'Median over sequences: window {fmt(np.median(med_w) if med_w else None)}, '
                    f'loop {fmt(np.median(med_l) if med_l else None)}', '']
    text = '\n'.join(out)
    (HERE.parent / 'results' / 'final_matrix.md').write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
