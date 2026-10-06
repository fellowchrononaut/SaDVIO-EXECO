#!/usr/bin/env python3
"""Compare two run labels cell by cell (sequence, mode): ATE, resets, KFs, marginalization and back-end time per KF.

Usage: compare_labels.py <label_before> <label_after> [<label_before> <label_after> ...]
Runs without metrics.json are evaluated first (eval_traj.evaluate). Prints one markdown table per pair.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_traj import evaluate  # noqa: E402

RUNS = HERE.parent / 'runs'


def profiler(rd):
    out = {}
    pf = rd / 'log_slam' / 'slam_profiler.txt'
    if pf.exists():
        for ln in pf.read_text().splitlines():
            if ':' in ln:
                k, v = ln.split(':', 1)
                try:
                    out[k.strip()] = float(v)
                except ValueError:
                    pass
    return out


def cell(label, seq, mode):
    vals = []
    for rd in sorted((RUNS / label / seq / mode).glob('run_*')):
        mf = rd / 'metrics.json'
        try:
            m = json.loads(mf.read_text()) if mf.exists() else evaluate(rd)
        except Exception:
            m = {}
        p = profiler(rd)
        vals.append((m.get('ate_rmse', np.nan), m.get('resets', np.nan), p.get('Number of keyframes', np.nan),
                     p.get('Marginalization dt', np.nan), p.get('Back end dt', np.nan)))
    return np.array(vals, dtype=float) if vals else None


def fmt(a, d=3):
    return '–' if a is None else ' / '.join(f'{x:.{d}f}' for x in a)


for before, after in zip(sys.argv[1::2], sys.argv[2::2]):
    print(f'\n{before} -> {after}\n')
    print('| seq | mode | ATE before (runs) | ATE after (runs) | resets b / a | KFs b / a | marg ms/KF b / a | BE ms/KF b / a |')
    print('|---|---|---|---|---|---|---|---|')
    for sd in sorted((RUNS / before).glob('*')):
        for md in sorted(sd.glob('*')):
            b, a = cell(before, sd.name, md.name), cell(after, sd.name, md.name)
            if b is None or a is None:
                continue
            mb, ma = np.nanmean(b, 0), np.nanmean(a, 0)
            print(f'| {sd.name} | {md.name} | {fmt(b[:, 0])} | {fmt(a[:, 0])} | {mb[1]:.1f} / {ma[1]:.1f} | '
                  f'{mb[2]:.0f} / {ma[2]:.0f} | {mb[3]:.1f} / {ma[3]:.1f} | {mb[4]:.1f} / {ma[4]:.1f} |')
