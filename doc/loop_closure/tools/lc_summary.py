#!/usr/bin/env python3
"""Summary of the loop closure matrix (tools/run_lc_matrix.sh), ground truth in evaluation only.

For every run of the variants (opt1, opt2, bowwin, cpu) x configurations (bimonovio, bimono, monovio, sadvio): ATE and
RPE of the window's trajectory (results.csv), of the loop-closed final trajectory (results_loop.csv) and of the online
output (results_loop_online.csv); loops accepted and correct (lc_runs_eval.loop_precision); cost from the profiler; the
run's coverage and end reason (a run cut short by the runner's stall timer would show a low coverage).
The baseline without loop closure is opt1's results.csv (option 1 leaves the window untouched).

Writes results/matrix_runs.json (one entry per run, cached: delete it to re-evaluate) and results/matrix_summary.md.
usage: lc_summary.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results'
VIO = HERE.parents[1] / 'vio_imu_fix'
sys.path.insert(0, str(VIO / 'tools'))
sys.path.insert(0, str(HERE))
from eval_traj import evaluate  # noqa: E402
from lc_runs_eval import loop_precision, profiler  # noqa: E402

VARIANTS = ['opt1', 'opt2', 'bowwin', 'cpu']
CONFIGS = ['bimonovio', 'bimono', 'monovio', 'sadvio']
METRICS = ['ate_rmse', 'rpe_t_1s', 'rpe_t_5s', 'rpe_r_1s_deg', 'coverage']
FILES = {'window': 'results.csv', 'final': 'results_loop.csv', 'online': 'results_loop_online.csv'}


def run_entry(rd):
    meta = json.loads((rd / 'run.json').read_text())
    e = {'end_reason': meta['end_reason'], 'wall_s': meta['wall_s']}
    for k, f in FILES.items():
        try:
            m = evaluate(rd, f)
            e[k] = {x: m.get(x) for x in METRICS + ['segments']}
        except Exception as ex:  # noqa: BLE001
            e[k] = {'error': str(ex)}
    e['loops'], e['loops_correct'] = loop_precision(rd, rd.parent.parent.name)
    p = profiler(rd)
    e['cost'] = {k: p.get(k) for k in ('Loop closure dt', 'Loop descriptor dt', 'Pose graph dt', 'Back end dt',
                                       'Front end dt', 'Number of keyframes')}
    return e


def main():
    cache_f = RES / 'matrix_runs.json'
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    for v in VARIANTS:
        for c in CONFIGS:
            for rd in sorted((VIO / 'runs' / f'lc_{v}_{c}').glob('*/*/run_*')):
                key = str(rd.relative_to(VIO / 'runs'))
                if key not in cache:
                    cache[key] = run_entry(rd)
                    cache_f.write_text(json.dumps(cache, indent=1))

    def cell(v, c, seq, part, metric):
        vals = [e[part].get(metric) for k, e in cache.items() if k.startswith(f'lc_{v}_{c}/{seq}/')
                and isinstance(e.get(part), dict) and e[part].get(metric) is not None]
        return float(np.mean(vals)) if vals else np.nan

    seqs = sorted({k.split('/')[1] for k in cache})
    out = ['# Loop closure matrix: summary', '',
           'Means over the runs of a cell (2 for opt1 / opt2, 1 for bowwin / cpu). Without loop closure = opt1 window '
           '(results.csv: option 1 leaves the window untouched). opt1 = bag-of-words, output corrected; opt2 = MegaLoc '
           '(GPU) + window corrected; bowwin = bag-of-words + window corrected; cpu = MegaLoc on the CPU + window.', '']
    for c in CONFIGS:
        out += [f'## {c}', '',
                '| seq | ATE without | opt1 final / online | opt2 window / final / online | bowwin window / final | '
                'cpu window / final | RPE 5 s without / opt1 final / opt2 final |',
                '|---|---|---|---|---|---|---|']
        med = {k: [] for k in ('wo', 'o1f', 'o1o', 'o2w', 'o2f', 'o2o', 'bww', 'bwf')}
        for s in seqs:
            wo = cell('opt1', c, s, 'window', 'ate_rmse')
            if np.isnan(wo):
                continue
            vals = {'wo': wo, 'o1f': cell('opt1', c, s, 'final', 'ate_rmse'),
                    'o1o': cell('opt1', c, s, 'online', 'ate_rmse'), 'o2w': cell('opt2', c, s, 'window', 'ate_rmse'),
                    'o2f': cell('opt2', c, s, 'final', 'ate_rmse'), 'o2o': cell('opt2', c, s, 'online', 'ate_rmse'),
                    'bww': cell('bowwin', c, s, 'window', 'ate_rmse'), 'bwf': cell('bowwin', c, s, 'final', 'ate_rmse')}
            for k in med:
                med[k].append(vals[k])
            cw, cf = cell('cpu', c, s, 'window', 'ate_rmse'), cell('cpu', c, s, 'final', 'ate_rmse')
            r5 = [cell('opt1', c, s, 'window', 'rpe_t_5s'), cell('opt1', c, s, 'final', 'rpe_t_5s'),
                  cell('opt2', c, s, 'final', 'rpe_t_5s')]
            f = lambda x: '–' if np.isnan(x) else f'{x:.3f}'  # noqa: E731
            out.append(f"| {s} | {f(vals['wo'])} | {f(vals['o1f'])} / {f(vals['o1o'])} | {f(vals['o2w'])} / "
                       f"{f(vals['o2f'])} / {f(vals['o2o'])} | {f(vals['bww'])} / {f(vals['bwf'])} | {f(cw)} / {f(cf)} | "
                       f"{' / '.join(f(x) for x in r5)} |")
        m = {k: np.nanmedian(v) if v else np.nan for k, v in med.items()}
        out += ['', f"Median over {len(med['wo'])} sequences: without {m['wo']:.3f}; opt1 final {m['o1f']:.3f} online "
                    f"{m['o1o']:.3f}; opt2 window {m['o2w']:.3f} final {m['o2f']:.3f} online {m['o2o']:.3f}; bowwin "
                    f"window {m['bww']:.3f} final {m['bwf']:.3f}", '']

    # Loops, cost and run health per variant
    out += ['## Loops, cost and run health', '',
            '| variant | runs | loops accepted | correct | loop module ms/KF (median) | descriptor ms/KF | '
            'pose graph ms/solve | runs with coverage < 0.9 of opt1 window |', '|---|---|---|---|---|---|---|---|']
    for v in VARIANTS:
        es = {k: e for k, e in cache.items() if k.startswith(f'lc_{v}_')}
        if not es:
            continue
        nl = sum(e['loops'] for e in es.values())
        nc = sum(e['loops_correct'] for e in es.values())
        md = lambda key: np.nanmedian([e['cost'].get(key) or np.nan for e in es.values()])  # noqa: E731
        low = 0
        for k, e in es.items():
            ref = [x['window'].get('coverage') for kk, x in cache.items() if kk.startswith('lc_opt1_')
                   and kk.split('/')[1:3] == k.split('/')[1:3] and x['window'].get('coverage')]
            cov = e['window'].get('coverage')
            if ref and cov is not None and cov < 0.9 * max(ref):
                low += 1
        out.append(f"| {v} | {len(es)} | {nl} | {nc} ({100 * nc / max(nl, 1):.1f} %) | {md('Loop closure dt'):.1f} | "
                   f"{md('Loop descriptor dt'):.1f} | {md('Pose graph dt'):.0f} | {low} |")
    text = '\n'.join(out) + '\n'
    (RES / 'matrix_summary.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
