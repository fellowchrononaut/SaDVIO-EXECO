#!/usr/bin/env python3
"""Data for the trajectory viewer page (GT vs stereo VO vs stereo VIO, with the KF images and their keypoints).

Per sequence, writes <out>/data/<seq>.json:
  frame_t       camera-0 timestamps relative to the first image [s] (video frame i = image i, 20 fps)
  gt            mocap positions at ~20 Hz, null between segments (gaps > 0.1 s)
  runs.{vo,vio} KF times [s], video frame index, raw positions, two SE3 alignments to the GT world
                ('ate': all associated KFs, as the ATE; 'start': rotation fitted on the first 20 s, translation
                pinned so that the first KF with GT sits on the GT: drift shows as divergence from a common start),
                per-KF position error for both, metrics, and the KF keypoints (log_slam/kf_features.csv, packed:
                base64 uint16 u*10, v*10, uint8 status; feat_off[k] = first keypoint of KF k)
The camera videos are encoded separately (320 px, 20 fps, one frame per image).
Usage: build_traj_viewer.py --out <dir> [--vo-label viewer_vo] [--vio-label viewer_vio] [seq ...]
"""
import argparse, base64, csv, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_traj import SEQ_DIRS, associate, evaluate, load_estimate, load_gt, read_csv, umeyama  # noqa: E402

EVAL = Path(__file__).resolve().parent.parent
RUNS = {'vo': ('bimono', 'Stereo VO', 'default config'),
        'vio': ('bimonovio', 'Stereo VIO', 'marginalization + time offset')}


def r3(a):
    return [[round(float(x), 3) for x in row] for row in a]


def nearest_frame(frame_ns, t_ns):
    """Index of the image nearest to each time (KF stamps pass through float64: ~256 ns resolution)."""
    i = np.clip(np.searchsorted(frame_ns, t_ns), 1, len(frame_ns) - 1)
    return np.where(np.abs(frame_ns[i - 1] - t_ns) <= np.abs(frame_ns[i] - t_ns), i - 1, i)


def run_data(seq, rd, frame_ns, t0_ns):
    t, p, R, _ = load_estimate(rd)
    fidx = nearest_frame(frame_ns, np.round(t * 1e9).astype(np.int64))
    tg, pg, Rg = load_gt(seq)
    ok, p_gt, _ = associate(t, tg, pg, Rg)
    align, err = {}, {}
    i0 = int(np.argmax(ok))  # first KF with ground truth
    for name, sel in (('ate', ok), ('start', ok & (t < t[i0] + 20))):
        _, Ra, ta = umeyama(p[sel], p_gt[sel], with_scale=False)
        if name == 'start':  # pin the translation: the first KF with GT sits on the GT
            ta = p_gt[i0] - Ra @ p[i0]
        align[name] = {'R': r3(Ra), 't': [round(float(x), 4) for x in ta]}
        e = np.linalg.norm((Ra @ p.T).T + ta - p_gt, axis=1)
        err[name] = [round(float(x), 3) if o else None for x, o in zip(e, ok)]
    m = json.loads((rd / "metrics.json").read_text()) if (rd / "metrics.json").exists() else evaluate(rd)
    metrics = {k: m.get(k) for k in ('ate_rmse', 'end_drift', 'rpe_t_1s', 'rpe_t_5s', 'rpe_r_1s_deg', 'coverage',
                                      'n_kf', 'resets')}

    # KF keypoints, in the order of the KFs
    feats = {}
    with open(rd / 'log_slam/kf_features.csv') as f:
        for row in csv.reader(f):
            if not row or not row[0].strip().isdigit():
                continue
            feats.setdefault(int(row[0]), []).append((float(row[1]), float(row[2]), int(row[3])))
    feats = {int(nearest_frame(frame_ns, np.array([k]))[0]): v for k, v in feats.items()}
    off, uv, st = [0], [], []
    for fi in fidx:
        for u, v, s in feats.get(int(fi), []):
            uv += [min(max(int(round(u * 10)), 0), 65535), min(max(int(round(v * 10)), 0), 65535)]
            st.append(s)
        off.append(len(st))
    return {'t': [round(float(x - t0_ns * 1e-9), 3) for x in t], 'frame': fidx.tolist(), 'p': r3(p),
            'align': align, 'err': err, 'metrics': metrics, 'feat_off': off,
            'feat_uv': base64.b64encode(np.array(uv, dtype='<u2').tobytes()).decode(),
            'feat_status': base64.b64encode(np.array(st, dtype=np.uint8).tobytes()).decode()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--vo-label', default='viewer_vo')
    ap.add_argument('--vio-label', default='viewer_vio')
    ap.add_argument('seqs', nargs='*', default=['room1', 'room2', 'room3', 'room4', 'room5', 'room6', 'magistrale2'])
    a = ap.parse_args()
    out = Path(a.out) / 'data'
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for seq in a.seqs:
        frame_ns = read_csv(SEQ_DIRS[seq] / 'cam0/data.csv', ncols=1)[:, 0].astype(np.int64)
        t0_ns = int(frame_ns[0])
        tg, pg, _ = load_gt(seq)
        tg_rel = tg - t0_ns * 1e-9
        gt, last_t = [], None
        for ti, pi in zip(tg_rel, pg):
            if last_t is not None and ti - last_t < 0.05:
                continue
            if last_t is not None and ti - last_t > 0.15:
                gt.append(None)
            gt.append([round(float(ti), 3)] + [round(float(x), 3) for x in pi])
            last_t = ti
        d = {'seq': seq, 'image_size': 512, 'fps': 20,
             'frame_t': [round((x - t0_ns) * 1e-9, 3) for x in frame_ns], 'gt': gt, 'runs': {}}
        for key, (mode, name, config) in RUNS.items():
            label = a.vo_label if key == 'vo' else a.vio_label
            rd = EVAL / 'runs' / label / seq / mode / 'run_00'
            d['runs'][key] = dict(run_data(seq, rd, frame_ns, t0_ns), name=name, config=config)
        (out / f'{seq}.json').write_text(json.dumps(d, separators=(',', ':')))
        index.append({'seq': seq, 'duration': d['frame_t'][-1],
                      **{f'{k}_ate': d['runs'][k]['metrics']['ate_rmse'] for k in RUNS}})
        print(f'{seq}: {(out / f"{seq}.json").stat().st_size / 1e6:.1f} MB')
    (out / 'index.json').write_text(json.dumps(index))


if __name__ == '__main__':
    main()
