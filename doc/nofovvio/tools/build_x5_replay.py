#!/usr/bin/env python3
"""Add Insta360 X5 runs to the presentation's SaDVIO replay viewer (doc/nofovvio/NOFOVVIO_LEDGER.md, items 1.4 and
3.3): the front-lens monovio run (baseline B, key 'vio') and optionally a two-lens nofovvio run (key 'nfv', shown with
the rear lens's video and keypoints). Same data format as doc/vio_imu_fix/tools/build_traj_viewer.py, with:
  gt     GUTWrap's reference front-camera positions (4 fps), rotated to z up (the reference has down = +y). With a
         measured scale (eval_x5.M_PER_UNIT: VID_012) it is drawn in true metres, so the VIO's scale error shows as a
         growing mismatch under the SE3 alignments; without one (VID_001) it is divided by the runs' mean Sim(3) scale,
         and the errors are metres only if the VIO's scale is right
  runs   'vio' (monovio, front lens) and 'nfv' (nofovvio); the KF positions are camera 0's (body + T_BS translation)
         for both, as the reference; the nfv monitor plays the rear lens (video <VID>_rear, kf_features_cam1.csv)
  err    per KF, against the reference interpolated linearly at the KF time (neighbours at most 0.6 s apart)
  plus   gt_label, note and metrics_display (the viewer shows these when present)

Writes <viewer>/data/<VID_xxx>.json, <viewer>/video/<VID_xxx>.mp4 (front lens) and <VID_xxx>_rear.mp4 (rear lens, with
a nofovvio run), one frame per image, 29.97 fps, 384 px, and adds or replaces the capture's entry in
<viewer>/data/index.json. Other entries are left untouched.
usage: build_x5_replay.py --viewer <sources/sadvio-replay> --run x5_012=<monovio run> [--nfv x5_012=<nofovvio run>]
                          [--run x5_001=... --nfv x5_001=...] [--skip-video]
"""
import argparse
import base64
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[1] / 'vio_imu_fix/tools'))  # after HERE: it has its own export_tum.py
from eval_traj import load_estimate  # noqa: E402
from eval_x5 import DATA, M_PER_UNIT, read_reference, umeyama  # noqa: E402
from export_tum import dataset_T_BS, dataset_text  # noqa: E402

FPS = 30000 / 1001
VIDEO_PX = 384
R_ZUP = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])  # reference world (down = +y) -> z up


def se3(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    return R, mu_d - R @ mu_s


def r3(a):
    return [[round(float(x), 3) for x in row] for row in a]


def nearest_frame(frame_ns, t_ns):
    i = np.clip(np.searchsorted(frame_ns, t_ns), 1, len(frame_ns) - 1)
    return np.where(np.abs(frame_ns[i - 1] - t_ns) <= np.abs(frame_ns[i] - t_ns), i - 1, i)


def interp_reference(t_ref, P_ref, t):
    """Reference position at times t (linear), and whether it is defined (inside, neighbours <= 0.6 s apart)"""
    i1 = np.clip(np.searchsorted(t_ref, t), 1, len(t_ref) - 1)
    i0 = i1 - 1
    ok = (t >= t_ref[0]) & (t <= t_ref[-1]) & (t_ref[i1] - t_ref[i0] <= 0.6)
    w = ((t - t_ref[i0]) / np.maximum(t_ref[i1] - t_ref[i0], 1e-9))[:, None]
    return (1 - w) * P_ref[i0] + w * P_ref[i1], ok


def encode_video(mav0, cam, frame_files, out):
    with tempfile.TemporaryDirectory() as tmp:
        for i, name in enumerate(frame_files):
            os.symlink(mav0 / cam / 'data' / name, Path(tmp) / f'{i:05d}.png')
        subprocess.run(['nice', '-n', '10', 'taskset', '-c', '8-23', 'ffmpeg', '-y', '-loglevel', 'error',
                        '-framerate', '30000/1001', '-i', f'{tmp}/%05d.png',
                        '-vf', f'scale={VIDEO_PX}:{VIDEO_PX}:flags=area,format=yuv420p', '-c:v', 'libx264',
                        '-preset', 'slow', '-crf', '26', '-g', '15', '-movflags', '+faststart', str(out)], check=True)


def kf_cam0_positions(rd):
    """KF times, camera 0 positions (body pose + camera 0's T_BS translation) and segments (one per re-initialization)"""
    t, p_body, R, _, seg_id = load_estimate(rd, with_segments=True)
    T_BS = dataset_T_BS(rd).get('camera_0', np.eye(4))
    return t, p_body + np.einsum('nij,j->ni', R, T_BS[:3, 3]), np.unique(seg_id, return_inverse=True)[1]


def imu_noise_label(rd):
    text = dataset_text(rd)
    m = re.search(r'gyroscope_noise_density:\s*([-0-9.eE+]+)', text)
    return 'TUM-VI placeholder IMU noise' if m and float(m.group(1)) == 0.00016 else "GUTWrap's measured IMU noise"


def run_entry(rd, key, frame_ns, t0_ns, t_ref, P_ref, to_m, mpu, features_file):
    t, p, seg = kf_cam0_positions(rd)
    fidx = nearest_frame(frame_ns, np.round(t * 1e9).astype(np.int64))
    P_at, ok = interp_reference(t_ref, P_ref, t)
    G_at = P_at * to_m
    sel = ok & (seg == 0)
    s, _, _ = umeyama(p[sel], P_at[sel])

    align = {'ate': [], 'start': []}
    err = {'ate': [None] * len(t), 'start': [None] * len(t)}
    ate = None
    for k in range(seg.max() + 1 if len(seg) else 0):
        okk = ok & (seg == k)
        if okk.sum() < 3:
            align['ate'].append(None)
            align['start'].append(None)
            continue
        i0 = int(np.argmax(okk))
        for name, sl in (('ate', okk), ('start', okk & (t < t[i0] + 20))):
            Ra, ta = se3(p[sl], G_at[sl])
            if name == 'start':
                ta = G_at[i0] - Ra @ p[i0]
            align[name].append({'R': r3(Ra), 't': [round(float(x), 4) for x in ta]})
            e = np.linalg.norm((Ra @ p.T).T + ta - G_at, axis=1)
            for i in np.where(okk)[0]:
                err[name][i] = round(float(e[i]), 3)
            if name == 'ate' and k == 0:
                ate = float(np.sqrt(np.mean(e[okk] ** 2)))

    m = json.loads((rd / 'metrics_x5.json').read_text()) if (rd / 'metrics_x5.json').exists() else {}
    resets = int(seg.max()) if len(seg) else 0
    metrics = {'ate_rmse': ate, 'end_drift': None, 'rpe_t_1s': None, 'rpe_t_5s': None, 'rpe_r_1s_deg': None,
               'coverage': m.get('coverage'), 'n_kf': int(len(t)), 'resets': resets}
    show = [['ATE (KF, SE3)' if mpu else 'ATE (KF)', f'{ate:.3f} m'],
            ['VIO scale', f'×{1 / (s * mpu[0]):.3f}'] if mpu else ['Scale', f'{s:.3f} units/m'],
            *([['ATE (Sim3)', f"{m['ate_m']:.3f} m"]] if mpu and 'ate_m' in m else []),
            ['Gravity', f"{m['gravity_deg']:.2f}°" if 'gravity_deg' in m else '—'],
            ['Rot. RMS', f"{m['rot_rms_deg']:.2f}°" if 'rot_rms_deg' in m else '—'],
            ['Keyframes', len(t)], ['Resets', resets]]

    feats = {}
    with open(rd / 'log_slam' / features_file) as f:
        for row in csv.reader(f):
            if row and row[0].strip().isdigit():
                feats.setdefault(int(row[0]), []).append((float(row[1]), float(row[2]), int(row[3])))
    feats = {int(nearest_frame(frame_ns, np.array([k]))[0]): v for k, v in feats.items()}
    off, uv, st = [0], [], []
    for fi in fidx:
        for u, v, s_ in feats.get(int(fi), []):
            uv += [min(max(int(round(u * 10)), 0), 65535), min(max(int(round(v * 10)), 0), 65535)]
            st.append(s_)
        off.append(len(st))

    ov = json.loads((rd / 'run.json').read_text()).get('overrides', {})
    px = int(round(1920 * float(ov.get('downsampling', 1))))
    return {'t': [round(float(x - t0_ns * 1e-9), 3) for x in t], 'frame': fidx.tolist(), 'p': r3(p),
            'seg': seg.tolist(), 'align': align, 'err': err, 'metrics': metrics, 'metrics_display': show,
            'feat_off': off,
            'feat_uv': base64.b64encode(np.array(uv, dtype='<u2').tobytes()).decode(),
            'feat_status': base64.b64encode(np.array(st, dtype=np.uint8).tobytes()).decode(),
            'config': f"KB4 · masks · {px} px"}, s, ate, px


def build(seq, rd, rd_nfv, viewer, skip_video=False):
    cap = 'VID_' + seq.split('_')[1]
    mav0 = DATA / cap / 'mav0'
    rows = [r for r in csv.reader((mav0 / 'cam0/data.csv').open()) if r and not r[0].startswith('#')]
    frame_ns = np.array([int(r[0]) for r in rows], dtype=np.int64)
    t0_ns = int(frame_ns[0])

    ref = read_reference(DATA / cap / 'reference_poses_cam0.csv')
    t_ref = np.array(sorted(ref)) * 1e-9
    P_ref = np.array([ref[k][1] for k in sorted(ref)]) @ R_ZUP.T

    # Reference in metres: the measured scale when there is one, else the runs' mean Sim(3) scale (KFs onto the
    # reference, first segment)
    mpu = M_PER_UNIT.get(cap)
    runs_in = [('vio', rd, 'kf_features.csv')] + ([('nfv', rd_nfv, 'kf_features_cam1.csv')] if rd_nfv else [])
    if mpu:
        to_m = mpu[0]
    else:
        scales = []
        for _, r, _ in runs_in:
            t, p, seg = kf_cam0_positions(r)
            P_at, ok = interp_reference(t_ref, P_ref, t)
            scales.append(umeyama(p[ok & (seg == 0)], P_at[ok & (seg == 0)])[0])
        to_m = 1 / float(np.mean(scales))

    runs, ates, s_of, px = {}, {}, {}, None
    for key, r, ff in runs_in:
        runs[key], s_of[key], ates[key], px = run_entry(r, key, frame_ns, t0_ns, t_ref, P_ref, to_m, mpu, ff)
        runs[key]['imu_noise'] = imu_noise_label(r)
    runs['vio'].update(name='Mono VIO (front lens)', short='MONO')
    if 'nfv' in runs:
        runs['nfv'].update(name='Two-lens VIO (nofovvio)', short='2-LENS', camera='rear lens',
                           video=f'{cap}_rear')

    gt, last = [], None
    for ti, pi in zip(t_ref - t0_ns * 1e-9, P_ref * to_m):
        if last is not None and ti - last > 0.6:
            gt.append(None)
        gt.append([round(float(ti), 3)] + [round(float(x), 3) for x in pi])
        last = ti

    note = (f'Insta360 X5 {cap}, SaDVIO with the IMU, KB4 fisheye model, the per-frame masks and the image-circle mask, '
            f'no time-offset estimation. Mono VIO (monovio): the front lens only ({runs["vio"]["imu_noise"]}). ')
    if 'nfv' in runs:
        note += ('Two-lens VIO (nofovvio): both lenses, each tracked on its own, one pose update over the landmarks '
                 f'of both ({runs["nfv"]["imu_noise"]}); its trajectory is the front camera\'s, as the reference, '
                 'and its keyframe panel shows the rear lens. ')
    note += "The reference is GUTWrap's SfM reconstruction of the front camera at 4 fps, gravity-aligned. "
    if mpu:
        ratios = ', '.join(f'{1 / (s_of[k] * mpu[0]):.3f}' for k in runs)
        note += (f'It is drawn in metres at {mpu[0]:.3f} m per SfM unit ({mpu[1]:.3f}-{mpu[2]:.3f}), measured on the '
                 'two 1 x 1 x 0.5 m boxes in the arena. The runs are aligned without scale, so any scale error shows: '
                 f'their lengths are {ratios} times the true ones, and the errors include it. The ATE after a '
                 'Sim(3) alignment, which removes the scale error, is on the metrics cards. ')
    else:
        note += (f'It is not metric yet: it is drawn divided by the runs\' mean Sim(3) scale ({1 / to_m:.3f} reference '
                 'units per VIO metre), so the errors read as metres only if the VIO\'s scale is right. ')
    note += 'Errors between reference frames use a linear interpolation of the reference.'
    if cap == 'VID_012':
        note += (' Near the end the camera leaves the arena: the front lens alone can lose tracking there, and the '
                 'IMU-only propagation can drift.')
    note += (' Keypoints as in the ROS image_kps view: blue = detected, no landmark; red = initialised landmark; '
             'green = resurrected map landmark. "Common start": rotation fitted on the first 20 s and the first '
             'keyframe pinned on the reference; "Whole run": the ATE fit.')
    d = {'seq': cap, 'dataset': 'Insta360 X5', 'image_size': [px, px], 'fps': FPS,
         'frame_t': [round((x - t0_ns) * 1e-9, 3) for x in frame_ns], 'gt': gt,
         'gt_label': 'Reference (SfM, metric from the boxes)' if mpu else 'Reference (SfM, scaled to the VIO)',
         'note': note, 'runs': runs}
    out = viewer / 'data' / f'{cap}.json'
    out.write_text(json.dumps(d, separators=(',', ':')))
    videos = [('cam0', f'{cap}.mp4')] + ([('cam1', f'{cap}_rear.mp4')] if 'nfv' in runs else [])
    for cam, name in videos:
        if not (skip_video and (viewer / 'video' / name).exists()):
            encode_video(mav0, cam, [r[1].strip() for r in rows], viewer / 'video' / name)
    print(f'{cap}: {out.stat().st_size / 1e6:.1f} MB data; KF ATE ' +
          ', '.join(f'{k} {v:.3f} m' for k, v in ates.items()))
    return {'seq': cap, 'dataset': 'Insta360 X5', 'duration': d['frame_t'][-1], 'vo_ate': None, 'vio_ate': ates['vio'],
            **({'nfv_ate': ates['nfv']} if 'nfv' in ates else {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--viewer', type=Path, required=True)
    ap.add_argument('--run', action='append', required=True, help='x5_012=<monovio run dir>')
    ap.add_argument('--nfv', action='append', default=[], help='x5_012=<nofovvio run dir> (optional)')
    ap.add_argument('--skip-video', action='store_true', help='keep existing videos')
    a = ap.parse_args()
    nfv = dict(item.split('=', 1) for item in a.nfv)
    index_path = a.viewer / 'data/index.json'
    index = json.loads(index_path.read_text())
    for item in a.run:
        seq, rd = item.split('=', 1)
        entry = build(seq, Path(rd), Path(nfv[seq]) if seq in nfv else None, a.viewer, a.skip_video)
        index = [e for e in index if e['seq'] != entry['seq']] + [entry]
    order = {'TUM-VI': 0, 'EuRoC': 1, 'Insta360 X5': 2}
    index.sort(key=lambda e: order.get(e['dataset'], 3))  # stable: keeps the order within a dataset
    index_path.write_text(json.dumps(index))


if __name__ == '__main__':
    main()
