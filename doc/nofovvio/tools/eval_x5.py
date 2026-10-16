#!/usr/bin/env python3
"""Evaluate a SaDVIO run on an Insta360 X5 capture against GUTWrap's reference trajectory (doc/nofovvio/NOFOVVIO_LEDGER.md,
items 1.3 / 2.3). The reference is evaluation only; it never enters SaDVIO.

Estimate: <run>/log_slam/trajectory_tum.txt from export_tum.py --frame cam0 (per frame, front camera, corrected by
the keyframes' final poses). Reference: <capture>/reference_poses_cam0.csv, front camera -> world at the 4 fps
reconstruction frames, on the same video clock (exact frame timestamps), world down = +y, SfM units.

Alignment: Sim3 (Umeyama) of the estimated camera positions onto the reference's, over the reference frames the
estimate covers. Reported:
  ate_ref_units      RMS position error after alignment, in reference (SfM) units
  ate_m_if_vio_metric  the same divided by the fitted scale: metres, if the VIO's scale is right
  scale_ref_per_m    fitted scale (reference units per estimated metre); with the cube's measurement, the reference's
                     metres per unit times this should be 1 for a metric VIO
  rot_rms_deg        RMS rotation error of the aligned camera orientations
  gravity_deg        angle between the estimate's down (-z of the SaDVIO world) mapped by the alignment and the
                     reference's down (+y): the VIO's gravity against the gravity-aligned reference
  coverage           fraction of reference frames with an estimated pose (same timestamp, +-1 ms)
  err_median / err_max / t_err_max_s   per-frame position error after alignment (reference units)
With a measured scale (M_PER_UNIT below, or --m-per-unit), also:
  m_per_unit         metres per reference unit (and its range), from a measured object
  ate_m              ATE in true metres (ate_ref_units x m_per_unit)
  vio_scale_ratio    VIO length / true length = 1 / (scale_ref_per_m x m_per_unit); 1 for a metric VIO; with the range
                     of the measured scale as vio_scale_ratio_range
  ate_gravity_m      ATE in metres after a gravity-only alignment (heading about gravity and a translation; no scale,
                     no roll / pitch): the estimate's metric, gravity-aligned world against the metric reference
--until T keeps the reference frames before T s (e.g. to leave out the end of VID_012, where tracking is lost and the
IMU-only propagation diverges); the alignment and every metric then use those frames only.

usage: eval_x5.py <run dir> [--capture VID_012] [--until T] [--out metrics_x5.json]   (the capture is taken from run.json's seq when omitted)
       writes <run>/metrics_x5.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

DATA = Path('/home/deos/s.jois/EXECO/Simulator_Validation/SaD_VIO_data/insta360x5')
# Measured metres per reference unit: (value, low, high). VID_012: GUTWrap's two 1 x 1 x 0.5 m boxes in the arena, v2
# from the SfM points on their printed side walls (VID_012/cube_corners_VID_012.csv, 2026-10-16; v1, 1.37 from the
# splat's top faces, was biased by soft edges). VID_001: not measured yet.
M_PER_UNIT = {'VID_012': (1.525, 1.507, 1.545)}


def quat_to_R(w, x, y, z):
    q = np.array([w, x, y, z]) / np.linalg.norm([w, x, y, z])
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def read_reference(path):
    """{ts: (R_w_c, t_w_c)} from "ts,qw,qx,qy,qz,tx,ty,tz" (camera -> world)"""
    out = {}
    for ln in path.read_text().splitlines():
        if not ln.strip() or ln.startswith('#'):
            continue
        v = [float(x) for x in ln.split(',')]
        out[int(v[0])] = (quat_to_R(*v[1:5]), np.array(v[5:8]))
    return out


def read_tum(path):
    """{ts_ns: (R_w_c, t_w_c)} from "t tx ty tz qx qy qz qw" (t in s)"""
    out = {}
    for ln in path.read_text().splitlines():
        v = ln.split()
        if len(v) != 8:
            continue
        t, tx, ty, tz, qx, qy, qz, qw = map(float, v)
        out[int(round(t * 1e9))] = (quat_to_R(qw, qx, qy, qz), np.array([tx, ty, tz]))
    return out


def umeyama(src, dst):
    """s, R, t minimising |dst - (s R src + t)|"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


# SaDVIO world (z up) -> reference world axes (down = +y): rotation of +90 deg about x
R_ZUP_TO_YDOWN = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])


def gravity_only_ate(P_ref_m, P_est):
    """RMS error after the heading (rotation about the reference's vertical y axis) and translation that best map the
    estimate (SaDVIO world, z up, metres) onto the metric reference"""
    Q = P_est @ R_ZUP_TO_YDOWN.T
    p, q = P_ref_m - P_ref_m.mean(0), Q - Q.mean(0)
    phi = np.arctan2((p[:, 0] * q[:, 2] - p[:, 2] * q[:, 0]).sum(), (p[:, 0] * q[:, 0] + p[:, 2] * q[:, 2]).sum())
    c, s = np.cos(phi), np.sin(phi)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    e = p - q @ Ry.T
    return float(np.sqrt((e ** 2).sum(1).mean()))


def metric(cap, m_per_unit, ate, s, P_ref, P_est):
    m = (m_per_unit, m_per_unit, m_per_unit) if m_per_unit else M_PER_UNIT.get(cap)
    if not m:
        return {}
    return dict(m_per_unit=m[0], m_per_unit_range=[m[1], m[2]], ate_m=round(ate * m[0], 4),
                vio_scale_ratio=round(1 / (s * m[0]), 4),
                vio_scale_ratio_range=[round(1 / (s * m[2]), 4), round(1 / (s * m[1]), 4)],
                ate_gravity_m=round(gravity_only_ate(P_ref * m[0], P_est), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('--capture', default=None, help='VID_012 or VID_001')
    ap.add_argument('--until', type=float, default=None, help='only reference frames before this time (s)')
    ap.add_argument('--out', default='metrics_x5.json')
    ap.add_argument('--m-per-unit', type=float, default=None, help='measured metres per reference unit (overrides '
                    'M_PER_UNIT)')
    a = ap.parse_args()
    cap = a.capture
    if cap is None:
        seq = json.loads((a.run / 'run.json').read_text())['seq']  # x5_012
        cap = 'VID_' + seq.split('_')[1]
    ref = read_reference(DATA / cap / 'reference_poses_cam0.csv')
    est = read_tum(a.run / 'log_slam/trajectory_tum.txt')
    est_ts = np.array(sorted(est))

    pairs = []
    for ts in sorted(ref):
        if a.until is not None and ts * 1e-9 >= a.until:
            break
        if len(est_ts) == 0:
            break
        i = np.searchsorted(est_ts, ts)
        best = min((j for j in (i - 1, i) if 0 <= j < len(est_ts)), key=lambda j: abs(est_ts[j] - ts))
        if abs(est_ts[best] - ts) <= 1_000_000:
            pairs.append((ref[ts], est[est_ts[best]], ts))
    out = dict(capture=cap, until_s=a.until, reference_frames=len(ref), matched=len(pairs),
               coverage=round(len(pairs) / max(1, len(ref)), 3))
    if len(pairs) >= 3:
        P_ref = np.array([p[0][1] for p in pairs])
        P_est = np.array([p[1][1] for p in pairs])
        s, R, t = umeyama(P_est, P_ref)
        err = P_ref - (s * (R @ P_est.T).T + t)
        e = np.linalg.norm(err, axis=1)
        ate = float(np.sqrt((e ** 2).mean()))
        rot = []
        for (R_ref, _), (R_est, _), _ in pairs:
            dR = R_ref.T @ R @ R_est
            rot.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
        down = R @ np.array([0, 0, -1.0])
        out.update(ate_ref_units=round(ate, 4), ate_m_if_vio_metric=round(ate / s, 4),
                   scale_ref_per_m=round(float(s), 4),
                   rot_rms_deg=round(float(np.sqrt(np.mean(np.square(rot)))), 3),
                   gravity_deg=round(float(np.degrees(np.arccos(np.clip(down @ [0, 1, 0], -1, 1)))), 3),
                   err_median=round(float(np.median(e)), 4), err_max=round(float(e.max()), 4),
                   t_err_max_s=round(pairs[int(e.argmax())][2] * 1e-9, 3),
                   **metric(cap, a.m_per_unit, ate, s, P_ref, P_est),
                   ref_path_length_ref_units=round(float(np.linalg.norm(np.diff(P_ref, axis=0), axis=1).sum()), 3),
                   t_first_s=round(pairs[0][2] * 1e-9, 3), t_last_s=round(pairs[-1][2] * 1e-9, 3))
    (a.run / a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
