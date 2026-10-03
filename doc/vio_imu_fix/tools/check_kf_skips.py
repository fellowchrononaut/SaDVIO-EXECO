#!/usr/bin/env python3
"""Count the IMU factors that skip a KF of the window: in log_slam/vio_diag.csv (one row per optimized KF), a factor
whose span (factor_dt) is longer than the gap to the previous logged KF by more than 1 ms. A legitimate skip is a
low-parallax bridge (the back end drops the previous KF and bridges the preintegration when the new KF's parallax is
< 0.5 deg, from log_slam/kf_votes.csv); any other skip is the signature of the multithreading race fixed in
SLAMCore::nextFrame (the front end anchored a frame to the KF before the one the back end had not yet added).
Usage: check_kf_skips.py <run_dir> [...]
"""
import csv, sys
from pathlib import Path

for rd in map(Path, sys.argv[1:]):
    rows = list(csv.reader(open(rd / 'log_slam/vio_diag.csv')))
    h = [c.strip() for c in rows[0]]
    it, idt = h.index('kf_timestamp (ns)'), h.index('factor_dt')
    votes = {}
    for r in list(csv.reader(open(rd / 'log_slam/kf_votes.csv')))[1:]:
        votes[int(r[0])] = float(r[2])
    prev, bridges, races = None, 0, []
    for r in rows[1:]:
        t_ns, dt = int(r[it]), float(r[idt])
        t = t_ns * 1e-9
        if prev is not None and dt > t - prev + 1e-3:
            if votes.get(t_ns, 1.0) < 0.5:
                bridges += 1
            else:
                races.append(round(t, 2))
        prev = t
    print(f'{rd}: {bridges} bridges, {len(races)} other KF skips {races[:6]}')
