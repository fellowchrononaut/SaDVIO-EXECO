#!/usr/bin/env python3
"""Live side-by-side view of two SaDVIO configs through the ROS 2 node, on one bag playback, in two RViz windows.

Each config runs its own vio_ros node in a namespace (/vo, /vio, ...): the sensor topics are absolute so both nodes
receive the same playback, the visualizer topics are relative so they do not collide. Each RViz window
(ros/launch/isae_slam.rviz with the topics prefixed by the namespace) shows one node; the two windows are placed
side by side. The nodes share the TF name "robot", so the RViz configs use the markers (frame "world"), not TF.
After the playback, the nodes are stopped and evaluated like run_sadvio_ros.py runs (results in runs/<label>/);
the RViz windows stay open until closed.

Usage: live_ros_compare.py --seq magistrale2 [--rate 1.0] [--display :20]
       [--node vo:bimono] [--node vio:bimonovio:marginalization=1,estimate_td=1]
"""
import argparse, json, os, re, shutil, subprocess, time
from pathlib import Path

import run_sadvio as rs
import run_sadvio_ros as rr

RVIZ_SRC = Path(__file__).resolve().parents[3] / 'ros/launch/isae_slam.rviz'


def rviz_config(ns: str, x: int, width: int, height: int) -> str:
    txt = RVIZ_SRC.read_text()
    txt = re.sub(r'(\n\s+Value: )/(?!/)', rf'\1/{ns}/', txt)  # topic names
    txt = re.sub(r'\n  Height: \d+', f'\n  Height: {height}', txt)
    txt = re.sub(r'\n  Width: \d+', f'\n  Width: {width}', txt)
    txt = re.sub(r'\n  X: \d+', f'\n  X: {x}', txt)
    txt = re.sub(r'\n  Y: \d+', '\n  Y: 0', txt)
    txt = re.sub(r'(\n\s+Distance: )[\d.]+', r'\g<1>40', txt)  # magistrale2 spans ~100 m
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='magistrale2')
    ap.add_argument('--rate', type=float, default=1.0)
    ap.add_argument('--display', default=':20')
    ap.add_argument('--label', default='live')
    ap.add_argument('--node', action='append', help='ns:mode[:key=value,...]')
    ap.add_argument('--stall', type=float, default=20)
    a = ap.parse_args()
    nodes = a.node or ['vo:bimono', 'vio:bimonovio:marginalization=1,estimate_td=1']

    runs, procs, viz = [], [], []
    width = 1920 // len(nodes)
    for k, spec in enumerate(nodes):
        ns, mode, *ov = spec.split(':')
        overrides = dict(kv.split('=', 1) for kv in ov[0].split(',')) if ov else {}
        overrides['enable_visu'] = 1
        if a.seq in rs.SEQ_DATASET:
            overrides.setdefault('dataset_id', rs.SEQ_DATASET[a.seq])
        result_dir = rs.EVAL / 'runs' / f'{a.label}_{ns}' / a.seq / mode / 'run_00'
        run_dir = rs.SCRATCH / result_dir.relative_to(rs.EVAL / 'runs')
        subprocess.run(['docker', 'exec', rs.CONTAINER, 'rm', '-rf', rs.to_container(run_dir)], check=True)
        if result_dir.exists():
            shutil.rmtree(result_dir)
        rs.write_config(run_dir / 'config', mode, overrides)
        (run_dir / f'sadvio_{ns}.rviz').write_text(rviz_config(ns, k * width, width, 1000))
        runs.append((ns, mode, overrides, run_dir, result_dir))

    t0 = time.time()
    for ns, mode, _, run_dir, _ in runs:
        procs.append(subprocess.Popen(
            ['docker', 'exec', *rr.ENV, '-e', 'EXECO_PERFRAME_LOG=1', '-w', rs.to_container(run_dir), rs.CONTAINER,
             'bash', '-c', f'{rr.SRC} && exec {rr.NODE} {rs.to_container(run_dir / "config")} '
                           f'--ros-args -r __ns:=/{ns} > stdout.log 2>&1']))
        viz.append(subprocess.Popen(
            ['docker', 'exec', *rr.ENV, '-e', f'DISPLAY={a.display}', '-e', 'XAUTHORITY=/root/.Xauthority',
             rs.CONTAINER, 'bash', '-c',
             f'{rr.SRC} && exec rviz2 -d {rs.to_container(run_dir / f"sadvio_{ns}.rviz")} '
             f'--ros-args -r __node:=rviz_{ns} > /tmp/rviz_{ns}.log 2>&1']))
    time.sleep(10)  # node start-up, RViz windows
    print(f'playing {a.seq} at rate {a.rate} ...', flush=True)
    topics = rr.RS_TOPICS if a.seq.startswith('rs_') else rr.TUMVI_TOPICS
    play = subprocess.run(['docker', 'exec', *rr.ENV, rs.CONTAINER, 'bash', '-c',
                           f'{rr.SRC} && ros2 bag play {rr.BAGS[a.seq]} --rate {a.rate} --topics {" ".join(topics)}'],
                          capture_output=True, text=True)
    t_play = time.time() - t0

    last, last_change = -1, time.time()
    while time.time() - last_change < a.stall and any(p.poll() is None for p in procs):
        time.sleep(2)
        size = sum(rs.output_size(r[3]) for r in runs)
        if size != last:
            last, last_change = size, time.time()
    for ns, mode, overrides, run_dir, result_dir in runs:
        subprocess.run(['docker', 'exec', rs.CONTAINER, 'pkill', '-f', f'vio_ros {rs.to_container(run_dir / "config")}'])
    for p in procs:
        p.wait(timeout=30)
    for ns, mode, overrides, run_dir, result_dir in runs:
        meta = dict(seq=a.seq, mode=mode, overrides=overrides, end_reason='stall', engine='ros-live', namespace=ns,
                    rate=a.rate, wall_s=round(time.time() - t0 - a.stall, 1), bag_play_s=round(t_play, 1),
                    bag_play_rc=play.returncode, binary=rr.NODE,
                    started=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0)))
        (run_dir / 'run.json').write_text(json.dumps(meta, indent=1))
        (run_dir / 'bag_play.log').write_text(play.stdout[-20000:] + play.stderr[-20000:])
        subprocess.run(['docker', 'exec', rs.CONTAINER, 'chown', '-R', f'{os.getuid()}:{os.getgid()}',
                        rs.to_container(run_dir)], check=True)
        result_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(result_dir))
        print(f'  {ns}: {result_dir.relative_to(rs.EVAL)}', flush=True)
    print('RViz windows stay open; close them when done.')


if __name__ == '__main__':
    main()
