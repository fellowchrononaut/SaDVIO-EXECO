#!/usr/bin/env python3
"""Run SaDVIO through its ROS 2 node (vio_ros) on a bag played in real time: the original RealSense bags, or the
TUM-VI sequences converted to ROS 2 bags by euroc_to_ros2bag.py (mav0/../ros2bag, mono16 images as in the official
TUM-VI bags).

Same config generation, scratch/results layout and evaluation as run_sadvio.py (offline engine), so ROS and
offline results are directly comparable. The node runs in the container's dense_devel colcon workspace; the bag is
played with `ros2 bag play` (rate 1, camera + IMU topics only) on an isolated ROS domain (localhost only).
The node is stopped once the bag has finished and its outputs have not grown for --stall seconds.

Usage: run_sadvio_ros.py --label ros_vo --seq rs_13_34_21 --mode bimono --runs 1 [--set key=value ...]
"""
import argparse, json, os, shutil, subprocess, time
from pathlib import Path

import run_sadvio as rs  # write_config, to_container, output_size, SCRATCH, EVAL, CONTAINER, SEQ_DATASET

WS = '/root/SaDVIO-Dense/SaDVIO-EXECO-dense_devel/ros'
NODE = f'{WS}/install/isae_slam_ros/lib/isae_slam_ros/vio_ros'
BAGS = {f'rs_{t}': f'/Simulator_Validation/RealSense_Captures_060526/{t}/rosbag2_2026_05_06-{t}' for t in rs.RS_TRAJS}
BAGS.update({seq: f'{rs.CROOT}/{Path(mav0).parent.as_posix()}/ros2bag' for seq, mav0 in rs.SEQUENCES.items()
             if seq.startswith(('room', 'magistrale')) and (rs.DATA / mav0).parent.joinpath('ros2bag').is_dir()})
RS_TOPICS = ['/camera/camera/infra1/image_rect_raw', '/camera/camera/infra2/image_rect_raw', '/camera/camera/imu']
TUMVI_TOPICS = ['/cam0/image_raw', '/cam1/image_raw', '/imu0']
ENV = ['-e', 'ROS_DOMAIN_ID=77', '-e', 'ROS_LOCALHOST_ONLY=1']
SRC = f'source /opt/ros/humble/setup.bash && source {WS}/install/setup.bash'


def run_once(result_dir: Path, seq: str, mode: str, overrides: dict, stall: float):
    run_dir = rs.SCRATCH / result_dir.relative_to(rs.EVAL / 'runs')
    if run_dir.exists():
        subprocess.run(['docker', 'exec', rs.CONTAINER, 'rm', '-rf', rs.to_container(run_dir)], check=True)
    if result_dir.exists():
        shutil.rmtree(result_dir)
    cfg_dir = run_dir / 'config'
    rs.write_config(cfg_dir, mode, overrides)
    ccfg = rs.to_container(cfg_dir)

    t0 = time.time()
    node = subprocess.Popen(['docker', 'exec', *ENV, '-e', 'EXECO_PERFRAME_LOG=1', '-w', rs.to_container(run_dir),
                             rs.CONTAINER, 'bash', '-c',
                             f'{SRC} && exec {NODE} {ccfg} > stdout.log 2>&1'])
    time.sleep(8)                                        # node start-up (config, sensors, subscriptions)
    play = subprocess.run(['docker', 'exec', *ENV, rs.CONTAINER, 'bash', '-c',
                           f'{SRC} && ros2 bag play {BAGS[seq]} --rate 1.0 --topics {" ".join(RS_TOPICS if seq.startswith("rs_") else TUMVI_TOPICS)}'],
                          capture_output=True, text=True)
    t_play = time.time() - t0
    last_size, last_change = -1, time.time()
    while node.poll() is None and time.time() - last_change < stall:
        time.sleep(2)
        size = rs.output_size(run_dir)
        if size != last_size:
            last_size, last_change = size, time.time()
    reason = f'exited({node.returncode})' if node.poll() is not None else 'stall'
    if node.poll() is None:
        subprocess.run(['docker', 'exec', rs.CONTAINER, 'pkill', '-f', f'vio_ros {ccfg}'])
        node.wait(timeout=30)
    meta = dict(seq=seq, mode=mode, overrides=overrides, end_reason=reason, engine='ros',
                wall_s=round(time.time() - t0 - (stall if reason == 'stall' else 0), 1), bag_play_s=round(t_play, 1),
                bag_play_rc=play.returncode, binary=NODE, started=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0)))
    (run_dir / 'run.json').write_text(json.dumps(meta, indent=1))
    (run_dir / 'bag_play.log').write_text(play.stdout[-20000:] + play.stderr[-20000:])
    subprocess.run(['docker', 'exec', rs.CONTAINER, 'chown', '-R', f'{os.getuid()}:{os.getgid()}',
                    rs.to_container(run_dir)], check=True)
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(result_dir))
    for d in [run_dir.parent, *run_dir.parent.parents]:
        if d == rs.SCRATCH.parent or any(d.iterdir()):
            break
        d.rmdir()
    print(f'  {result_dir.relative_to(rs.EVAL)}: {reason}, bag play rc {play.returncode}, ~{meta["wall_s"]:.0f} s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--seq', required=True, choices=sorted(BAGS))
    ap.add_argument('--mode', required=True, choices=rs.MODES)
    ap.add_argument('--runs', type=int, default=1)
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--stall', type=float, default=15)
    a = ap.parse_args()
    overrides = dict(kv.split('=', 1) for kv in a.set)
    if a.seq in rs.SEQ_DATASET:
        overrides.setdefault('dataset_id', rs.SEQ_DATASET[a.seq])
    for i in range(a.runs):
        run_once(rs.EVAL / 'runs' / a.label / a.seq / a.mode / f'run_{i:02d}', a.seq, a.mode, overrides, a.stall)


if __name__ == '__main__':
    main()
