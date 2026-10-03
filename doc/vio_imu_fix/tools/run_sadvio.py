#!/usr/bin/env python3
"""Run SaDVIO (container dense_devel build, offline EuRoC reader) on a TUM-VI sequence.

The binary runs in a scratch folder under the data folder (the only host folder the container mounts) and the
results are then moved to this evaluation folder, so the data folder only holds what the container needs:
  runs/<label>/<sequence>/<mode>/run_NN/      (next to tools/)
  config/config.yaml, config/dataset/tumvi_512_ds.yaml   generated config used by the run
  log_slam/                                              SaDVIO outputs (results.csv, ...)
  stdout.log                                             binary console output
  run.json                                               run metadata (options, wall time, end reason)

The binary never exits at end of data, so the run is stopped once no output file has grown for
--stall seconds (or at --timeout). Ground truth is not used here (see eval_traj.py).

Usage:
  run_sadvio.py --label baseline --seq room1 --mode bimonovio --runs 3 [--set key=value ...]
"""
import argparse, json, os, shutil, subprocess, time
from pathlib import Path

EVAL       = Path(__file__).resolve().parent.parent                       # configs/, runs/ (results)
DATA       = Path('/home/deos/s.jois/EXECO/Simulator_Validation/SaD_VIO_data')  # datasets, mounted in the container
SCRATCH    = DATA / 'scratch'                                               # working folder of the running binary
CROOT      = '/Simulator_Validation/SaD_VIO_data'          # DATA inside the container
CONTAINER  = 'sad_vio_dense'
BIN        = '/root/SaDVIO-Dense/SaDVIO-EXECO-dense_devel/cpp/build/isaeslam'
SEQUENCES  = {'room1': 'tumvi/dataset-room1_512_16/mav0',
              'magistrale2': 'tumvi/dataset-magistrale2_512_16/mav0'}
SEQUENCES.update({f'room{i}': f'tumvi/dataset-room{i}_512_16/mav0' for i in range(2, 7)})
RS_TRAJS   = ['13_18_29', '13_21_59', '13_30_22', '13_31_26', '13_34_21', '13_36_44', '13_41_28', '13_42_11', '13_45_18']
SEQUENCES.update({f'rs_{t}': f'realsense/{t}/mav0' for t in RS_TRAJS})   # real RealSense D455 (prepare_realsense.py)
SEQ_DATASET = {f'rs_{t}': '"realsense_d455_vio"' for t in RS_TRAJS}        # default dataset_id per sequence
SEQUENCES['rs_13_18_29_synthimu'] = 'realsense/13_18_29_synthimu/mav0'   # diagnostic: IMU synthesised from GT
SEQUENCES['rs_hyb_realgyro'] = 'realsense/13_18_29_hyb_realgyro/mav0'
SEQ_DATASET['rs_hyb_realgyro'] = '"realsense_d455_vio"'
SEQUENCES['rs_hyb_realacc'] = 'realsense/13_18_29_hyb_realacc/mav0'
SEQ_DATASET['rs_hyb_realacc'] = '"realsense_d455_vio"'
SEQUENCES['rs_accinterp'] = 'realsense/13_18_29_accinterp/mav0'
SEQ_DATASET['rs_accinterp'] = '"realsense_d455_vio"'
SEQUENCES['rs_synth_sh-30'] = 'realsense/13_18_29_synth_sh-30/mav0'
SEQ_DATASET['rs_synth_sh-30'] = '"realsense_d455_vio"'
SEQUENCES['rs_synth_sh-15'] = 'realsense/13_18_29_synth_sh-15/mav0'
SEQ_DATASET['rs_synth_sh-15'] = '"realsense_d455_vio"'
SEQUENCES['rs_synth_sh15'] = 'realsense/13_18_29_synth_sh15/mav0'
SEQ_DATASET['rs_synth_sh15'] = '"realsense_d455_vio"'
SEQUENCES['rs_synth_sh30'] = 'realsense/13_18_29_synth_sh30/mav0'
SEQ_DATASET['rs_synth_sh30'] = '"realsense_d455_vio"'
SEQ_DATASET['rs_13_18_29_synthimu'] = '"realsense_d455_vio"'
MODES      = ('bimono', 'bimonovio', 'mono', 'monovio')


def to_container(p: Path) -> str:
    return CROOT + '/' + p.relative_to(DATA).as_posix()


def write_config(cfg_dir: Path, mode: str, overrides: dict):
    cfg_dir.joinpath('dataset').mkdir(parents=True, exist_ok=True)
    for y in (EVAL / 'configs/dataset').glob('*.yaml'):
        shutil.copy(y, cfg_dir / 'dataset' / y.name)
    lines = (EVAL / 'configs/base_config.yaml').read_text().splitlines()
    overrides = dict(overrides, slam_mode=f'"{mode}"')
    out, seen = [], set()
    for ln in lines:
        key = ln.split(':', 1)[0].strip() if ':' in ln and not ln.lstrip().startswith('#') else None
        if key in overrides and not ln.startswith(' '):
            out.append(f'{key} : {overrides[key]}')
            seen.add(key)
        else:
            out.append(ln)
    missing = set(overrides) - seen
    if missing:
        raise SystemExit(f'unknown config keys: {sorted(missing)}')
    (cfg_dir / 'config.yaml').write_text('\n'.join(out) + '\n')


def output_size(run_dir: Path) -> int:
    log = run_dir / 'log_slam'
    total = (run_dir / 'stdout.log').stat().st_size if (run_dir / 'stdout.log').exists() else 0
    if log.is_dir():
        total += sum(f.stat().st_size for f in log.iterdir() if f.is_file())
    return total


def run_once(result_dir: Path, seq: str, mode: str, overrides: dict, stall: float, timeout: float, binary: str = BIN,
             env: tuple = ()):
    run_dir = SCRATCH / result_dir.relative_to(EVAL / 'runs')
    if run_dir.exists():
        subprocess.run(['docker', 'exec', CONTAINER, 'rm', '-rf', to_container(run_dir)], check=True)
    if result_dir.exists():
        shutil.rmtree(result_dir)
    cfg_dir = run_dir / 'config'
    write_config(cfg_dir, mode, overrides)
    mav0 = to_container(DATA / SEQUENCES[seq])
    cmd = ['docker', 'exec', '-e', 'EXECO_PERFRAME_LOG=1', *env, '-w', to_container(run_dir), CONTAINER,
           'bash', '-c', f'exec {binary} {to_container(cfg_dir)} {mav0} > stdout.log 2>&1']
    t0 = time.time()
    proc = subprocess.Popen(cmd)
    last_size, last_change, reason = -1, time.time(), 'stall'
    while True:
        time.sleep(2)
        if proc.poll() is not None:
            reason = f'exited({proc.returncode})'
            break
        size = output_size(run_dir)
        if size != last_size:
            last_size, last_change = size, time.time()
        elif time.time() - last_change > stall:
            break
        if time.time() - t0 > timeout:
            reason = 'timeout'
            break
    if proc.poll() is None:
        subprocess.run(['docker', 'exec', CONTAINER, 'pkill', '-f', to_container(cfg_dir)])
        proc.wait(timeout=30)
    wall = time.time() - t0 - (stall if reason == 'stall' else 0)
    meta = dict(seq=seq, mode=mode, overrides=overrides, end_reason=reason, wall_s=round(wall, 1), binary=binary,
                started=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0)))
    (run_dir / 'run.json').write_text(json.dumps(meta, indent=1))

    # Hand the container's (root-owned) outputs back to the host user, move them out of the data folder
    subprocess.run(['docker', 'exec', CONTAINER, 'chown', '-R', f'{os.getuid()}:{os.getgid()}', to_container(run_dir)],
                   check=True)
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(result_dir))
    for d in [run_dir.parent, *run_dir.parent.parents]:
        if d == SCRATCH.parent or any(d.iterdir()):
            break
        d.rmdir()
    print(f'  {result_dir.relative_to(EVAL)}: {reason}, ~{wall:.0f} s')
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--seq', required=True, choices=SEQUENCES)
    ap.add_argument('--mode', required=True, choices=MODES)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--first', type=int, default=0, help='index of the first run folder')
    ap.add_argument('--set', action='append', default=[], help='config override key=value')
    ap.add_argument('--stall', type=float, default=30)
    ap.add_argument('--timeout', type=float, default=3600)
    ap.add_argument('--bin', default=BIN, help='isaeslam binary inside the container')
    ap.add_argument('--kf-features', action='store_true', help='log the KF features (log_slam/kf_features.csv)')
    a = ap.parse_args()
    overrides = dict(kv.split('=', 1) for kv in a.set)
    if a.seq in SEQ_DATASET:
        overrides.setdefault('dataset_id', SEQ_DATASET[a.seq])
    for i in range(a.first, a.first + a.runs):
        run_once(EVAL / 'runs' / a.label / a.seq / a.mode / f'run_{i:02d}', a.seq, a.mode, overrides,
                 a.stall, a.timeout, a.bin, ('-e', 'EXECO_KF_FEATURES_LOG=1') if a.kf_features else ())


if __name__ == '__main__':
    main()
