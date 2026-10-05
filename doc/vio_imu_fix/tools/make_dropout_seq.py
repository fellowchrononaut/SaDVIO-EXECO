#!/usr/bin/env python3
"""Make a copy of a sequence (EuRoC folder format) whose camera images are black in given time windows: a forced
visual dropout (lens covered, lights off) while the IMU keeps running. Everything else is a relative symlink to the
original sequence (images outside the windows, IMU, ground truth, calibration), so the copy costs a few MB.

  make_dropout_seq.py <src mav0 dir> <dst mav0 dir> <start_s:duration_s> [...]

start_s is counted from the first image of cam0. Both cameras go black in the same windows.
"""
import os
import struct
import sys
import zlib
from pathlib import Path


def png_size(path):
    b = path.read_bytes()[:26]
    w, h, depth, ctype = struct.unpack('>IIBB', b[16:26])
    return w, h, depth, ctype


def black_png(w, h, depth):
    """Grayscale PNG of zeros (8 or 16 bit)."""
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    row = b'\x00' + b'\x00' * (w * depth // 8)  # filter byte + pixels
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, depth, 0, 0, 0, 0)) +
            chunk(b'IDAT', zlib.compress(row * h, 9)) + chunk(b'IEND', b''))


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    src, dst = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    windows = [tuple(float(x) for x in w.split(':')) for w in sys.argv[3:]]
    dst.mkdir(parents=True, exist_ok=True)

    t0 = None
    for line in (src / 'cam0/data.csv').read_text().splitlines():
        if line and not line.startswith('#'):
            t0 = int(line.split(',')[0])
            break

    def link(target, name):
        if name.is_symlink() or name.exists():
            name.unlink()
        name.symlink_to(os.path.relpath(target, name.parent))

    for entry in src.iterdir():
        if entry.name not in ('cam0', 'cam1'):
            link(entry, dst / entry.name)

    n_black = 0
    for cam in ('cam0', 'cam1'):
        (dst / cam / 'data').mkdir(parents=True, exist_ok=True)
        for entry in (src / cam).iterdir():
            if entry.name != 'data':
                link(entry, dst / cam / entry.name)
        black = None
        for img in sorted((src / cam / 'data').iterdir()):
            t = (int(img.stem) - t0) * 1e-9
            out = dst / cam / 'data' / img.name
            if any(s <= t < s + d for s, d in windows):
                if black is None:
                    w, h, depth, ctype = png_size(img)
                    assert ctype == 0, f'{img}: only grayscale images are supported'
                    black = black_png(w, h, depth)
                if out.is_symlink() or out.exists():
                    out.unlink()
                out.write_bytes(black)
                n_black += 1
            else:
                link(img, out)
    print(f'{dst}: {n_black} black images in {len(windows)} windows')


if __name__ == '__main__':
    main()
