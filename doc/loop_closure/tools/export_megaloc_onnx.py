#!/usr/bin/env python3
"""Export MegaLoc (Berton & Masone 2025, torch.hub gmberton/MegaLoc) to ONNX for SaDVIO's learned loop detector.

Runs in the hloc container, fed on stdin:
  docker exec -i hloc_simval python3 - <out_dir> <H>x<W> [<H>x<W> ...] < export_megaloc_onnx.py
One file per input size (fixed shapes, multiples of 14: MegaLoc then does no resize of its own), e.g. 322x322 for
TUM-VI (512 x 512) and 322x504 for EuRoC (480 x 752):
  <out_dir>/megaloc_<H>x<W>.onnx   input "image" [1, 3, H, W] float32, RGB normalised with the ImageNet mean / std;
                                   output "descriptor" [1, 8448], L2-normalised
The TorchScript exporter is used (dynamo=False): it needs no onnx package. A check compares the exported model's
output with PyTorch's when onnxruntime is available.
"""
import sys
from pathlib import Path

import torch


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    net = torch.hub.load("gmberton/MegaLoc", "get_trained_model").eval()
    for size in sys.argv[2:]:
        h, w = (int(v) for v in size.split("x"))
        assert h % 14 == 0 and w % 14 == 0, "sizes must be multiples of 14"
        x = torch.randn(1, 3, h, w)
        f = out / f"megaloc_{h}x{w}.onnx"
        with torch.no_grad():
            torch.onnx.export(net, (x,), str(f), input_names=["image"], output_names=["descriptor"],
                              opset_version=17, dynamo=False)
        print(f"{f}: {f.stat().st_size / 1e6:.0f} MB", flush=True)
        try:
            import onnxruntime as ort
            ref = net(x).detach().numpy()
            got = ort.InferenceSession(str(f)).run(None, {"image": x.numpy()})[0]
            print(f"  max |onnx - torch| = {abs(got - ref).max():.2e}", flush=True)
        except ImportError:
            pass


if __name__ == "__main__":
    main()
