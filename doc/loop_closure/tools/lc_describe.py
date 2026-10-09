#!/usr/bin/env python3
"""Learned global descriptors for the loop-detection benchmark (phase 0): keyframe-to-keyframe similarity matrices.

Runs in the hloc container (AnyLoc at /AnyLoc, torch.hub cache with DINOv2 / MegaLoc), fed on stdin:
  docker exec -i hloc_simval python3 - <args> < lc_describe.py
Reads the keyframe lists written by make_kf_sets.py (images only, no ground truth) and writes, per method and
sequence, the cosine similarity of every keyframe pair (float16) to <out>/<method>/<seq>_sim.npy, plus
<out>/<method>/meta.json (descriptor size, time per image).

Methods:
  anyloc_g   AnyLoc as published: DINOv2 ViT-G/14, layer 31 value facet, hard VLAD with 32 clusters, AnyLoc's
             cached "indoor" vocabulary (49 152-D)
  anyloc_b   the same recipe on DINOv2 ViT-B/14 (layer 11 value facet, 24 576-D)
  anyloc_s   the same recipe on DINOv2 ViT-S/14 (layer 11 value facet, 12 288-D)
             (b and s have no published vocabulary: one is fitted on the other dataset's keyframes, so EuRoC
             queries use a vocabulary from TUM-VI and the reverse; nothing is fitted on the sequence it describes)
  megaloc    MegaLoc (DINOv2-B + SALAD-type head, 8448-D), shorter side resized to 322 px
Images keep their size (cropped to multiples of 14) for AnyLoc.

Timing mode (--time cpu|cuda): describes --n images of the first sequence only and reports ms per image, with
torch limited to --threads threads (pin the process with taskset from outside).

usage: lc_describe.py --lists DIR --out DIR --methods anyloc_g,megaloc [--seqs room1,MH_01] [--time cpu --n 30]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image

sys.path.insert(0, "/AnyLoc")
from utilities import DinoV2ExtractFeatures, VLAD  # noqa: E402

NC = 32
ANYLOC = {"anyloc_g": ("dinov2_vitg14", 31), "anyloc_b": ("dinov2_vitb14", 11), "anyloc_s": ("dinov2_vits14", 11)}
TUMVI = [f"room{i}" for i in range(1, 7)]
NORM = tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def open_rgb(path):
    """8-bit RGB: TUM-VI's 16-bit images are scaled down (a plain convert saturates them to white)."""
    img = Image.open(path)
    if img.mode in ("I;16", "I;16B", "I"):
        img = Image.fromarray((np.asarray(img, dtype=np.uint32) >> 8).astype(np.uint8))
    return img.convert("RGB")


def load_anyloc_input(path):
    x = NORM(tvf.functional.to_tensor(open_rgb(path)))
    h, w = (x.shape[1] // 14) * 14, (x.shape[2] // 14) * 14
    return x[:, :h, :w][None]


def load_megaloc_input(path):
    img = open_rgb(path)
    s = 322 / min(img.size)
    w, h = round(img.width * s / 14) * 14, round(img.height * s / 14) * 14
    return NORM(tvf.functional.to_tensor(img.resize((w, h), Image.BICUBIC)))[None]


class AnyLocMethod:
    def __init__(self, name, dev, out):
        model, layer = ANYLOC[name]
        self.name, self.dev = name, dev
        self.ext = DinoV2ExtractFeatures(model, layer, "value", device=dev)
        self.vlads = {}
        if name == "anyloc_g":
            v = VLAD(NC, desc_dim=None, cache_dir=f"/AnyLoc/cache/vocabulary/dinov2_vitg14/l31_value_c{NC}/indoor")
            v.fit(None)
            self.vlads = {"any": v}
        self.vocab_dir = out / name / "vocab"

    def patches(self, path):
        with torch.no_grad():
            return self.ext(load_anyloc_input(path).to(self.dev)).squeeze(0).float().cpu()

    def fit_cross_vocabs(self, lists):
        """Vocabulary per dataset, fitted on the other dataset's keyframes (every 10th, 400 patches each)."""
        rng = np.random.default_rng(0)
        for target, source in (("tumvi", [s for s in lists if s not in TUMVI]), ("euroc", [s for s in lists if s in TUMVI])):
            cache = self.vocab_dir / f"for_{target}"
            v = VLAD(NC, desc_dim=None, cache_dir=str(cache))
            if not (cache / "c_centers.pt").exists():
                train = []
                for s in source:
                    for p in lists[s][::10]:
                        f = self.patches(p)
                        train.append(f[rng.choice(len(f), min(len(f), 400), replace=False)])
                cache.mkdir(parents=True, exist_ok=True)
                v.fit(torch.cat(train))
            else:
                v.fit(None)
            self.vlads[target] = v

    def describe(self, seq, path):
        v = self.vlads.get("any") or self.vlads["tumvi" if seq in TUMVI else "euroc"]
        return v.generate(self.patches(path)).numpy()


class MegaLocMethod:
    def __init__(self, dev):
        self.dev = dev
        self.net = torch.hub.load("gmberton/MegaLoc", "get_trained_model").eval().to(dev)

    def describe(self, seq, path):
        with torch.no_grad():
            return self.net(load_megaloc_input(path).to(self.dev)).float().squeeze(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default="anyloc_g,anyloc_b,anyloc_s,megaloc")
    ap.add_argument("--seqs", default="")
    ap.add_argument("--time", default="", help="cpu or cuda: timing only")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    lists = {f.stem: f.read_text().split() for f in sorted(Path(a.lists).glob("*.txt"))}
    seqs = a.seqs.split(",") if a.seqs else list(lists)
    out = Path(a.out)
    dev = a.time or "cuda"
    if dev == "cpu":
        torch.set_num_threads(a.threads)
    for name in a.methods.split(","):
        m = MegaLocMethod(dev) if name == "megaloc" else AnyLocMethod(name, dev, out)
        if isinstance(m, AnyLocMethod) and name != "anyloc_g":
            m.fit_cross_vocabs(lists)
        if a.time:
            files = lists[seqs[0]][: a.n + 2]
            m.describe(seqs[0], files[0]), m.describe(seqs[0], files[1])  # warm-up
            t0 = time.time()
            for f in files[2:]:
                d = m.describe(seqs[0], f)
            ms = (time.time() - t0) * 1e3 / (len(files) - 2)
            print(json.dumps({"method": name, "device": dev, "threads": a.threads if dev == "cpu" else None,
                              "seq": seqs[0], "ms_per_image": round(ms, 1), "dim": int(d.size)}), flush=True)
            continue
        (out / name).mkdir(parents=True, exist_ok=True)
        meta = {"method": name, "per_seq": {}}
        for s in seqs:
            t0 = time.time()
            G = np.stack([m.describe(s, f) for f in lists[s]]).astype(np.float32)
            ms = (time.time() - t0) * 1e3 / len(G)
            G /= np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-12)
            np.save(out / name / f"{s}_sim.npy", (G @ G.T).astype(np.float16))
            meta["dim"] = int(G.shape[1])
            meta["per_seq"][s] = {"n": len(G), "ms_per_image_gpu": round(ms, 1)}
            print(f"{name} {s}: {len(G)} keyframes, {ms:.1f} ms per image", flush=True)
        (out / name / "meta.json").write_text(json.dumps(meta, indent=1))
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
