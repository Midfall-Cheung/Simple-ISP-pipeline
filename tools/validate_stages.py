"""Quantitative stage validation on a synthetic ground truth.

A libraw-rendered linear RGB frame is treated as ground truth, sampled
through the CFA pattern and then reconstructed. That isolates each stage:

* demosaic: bilinear vs Hamilton-Adams PSNR/SSIM against the ground truth;
* RAW-domain NR: the noisy mosaic is denoised before demosaicing, and the
  RGB-domain variant after; both are scored against the clean reconstruction.

Usage::

    python -m tools.validate_stages --input data/backlit_window_nikon.nef
    python -m tools.validate_stages --input data/a0001-jmac_DSC1459.dng --noise 0.01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from modules import demosaic, raw_nr
from utils.cfa import mosaic_from_rgb
from utils.metrics import psnr, ssim
from utils.raw_loader import load_dng


def render_linear_reference(path: Path) -> np.ndarray:
    import rawpy

    with rawpy.imread(str(path)) as raw:
        return raw.postprocess(
            use_camera_wb=True, no_auto_bright=True, output_bps=16, gamma=(1, 1)
        ).astype(np.float32) / 65535.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--crop", type=int, nargs=4, default=[600, 1500, 1200, 2000],
                        metavar=("Y0", "Y1", "X0", "X1"))
    parser.add_argument("--noise", type=float, default=0.01, help="additive noise sigma")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--sigma-r", type=float, default=None,
        help="bilateral σr; defaults to 1.5x the known added-noise sigma",
    )
    args = parser.parse_args(argv)

    frame = load_dng(args.input)
    truth_full = render_linear_reference(args.input)
    y0, y1, x0, x1 = args.crop
    ground_truth = np.clip(truth_full[y0:y1, x0:x1], 0.0, 1.0)

    mosaic = mosaic_from_rgb(ground_truth, frame.pattern, frame.color_desc)

    print(f"file        : {args.input.name}")
    print(f"crop        : {ground_truth.shape[1]}x{ground_truth.shape[0]}")
    print(f"CFA         : {frame.cfa_string}")
    print("\n[demosaic] PSNR / SSIM vs ground truth")
    for method in ("bilinear", "hamilton"):
        rgb = demosaic.demosaic(mosaic, frame.pattern, frame.color_desc, method=method)
        print(f"  {method:<9s} {psnr(ground_truth, rgb):6.2f} dB   SSIM {ssim(ground_truth, rgb):.4f}")

    rng = np.random.default_rng(args.seed)
    noisy = np.clip(mosaic + rng.normal(0.0, args.noise, mosaic.shape), 0.0, 1.0).astype(np.float32)
    clean = demosaic.hamilton_adams_demosaic(mosaic, frame.pattern, frame.color_desc)
    print(f"\n[denoise] additive noise sigma={args.noise:g}; PSNR / SSIM vs clean reconstruction")
    sigma_r = args.sigma_r if args.sigma_r is not None else 1.5 * args.noise
    variants = {}
    variants["no NR"] = demosaic.hamilton_adams_demosaic(noisy, frame.pattern, frame.color_desc)
    denoised, sigma_r = raw_nr.denoise_raw(noisy, frame.pattern, sigma_color=sigma_r)
    variants["RAW NR"] = demosaic.hamilton_adams_demosaic(denoised, frame.pattern, frame.color_desc)
    rgb_noisy = demosaic.hamilton_adams_demosaic(noisy, frame.pattern, frame.color_desc)
    rgb_denoised, _ = raw_nr.denoise_rgb(rgb_noisy, sigma_color=sigma_r)
    variants["RGB NR"] = rgb_denoised
    for name, rgb in variants.items():
        print(f"  {name:<7s} {psnr(clean, rgb):6.2f} dB   SSIM {ssim(clean, rgb):.4f}")
    print(f"\n  (bilateral sigma_r = {sigma_r:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
