#!/usr/bin/env python3
"""End-to-end software ISP pipeline.

RAW (DNG) -> BLC -> DPC -> LSC -> AWB -> demosaic -> CCM -> gamma -> PNG

Usage::

    # process every DNG in data/ and write results/<stem>_isp.png
    python main.py

    python main.py --input data/a0001-jmac_DSC1459.dng --awb gray_world
    python main.py --lsc auto --no-dpc --outdir results
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from modules import awb, blc, ccm_gamma, demosaic, dpc, lsc  # noqa: E402
from utils.raw_loader import RawFrame, camera_to_srgb_matrix, load_dng  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    """Knobs for the software ISP, one field per pipeline stage."""

    use_blc: bool = True
    use_dpc: bool = True
    dpc_threshold: float = 0.05
    lsc_mode: str = "off"  # off | auto | explicit
    lsc_k1: float = 0.0
    lsc_k2: float = 0.0
    awb_mode: str = "camera"  # camera | gray_world | none
    demosaic: str = "bilinear"
    gamma: float = 2.2
    gamma_mode: str = "gamma"  # gamma | srgb


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(
    frame: RawFrame,
    config: PipelineConfig,
    ccm: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Run the ISP chain on one frame and return every intermediate result."""
    timings: Dict[str, float] = {}
    info: Dict[str, object] = {"input": str(frame.path), "cfa": frame.cfa_string}

    tic = time.perf_counter()
    if config.use_blc:
        mosaic = blc.black_level_calibration(
            frame.mosaic, frame.pattern, frame.black_levels, frame.white_level
        )
    else:
        mosaic = np.clip(frame.mosaic / float(frame.white_level), 0.0, 1.0).astype(np.float32)
    timings["blc"] = time.perf_counter() - tic

    tic = time.perf_counter()
    defect_count = 0
    if config.use_dpc:
        mosaic, defect_mask = dpc.correct_defects(
            mosaic, frame.pattern, threshold=config.dpc_threshold
        )
        defect_count = int(defect_mask.sum())
    timings["dpc"] = time.perf_counter() - tic
    info["defect_count"] = defect_count

    tic = time.perf_counter()
    if config.lsc_mode == "auto":
        k1, k2 = lsc.estimate_coefficients(mosaic)
    elif config.lsc_mode == "explicit":
        k1, k2 = config.lsc_k1, config.lsc_k2
    else:
        k1, k2 = 0.0, 0.0
    if config.lsc_mode != "off":
        mosaic = lsc.apply_lsc(mosaic, k1, k2)
    timings["lsc"] = time.perf_counter() - tic
    info.update({"lsc_k1": k1, "lsc_k2": k2, "lsc_mode": config.lsc_mode})

    tic = time.perf_counter()
    if config.awb_mode == "camera":
        gains = frame.wb_gains()
    elif config.awb_mode == "gray_world":
        gains = awb.gray_world_gains(mosaic, frame.pattern, frame.color_desc)
    elif config.awb_mode == "none":
        gains = None
    else:
        raise ValueError(f"unknown AWB mode: {config.awb_mode!r}")
    if gains is not None:
        mosaic = awb.apply_gains(mosaic, frame.pattern, gains)
    timings["awb"] = time.perf_counter() - tic
    info.update({"awb_mode": config.awb_mode, "wb_gains": None if gains is None else gains.tolist()})

    tic = time.perf_counter()
    rgb_linear = demosaic.demosaic(
        mosaic, frame.pattern, frame.color_desc, method=config.demosaic
    )
    timings["demosaic"] = time.perf_counter() - tic

    tic = time.perf_counter()
    if ccm is None:
        ccm = camera_to_srgb_matrix(frame)
    image = ccm_gamma.color_finish(
        rgb_linear, ccm, gamma=config.gamma, gamma_mode=config.gamma_mode
    )
    timings["ccm_gamma"] = time.perf_counter() - tic

    return {
        "mosaic": mosaic,
        "rgb_linear": rgb_linear,
        "image": image,
        "ccm": ccm,
        "timings": timings,
        "info": info,
    }


# --------------------------------------------------------------------------- #
# Visualisation helpers
# --------------------------------------------------------------------------- #
def _downscale(image: np.ndarray, max_width: int = 1000) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / float(w)
    return cv2.resize(image, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def _darkest_window(gray: np.ndarray, size: int) -> Tuple[int, int]:
    """Top-left corner of the lowest-mean ``size`` x ``size`` window."""
    stride = 4
    small = gray[::stride, ::stride].astype(np.float32)
    win = max(1, size // stride)
    if small.shape[0] <= win or small.shape[1] <= win:
        return 0, 0
    integral = cv2.integral(small)
    sums = (
        integral[win:, win:]
        - integral[:-win, win:]
        - integral[win:, :-win]
        + integral[:-win, :-win]
    )
    y, x = np.unravel_index(int(np.argmin(sums)), sums.shape)
    return int(y * stride), int(x * stride)


def _shadow_stats(image: np.ndarray) -> Tuple[float, float, float]:
    rgb = image.astype(np.float32) / 255.0
    means = rgb.reshape(-1, 3).mean(axis=0)
    return float(means[0]), float(means[1]), float(means[2])


def make_blc_comparison(
    frame: RawFrame,
    config: PipelineConfig,
    out_path: Path,
) -> Path:
    """Render 'BLC off vs BLC on' including a shadow-region zoom."""
    demo_frame = frame
    pedestal = None
    if float(np.max(frame.black_levels)) <= 0.0:
        # This DNG stores no BlackLevel tag. Add a synthetic sensor pedestal so
        # the comparison demonstrates the algorithm instead of a no-op.
        pedestal = 256.0
        demo_frame = dataclasses.replace(
            frame,
            mosaic=blc.add_pedestal(frame.mosaic, frame.pattern, pedestal),
            black_levels=np.full_like(frame.black_levels, pedestal),
            meta=dict(frame.meta),
        )

    off_result = run_pipeline(demo_frame, dataclasses.replace(config, use_blc=False))
    on_result = run_pipeline(demo_frame, dataclasses.replace(config, use_blc=True))

    off_img = off_result["image"]
    on_img = on_result["image"]

    gray = cv2.cvtColor(_downscale(on_img, 1200), cv2.COLOR_RGB2GRAY)
    crop_size = int(min(gray.shape) * 0.28)
    cy, cx = _darkest_window(gray, crop_size)
    scale = on_img.shape[1] / float(gray.shape[1])
    cy, cx = int(cy * scale), int(cx * scale)
    crop_size = int(min(on_img.shape[:2]) * 0.28)
    cy = min(cy, on_img.shape[0] - crop_size)
    cx = min(cx, on_img.shape[1] - crop_size)

    def crop(img):
        return img[cy : cy + crop_size, cx : cx + crop_size]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=130)
    panels = [
        (axes[0, 0], _downscale(off_img, 900), "BLC off"),
        (axes[0, 1], _downscale(on_img, 900), "BLC on"),
        (axes[1, 0], crop(off_img), "BLC off - shadow zoom"),
        (axes[1, 1], crop(on_img), "BLC on - shadow zoom"),
    ]
    for ax, img, title in panels:
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for ax, img in ((axes[1, 0], crop(off_img)), (axes[1, 1], crop(on_img))):
        r, g, b = _shadow_stats(img)
        ax.text(
            0.5,
            -0.02,
            f"crop mean RGB = ({r:.3f}, {g:.3f}, {b:.3f})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="#333333",
        )

    pattern_note = f"CFA {frame.cfa_string}"
    if pedestal is not None:
        pattern_note += (
            f" | DNG has no BlackLevel tag, {int(pedestal)}-count pedestal injected for this demo"
        )
    fig.suptitle(
        "Black level calibration - shadows lifted and coloured by the sensor pedestal\n"
        f"{frame.path.name}  ({pattern_note})",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Reporting / CLI
# --------------------------------------------------------------------------- #
def _print_summary(stem: str, result: dict, config: PipelineConfig) -> None:
    info = result["info"]
    timings = result["timings"]
    total = sum(timings.values())
    print(f"\n[{stem}] pipeline summary")
    print(f"  CFA pattern      : {info['cfa']}")
    print(f"  BLC              : {'on' if config.use_blc else 'off'}")
    print(f"  DPC              : {info['defect_count']} defect pixel(s) corrected")
    print(
        f"  LSC              : {info['lsc_mode']} (k1={info['lsc_k1']:.4f}, k2={info['lsc_k2']:.4f})"
    )
    print(f"  AWB              : {info['awb_mode']} gains={info['wb_gains']}")
    print("  stage timings (ms):")
    for name, seconds in timings.items():
        print(f"    {name:<11s} {seconds * 1000:7.1f}")
    print(f"    {'total':<11s} {total * 1000:7.1f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the software ISP pipeline on RAW/DNG files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, nargs="*", default=None, help="RAW/DNG files")
    parser.add_argument("--outdir", type=Path, default=RESULTS_DIR, help="output directory")
    parser.add_argument("--no-blc", action="store_true", help="disable black level calibration")
    parser.add_argument("--no-dpc", action="store_true", help="disable defective pixel correction")
    parser.add_argument(
        "--dpc-threshold", type=float, default=0.05, help="DPC detection threshold in [0,1]"
    )
    parser.add_argument(
        "--lsc",
        choices=["off", "auto", "explicit"],
        default="off",
        help="lens shading correction mode",
    )
    parser.add_argument("--lsc-k1", type=float, default=0.0, help="radial gain coefficient k1")
    parser.add_argument("--lsc-k2", type=float, default=0.0, help="radial gain coefficient k2")
    parser.add_argument("--awb", choices=["camera", "gray_world", "none"], default="camera")
    parser.add_argument("--demosaic", choices=["bilinear"], default="bilinear")
    parser.add_argument("--gamma", type=float, default=2.2, help="gamma exponent")
    parser.add_argument("--gamma-mode", choices=["gamma", "srgb"], default="gamma")
    parser.add_argument(
        "--no-comparison", action="store_true", help="skip the BLC on/off comparison figure"
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        use_blc=not args.no_blc,
        use_dpc=not args.no_dpc,
        dpc_threshold=args.dpc_threshold,
        lsc_mode=args.lsc,
        lsc_k1=args.lsc_k1,
        lsc_k2=args.lsc_k2,
        awb_mode=args.awb,
        demosaic=args.demosaic,
        gamma=args.gamma,
        gamma_mode=args.gamma_mode,
    )


def discover_inputs(paths: Optional[List[Path]]) -> List[Path]:
    if paths:
        return [Path(p) for p in paths]
    found = sorted(DATA_DIR.glob("*.dng")) + sorted(DATA_DIR.glob("*.DNG"))
    if not found:
        raise SystemExit(
            f"no .dng files found in {DATA_DIR}; drop a RAW file there or pass --input"
        )
    return found


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    inputs = discover_inputs(args.input)
    config = config_from_args(args)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for path in inputs:
        frame = load_dng(path)
        result = run_pipeline(frame, config)
        stem = path.stem

        out_image = outdir / f"{stem}_isp.png"
        cv2.imwrite(
            str(out_image),
            cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, 9],
        )
        print(f"[{stem}] wrote {out_image}")

        if not args.no_comparison:
            suffix = "" if len(inputs) == 1 else f"_{stem}"
            comp_path = outdir / f"blc_comparison{suffix}.png"
            make_blc_comparison(frame, config, comp_path)
            print(f"[{stem}] wrote {comp_path}")

        _print_summary(stem, result, config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
