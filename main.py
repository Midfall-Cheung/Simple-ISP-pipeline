#!/usr/bin/env python3
"""End-to-end software ISP pipeline.

RAW (DNG/NEF/...) -> BLC -> DPC -> RAW NR -> LSC -> AWB -> demosaic
                  -> CCM -> LTM -> gamma -> YUV sharpen -> PNG

Usage::

    # process every RAW in data/ and write results/<stem>_isp.png
    python main.py

    # low-light pipeline with RAW-domain denoising + Hamilton-Adams demosaic
    python main.py --raw-nr --demosaic hamilton --sharpen

    # backlit scene: local tone mapping to recover the shadows
    python main.py --ltm --ltm-compress 0.6

    # stage-3 report figures + objective metrics against the libraw reference
    python main.py --input data/backlit_window_nikon.nef --figures all --metrics
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from modules import awb, blc, ccm_gamma, demosaic, dpc, lsc, raw_nr, yuv_post  # noqa: E402
from utils import metrics as metrics_utils  # noqa: E402
from utils.raw_loader import RawFrame, camera_to_srgb_matrix, load_dng  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# Any libraw-supported RAW extension counts for auto-discovery.
RAW_SUFFIXES = {
    ".dng", ".nef", ".arw", ".cr2", ".cr3", ".raf", ".orf",
    ".rw2", ".pef", ".srw", ".raw", ".3fr", ".mrw",
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    """Knobs for the software ISP, one field per pipeline stage."""

    # RAW domain
    use_blc: bool = True
    use_dpc: bool = True
    dpc_threshold: float = 0.05
    raw_nr: bool = False
    raw_nr_strength: float = 1.5
    raw_nr_sigma: Optional[float] = None
    lsc_mode: str = "off"  # off | auto | explicit
    lsc_k1: float = 0.0
    lsc_k2: float = 0.0
    awb_mode: str = "camera"  # camera | gray_world | none

    # demosaic / colour
    demosaic: str = "bilinear"  # bilinear | hamilton
    rgb_nr: bool = False  # comparison-only: bilateral after demosaic
    ltm: bool = False
    ltm_compress: float = 0.6
    ltm_radius: int = 24
    ltm_exposure: float = 1.0

    # display
    gamma: float = 2.2
    gamma_mode: str = "gamma"  # gamma | srgb
    sharpen: bool = False
    sharpen_amount: float = 0.6
    sharpen_radius: float = 1.2
    sharpen_coring: float = 0.01


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

    # ---- RAW domain -------------------------------------------------------
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
    sigma_used: Optional[float] = None
    if config.raw_nr:
        mosaic, sigma_used = raw_nr.denoise_raw(
            mosaic, frame.pattern,
            sigma_color=config.raw_nr_sigma,
            strength=config.raw_nr_strength,
        )
    timings["raw_nr"] = time.perf_counter() - tic
    info["raw_nr_sigma"] = sigma_used

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

    # ---- RGB domain -------------------------------------------------------
    tic = time.perf_counter()
    rgb_linear = demosaic.demosaic(
        mosaic, frame.pattern, frame.color_desc, method=config.demosaic
    )
    timings["demosaic"] = time.perf_counter() - tic

    tic = time.perf_counter()
    if config.rgb_nr:
        rgb_linear, _ = raw_nr.denoise_rgb(
            rgb_linear,
            sigma_color=config.raw_nr_sigma or (sigma_used or None),
            strength=config.raw_nr_strength,
        )
    timings["rgb_nr"] = time.perf_counter() - tic

    tic = time.perf_counter()
    if ccm is None:
        ccm = camera_to_srgb_matrix(frame)
    corrected = ccm_gamma.apply_ccm(rgb_linear, ccm)
    np.clip(corrected, 0.0, 1.0, out=corrected)
    if config.ltm:
        corrected = yuv_post.local_tone_map(
            corrected,
            radius=config.ltm_radius,
            compress=config.ltm_compress,
            exposure=config.ltm_exposure,
        )
    timings["ccm"] = time.perf_counter() - tic

    tic = time.perf_counter()
    display = ccm_gamma.gamma_encode(corrected, gamma=config.gamma, mode=config.gamma_mode)
    if config.sharpen:
        display = yuv_post.unsharp_mask_yuv(
            display,
            amount=config.sharpen_amount,
            radius=config.sharpen_radius,
            coring=config.sharpen_coring,
        )
    image = ccm_gamma.to_uint8(display)
    timings["gamma_sharpen"] = time.perf_counter() - tic

    return {
        "mosaic": mosaic,
        "rgb_linear": rgb_linear,
        "corrected": corrected,
        "display": display,
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


def _pick_crop(
    image: np.ndarray,
    size: int = 480,
    mean_band: Tuple[float, float] = (4.0, 60.0),
) -> Tuple[int, int, int]:
    """Pick the most textured crop whose mean luma sits inside ``mean_band``."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    step = 8
    small = gray[::step, ::step]
    win = max(8, size // step)
    if small.shape[0] <= win or small.shape[1] <= win:
        return 0, 0, size
    mean = cv2.boxFilter(small, -1, (win, win))
    grad = np.abs(cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(
        cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    )
    grad = cv2.boxFilter(grad, -1, (win, win))
    valid = (mean >= mean_band[0]) & (mean <= mean_band[1])
    if not valid.any():
        valid = np.ones_like(mean, dtype=bool)
    score = np.where(valid, grad, -1.0)
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    y0 = min(max(int(y * step) - size // 2, 0), image.shape[0] - size)
    x0 = min(max(int(x * step) - size // 2, 0), image.shape[1] - size)
    return y0, x0, size


def _boost(image: np.ndarray, ev: float) -> np.ndarray:
    """Exposure boost on a display-referred uint8 image (for shadow zooms)."""
    linear = metrics_utils.srgb_to_linear(image.astype(np.float32) / 255.0) * (2.0**ev)
    return np.clip(
        np.rint(metrics_utils.linear_to_srgb(linear) * 255.0), 0, 255
    ).astype(np.uint8)


def _mean_luma(image: np.ndarray) -> float:
    return float(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).mean())


def _shadow_mean(image: np.ndarray, size: int = 400) -> float:
    y0, x0, _ = _pick_crop(image, size)
    gray = cv2.cvtColor(image[y0 : y0 + size, x0 : x0 + size], cv2.COLOR_RGB2GRAY)
    return float(gray.mean())


def _hf_sigma(image: np.ndarray) -> float:
    """High-frequency energy of a patch (0-255 units): residual noise + grain."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    high_pass = gray - cv2.GaussianBlur(gray, (0, 0), 1.2)
    return float(high_pass.std())


def _shadow_stats(image: np.ndarray) -> Tuple[float, float, float]:
    rgb = image.astype(np.float32) / 255.0
    means = rgb.reshape(-1, 3).mean(axis=0)
    return float(means[0]), float(means[1]), float(means[2])


# --------------------------------------------------------------------------- #
# Comparison figures
# --------------------------------------------------------------------------- #
def make_blc_comparison(frame: RawFrame, config: PipelineConfig, out_path: Path) -> Path:
    """Render 'BLC off vs BLC on' including a shadow-region zoom."""
    demo_frame = frame
    pedestal = None
    if float(np.max(frame.black_levels)) <= 0.0:
        # This file stores no BlackLevel tag. Add a synthetic sensor pedestal so
        # the comparison demonstrates the algorithm instead of a no-op.
        pedestal = 256.0
        demo_frame = dataclasses.replace(
            frame,
            mosaic=blc.add_pedestal(frame.mosaic, frame.pattern, pedestal),
            black_levels=np.full_like(frame.black_levels, pedestal),
            meta=dict(frame.meta),
        )

    off_img = run_pipeline(demo_frame, dataclasses.replace(config, use_blc=False))["image"]
    on_img = run_pipeline(demo_frame, dataclasses.replace(config, use_blc=True))["image"]

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
            0.5, -0.02,
            f"crop mean RGB = ({r:.3f}, {g:.3f}, {b:.3f})",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#333333",
        )

    pattern_note = f"CFA {frame.cfa_string}"
    if pedestal is not None:
        pattern_note += (
            f" | no BlackLevel tag, {int(pedestal)}-count pedestal injected for this demo"
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


def make_nr_comparison(frame: RawFrame, config: PipelineConfig, out_path: Path) -> Path:
    """RAW-domain vs RGB-domain denoising, on a boosted shadow crop."""
    base_cfg = dataclasses.replace(
        config, raw_nr=False, rgb_nr=False, ltm=False, sharpen=False
    )
    no_nr = run_pipeline(frame, base_cfg)
    raw_result = run_pipeline(frame, dataclasses.replace(base_cfg, raw_nr=True))
    sigma = raw_result["info"]["raw_nr_sigma"]
    rgb_result = run_pipeline(
        frame, dataclasses.replace(base_cfg, rgb_nr=True, raw_nr_sigma=sigma)
    )

    images = {
        "no NR": no_nr["image"],
        "RAW-domain NR": raw_result["image"],
        "RGB-domain NR": rgb_result["image"],
    }
    y0, x0, size = _pick_crop(images["no NR"], 460, mean_band=(9, 24))
    ev = 3.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), dpi=130)
    for ax, (title, img) in zip(axes, images.items()):
        patch = _boost(img[y0 : y0 + size, x0 : x0 + size], ev)
        ax.imshow(patch)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        noise = _hf_sigma(img[y0 : y0 + size, x0 : x0 + size])
        ax.text(
            0.5, -0.03, f"crop HF σ = {noise:.2f} / 255",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#333333",
        )
    fig.suptitle(
        f"RAW-domain vs RGB-domain denoising - shadow crop, +{ev:g} EV for visibility\n"
        f"{frame.path.name}  (σr = {sigma:.4f}, RAW noise σ ≈ "
        f"{raw_nr.estimate_noise_sigma(no_nr['mosaic'], frame.pattern):.4f})",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_demosaic_comparison(frame: RawFrame, config: PipelineConfig, out_path: Path) -> Path:
    """Bilinear vs Hamilton-Adams on a high-frequency crop, 3x zoom."""
    base_cfg = dataclasses.replace(config, ltm=False, sharpen=False, raw_nr=False, rgb_nr=False)
    bilinear = run_pipeline(frame, dataclasses.replace(base_cfg, demosaic="bilinear"))["image"]
    hamilton = run_pipeline(frame, dataclasses.replace(base_cfg, demosaic="hamilton"))["image"]

    # Pick the crop where the two interpolators disagree most (that is where
    # zipper artefacts and false colour live), while avoiding clipped areas.
    diff = np.abs(bilinear.astype(np.float32) - hamilton.astype(np.float32)).mean(axis=2)
    size = 180
    score = cv2.boxFilter(diff, -1, (size, size))
    gray = cv2.cvtColor(bilinear, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.boxFilter(gray, -1, (size, size))
    score = np.where((mean > 15) & (mean < 240), score, -1.0)
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    y0 = min(max(int(y) - size // 2, 0), bilinear.shape[0] - size)
    x0 = min(max(int(x) - size // 2, 0), bilinear.shape[1] - size)
    zoom = 4
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=130)
    for ax, img, title in (
        (axes[0], bilinear, "Bilinear (zipper / false colour)"),
        (axes[1], hamilton, "Hamilton-Adams (gradient directed)"),
    ):
        patch = cv2.resize(
            img[y0 : y0 + size, x0 : x0 + size], None, fx=zoom, fy=zoom,
            interpolation=cv2.INTER_NEAREST,
        )
        ax.imshow(patch)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(
        f"Demosaic comparison - {size}x{size} crop at {zoom}x\n{frame.path.name}",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_ltm_comparison(frame: RawFrame, config: PipelineConfig, out_path: Path) -> Path:
    """LTM off vs on: full frame plus a shadow crop."""
    base_cfg = dataclasses.replace(config, ltm=False, sharpen=False)
    off_img = run_pipeline(frame, base_cfg)["image"]
    on_img = run_pipeline(frame, dataclasses.replace(base_cfg, ltm=True))["image"]

    y0, x0, size = _pick_crop(off_img, 420, mean_band=(3, 45))
    crop_off = off_img[y0 : y0 + size, x0 : x0 + size]
    crop_on = on_img[y0 : y0 + size, x0 : x0 + size]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.6), dpi=130)
    panels = [
        (axes[0, 0], _downscale(off_img, 900), "LTM off"),
        (axes[0, 1], _downscale(on_img, 900), "LTM on"),
        (axes[1, 0], crop_off, "LTM off - shadow crop"),
        (axes[1, 1], crop_on, "LTM on - shadow crop"),
    ]
    for ax, img, title in panels:
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for ax, img in ((axes[1, 0], crop_off), (axes[1, 1], crop_on)):
        ax.text(
            0.5, -0.02,
            f"crop mean luma = {_mean_luma(img):.1f} / 255",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#333333",
        )
    clipped_off = 100.0 * float(np.mean(off_img == 255))
    clipped_on = 100.0 * float(np.mean(on_img == 255))
    fig.suptitle(
        "Local tone mapping (guided-filter base/detail) - shadows recovered, highlights kept\n"
        f"{frame.path.name}  (compress={config.ltm_compress:g}, "
        f"clipped pixels {clipped_off:.2f}% -> {clipped_on:.2f}%)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(frame: RawFrame, config: PipelineConfig) -> Dict[str, object]:
    """Compare the baseline pipeline and the configured upgrades to libraw."""
    reference = metrics_utils.reference_from_raw(frame.path)
    baseline_cfg = dataclasses.replace(
        config, raw_nr=False, rgb_nr=False, ltm=False, sharpen=False, demosaic="bilinear"
    )
    baseline = run_pipeline(frame, baseline_cfg)["image"]
    upgraded = run_pipeline(frame, config)["image"]
    baseline_metrics = metrics_utils.evaluate(reference, baseline, name="baseline")
    upgraded_metrics = metrics_utils.evaluate(reference, upgraded, name="upgraded")
    upgraded_metrics["delta_vs_baseline"] = {
        "psnr_db": upgraded_metrics["psnr_db"] - baseline_metrics["psnr_db"],
        "ssim": upgraded_metrics["ssim"] - baseline_metrics["ssim"],
        "delta_e_mean": upgraded_metrics["delta_e_mean"] - baseline_metrics["delta_e_mean"],
    }
    return {
        "scene": frame.path.name,
        "baseline": baseline_metrics,
        "upgraded": upgraded_metrics,
        "config": dataclasses.asdict(config),
    }


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
    nr_state = "off"
    if config.raw_nr:
        nr_state = f"on (sigma_r={info['raw_nr_sigma']:.4f})"
    elif config.rgb_nr:
        nr_state = "RGB-domain (comparison)"
    print(f"  RAW NR           : {nr_state}")
    print(
        f"  LSC              : {info['lsc_mode']} (k1={info['lsc_k1']:.4f}, k2={info['lsc_k2']:.4f})"
    )
    print(f"  AWB              : {info['awb_mode']} gains={info['wb_gains']}")
    print(f"  Demosaic         : {config.demosaic}")
    print(f"  LTM              : {'on' if config.ltm else 'off'} (compress={config.ltm_compress:g})")
    print(f"  Sharpen (YUV)    : {'on' if config.sharpen else 'off'}")
    print("  stage timings (ms):")
    for name, seconds in timings.items():
        if seconds <= 0.0:
            continue
        print(f"    {name:<13s} {seconds * 1000:7.1f}")
    print(f"    {'total':<13s} {total * 1000:7.1f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the software ISP pipeline on RAW/DNG files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, nargs="*", default=None, help="RAW files")
    parser.add_argument("--outdir", type=Path, default=RESULTS_DIR, help="output directory")
    parser.add_argument("--no-blc", action="store_true", help="disable black level calibration")
    parser.add_argument("--no-dpc", action="store_true", help="disable defective pixel correction")
    parser.add_argument(
        "--dpc-threshold", type=float, default=0.05, help="DPC detection threshold in [0,1]"
    )
    parser.add_argument("--raw-nr", action="store_true", help="enable RAW-domain bilateral NR")
    parser.add_argument(
        "--raw-nr-strength", type=float, default=1.5, help="σr multiplier over estimated noise"
    )
    parser.add_argument(
        "--raw-nr-sigma", type=float, default=None, help="explicit bilateral σr (overrides auto)"
    )
    parser.add_argument(
        "--lsc", choices=["off", "auto", "explicit"], default="off",
        help="lens shading correction mode",
    )
    parser.add_argument("--lsc-k1", type=float, default=0.0, help="radial gain coefficient k1")
    parser.add_argument("--lsc-k2", type=float, default=0.0, help="radial gain coefficient k2")
    parser.add_argument("--awb", choices=["camera", "gray_world", "none"], default="camera")
    parser.add_argument(
        "--demosaic", choices=["bilinear", "hamilton"], default="bilinear",
        help="demosaic algorithm",
    )
    parser.add_argument("--ltm", action="store_true", help="enable guided-filter local tone mapping")
    parser.add_argument("--ltm-compress", type=float, default=0.6, help="log-base compression")
    parser.add_argument("--ltm-radius", type=int, default=24, help="guided filter radius")
    parser.add_argument("--ltm-exposure", type=float, default=1.0, help="pre-LTM exposure gain")
    parser.add_argument("--sharpen", action="store_true", help="enable YUV unsharp mask")
    parser.add_argument("--sharpen-amount", type=float, default=0.6)
    parser.add_argument("--sharpen-radius", type=float, default=1.2)
    parser.add_argument("--sharpen-coring", type=float, default=0.01, help="noise dead-zone")
    parser.add_argument("--gamma", type=float, default=2.2, help="gamma exponent")
    parser.add_argument("--gamma-mode", choices=["gamma", "srgb"], default="gamma")
    parser.add_argument(
        "--figures", choices=["none", "blc", "nr", "demosaic", "ltm", "all"], default="none",
        help="generate stage comparison figures",
    )
    parser.add_argument(
        "--metrics", action="store_true", help="evaluate against the libraw reference"
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        use_blc=not args.no_blc,
        use_dpc=not args.no_dpc,
        dpc_threshold=args.dpc_threshold,
        raw_nr=args.raw_nr,
        raw_nr_strength=args.raw_nr_strength,
        raw_nr_sigma=args.raw_nr_sigma,
        lsc_mode=args.lsc,
        lsc_k1=args.lsc_k1,
        lsc_k2=args.lsc_k2,
        awb_mode=args.awb,
        demosaic=args.demosaic,
        ltm=args.ltm,
        ltm_compress=args.ltm_compress,
        ltm_radius=args.ltm_radius,
        ltm_exposure=args.ltm_exposure,
        gamma=args.gamma,
        gamma_mode=args.gamma_mode,
        sharpen=args.sharpen,
        sharpen_amount=args.sharpen_amount,
        sharpen_radius=args.sharpen_radius,
        sharpen_coring=args.sharpen_coring,
    )


def discover_inputs(paths: Optional[List[Path]]) -> List[Path]:
    if paths:
        return [Path(p) for p in paths]
    found = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in RAW_SUFFIXES)
    if not found:
        raise SystemExit(
            f"no RAW files found in {DATA_DIR}; drop a DNG/NEF/ARW there or pass --input"
        )
    return found


def _figure_paths(outdir: Path, stem: str, single: bool, kinds: List[str]) -> Dict[str, Path]:
    return {
        kind: outdir / (f"{kind}_comparison.png" if single else f"{stem}_{kind}_comparison.png")
        for kind in kinds
    }


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    inputs = discover_inputs(args.input)
    config = config_from_args(args)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    single = len(inputs) == 1

    all_metrics: List[Dict[str, object]] = []

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

        figures = {
            "blc": make_blc_comparison,
            "nr": make_nr_comparison,
            "demosaic": make_demosaic_comparison,
            "ltm": make_ltm_comparison,
        }
        kinds = list(figures) if args.figures == "all" else (
            [] if args.figures == "none" else [args.figures]
        )
        paths = _figure_paths(outdir, stem, single, kinds)
        for kind in kinds:
            figures[kind](frame, config, paths[kind])
            print(f"[{stem}] wrote {paths[kind]}")

        _print_summary(stem, result, config)

        if args.metrics:
            report = compute_metrics(frame, config)
            all_metrics.append(report)
            print(
                f"  metrics vs libraw reference | baseline: "
                f"{metrics_utils.format_metrics(report['baseline'])}"
            )
            print(
                f"                              | upgraded: "
                f"{metrics_utils.format_metrics(report['upgraded'])}"
            )

    if all_metrics:
        metrics_path = outdir / "metrics.json"
        existing: List[Dict[str, object]] = []
        if metrics_path.exists():
            try:
                existing = json.loads(metrics_path.read_text())
            except json.JSONDecodeError:
                existing = []
        by_scene = {item.get("scene"): item for item in existing if isinstance(item, dict)}
        for item in all_metrics:
            by_scene[item["scene"]] = item
        metrics_path.write_text(json.dumps(list(by_scene.values()), indent=2, ensure_ascii=False))
        print(f"\n[metrics] written to {metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
