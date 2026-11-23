"""Benchmark the NumPy vs C++/OpenMP kernels and the full pipeline.

Usage::

    python -m tools.benchmark                 # all RAW files in data/
    python -m tools.benchmark --repeat 5

Writes ``results/benchmark.json`` and prints a Markdown table ready to paste
into the README.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

import main as pipeline
from modules import accel, blc, ccm_gamma, demosaic, dpc, raw_nr, yuv_post
from utils.raw_loader import camera_to_srgb_matrix, load_dng

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_SUFFIXES = {".dng", ".nef", ".arw", ".cr2", ".cr3", ".raf", ".orf", ".rw2", ".pef"}


def timed(fn, repeat: int = 3) -> float:
    """Median wall time in milliseconds (one warm-up run excluded)."""
    fn()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(statistics.median(samples))


def benchmark_scene(path: Path, repeat: int) -> dict:
    frame = load_dng(path)
    ccm = camera_to_srgb_matrix(frame)
    gains = frame.wb_gains()

    mosaic = blc.black_level_calibration(
        frame.mosaic, frame.pattern, frame.black_levels, frame.white_level
    )
    corrected = dpc.correct_defects(mosaic, frame.pattern)[0]
    wb = corrected.copy()
    from modules import awb

    wb = awb.apply_gains(wb, frame.pattern, gains)

    result = {
        "scene": path.name,
        "shape": list(frame.mosaic.shape),
        "cfa": frame.cfa_string,
        "stages": {},
    }
    row = {}

    row["BLC (numpy)"] = timed(
        lambda: blc.black_level_calibration(
            frame.mosaic, frame.pattern, frame.black_levels, frame.white_level
        ),
        repeat,
    )
    row["DPC (opencv)"] = timed(
        lambda: dpc.correct_defects(mosaic, frame.pattern), repeat
    )

    row["Bilateral (python/cv2)"] = timed(
        lambda: raw_nr.denoise_raw(corrected, frame.pattern, sigma_color=0.01, backend="python"),
        repeat,
    )
    if accel.available():
        row["Bilateral (cpp/omp)"] = timed(
            lambda: raw_nr.denoise_raw(corrected, frame.pattern, sigma_color=0.01, backend="cpp"),
            repeat,
        )

    row["HA demosaic (python)"] = timed(
        lambda: demosaic.hamilton_adams_demosaic(wb, frame.pattern, frame.color_desc, backend="python"),
        repeat,
    )
    if accel.available():
        row["HA demosaic (cpp/omp)"] = timed(
            lambda: demosaic.hamilton_adams_demosaic(wb, frame.pattern, frame.color_desc, backend="cpp"),
            repeat,
        )

    rgb = demosaic.hamilton_adams_demosaic(wb, frame.pattern, frame.color_desc)
    linear = ccm_gamma.apply_ccm(rgb, ccm)
    row["LTM (numpy/cv2)"] = timed(lambda: yuv_post.local_tone_map(linear), repeat)

    cfg_python = pipeline.PipelineConfig(
        raw_nr=False, demosaic="hamilton", backend="python", ltm=False, sharpen=False
    )
    cfg_cpp = pipeline.PipelineConfig(
        raw_nr=False, demosaic="hamilton", backend="cpp", ltm=False, sharpen=False
    )
    row["Full pipeline (python)"] = timed(
        lambda: pipeline.run_pipeline(frame, cfg_python), max(1, repeat - 1)
    )
    if accel.available():
        row["Full pipeline (cpp/omp)"] = timed(
            lambda: pipeline.run_pipeline(frame, cfg_cpp), max(1, repeat - 1)
        )

    result["stages"] = {k: round(v, 1) for k, v in row.items()}
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="*", default=None)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--outdir", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    inputs = args.input or sorted(
        p for p in DATA_DIR.iterdir() if p.suffix.lower() in RAW_SUFFIXES
    )
    if not inputs:
        raise SystemExit("no RAW files found")

    print(f"backend: {accel.info()}")
    print(f"platform: {platform.platform()} | {platform.machine()}\n")

    records = [benchmark_scene(Path(p), args.repeat) for p in inputs]

    # Markdown table
    print("| 场景 | 阶段 | Python (ms) | C++/OpenMP (ms) | 加速比 |")
    print("| --- | --- | ---: | ---: | ---: |")
    for rec in records:
        stages = rec["stages"]
        pairs = [
            ("RAW 双边降噪", "Bilateral (python/cv2)", "Bilateral (cpp/omp)"),
            ("Hamilton-Adams 去马赛克", "HA demosaic (python)", "HA demosaic (cpp/omp)"),
            ("完整流水线（含读盘）", "Full pipeline (python)", "Full pipeline (cpp/omp)"),
        ]
        for label, py_key, cpp_key in pairs:
            if cpp_key not in stages:
                continue
            py, cpp = stages[py_key], stages[cpp_key]
            print(f"| {rec['scene']} | {label} | {py:.1f} | {cpp:.1f} | {py / cpp:.2f}x |")

    summary = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "accel": accel.info(),
        "repeat": args.repeat,
        "scenes": records,
    }
    out = Path(args.outdir) / "benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[benchmark] written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
