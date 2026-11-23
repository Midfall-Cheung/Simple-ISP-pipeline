"""Stage-1 metadata verification: dump the calibration data of a RAW/DNG file.

Usage::

    python -m tools.inspect_raw data/a0001-jmac_DSC1459.dng
    python -m tools.inspect_raw data/*.dng --json results/dng_metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils.raw_loader import camera_to_srgb_from_dng, camera_to_srgb_matrix, load_dng


def format_report(info: dict, ccm: np.ndarray) -> str:
    lines = [
        "=" * 78,
        f"RAW FILE : {info['file']}",
        "=" * 78,
        f"mosaic shape            : {info['shape']}  (single-channel Bayer)",
        f"CFA pattern             : {info['cfa_pattern']}   "
        f"(pattern index {info['pattern_index']}, color_desc '{info['color_desc']}')",
        f"black_level_per_channel : {info['black_level_per_channel']}",
        f"white_level             : {info['white_level']:.0f}",
        f"camera_whitebalance     : {info['camera_whitebalance']}",
        f"WB gains / green        : {info['wb_gains_normalised_to_green']}",
    ]
    lines.append("CCM (camera -> linear sRGB, libraw rgb_cam, row-normalised):")
    for row in ccm:
        lines.append("    [" + "  ".join(f"{v: .6f}" for v in row) + "]")
    if info.get("dng_color_matrix_xyz_to_cam"):
        lines.append("DNG ColorMatrix2 (XYZ -> camera, D65):")
        for row in info["dng_color_matrix_xyz_to_cam"]:
            lines.append("    [" + "  ".join(f"{v: .6f}" for v in row) + "]")
    lines.append("per-colour mosaic statistics (raw counts):")
    for name, st in info["channel_stats"].items():
        lines.append(
            f"    {name}: n={st['count']:>8d}  min={st['min']:>6.1f}  "
            f"max={st['max']:>6.1f}  median={st['median']:>7.1f}"
        )
    sizes = info.get("sizes") or {}
    lines.append(
        "sizes                   : "
        f"{sizes.get('width')}x{sizes.get('height')} visible, "
        f"crop {sizes.get('crop_width')}x{sizes.get('crop_height')} @ "
        f"({sizes.get('crop_left_margin')},{sizes.get('crop_top_margin')})"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="one or more RAW/DNG files")
    parser.add_argument("--json", type=Path, default=None, help="dump all metadata to JSON")
    args = parser.parse_args(argv)

    records = []
    for path in args.inputs:
        frame = load_dng(path)
        info = frame.describe()
        ccm = camera_to_srgb_matrix(frame)
        print(format_report(info, ccm))
        info["ccm_cam_to_srgb"] = ccm.round(6).tolist()
        if frame.dng_color_matrix is not None:
            derived = camera_to_srgb_from_dng(frame.dng_color_matrix)
            info["ccm_from_dng_colormatrix"] = derived.round(6).tolist()
            info["ccm_crosscheck_max_abs_diff"] = float(np.abs(derived - ccm).max())
        records.append(info)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(records, indent=2, ensure_ascii=False))
        print(f"\n[inspect_raw] metadata written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
