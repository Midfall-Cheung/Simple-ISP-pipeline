"""DNG/RAW loading and calibration-metadata extraction (Stage 1).

Only the pieces an ISP actually needs are surfaced:

* the single-channel Bayer mosaic (``uint16`` counts),
* its 2x2 CFA phase pattern + channel descriptor (e.g. ``BGGR``),
* per-channel black / white levels,
* as-shot white-balance multipliers,
* the camera -> linear sRGB colour matrix used as the CCM (libraw's
  ``rgb_cam``, derived from the DNG colour matrices).

Everything downstream is phase-aware, so the pattern kept here is the single
source of truth for where R / Gr / Gb / B sit in the mosaic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from utils.cfa import channel_map as _channel_map, color_groups

try:  # rawpy is only required when a DNG is actually opened
    import rawpy
except ImportError:  # pragma: no cover - lets pure-numpy helpers be imported
    rawpy = None

# IEC 61966-2-1 sRGB primaries: linear sRGB -> XYZ (D65).
SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)

# XYZ (D65) -> linear sRGB, i.e. the inverse of the matrix above.
XYZ_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ).astype(np.float32)


@dataclass
class RawFrame:
    """A Bayer mosaic plus every calibration number needed by the pipeline."""

    path: Path
    mosaic: np.ndarray  # (H, W) float32, raw sensor counts
    pattern: np.ndarray  # (2, 2) channel index per CFA phase
    color_desc: str  # e.g. "RGBG" (index -> colour name)
    black_levels: np.ndarray  # (4,) ordered like ``color_desc``
    white_level: float
    white_balance: np.ndarray  # (4,) as-shot multipliers, ordered like color_desc
    camera_to_srgb: np.ndarray  # (3, 3) camera RGB -> linear sRGB (libraw rgb_cam)
    dng_color_matrix: Optional[np.ndarray] = None  # (3, 3) DNG XYZ -> camera (D65)
    meta: Dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------------ info
    @property
    def shape(self) -> tuple:
        return self.mosaic.shape

    @property
    def cfa_names(self) -> List[str]:
        """Colour name for every entry of ``pattern`` (index order)."""
        return list(self.color_desc)

    @property
    def cfa_string(self) -> str:
        """Pattern in row-major order, e.g. ``BGGR`` / ``RGGB``."""
        return "".join(self.cfa_names[i] for i in self.pattern.ravel())

    @property
    def channel_map(self) -> np.ndarray:
        """(H, W) map from pixel -> channel index, derived from ``pattern``."""
        cached = self.meta.get("_channel_map")
        if cached is None:
            cached = _channel_map(self.pattern, self.mosaic.shape)
            self.meta["_channel_map"] = cached
        return cached

    def channel_index(self, name: str) -> int:
        """Index of colour ``name`` in ``color_desc`` (first match)."""
        try:
            return self.color_desc.index(name)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"{name!r} not in color_desc {self.color_desc!r}") from exc

    def green_indices(self) -> tuple:
        return tuple(i for i, n in enumerate(self.color_desc) if n == "G")

    def wb_gains(self, reference: str = "G") -> np.ndarray:
        """As-shot white-balance multipliers normalised to the reference colour.

        Some files store a placeholder ``0`` for the second green sample
        (G2); such entries are filled from the reference colour instead of
        being averaged in, which would otherwise halve the gain.
        """
        gains = np.asarray(self.white_balance, dtype=np.float32).copy()
        groups = color_groups(self.color_desc)
        ref_idx = groups.get(reference) or groups["G"]
        valid = [float(gains[i]) for i in ref_idx if gains[i] > 0]
        if not valid:
            raise ValueError(f"invalid white balance data: {self.white_balance}")
        ref = float(np.mean(valid))
        gains[gains <= 0] = ref
        return gains / ref

    # ------------------------------------------------------------- reporting
    def describe(self) -> Dict[str, object]:
        """Human-readable summary used by ``tools/inspect_raw.py``."""
        ch = self.channel_map
        stats = {}
        for name in "RGB":
            idx = [i for i, n in enumerate(self.color_desc) if n == name]
            vals = self.mosaic[np.isin(ch, idx)]
            stats[name] = {
                "count": int(vals.size),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "median": float(np.median(vals)),
            }
        return {
            "file": str(self.path),
            "shape": tuple(int(s) for s in self.mosaic.shape),
            "cfa_pattern": self.cfa_string,
            "color_desc": self.color_desc,
            "pattern_index": self.pattern.tolist(),
            "black_level_per_channel": [float(v) for v in self.black_levels],
            "white_level": float(self.white_level),
            "camera_whitebalance": [float(v) for v in self.white_balance],
            "wb_gains_normalised_to_green": [float(v) for v in self.wb_gains()],
            "camera_to_srgb": self.camera_to_srgb.round(6).tolist(),
            "dng_color_matrix_xyz_to_cam": (
                None if self.dng_color_matrix is None else self.dng_color_matrix.round(6).tolist()
            ),
            "sizes": self.meta.get("sizes", {}),
            "channel_stats": stats,
        }


def load_dng(path: str | Path, crop_margins: bool = False) -> RawFrame:
    """Open a DNG/RAW file and return a :class:`RawFrame`.

    Parameters
    ----------
    path:
        File to open (anything libraw understands, not only ``.dng``).
    crop_margins:
        When ``True`` the sensor active area reported by libraw is cropped.
        The crop offsets must stay even, otherwise the CFA phase would shift;
        that case is rejected explicitly instead of silently mis-aligning.
    """
    if rawpy is None:  # pragma: no cover
        raise ImportError("rawpy is required to load RAW files: pip install rawpy")

    path = Path(path)
    with rawpy.imread(str(path)) as raw:
        mosaic = raw.raw_image_visible.astype(np.float32, copy=True)
        pattern = np.array(raw.raw_pattern, dtype=np.uint8, copy=True)
        color_desc = raw.color_desc.decode("ascii")
        black = np.asarray(raw.black_level_per_channel, dtype=np.float32).copy()
        white = float(raw.white_level)
        wb = np.asarray(raw.camera_whitebalance, dtype=np.float32).copy()
        color_matrix = np.asarray(raw.color_matrix, dtype=np.float32).copy()
        dng_cm = np.asarray(raw.rgb_xyz_matrix, dtype=np.float32)[:3, :3].copy()
        sizes = raw.sizes
        sizes_dict = {
            "raw_height": int(sizes.raw_height),
            "raw_width": int(sizes.raw_width),
            "height": int(sizes.height),
            "width": int(sizes.width),
            "top_margin": int(sizes.top_margin),
            "left_margin": int(sizes.left_margin),
            "crop_left_margin": int(sizes.crop_left_margin),
            "crop_top_margin": int(sizes.crop_top_margin),
            "crop_width": int(sizes.crop_width),
            "crop_height": int(sizes.crop_height),
        }
        # Sanity: the pattern we store must reproduce libraw's own colour map.
        libraw_map = raw.raw_colors_visible
        expected = _channel_map(pattern, mosaic.shape)
        if libraw_map.shape == expected.shape and not np.array_equal(libraw_map, expected):
            raise RuntimeError(
                "CFA phase mismatch between raw_pattern and raw_colors_visible"
            )

    if crop_margins:
        top, left = sizes_dict["crop_top_margin"], sizes_dict["crop_left_margin"]
        height, width = sizes_dict["crop_height"], sizes_dict["crop_width"]
        if top % 2 or left % 2:
            raise ValueError("odd crop margins would break CFA phase alignment")
        mosaic = mosaic[top : top + height, left : left + width]
        sizes_dict["cropped"] = True

    if black.size < len(color_desc):
        raise ValueError(f"unexpected black level count: {black.size}")

    # libraw exposes colour matrix 1 (Standard Light A) / 2 (D65) already
    # adapted into a camera -> linear sRGB matrix ("rgb_cam" in dcraw terms).
    camera_to_srgb = color_matrix[:3, :3].astype(np.float32)
    if dng_cm.size == 9 and np.allclose(dng_cm, 0):
        dng_cm = None

    # Some formats (e.g. several NEFs) come back with a zero rgb_cam even
    # though the XYZ -> camera matrix is present; derive the CCM instead of
    # silently producing a black image.
    if not np.any(camera_to_srgb) and dng_cm is not None:
        camera_to_srgb = camera_to_srgb_from_dng(dng_cm)
    if not np.any(camera_to_srgb):
        raise ValueError(f"no usable colour matrix found in {path}")

    meta: Dict[str, object] = {"sizes": sizes_dict}
    meta["_channel_map"] = _channel_map(pattern, mosaic.shape)

    return RawFrame(
        path=path,
        mosaic=mosaic,
        pattern=pattern,
        color_desc=color_desc,
        black_levels=black,
        white_level=white,
        white_balance=wb,
        camera_to_srgb=camera_to_srgb,
        dng_color_matrix=dng_cm,
        meta=meta,
    )


def camera_to_srgb_matrix(frame: RawFrame, normalize_rows: bool = True) -> np.ndarray:
    """CCM that maps white-balanced camera RGB to linear sRGB.

    ``libraw``'s ``color_matrix`` is the ``rgb_cam`` matrix (camera -> sRGB),
    built from the DNG ``ColorMatrix1``/``ColorMatrix2`` tags and already
    adapted to the shot's illuminant. Rows are normalised so white stays
    white after white balance.
    """
    ccm = np.asarray(frame.camera_to_srgb, dtype=np.float32).copy()
    if normalize_rows:
        sums = ccm.sum(axis=1, keepdims=True)
        sums[np.abs(sums) < 1e-6] = 1.0
        ccm = ccm / sums
    return ccm.astype(np.float32)


def camera_to_srgb_from_dng(xyz_to_cam: np.ndarray) -> np.ndarray:
    """Derive camera -> sRGB from a DNG XYZ -> camera matrix (dcraw method).

    dcraw's ``cam_xyz_coeff`` composes the calibration matrix with the sRGB
    primaries, normalises each camera channel so the reference white maps to
    ``(1, 1, 1)``, and inverts. The result matches libraw's ``rgb_cam``, which
    is why the pipeline can use ``RawFrame.camera_to_srgb`` directly.
    """
    xyz_to_cam = np.asarray(xyz_to_cam, dtype=np.float32)[:3, :3]
    cam_rgb = xyz_to_cam @ SRGB_TO_XYZ
    sums = cam_rgb.sum(axis=1, keepdims=True)
    sums[np.abs(sums) < 1e-6] = 1.0
    return np.linalg.inv(cam_rgb / sums).astype(np.float32)


def load_frame_and_ccm(path: str | Path, crop_margins: bool = False):
    frame = load_dng(path, crop_margins=crop_margins)
    return frame, camera_to_srgb_matrix(frame)
