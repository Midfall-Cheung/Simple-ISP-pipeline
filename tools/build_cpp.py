"""Build the C++/OpenMP acceleration module (``modules/_isp_accel*.so``).

Usage::

    python -m tools.build_cpp
    python -m tools.build_cpp --compiler /opt/homebrew/bin/g++-13

Requires a C++17 compiler, ``pybind11`` (pip) and OpenMP. On macOS OpenMP
comes from Homebrew's ``libomp``::

    brew install libomp
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "csrc" / "isp_accel.cpp"
OUTPUT = ROOT / "modules" / f"_isp_accel{sysconfig.get_config_var('EXT_SUFFIX')}"


def _brew_prefix(formula: str) -> Path | None:
    try:
        out = subprocess.run(
            ["brew", "--prefix", formula], capture_output=True, text=True, check=False
        )
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip())
    except FileNotFoundError:
        pass
    for candidate in (Path("/opt/homebrew/opt") / formula, Path("/usr/local/opt") / formula):
        if candidate.exists():
            return candidate
    return None


def find_openmp() -> Path | None:
    for formula in ("libomp", "llvm"):
        prefix = _brew_prefix(formula)
        if prefix and (prefix / "lib").exists() and (prefix / "include").exists():
            return prefix
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default=None, help="C++ compiler (default: clang++/g++)")
    parser.add_argument("--no-openmp", action="store_true", help="build without OpenMP")
    parser.add_argument("--jobs", type=int, default=0, help="unused, kept for parity")
    args = parser.parse_args(argv)

    try:
        import pybind11
    except ImportError:
        print("pybind11 is missing: pip install pybind11", file=sys.stderr)
        return 2

    compiler = args.compiler or ("clang++" if platform.system() == "Darwin" else "g++")
    openmp = None if args.no_openmp else find_openmp()

    cmd = [
        compiler, "-O3", "-std=c++17", "-fPIC", "-shared", "-ffp-contract=off",
        "-I", pybind11.get_include(),
        "-I", sysconfig.get_paths()["include"],
        str(SOURCE), "-o", str(OUTPUT),
    ]
    if platform.system() == "Darwin":
        cmd += ["-Wl,-undefined,dynamic_lookup"]
        try:
            sdk = subprocess.run(
                ["xcrun", "--show-sdk-path"], capture_output=True, text=True, check=False
            ).stdout.strip()
            if sdk:
                cmd += ["-isysroot", sdk]
                # Some CommandLineTools installs ship the libc++ headers only
                # inside the SDK; point the compiler at them explicitly.
                libcxx = Path(sdk) / "usr" / "include" / "c++" / "v1"
                if libcxx.is_dir():
                    cmd += ["-I", str(libcxx)]
        except FileNotFoundError:
            pass
    if openmp:
        cmd += [
            "-Xpreprocessor", "-fopenmp",
            "-I", str(openmp / "include"),
            "-L", str(openmp / "lib"),
            "-lomp",
        ]
        print(f"OpenMP: {openmp}")
    else:
        print("OpenMP: disabled (build without it)")

    print("$ " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        print("build failed", file=sys.stderr)
        return result.returncode
    print(f"built {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
