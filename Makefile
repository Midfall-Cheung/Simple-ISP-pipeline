PY ?= .venv/bin/python

.PHONY: help install dev build demo bench check

help:
	@echo "make install  - install runtime requirements"
	@echo "make dev      - install build extras (pybind11)"
	@echo "make build    - build the C++/OpenMP extension"
	@echo "make demo     - run the full ISP on data/* with all upgrades"
	@echo "make bench    - benchmark NumPy vs C++/OpenMP"
	@echo "make check    - run the metric self-tests"

install:
	$(PY) -m pip install -r requirements.txt

dev:
	$(PY) -m pip install -r requirements-dev.txt

build:
	$(PY) -m tools.build_cpp

demo:
	$(PY) main.py --raw-nr --demosaic hamilton --ltm --sharpen --figures all --metrics

bench:
	$(PY) -m tools.benchmark

check:
	$(PY) -m utils.metrics
