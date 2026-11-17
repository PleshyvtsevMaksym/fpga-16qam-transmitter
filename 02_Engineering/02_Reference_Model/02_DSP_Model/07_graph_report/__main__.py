"""Command line: py -m 07_graph_report [--major]"""

import sys
from importlib import import_module

import matplotlib

matplotlib.use("Agg")    # write files only, open no windows

_report = import_module("07_graph_report")
build_report, GRAPHS_DIR = _report.build_report, _report.GRAPHS_DIR    # noqa: E402

if __name__ == "__main__":
    build_report(GRAPHS_DIR, new_major="--major" in sys.argv[1:])
