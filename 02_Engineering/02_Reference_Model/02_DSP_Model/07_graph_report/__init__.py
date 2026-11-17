"""
Report: save every figure of the model as a separate PDF page with the run
parameters, measured values and a description, into versioned folders.

    py -m 07_graph_report            -> results/DSP_Graphs/v0.N (next minor version)
    py -m 07_graph_report --major    -> results/DSP_Graphs/vN.0 (next major version)

main.py only shows the figures; this command only writes the report.
"""

from importlib import import_module

_build = import_module("07_graph_report.04_build_report")
build_report = _build.build_report
collect_data = _build.collect_data
GRAPHS_DIR = import_module("07_graph_report.01_report_context").GRAPHS_DIR
