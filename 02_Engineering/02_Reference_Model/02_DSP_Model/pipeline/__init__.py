"""
Signal pipelines of the transmitter model.

    run_fpga_pipeline              ideal FPGA transmitter, stages 01-07 -> IF samples
    run_fpga_fixed_point_pipeline  bit-true FPGA (integer word widths) -> DAC codes
    run_hardware_pipeline  IF samples -> DAC, filter, noise -> transmitter output
    print_pipeline_report  terminal report of the FPGA pipeline
    print_fixed_point_report  word widths and checks of the bit-true FPGA
"""

from pipeline.fpga_pipeline import run_fpga_pipeline, run_fpga_fixed_point_pipeline
from pipeline.hardware_pipeline import run_hardware_pipeline
from pipeline.pipeline_report import print_pipeline_report, print_fixed_point_report
