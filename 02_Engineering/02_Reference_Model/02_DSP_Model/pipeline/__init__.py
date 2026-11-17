"""
Signal pipelines of the transmitter model.

    run_fpga_pipeline      ideal FPGA transmitter, stages 01-07 -> IF samples
    run_hardware_pipeline  IF samples -> DAC, filter, noise -> transmitter output
    print_pipeline_report  terminal report of the FPGA pipeline
"""

from pipeline.fpga_pipeline import run_fpga_pipeline
from pipeline.hardware_pipeline import run_hardware_pipeline
from pipeline.pipeline_report import print_pipeline_report
