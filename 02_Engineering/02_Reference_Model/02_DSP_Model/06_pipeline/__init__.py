"""
Pipelines of the transmitter model, one per model folder (01-05).

    run_ideal_reference_pipeline   01 ideal reference -> ideal IF samples
    run_bit_true_fpga_pipeline     02 bit-true FPGA -> DAC codes (+ IF in ideal units)
    run_physical_dac_pipeline      03 physical DAC -> DAC output
    run_analog_output_pipeline     04 analog output -> transmitter output
    run_measurement_pipeline       05 instruments -> spectrum, symbols, EVM
    print_pipeline_report          terminal report of 01
    print_fixed_point_report       terminal report of 02

The file names start with a digit, so they are loaded with
importlib.import_module() and re-exported here:

    from pipeline import run_ideal_reference_pipeline
"""

from importlib import import_module

run_ideal_reference_pipeline = import_module(
    "06_pipeline.01_ideal_reference_pipeline").run_ideal_reference_pipeline
run_bit_true_fpga_pipeline = import_module(
    "06_pipeline.02_bit_true_fpga_pipeline").run_bit_true_fpga_pipeline
run_physical_dac_pipeline = import_module(
    "06_pipeline.03_physical_dac_pipeline").run_physical_dac_pipeline
run_analog_output_pipeline = import_module(
    "06_pipeline.04_analog_output_pipeline").run_analog_output_pipeline
run_measurement_pipeline = import_module(
    "06_pipeline.05_measurement_pipeline").run_measurement_pipeline

_report = import_module("06_pipeline.pipeline_report")
print_pipeline_report = _report.print_pipeline_report
print_fixed_point_report = _report.print_fixed_point_report
