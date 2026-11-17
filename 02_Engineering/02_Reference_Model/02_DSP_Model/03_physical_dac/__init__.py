"""
03 Physical DAC model: from FPGA codes to the analog DAC output.

Files in physical order, every function takes datasheet parameters:
    01 quantization (code-to-analog)   02 nonlinearity   03 offset/gain error
    04 clock jitter   05 phase noise   06 zero-order hold   07 DAC noise
The blocks are connected in pipeline/03_physical_dac_pipeline.py.
"""

from importlib import import_module

_quantization = import_module("03_physical_dac.01_dac_quantization")
_clock_jitter = import_module("03_physical_dac.04_clock_jitter")
_dac_noise = import_module("03_physical_dac.07_dac_noise")

quantize_dac = _quantization.quantize_dac
compute_full_scale = _quantization.compute_full_scale
add_clock_jitter = _clock_jitter.add_clock_jitter
add_dac_noise = _dac_noise.add_dac_noise
