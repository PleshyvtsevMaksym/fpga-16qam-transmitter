"""
Hardware impairments after the FPGA, in physical order (see pipeline/hardware_pipeline.py).

Every function takes datasheet parameters. The module file names start
with a digit, so they are loaded with importlib.import_module().
"""

from importlib import import_module

_awgn = import_module("hardware.01_awgn")
_dac_quantization = import_module("hardware.02_dac_quantization")
_clock_jitter = import_module("hardware.03_clock_jitter")

add_awgn = _awgn.add_awgn
add_dac_noise = _awgn.add_dac_noise
quantize_dac = _dac_quantization.quantize_dac
compute_full_scale = _dac_quantization.compute_full_scale
add_clock_jitter = _clock_jitter.add_clock_jitter
