"""
03 Physical DAC model: from FPGA codes to the analog DAC output.

Files in physical order, every function takes datasheet parameters:
    01 quantization (code-to-analog)   02 nonlinearity   03 offset/gain error
    04 clock jitter   05 phase noise   06 output dynamics (ZOH, glitches, settling)
    07 DAC noise
The blocks are connected in pipeline/03_physical_dac_pipeline.py.
"""

from importlib import import_module

_quantization = import_module("03_physical_dac.01_dac_quantization")
_nonlinearity = import_module("03_physical_dac.02_dac_nonlinearity")
_offset_gain = import_module("03_physical_dac.03_offset_gain_error")
_clock_jitter = import_module("03_physical_dac.04_clock_jitter")
_phase_noise = import_module("03_physical_dac.05_phase_noise")
_output_dynamics = import_module("03_physical_dac.06_output_dynamics")
_dac_noise = import_module("03_physical_dac.07_dac_noise")

quantize_dac = _quantization.quantize_dac
compute_full_scale = _quantization.compute_full_scale
apply_dac_nonlinearity = _nonlinearity.apply_dac_nonlinearity
apply_offset_gain_error = _offset_gain.apply_offset_gain_error
add_clock_jitter = _clock_jitter.add_clock_jitter
add_clock_phase_noise = _phase_noise.add_clock_phase_noise
apply_zero_order_hold = _output_dynamics.apply_zero_order_hold
add_dac_glitches = _output_dynamics.add_dac_glitches
apply_settling = _output_dynamics.apply_settling
add_dac_noise = _dac_noise.add_dac_noise
