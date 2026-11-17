"""
02 Bit-true FPGA model: integer arithmetic with the FPGA word widths.

    fpga = import_module("02_bit_true_fpga")

The blocks are connected in pipeline/02_bit_true_fpga_pipeline.py.
"""

from importlib import import_module

_arithmetic = import_module("02_bit_true_fpga.01_fixed_point_arithmetic")
_nco = import_module("02_bit_true_fpga.02_nco")

signed_range = _arithmetic.signed_range
required_bits = _arithmetic.required_bits
quantize_coefficients = _arithmetic.quantize_coefficients
shift_right = _arithmetic.shift_right
saturate = _arithmetic.saturate
generate_nco = _nco.generate_nco
