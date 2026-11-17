"""
FPGA stages of the 16-QAM transmitter (ideal floating-point reference and
bit-true fixed-point blocks), listed in processing order.

The module file names start with a digit (01_..., 02_...), which the
normal `import` statement cannot handle. importlib.import_module() loads
them by their string name, and the functions are re-exported here, so
other code can simply write:

    from fpga import prbs15_generator, design_rrc
"""

from importlib import import_module

_prbs15 = import_module("fpga.01_prbs15_generation")
_mapping = import_module("fpga.02_16qam_gray_mapping")
_normalization = import_module("fpga.03_symbol_normalization")
_upsampling = import_module("fpga.04_zero_insertion_upsampling")
_rrc = import_module("fpga.05_rrc_filter_design")
_fir = import_module("fpga.06_fir_pulse_shaping")
_if_modulation = import_module("fpga.07_quadrature_if_modulation")

prbs15_generator = _prbs15.prbs15_generator
qam16_gray_mapping = _mapping.qam16_gray_mapping
qam16_gray_demapping = _mapping.qam16_gray_demapping
normalize_symbols = _normalization.normalize_symbols
upsample_symbols = _upsampling.upsample_symbols
design_rrc = _rrc.design_rrc
apply_fir = _fir.apply_fir
modulate_to_if = _if_modulation.modulate_to_if

# Bit-true fixed-point blocks (what the FPGA really computes)
_fixed_point = import_module("fpga.08_fixed_point_arithmetic")
_nco = import_module("fpga.09_nco")

signed_range = _fixed_point.signed_range
required_bits = _fixed_point.required_bits
quantize_coefficients = _fixed_point.quantize_coefficients
shift_right = _fixed_point.shift_right
saturate = _fixed_point.saturate
generate_nco = _nco.generate_nco
