"""
DSP stages of the 16-QAM transmitter, listed in processing order.

The module file names start with a digit (01_..., 02_...), which the
normal `import` statement cannot handle. importlib.import_module() loads
them by their string name, and the functions are re-exported here, so
other code can simply write:

    from dsp import prbs15_generator, design_rrc
"""

from importlib import import_module

_prbs15 = import_module("dsp.01_prbs15_generation")
_mapping = import_module("dsp.02_16qam_gray_mapping")
_normalization = import_module("dsp.03_symbol_normalization")
_upsampling = import_module("dsp.04_zero_insertion_upsampling")
_rrc = import_module("dsp.05_rrc_filter_design")
_fir = import_module("dsp.06_fir_pulse_shaping")
_if_modulation = import_module("dsp.07_quadrature_if_modulation")

prbs15_generator = _prbs15.prbs15_generator
qam16_gray_mapping = _mapping.qam16_gray_mapping
qam16_gray_demapping = _mapping.qam16_gray_demapping
normalize_symbols = _normalization.normalize_symbols
upsample_symbols = _upsampling.upsample_symbols
design_rrc = _rrc.design_rrc
apply_fir = _fir.apply_fir
modulate_to_if = _if_modulation.modulate_to_if
