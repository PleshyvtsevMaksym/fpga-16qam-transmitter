"""
01 Ideal reference model: the 16-QAM transmitter in floating point.

Folder and file names start with a digit, so they are loaded with
importlib.import_module() and re-exported here:

    ideal = import_module("01_ideal_reference")
    rrc = ideal.design_rrc(0.35, 8, 100)

The stages are connected in pipeline/01_ideal_reference_pipeline.py.
"""

from importlib import import_module

_prbs15 = import_module("01_ideal_reference.01_prbs15_generation")
_mapping = import_module("01_ideal_reference.02_16qam_gray_mapping")
_normalization = import_module("01_ideal_reference.03_symbol_normalization")
_upsampling = import_module("01_ideal_reference.04_zero_insertion_upsampling")
_rrc = import_module("01_ideal_reference.05_rrc_filter_design")
_fir = import_module("01_ideal_reference.06_fir_pulse_shaping")
_if_modulation = import_module("01_ideal_reference.07_quadrature_if_modulation")

prbs15_generator = _prbs15.prbs15_generator
qam16_gray_mapping = _mapping.qam16_gray_mapping
qam16_gray_demapping = _mapping.qam16_gray_demapping
normalize_symbols = _normalization.normalize_symbols
upsample_symbols = _upsampling.upsample_symbols
design_rrc = _rrc.design_rrc
apply_fir = _fir.apply_fir
modulate_to_if = _if_modulation.modulate_to_if
