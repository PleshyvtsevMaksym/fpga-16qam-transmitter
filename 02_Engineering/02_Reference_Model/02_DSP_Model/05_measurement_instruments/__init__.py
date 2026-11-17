"""
05 Virtual measurement instruments: they only measure and never change
the signal, like the instruments in the laboratory.

    01 spectrum analyzer   02 vector signal analyzer (EVM)   03 RF metrics   04 oscilloscope
The measurements are run in pipeline/05_measurement_pipeline.py.
"""

from importlib import import_module

_spectrum = import_module("05_measurement_instruments.01_spectrum_analyzer")
_vsa = import_module("05_measurement_instruments.02_vector_signal_analyzer")

compute_spectrum = _spectrum.compute_spectrum
compute_psd = _spectrum.compute_psd
demodulate_if = _vsa.demodulate_if
compute_evm = _vsa.compute_evm
