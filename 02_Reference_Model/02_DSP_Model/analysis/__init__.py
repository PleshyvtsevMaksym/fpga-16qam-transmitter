"""
Virtual measurement instruments of the 16-QAM model.

They only measure a signal and never change it, like the spectrum
analyzer and the vector signal analyzer in the laboratory.

The module file names start with a digit, so they are loaded with
importlib.import_module() and re-exported here:

    from analysis import compute_spectrum, compute_evm
"""

from importlib import import_module

_spectrum = import_module("analysis.01_fft_spectrum")
_evm = import_module("analysis.02_evm")

compute_spectrum = _spectrum.compute_spectrum
demodulate_if = _evm.demodulate_if
compute_evm = _evm.compute_evm
