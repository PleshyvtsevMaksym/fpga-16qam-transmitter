"""
05 Measurement pipeline: what the laboratory instruments report.

    transmitter output
      -> 01 spectrum analyzer         amplitude spectrum
      -> 02 vector signal analyzer    sync, symbols, EVM
      -> 03 RF metrics, 04 oscilloscope model           (later)
"""

from importlib import import_module

import numpy as np

_instruments = import_module("05_measurement_instruments")
compute_spectrum = _instruments.compute_spectrum
demodulate_if = _instruments.demodulate_if
compute_evm = _instruments.compute_evm


def run_measurement_pipeline(
    samples: np.ndarray,
    rrc_coefficients: np.ndarray,
    reference_symbols: np.ndarray,
    sampling_rate: float,
    intermediate_frequency: float,
    samples_per_symbol: int
) -> dict:
    """
    Measure a transmitter output like the laboratory instruments do.

    Parameters
    ----------
    samples : np.ndarray
        Real transmitter output samples.
    rrc_coefficients : np.ndarray
        Ideal RRC coefficients (the analyzer's matched filter).
    reference_symbols : np.ndarray
        Known transmitted symbols at Es = 1 ("known data" mode of the VSA).
    sampling_rate : float
        Sample rate in Hz.
    intermediate_frequency : float
        Carrier frequency in Hz.
    samples_per_symbol : int
        Samples per symbol.

    Returns
    -------
    dict
        frequencies, spectrum_db (spectrum analyzer); symbols, sync,
        evm_percent (vector signal analyzer).
    """

    # 1. Spectrum analyzer
    frequencies, spectrum_db = compute_spectrum(samples, sampling_rate)

    # 2. Vector signal analyzer: sync, symbols, EVM
    symbols, sync = demodulate_if(
        samples, rrc_coefficients, sampling_rate,
        intermediate_frequency, samples_per_symbol, reference_symbols,
    )
    evm_percent = compute_evm(symbols, reference_symbols)

    return {
        "frequencies": frequencies,
        "spectrum_db": spectrum_db,
        "symbols": symbols,
        "sync": sync,
        "evm_percent": evm_percent,
    }
