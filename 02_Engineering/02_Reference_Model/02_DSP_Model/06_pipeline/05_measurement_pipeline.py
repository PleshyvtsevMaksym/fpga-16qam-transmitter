"""
05 Measurement pipeline: what the laboratory instruments report.

    transmitter output
      -> 01 spectrum analyzer         PSD in dBFS/Hz (RBW)
      -> 02 vector signal analyzer    sync, symbols, EVM
      -> 03 RF metrics, 04 oscilloscope model           (later)
"""

from importlib import import_module

import numpy as np

_instruments = import_module("05_measurement_instruments")
design_rrc = import_module("01_ideal_reference").design_rrc
compute_psd = _instruments.compute_psd
demodulate_if = _instruments.demodulate_if
compute_evm = _instruments.compute_evm


def run_measurement_pipeline(
    samples: np.ndarray,
    reference_symbols: np.ndarray,
    sampling_rate: float,
    intermediate_frequency: float,
    symbol_rate: float,
    rrc_rolloff: float,
    rrc_span: int,
    rbw: float,
    reference_level: float
) -> dict:
    """
    Measure a transmitter output like the laboratory instruments do.

    Parameters
    ----------
    samples : np.ndarray
        Real transmitter output samples.
    reference_symbols : np.ndarray
        Known transmitted symbols at Es = 1 ("known data" mode of the VSA).
    sampling_rate : float
        Sample rate of the samples in Hz (Fs or Fs_a), integer multiple of Rs.
    intermediate_frequency : float
        Carrier frequency in Hz.
    symbol_rate : float
        Symbol rate Rs in Hz.
    rrc_rolloff, rrc_span : float, int
        VSA measurement filter (RRC) settings.
    rbw : float
        Spectrum analyzer resolution bandwidth in Hz.
    reference_level : float
        Peak of a 0 dBFS sine (DAC full scale), signal units.

    Returns
    -------
    dict
        frequencies, psd_dbfs_hz (spectrum analyzer); symbols, sync,
        evm_percent (vector signal analyzer).
    """

    # 1. Spectrum analyzer: PSD in dBFS/Hz
    frequencies, psd_dbfs_hz = compute_psd(samples, sampling_rate, rbw, reference_level)

    # 2. VSA measurement filter (RRC) at the sample rate of the input
    samples_per_symbol = int(round(sampling_rate / symbol_rate))
    if not np.isclose(samples_per_symbol * symbol_rate, sampling_rate):
        raise ValueError("Sampling rate must be an integer multiple of the symbol rate.")
    rrc_coefficients = design_rrc(rrc_rolloff, rrc_span, samples_per_symbol)

    # 3. Vector signal analyzer: sync, symbols, EVM
    symbols, sync = demodulate_if(
        samples, rrc_coefficients, sampling_rate,
        intermediate_frequency, samples_per_symbol, reference_symbols,
    )
    evm_percent = compute_evm(symbols, reference_symbols)

    return {
        "frequencies": frequencies,
        "psd_dbfs_hz": psd_dbfs_hz,
        "symbols": symbols,
        "sync": sync,
        "evm_percent": evm_percent,
    }
