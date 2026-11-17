"""
03 Physical DAC pipeline: FPGA output -> analog DAC output.

Physical order, every step takes datasheet values:

    FPGA samples / codes
      -> 01 quantization            DAC bits, back-off
      -> 02 nonlinearity            SFDR, HD2/HD3               (later)
      -> 03 offset and gain error   % FSR                       (later)
      -> 04 clock jitter            oscillator + DAC RMS jitter
      -> 05 phase noise             oscillator L(f)             (later)
      -> 06 zero-order hold         analog time grid            (later)
      -> 07 DAC output noise        NSD in dBFS/Hz
      = DAC output (goes to 04 analog output)
"""

from importlib import import_module

import numpy as np

_dac = import_module("03_physical_dac")
quantize_dac = _dac.quantize_dac
compute_full_scale = _dac.compute_full_scale
add_clock_jitter = _dac.add_clock_jitter
add_dac_noise = _dac.add_dac_noise


def run_physical_dac_pipeline(
    if_samples: np.ndarray,
    sampling_rate: float,
    dac_bits: int | None = None,
    dac_backoff_db: float = 0.0,
    dac_nsd_dbfs_hz: float | None = None,
    clock_jitter_rms: float | None = None,
    dac_jitter_rms: float | None = None,
    seed: int | None = None,
    full_scale: float | None = None
) -> np.ndarray:
    """
    Turn FPGA output samples into the analog output of a real DAC.

    Every impairment set to None is switched off, so the same function
    gives the ideal output, one effect at a time, or the full hardware.

    Parameters
    ----------
    if_samples : np.ndarray
        IF samples from the FPGA (01 ideal reference or 02 bit-true FPGA).
    sampling_rate : float
        DAC sample rate in Hz.
    dac_bits : int, optional
        DAC resolution N. None means an ideal DAC (no quantization).
    dac_backoff_db : float, optional
        Signal peak below DAC full scale in dB (0 = peak at full scale).
    dac_nsd_dbfs_hz : float, optional
        DAC output noise spectral density in dBFS/Hz. None = no noise.
    clock_jitter_rms : float, optional
        RMS jitter of the clock oscillator in seconds. None = ideal clock.
    dac_jitter_rms : float, optional
        RMS internal jitter of the DAC in seconds. None = not given.
    seed : int, optional
        Seed for the random generators, makes results repeatable.
        Every random effect gets its own seed (seed, seed + 1, ...).
    full_scale : float, optional
        DAC full scale in signal units when it is already fixed by the
        FPGA (bit-true pipeline output). None = compute it from the signal
        peak and dac_backoff_db (ideal floating-point FPGA).

    Returns
    -------
    np.ndarray
        DAC output samples, same length as the input.
    """

    # 0. Separate random streams for the random effects
    def effect_seed(index):
        return None if seed is None else seed + index

    # 1. DAC full scale set by the FPGA output scaling (back-off)
    if full_scale is None:
        full_scale = compute_full_scale(if_samples, dac_backoff_db)

    # 2. Quantization: the DAC can only output 2^N discrete levels
    signal = quantize_dac(if_samples, dac_bits, full_scale)

    # 3. Nonlinearity, offset and gain error - added later (03-02, 03-03)

    # 4. Clock jitter: DAC steps happen at slightly wrong instants
    signal = add_clock_jitter(
        signal, clock_jitter_rms, sampling_rate,
        dac_jitter_rms=dac_jitter_rms, seed=effect_seed(0),
    )

    # 5. Phase noise, zero-order hold - added later (03-05, 03-06)

    # 6. DAC output noise floor
    signal = add_dac_noise(
        signal, dac_nsd_dbfs_hz, full_scale, sampling_rate, seed=effect_seed(1),
    )

    return signal
