"""
Hardware pipeline: what happens to the signal after the FPGA.

The FPGA pipeline (run_fpga_pipeline) ends with the IF samples that the
FPGA sends to the DAC. This module adds what happens to that signal in
real hardware, in the physical order. Every step takes datasheet values:

    IF samples (FPGA)
      -> 02 quantization            DAC bits, back-off
      -> 05 nonlinearity            SFDR, HD2/HD3               (later)
      -> 06 offset and gain error   % FSR                       (later)
      -> 03 clock jitter            oscillator + DAC RMS jitter
      -> 04 phase noise             oscillator L(f)             (later)
      -> 07 zero-order hold         analog time grid            (later)
      -> 01 DAC output noise        NSD in dBFS/Hz
      -> 08 reconstruction filter   type, order, cutoff         (later)
      -> 09 output stage            full-scale current, load    (later)
      = transmitter output signal
"""

import numpy as np

from hardware import add_clock_jitter, add_dac_noise, compute_full_scale, quantize_dac


def run_hardware_pipeline(
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
    Turn ideal FPGA IF samples into a realistic transmitter output.

    Every impairment set to None is switched off, so the same function
    gives the ideal output, one effect at a time, or the full hardware.

    Parameters
    ----------
    if_samples : np.ndarray
        Real IF samples from the ideal pipeline (stage 07 output).
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
        Transmitter output samples, same length as the input.
    """

    # 0. Separate random streams for the random effects
    def effect_seed(index):
        return None if seed is None else seed + index

    # 1. DAC full scale set by the FPGA output scaling (back-off)
    if full_scale is None:
        full_scale = compute_full_scale(if_samples, dac_backoff_db)

    # 2. Quantization: the DAC can only output 2^N discrete levels
    signal = quantize_dac(if_samples, dac_bits, full_scale)

    # 3. Nonlinearity, offset and gain error - added later (hardware 05, 06)

    # 4. Clock jitter: DAC steps happen at slightly wrong instants
    signal = add_clock_jitter(
        signal, clock_jitter_rms, sampling_rate,
        dac_jitter_rms=dac_jitter_rms, seed=effect_seed(0),
    )

    # 5. Phase noise, zero-order hold - added later (hardware 04, 07)

    # 6. DAC output noise floor
    signal = add_dac_noise(
        signal, dac_nsd_dbfs_hz, full_scale, sampling_rate, seed=effect_seed(1),
    )

    # 7. Reconstruction filter, output stage - added later (hardware 08, 09)

    return signal
