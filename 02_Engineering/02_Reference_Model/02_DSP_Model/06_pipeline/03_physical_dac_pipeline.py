"""
03 Physical DAC pipeline: FPGA output -> analog DAC output.

Physical order, every step takes datasheet values:

    FPGA samples / codes                                       (Fs)
      -> 01 quantization            DAC bits, back-off
      -> 02 nonlinearity            HD2/HD3 in dBc
      -> 03 offset and gain error   % FSR
      -> 04 clock jitter            oscillator + DAC RMS jitter
      -> 05 phase noise             oscillator L(f) mask
      -> 06 output dynamics         ZOH -> glitches -> settling (Fs_a = M * Fs)
      -> 07 DAC output noise        NSD in dBFS/Hz              (Fs_a)
      = DAC output on the analog grid (goes to 04 analog output)
"""

from importlib import import_module

import numpy as np

_dac = import_module("03_physical_dac")
quantize_dac = _dac.quantize_dac
compute_full_scale = _dac.compute_full_scale
apply_dac_nonlinearity = _dac.apply_dac_nonlinearity
apply_offset_gain_error = _dac.apply_offset_gain_error
add_clock_jitter = _dac.add_clock_jitter
add_clock_phase_noise = _dac.add_clock_phase_noise
apply_zero_order_hold = _dac.apply_zero_order_hold
add_dac_glitches = _dac.add_dac_glitches
apply_settling = _dac.apply_settling
add_dac_noise = _dac.add_dac_noise


def run_physical_dac_pipeline(
    if_samples: np.ndarray,
    sampling_rate: float,
    full_scale: float | None,
    dac_bits: int | None,
    dac_backoff_db: float,
    dac_hd2_dbc: float | None,
    dac_hd3_dbc: float | None,
    dac_hd_test_level_dbfs: float,
    dac_offset_error_pct_fsr: float | None,
    dac_gain_error_pct_fsr: float | None,
    clock_jitter_rms: float | None,
    dac_jitter_rms: float | None,
    clock_phase_noise_offsets_hz: list | None,
    clock_phase_noise_dbc_hz: list | None,
    analog_oversampling: int,
    dac_full_scale_voltage: float,
    dac_glitch_impulse_vs: float | None,
    dac_glitch_duration_s: float,
    dac_settling_time_s: float | None,
    dac_settling_accuracy: float,
    dac_nsd_dbfs_hz: float | None,
    seed: int | None
) -> dict:
    """
    Turn FPGA output samples into the analog output of a real DAC.

    Every impairment set to None is switched off.

    Parameters
    ----------
    if_samples : np.ndarray
        IF samples from the FPGA (01 or 02), at Fs.
    sampling_rate : float
        DAC clock Fs in Hz.
    full_scale : float or None
        DAC full scale in signal units; None = from signal peak and back-off.
    dac_bits, dac_backoff_db : int or None, float
        DAC resolution (None = ideal) and back-off in dB.
    dac_hd2_dbc, dac_hd3_dbc, dac_hd_test_level_dbfs : float
        Harmonic distortion in dBc and its test level in dBFS.
    dac_offset_error_pct_fsr, dac_gain_error_pct_fsr : float or None
        Offset and gain error in % FSR.
    clock_jitter_rms, dac_jitter_rms : float or None
        Oscillator and DAC RMS jitter in s.
    clock_phase_noise_offsets_hz, clock_phase_noise_dbc_hz : list or None
        Oscillator L(f) mask.
    analog_oversampling : int
        Analog grid factor M.
    dac_full_scale_voltage : float
        Output peak voltage at full scale in V.
    dac_glitch_impulse_vs, dac_glitch_duration_s : float
        Glitch impulse in V*s (None = off) and duration in s.
    dac_settling_time_s, dac_settling_accuracy : float
        Settling time in s (None = off) and its accuracy fraction.
    dac_nsd_dbfs_hz : float or None
        DAC noise spectral density in dBFS/Hz.
    seed : int or None
        Base random seed (each random effect gets seed + k).

    Returns
    -------
    dict
        Signal after every step ("quantized", "static", "clocked" at Fs;
        "held", "glitched", "output" at Fs_a), "analog_sampling_rate",
        "full_scale".
    """

    # 0. Separate random streams for the random effects
    def effect_seed(index):
        return None if seed is None else seed + index

    # 1. DAC full scale set by the FPGA output scaling (back-off)
    if full_scale is None:
        full_scale = compute_full_scale(if_samples, dac_backoff_db)

    # 2. Static transfer curve: quantization, nonlinearity, offset/gain
    quantized = quantize_dac(if_samples, dac_bits, full_scale)
    signal = apply_dac_nonlinearity(
        quantized, full_scale, dac_hd2_dbc, dac_hd3_dbc, dac_hd_test_level_dbfs,
    )
    signal = apply_offset_gain_error(
        signal, full_scale, dac_offset_error_pct_fsr, dac_gain_error_pct_fsr,
    )
    static = signal

    # 3. Clock: white jitter and phase noise of the update instants
    signal = add_clock_jitter(
        signal, clock_jitter_rms, sampling_rate, dac_jitter_rms, effect_seed(0),
    )
    signal = add_clock_phase_noise(
        signal, sampling_rate, clock_phase_noise_offsets_hz,
        clock_phase_noise_dbc_hz, effect_seed(2),
    )
    clocked = signal

    # 4. Output dynamics on the analog grid: ZOH -> glitches -> settling
    analog_rate = analog_oversampling * sampling_rate
    held = apply_zero_order_hold(signal, analog_oversampling)

    # Glitches depend on the digital codes (bit toggles), not on the analog
    # level after nonlinearity/offset/gain: count them on the quantized codes
    held_codes = apply_zero_order_hold(quantized, analog_oversampling)
    glitches = add_dac_glitches(
        held_codes, analog_oversampling, analog_rate, full_scale, dac_bits,
        dac_glitch_impulse_vs, dac_glitch_duration_s, dac_full_scale_voltage,
    ) - held_codes
    signal = held + glitches
    glitched = signal
    signal = apply_settling(
        signal, analog_rate, dac_settling_time_s, dac_settling_accuracy,
    )

    # 5. DAC output noise floor over the analog bandwidth
    signal = add_dac_noise(
        signal, dac_nsd_dbfs_hz, full_scale, analog_rate, effect_seed(1),
    )

    return {
        "full_scale": full_scale,
        "analog_sampling_rate": analog_rate,
        "quantized": quantized,     # 01, Fs
        "static": static,           # 01-03 transfer curve, Fs
        "clocked": clocked,         # 04-05 clock, Fs
        "held": held,               # 06 ZOH, Fs_a
        "glitched": glitched,       # 06 glitches, Fs_a
        "output": signal,           # 06 settling + 07 noise, Fs_a
    }
