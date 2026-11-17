"""03-04 DAC clock jitter (random white jitter of the update instants)."""

import numpy as np


def add_clock_jitter(
    samples: np.ndarray,
    clock_jitter_rms: float | None,
    sampling_rate: float,
    dac_jitter_rms: float | None = None,
    seed: int | None = None
) -> np.ndarray:
    """
    Add the amplitude error caused by random jitter of the DAC clock.

    Two datasheet values add up as independent random variables:
        - clock oscillator: "RMS phase jitter" (integrated, e.g. 12 kHz-20 MHz)
        - DAC: internal "aperture jitter" / clock input jitter
    sigma_t = sqrt(clock_jitter_rms^2 + dac_jitter_rms^2)

    A DAC holds each value for one clock period (NRZ). The step from
    x[n-1] to x[n] happens at n * Ts + dt_n instead of n * Ts, where dt_n
    is white Gaussian with RMS value sigma_t. A late edge (dt_n > 0) keeps
    the old value for dt_n seconds: an error pulse of height
    -(x[n] - x[n-1]) and width dt_n. Over one sample period it is
    equivalent to the sample error

        e[n] = -(x[n] - x[n-1]) * dt_n / Ts,   x[-1] = 0 (DAC at mid-scale)

    For a sine of frequency f this gives
    SNR_jitter ~ -20 * log10(2 * pi * f * sigma_t) dB.

    Only the white (wideband) part of the jitter is modelled here. Slow,
    correlated jitter is the phase noise of the clock (03-05).

    Parameters
    ----------
    samples : np.ndarray
        Real samples at the DAC output (after quantization).
    clock_jitter_rms : float or None
        RMS jitter of the clock oscillator in seconds (e.g. 1e-12 = 1 ps).
        None means an ideal clock.
    sampling_rate : float
        DAC sample rate in Hz.
    dac_jitter_rms : float or None, optional
        RMS internal jitter of the DAC in seconds. None means not given.
    seed : int, optional
        Seed of the random generator, so the same jitter is produced
        on every run.

    Returns
    -------
    np.ndarray
        Samples with jitter error added, same length, same units
        as the input.
    """

    # 1. Input validation
    if not isinstance(samples, np.ndarray):
        raise TypeError("Samples must be a NumPy array.")

    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Samples must be a non-empty 1D array.")

    if np.iscomplexobj(samples):
        raise TypeError("Samples must be real DAC output samples.")

    for jitter in (clock_jitter_rms, dac_jitter_rms):
        if jitter is not None and (not np.isfinite(jitter) or jitter < 0):
            raise ValueError("Jitter must be a non-negative finite number.")

    # 2. Ideal clock and ideal DAC timing
    if clock_jitter_rms is None and dac_jitter_rms is None:
        return samples.copy()

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive and finite.")

    # 3. Total RMS jitter (independent sources add in power)
    jitter_rms = np.sqrt((clock_jitter_rms or 0.0) ** 2 + (dac_jitter_rms or 0.0) ** 2)

    # 4. Random timing error of every clock edge
    rng = np.random.default_rng(seed)
    time_errors = rng.normal(loc=0.0, scale=jitter_rms, size=samples.size)

    # 5. Size of every DAC step (output starts from mid-scale, 0)
    steps = np.diff(samples, prepend=0.0)

    # 6. Amplitude error of a late or early step
    amplitude_errors = -steps * time_errors * sampling_rate

    return samples + amplitude_errors
