"""03-04 Clock jitter: random white jitter of the DAC update instants."""

import numpy as np


def add_clock_jitter(
    samples: np.ndarray,
    clock_jitter_rms: float | None,
    sampling_rate: float,
    dac_jitter_rms: float | None,
    seed: int | None
) -> np.ndarray:
    """
    Add the error of white random jitter of the DAC clock edges.

    Every DAC step x[n] - x[n-1] happens dt_n too late or early,
    dt_n ~ N(0, sigma_t), sigma_t = sqrt(clock^2 + dac^2):
    e[n] = -(x[n] - x[n-1]) * dt_n * Fs.

    Parameters
    ----------
    samples : np.ndarray
        Real DAC samples at the DAC rate.
    clock_jitter_rms : float or None
        Oscillator RMS jitter in s. None = ideal clock.
    sampling_rate : float
        DAC sample rate Fs in Hz.
    dac_jitter_rms : float or None
        DAC internal (aperture) RMS jitter in s. None = not given.
    seed : int or None
        Random seed (repeatable results).

    Returns
    -------
    np.ndarray
        Samples with jitter error, same length and units.
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
