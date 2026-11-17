"""03-07 DAC noise: white output noise of the DAC from its datasheet NSD."""

import numpy as np


def add_dac_noise(
    samples: np.ndarray,
    nsd_dbfs_hz: float | None,
    full_scale: float,
    sampling_rate: float,
    seed: int | None
) -> np.ndarray:
    """
    Add the white output noise of the DAC given by its NSD in dBFS/Hz.

    P_noise = (full_scale^2 / 2) * 10^(NSD / 10) * Fs / 2.

    Parameters
    ----------
    samples : np.ndarray
        Real DAC output samples.
    nsd_dbfs_hz : float or None
        Noise spectral density in dBFS/Hz. None = noiseless DAC.
    full_scale : float
        Signal value at DAC full scale.
    sampling_rate : float
        Sample rate of the samples in Hz (analog grid rate after 03-06).
    seed : int or None
        Random seed (repeatable results).

    Returns
    -------
    np.ndarray
        Samples with DAC noise, same length and units.
    """

    # 1. Input validation
    if not isinstance(samples, np.ndarray):
        raise TypeError("Samples must be a NumPy array.")

    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Samples must be a non-empty 1D array.")

    if np.iscomplexobj(samples):
        raise TypeError("Samples must be real DAC output samples.")

    # 2. Noiseless DAC
    if nsd_dbfs_hz is None:
        return samples.copy()

    if not np.isfinite(full_scale) or full_scale <= 0:
        raise ValueError("Full scale must be a positive finite number.")

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive and finite.")

    # 3. Full-scale sine power (0 dBFS reference)
    p_full_scale = full_scale ** 2 / 2

    # 4. Noise power in the first Nyquist zone
    p_noise = p_full_scale * 10 ** (nsd_dbfs_hz / 10) * sampling_rate / 2

    # 5. Generate white Gaussian noise
    rng = np.random.default_rng(seed)
    noise = np.sqrt(p_noise) * rng.standard_normal(samples.size)

    # 6. Add noise
    return samples + noise
