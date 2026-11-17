"""03-07 DAC noise: white output noise of the DAC from its datasheet NSD."""

import numpy as np


def add_dac_noise(
    samples: np.ndarray,
    nsd_dbfs_hz: float | None,
    full_scale: float,
    sampling_rate: float,
    seed: int | None = None
) -> np.ndarray:
    """
    Add the white output noise of a DAC given by its noise spectral density.

    DAC datasheets give the noise floor as NSD (noise spectral density),
    in dBFS/Hz (relative to a full-scale sine) or in dBm/Hz (absolute power
    into the specified load). The two are linked by the full-scale power:

        NSD_dBFS/Hz = NSD_dBm/Hz - P_FS_dBm

    Reference power: a full-scale sine, P_FS = full_scale^2 / 2.
    The noise is white over the first Nyquist zone 0 ... Fs/2:

        P_noise = P_FS * 10^(NSD / 10) * Fs / 2
        sigma   = sqrt(P_noise)

    Noise outside 0 ... Fs/2 is not represented at this sample rate; it is
    added later on the analog time grid (03-06, 04-01).

    Parameters
    ----------
    samples : np.ndarray
        Real samples at the DAC output.
    nsd_dbfs_hz : float or None
        Noise spectral density in dBFS/Hz (for example -150).
        None returns a clean copy (noiseless DAC).
    full_scale : float
        Signal value at DAC full scale (same units as the samples).
    sampling_rate : float
        DAC sample rate Fs in Hz.
    seed : int, optional
        Seed for the random number generator, makes results repeatable.

    Returns
    -------
    np.ndarray
        Samples with DAC output noise, same length as the input.
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
