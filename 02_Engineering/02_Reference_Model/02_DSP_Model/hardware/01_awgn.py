"""
Hardware 01: Additive white Gaussian noise (AWGN).

add_awgn       noise at a given SNR per sample (theory illustrations)
add_dac_noise  white output noise of the DAC from its datasheet NSD
"""

import numpy as np


def add_awgn(
    samples: np.ndarray,
    snr_db: float | None,
    seed: int | None = None
) -> np.ndarray:
    """
    Add white Gaussian noise to samples at a given signal-to-noise ratio.

    P_signal = mean(|x|^2)
    P_noise  = P_signal / 10^(SNR_dB / 10)

    For complex samples the noise power is split equally between I and Q
    (sigma = sqrt(P_noise / 2) per component); for real samples
    sigma = sqrt(P_noise).

    Parameters
    ----------
    samples : np.ndarray
        Real or complex 1D samples taken from any DSP stage.
    snr_db : float or None
        Signal-to-noise ratio per sample in dB. None returns a clean copy.
    seed : int, optional
        Seed for the random number generator, makes results repeatable.

    Returns
    -------
    np.ndarray
        Samples with added noise, same length and type as the input.
    """

    # 1. Input validation
    samples = np.asarray(samples)

    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Samples must be a non-empty 1D array.")

    # 2. No noise requested -> clean copy
    if snr_db is None:
        return samples.copy()

    # 3. Signal power
    p_signal = np.mean(np.abs(samples) ** 2)

    # 4. Noise power from SNR
    snr_linear = 10 ** (snr_db / 10)
    p_noise = p_signal / snr_linear

    # 5. Random generator (seed only makes results repeatable)
    rng = np.random.default_rng(seed)

    # 6. Generate noise
    if np.iscomplexobj(samples):
        sigma = np.sqrt(p_noise / 2)
        noise = sigma * (
            rng.standard_normal(samples.size)
            + 1j * rng.standard_normal(samples.size)
        )
    else:
        sigma = np.sqrt(p_noise)
        noise = sigma * rng.standard_normal(samples.size)

    # 7. Add noise
    return samples + noise


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
    added later on the analog time grid (hardware 07, 08).

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
