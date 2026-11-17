"""04-03 Analog noise: additive white Gaussian noise (AWGN) at a given SNR.

Used now for theory illustrations; later the output-stage noise
(amplifier noise density, nV/sqrt(Hz), into the load) is added here.
"""

import numpy as np


def add_awgn(
    samples: np.ndarray,
    snr_db: float | None,
    seed: int | None
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
    seed : int or None
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
