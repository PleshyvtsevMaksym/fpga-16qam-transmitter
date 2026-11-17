"""05-01 FFT amplitude spectrum (virtual spectrum analyzer)."""

import numpy as np


def compute_spectrum(
    samples: np.ndarray,
    sampling_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the normalized two-sided amplitude spectrum in dB.

    The magnitude is |X[k]| / N of a single FFT over the whole signal
    (rectangular window), so a complex tone of amplitude A gives A and a
    real cosine of amplitude A gives A / 2 in each of its two bins.
    This is an amplitude spectrum, not a calibrated PSD.

    Parameters
    ----------
    samples : np.ndarray
        Input real or complex signal samples.
    sampling_rate : float
        Sampling frequency in Hz.

    Returns
    -------
    frequencies : np.ndarray
        Shifted frequency bins in Hz, from -Fs/2 upwards.
    magnitude_db : np.ndarray
        Normalized two-sided amplitude spectrum in dB (dB re 1).
    """

    # 1. Input validation
    if not isinstance(samples, np.ndarray):
        raise TypeError("Samples must be a NumPy array.")

    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Samples must be a non-empty 1D array.")

    if not np.issubdtype(samples.dtype, np.number):
        raise TypeError("Samples must contain numeric values.")

    if not np.all(np.isfinite(samples)):
        raise ValueError("Samples must contain finite values.")

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive and finite.")

    # 2. Compute FFT
    N = len(samples)
    X = np.fft.fft(samples)

    # 3. Generate frequency bins
    frequencies = np.fft.fftfreq(N, d=1 / sampling_rate)

    # 4. Shift frequencies and FFT coefficients
    frequencies = np.fft.fftshift(frequencies)
    X = np.fft.fftshift(X)

    # 5. Calculate normalized magnitude
    magnitude = np.abs(X) / N

    # 6. Convert magnitude to dB
    magnitude_db = 20 * np.log10(
        np.maximum(magnitude, np.finfo(float).tiny)
    )

    return frequencies, magnitude_db
