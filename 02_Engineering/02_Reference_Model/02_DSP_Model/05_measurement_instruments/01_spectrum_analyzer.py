"""05-01 Spectrum analyzer: FFT amplitude spectrum."""

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


def compute_psd(
    samples: np.ndarray,
    sampling_rate: float,
    rbw: float,
    reference_level: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Measure the one-sided power spectral density like a spectrum analyzer.

    Welch average of Hann-windowed segments (50 % overlap), segment length
    set by the resolution bandwidth RBW = 1.5 * Fs / N_seg; result in
    dBFS/Hz, 0 dBFS = sine with peak reference_level.

    Parameters
    ----------
    samples : np.ndarray
        Real signal samples.
    sampling_rate : float
        Sampling frequency in Hz.
    rbw : float
        Resolution bandwidth in Hz.
    reference_level : float
        Peak value of a 0 dBFS sine (DAC full scale), signal units.

    Returns
    -------
    frequencies : np.ndarray
        Frequencies 0 ... Fs/2 in Hz.
    psd_dbfs_hz : np.ndarray
        Power spectral density in dBFS/Hz.
    """

    # 1. Input validation
    if not isinstance(samples, np.ndarray) or samples.ndim != 1 or samples.size == 0:
        raise ValueError("Samples must be a non-empty 1D NumPy array.")
    if np.iscomplexobj(samples):
        raise TypeError("Samples must be real.")
    for name, value in (("Sampling rate", sampling_rate), ("RBW", rbw),
                        ("Reference level", reference_level)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")

    # 2. Segment length from the RBW (Hann ENBW = 1.5 bins), power of two
    segment = 1 << int(np.ceil(np.log2(1.5 * sampling_rate / rbw)))
    segment = min(segment, 1 << int(np.floor(np.log2(samples.size))))
    window = np.hanning(segment)
    step = segment // 2
    starts = np.arange(0, samples.size - segment + 1, step)

    # 3. Averaged periodogram (Welch), one-sided
    power = np.zeros(segment // 2 + 1)
    for start in starts:
        spectrum = np.fft.rfft(samples[start:start + segment] * window)
        power += np.abs(spectrum) ** 2
    psd = power / starts.size / (sampling_rate * np.sum(window ** 2))
    psd[1:-1] *= 2

    # 4. dBFS/Hz relative to a full-scale sine (power reference_level^2 / 2)
    frequencies = np.fft.rfftfreq(segment, d=1 / sampling_rate)
    psd_dbfs_hz = 10 * np.log10(np.maximum(psd / (reference_level ** 2 / 2), 1e-30))

    return frequencies, psd_dbfs_hz
