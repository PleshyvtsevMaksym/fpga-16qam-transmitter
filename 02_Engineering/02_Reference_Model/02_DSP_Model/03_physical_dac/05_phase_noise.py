"""03-05 Phase noise: clock oscillator mask L(f) in dBc/Hz."""

import numpy as np


def add_clock_phase_noise(
    samples: np.ndarray,
    sampling_rate: float,
    offset_frequencies: np.ndarray | None,
    phase_noise_dbc_hz: np.ndarray | None,
    seed: int | None
) -> np.ndarray:
    """
    Add the error of the clock phase noise given by the datasheet mask L(f).

    L(f) is interpolated in dB over log10(f), turned into a random clock
    phase phi[n] with PSD 2 * 10^(L/10), and applied as timing error
    dt = phi / (2 pi Fs) to the DAC steps (as in 03-04).

    Parameters
    ----------
    samples : np.ndarray
        Real DAC samples at the DAC rate.
    sampling_rate : float
        DAC clock = sample rate Fs in Hz.
    offset_frequencies : array or None
        Mask offsets in Hz, increasing. None = no phase noise.
    phase_noise_dbc_hz : array or None
        L(f) at these offsets in dBc/Hz.
    seed : int or None
        Random seed (repeatable results).

    Returns
    -------
    np.ndarray
        Samples with phase noise error, same length and units.
    """

    # 1. Input validation
    if not isinstance(samples, np.ndarray):
        raise TypeError("Samples must be a NumPy array.")

    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Samples must be a non-empty 1D array.")

    if np.iscomplexobj(samples):
        raise TypeError("Samples must be real DAC output samples.")

    # 2. Ideal clock bypass
    if offset_frequencies is None or phase_noise_dbc_hz is None:
        return samples.copy()

    offset_frequencies = np.asarray(offset_frequencies, dtype=float)
    phase_noise_dbc_hz = np.asarray(phase_noise_dbc_hz, dtype=float)

    if offset_frequencies.ndim != 1 or offset_frequencies.size == 0:
        raise ValueError("Offset frequencies must be a non-empty 1D array.")

    if phase_noise_dbc_hz.shape != offset_frequencies.shape:
        raise ValueError("Phase noise mask must have one value per offset frequency.")

    if np.any(offset_frequencies <= 0) or np.any(np.diff(offset_frequencies) <= 0):
        raise ValueError("Offset frequencies must be positive and increasing.")

    if not np.all(np.isfinite(phase_noise_dbc_hz)):
        raise ValueError("Phase noise values must be finite (dBc/Hz).")

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive and finite.")

    # 3. White Gaussian noise and its spectrum
    rng = np.random.default_rng(seed)
    white_spectrum = np.fft.rfft(rng.standard_normal(samples.size))

    # 4. L(f) on the positive FFT bins (linear in dB over log10(f))
    frequency_bins = np.fft.rfftfreq(samples.size, d=1 / sampling_rate)
    positive = frequency_bins > 0

    mask_dbc_hz = np.interp(
        np.log10(frequency_bins[positive]),
        np.log10(offset_frequencies),
        phase_noise_dbc_hz,
    )

    # 5. One-sided phase PSD in rad^2/Hz (DC bin stays 0)
    phase_psd = np.zeros_like(frequency_bins)
    phase_psd[positive] = 2 * 10 ** (mask_dbc_hz / 10)

    # 6. Shape the white noise into the clock phase phi[n]
    phase_spectrum = np.sqrt(phase_psd * sampling_rate / 2) * white_spectrum
    phase_noise = np.fft.irfft(phase_spectrum, n=samples.size)

    # 7. Clock phase -> timing error of every edge
    time_errors = phase_noise / (2 * np.pi * sampling_rate)

    # 8. Amplitude error of the late or early DAC steps (as in 03-04)
    steps = np.diff(samples, prepend=0.0)
    amplitude_errors = -steps * time_errors * sampling_rate

    return samples + amplitude_errors
