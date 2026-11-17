"""01-07 Quadrature IF modulation: complex baseband onto a digital IF."""

import numpy as np


def modulate_to_if(
    samples: np.ndarray,
    sampling_rate: float,
    intermediate_frequency: float
) -> np.ndarray:
    """
    Modulate complex baseband samples onto a real intermediate frequency.

    x[n] = I[n] * cos(2*pi*fc*n/Fs) - Q[n] * sin(2*pi*fc*n/Fs),
    with an absolute sample index n, so the carrier phase is continuous
    over the whole signal.

    Parameters
    ----------
    samples : np.ndarray
        Complex baseband samples I + jQ.
    sampling_rate : float
        Sampling frequency Fs in Hz.
    intermediate_frequency : float
        Carrier frequency fc in Hz, 0 < fc < Fs / 2.

    Returns
    -------
    np.ndarray
        Real-valued IF samples, same length as the input.
    """

    # 1. Input validation
    if not isinstance(samples, np.ndarray):
        raise TypeError("Samples must be a NumPy array.")

    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Samples must be a non-empty 1D array.")

    if not np.issubdtype(samples.dtype, np.complexfloating):
        raise TypeError("Samples must contain complex values.")

    if not np.all(np.isfinite(samples)):
        raise ValueError("Samples must contain finite values.")

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive and finite.")

    if not np.isfinite(intermediate_frequency) or not (
        0 < intermediate_frequency < sampling_rate / 2
    ):
        raise ValueError("Intermediate frequency must be between 0 and Nyquist frequency.")

    # 2. Generate sample indices
    n = np.arange(len(samples))

    # 3. Calculate carrier phase
    delta = 2 * np.pi * (intermediate_frequency / sampling_rate) * n

    # 4. Quadrature modulation
    i_data = np.real(samples) * np.cos(delta)
    q_data = np.imag(samples) * np.sin(delta)

    # 5. Combine I/Q components
    if_samples = i_data - q_data

    return if_samples
