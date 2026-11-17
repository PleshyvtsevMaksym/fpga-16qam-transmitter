"""Analysis 02: IF demodulation and EVM (virtual vector signal analyzer)."""

import numpy as np


def demodulate_if(
    samples: np.ndarray,
    rrc_coefficients: np.ndarray,
    sampling_rate: float,
    intermediate_frequency: float,
    samples_per_symbol: int,
    num_symbols: int
) -> np.ndarray:
    """
    Recover one complex symbol per symbol period from a real IF signal.

    The transmitter steps are reversed:
    baseband[n] = 2 * x[n] * exp(-j*2*pi*fc*n/Fs)   (the factor 2 restores
    the baseband amplitude; the image at -2 fc is removed by the filter),
    then the matched filter (time-reversed transmit RRC) and one sample
    per symbol. Each RRC delays the signal by (L - 1) / 2 samples, so the
    first symbol is at index L - 1 after both filters.

    Parameters
    ----------
    samples : np.ndarray
        Real IF samples (FPGA output or transmitter output).
    rrc_coefficients : np.ndarray
        Transmit RRC coefficients, length L.
    sampling_rate : float
        Sampling frequency Fs in Hz.
    intermediate_frequency : float
        Carrier frequency fc in Hz, 0 < fc < Fs / 2.
    samples_per_symbol : int
        Samples per symbol (Fs / Rs).
    num_symbols : int
        Number of symbols to recover.

    Returns
    -------
    np.ndarray
        Complex received symbols at the Es = 1 scale, one per symbol.
    """

    # 1. Input validation
    if not isinstance(samples, np.ndarray):
        raise TypeError("Samples must be a NumPy array.")

    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Samples must be a non-empty 1D array.")

    if np.iscomplexobj(samples):
        raise TypeError("Samples must be real IF samples.")

    if not np.all(np.isfinite(samples)):
        raise ValueError("Samples must contain finite values.")

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive and finite.")

    if not np.isfinite(intermediate_frequency) or not (
        0 < intermediate_frequency < sampling_rate / 2
    ):
        raise ValueError("Intermediate frequency must be between 0 and Nyquist frequency.")

    if samples_per_symbol < 1 or num_symbols < 1:
        raise ValueError("Samples per symbol and number of symbols must be positive.")

    # 2. Sample indices
    n = np.arange(len(samples))

    # 3. Downconversion to baseband (x2 restores the amplitude)
    baseband = 2 * samples * np.exp(
        -1j * 2 * np.pi * (intermediate_frequency / sampling_rate) * n
    )

    # 4. Matched filter: time-reversed transmit RRC
    matched = np.convolve(baseband, rrc_coefficients[::-1])

    # 5. Symbol timing: two RRC filters delay by L - 1 samples
    total_delay = len(rrc_coefficients) - 1
    positions = total_delay + samples_per_symbol * np.arange(num_symbols)

    if positions[-1] >= len(matched):
        raise ValueError("Signal is too short for the requested number of symbols.")

    # 6. One sample per symbol
    return matched[positions]


def compute_evm(received: np.ndarray, reference: np.ndarray) -> float:
    """
    Compute the RMS error vector magnitude in percent.

    EVM = sqrt( mean(|received - reference|^2) / mean(|reference|^2) ) * 100 %

    Parameters
    ----------
    received : np.ndarray
        Received complex symbols (for example from demodulate_if).
    reference : np.ndarray
        Ideal transmitted symbols at the same scale, same length.

    Returns
    -------
    float
        RMS EVM in percent.
    """

    # 1. Input validation
    received = np.asarray(received)
    reference = np.asarray(reference)

    if received.size == 0 or received.shape != reference.shape:
        raise ValueError("Received and reference must be non-empty and have the same shape.")

    # 2. Error vector
    error = received - reference

    # 3. RMS EVM relative to the reference power
    return 100 * np.sqrt(np.mean(np.abs(error) ** 2) / np.mean(np.abs(reference) ** 2))
