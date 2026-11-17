"""Analysis 02: IF demodulation and EVM (virtual vector signal analyzer)."""

import numpy as np


def demodulate_if(
    samples: np.ndarray,
    rrc_coefficients: np.ndarray,
    sampling_rate: float,
    intermediate_frequency: float,
    samples_per_symbol: int,
    reference_symbols: np.ndarray
) -> tuple[np.ndarray, dict]:
    """
    Recover symbols from a real IF signal like a vector signal analyzer.

    A laboratory VSA does not know the delay, gain or phase of the device
    under test. It finds them from the known transmitted sequence
    ("known data" mode) and corrects them before measuring EVM:

    1. Downconversion: baseband[n] = 2 * x[n] * exp(-j*2*pi*fc*n/Fs)
       (factor 2 restores the amplitude; the image at -2 fc is removed
       by the matched filter).
    2. Matched filter: time-reversed transmit RRC.
    3. Timing and frame sync: the delay d (in samples) that maximizes
           |c[d]| = | sum_k matched[d + sps*k] * conj(ref[k]) |
       computed for all d at once with an FFT cross-correlation.
       For the ideal transmitter d = L - 1 (two RRC filters).
    4. Gain and phase: complex least-squares factor
           g = sum(conj(rx) * ref) / sum(|rx|^2),   rx_corrected = g * rx
       The device gain is 1/|g| and its phase rotation is -angle(g).

    Timing is found to the nearest sample (Ts / Tsym = 1 % of a symbol);
    a fractional delay and a frequency offset are not corrected.

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
    reference_symbols : np.ndarray
        Known transmitted symbols (Es = 1), one per symbol to recover.

    Returns
    -------
    symbols : np.ndarray
        Recovered complex symbols after delay, gain and phase correction,
        same length as reference_symbols.
    sync : dict
        Measured "delay_samples" (int), "gain" (linear) and
        "phase_deg" (degrees) of the device under test.
    """

    # 1. Input validation
    samples = np.asarray(samples)
    rrc_coefficients = np.asarray(rrc_coefficients)
    reference_symbols = np.asarray(reference_symbols)

    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Samples must be a non-empty 1D array.")

    if np.iscomplexobj(samples):
        raise ValueError("IF samples must be real.")

    if rrc_coefficients.ndim != 1 or rrc_coefficients.size == 0:
        raise ValueError("RRC coefficients must be a non-empty 1D array.")

    if reference_symbols.ndim != 1 or reference_symbols.size == 0:
        raise ValueError("Reference symbols must be a non-empty 1D array.")

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive and finite.")

    if not np.isfinite(intermediate_frequency) or not (
        0 < intermediate_frequency < sampling_rate / 2
    ):
        raise ValueError("Intermediate frequency must be between 0 and Nyquist frequency.")

    if samples_per_symbol < 1:
        raise ValueError("Samples per symbol must be positive.")

    num_symbols = reference_symbols.size
    frame_length = samples_per_symbol * (num_symbols - 1) + 1

    # 2. Downconversion to baseband (x2 restores the amplitude)
    n = np.arange(len(samples))
    baseband = 2 * samples * np.exp(
        -1j * 2 * np.pi * (intermediate_frequency / sampling_rate) * n
    )

    # 3. Matched filter: time-reversed transmit RRC
    matched = np.convolve(baseband, rrc_coefficients[::-1])

    if len(matched) < frame_length:
        raise ValueError("Signal is too short for the number of reference symbols.")

    # 4. Timing and frame sync: cross-correlation with the known symbols
    reference_train = np.zeros(frame_length, dtype=complex)
    reference_train[::samples_per_symbol] = reference_symbols

    n_fft = 1 << int(np.ceil(np.log2(len(matched) + frame_length)))
    correlation = np.fft.ifft(
        np.fft.fft(matched, n_fft) * np.conj(np.fft.fft(reference_train, n_fft))
    )

    max_delay = len(matched) - frame_length
    delay = int(np.argmax(np.abs(correlation[:max_delay + 1])))

    # 5. One sample per symbol at the found delay
    received = matched[delay + samples_per_symbol * np.arange(num_symbols)]

    # 6. Gain and phase correction (complex least squares)
    g = np.sum(np.conj(received) * reference_symbols) / np.sum(np.abs(received) ** 2)

    sync = {
        "delay_samples": delay,
        "gain": float(1 / np.abs(g)),
        "phase_deg": float(-np.degrees(np.angle(g))),
    }

    return g * received, sync


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
