"""03-01 DAC quantization: ideal N-bit DAC with two's complement codes."""

import numbers

import numpy as np


def quantize_dac(
    samples: np.ndarray,
    num_bits: int | None,
    full_scale: float
) -> np.ndarray:
    """
    Quantize samples to the 2^N output levels of an ideal N-bit DAC.

    Two's complement codes in -2^(N-1) ... 2^(N-1) - 1 (saturated),
    LSB = 2 * full_scale / 2^N.

    Parameters
    ----------
    samples : np.ndarray
        Real IF samples from the FPGA.
    num_bits : int or None
        DAC resolution N. None = ideal DAC (clean copy).
    full_scale : float
        Signal value at DAC full scale.

    Returns
    -------
    np.ndarray
        Quantized samples, same length and units.
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

    if num_bits is not None:
        if not isinstance(num_bits, numbers.Integral) or num_bits <= 0:
            raise ValueError("DAC resolution must be a positive integer.")

        if not np.isfinite(full_scale) or full_scale <= 0:
            raise ValueError("Full scale must be a positive finite number.")

    # 2. Ideal DAC bypass
    if num_bits is None:
        return samples.copy()

    # 3. Calculate LSB
    delta = 2 * full_scale / (2 ** num_bits)

    # 4. Convert samples to digital codes
    digital_codes = np.round(samples / delta)

    # 5. Saturate digital codes (two's complement range)
    min_code = -(2 ** (num_bits - 1))
    max_code = 2 ** (num_bits - 1) - 1
    digital_codes = np.clip(digital_codes, min_code, max_code)

    # 6. Convert codes back to analog levels
    quantized_samples = digital_codes * delta

    return quantized_samples


def compute_full_scale(samples: np.ndarray, backoff_db: float) -> float:
    """
    Compute the DAC full scale from the signal peak and the back-off.

    Parameters
    ----------
    samples : np.ndarray
        Real IF samples from the FPGA.
    backoff_db : float
        Peak below full scale in dB, >= 0.

    Returns
    -------
    float
        Signal value at DAC full scale: max(|x|) * 10^(backoff / 20).
    """

    # 1. Input validation
    if not isinstance(samples, np.ndarray) or samples.ndim != 1 or samples.size == 0:
        raise ValueError("Samples must be a non-empty 1D NumPy array.")

    if not np.isfinite(backoff_db) or backoff_db < 0:
        raise ValueError("Back-off must be a non-negative finite number in dB.")

    # 2. Signal peak
    peak = np.max(np.abs(samples))

    if peak == 0:
        raise ValueError("Samples must not be all zero.")

    # 3. Full scale above the peak by the back-off
    return float(peak * 10 ** (backoff_db / 20))
