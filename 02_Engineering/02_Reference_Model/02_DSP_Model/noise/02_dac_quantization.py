"""Noise 02: DAC quantization (ideal N-bit DAC, two's complement codes)."""

import numbers

import numpy as np


def quantize_dac(
    samples: np.ndarray,
    num_bits: int | None,
    full_scale: float
) -> np.ndarray:
    """
    Quantize a real signal to the output levels of an ideal N-bit DAC.

    The bipolar range [-full_scale, +full_scale) is divided into 2^N
    codes of width delta = 2 * full_scale / 2^N (LSB). Each sample is
    rounded to the nearest code, codes are limited to the two's complement
    range -2^(N-1) ... 2^(N-1) - 1 (saturation), and converted back to
    the analog value code * delta.

    Parameters
    ----------
    samples : np.ndarray
        Real IF samples sent from the FPGA to the DAC.
    num_bits : int or None
        DAC resolution N. None returns a clean copy (ideal DAC).
    full_scale : float
        Signal value that corresponds to the DAC full scale (same units
        as the samples). Choosing it sets the back-off of the signal.

    Returns
    -------
    np.ndarray
        Quantized samples, same length, same units as the input.
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

    The FPGA scales its output so that the signal peak sits backoff_db
    below the DAC full scale (0 dB = peak exactly at full scale):

        full_scale = max(|x|) * 10^(backoff_db / 20)

    More back-off leaves headroom for gain/offset errors and nonlinearity,
    but uses fewer DAC codes, so the quantization SNR drops by the same
    number of dB.

    Parameters
    ----------
    samples : np.ndarray
        Real IF samples sent from the FPGA to the DAC.
    backoff_db : float
        Peak back-off from full scale in dB, >= 0.

    Returns
    -------
    float
        Signal value that corresponds to the DAC full scale.
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
