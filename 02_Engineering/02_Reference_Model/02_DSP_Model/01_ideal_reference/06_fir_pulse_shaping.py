"""01-06 FIR pulse shaping: full linear convolution with the RRC."""

import numpy as np


def apply_fir(samples: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """
    Apply FIR filtering using full linear convolution and validate the output.

    y[n] = sum_k h[k] * x[n - k]. The "full" mode keeps the filter
    start-up and ending transients, so the output length is
    len(samples) + len(coefficients) - 1.

    Parameters
    ----------
    samples : np.ndarray
        Input real or complex signal samples.
    coefficients : np.ndarray
        FIR filter coefficients.

    Returns
    -------
    np.ndarray
        Validated filtered signal samples.
    """

    # 1. Input validation
    if samples is None or coefficients is None:
        raise ValueError("Samples and coefficients cannot be None.")

    samples = np.asarray(samples)
    coefficients = np.asarray(coefficients)

    if samples.ndim != 1 or coefficients.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")

    if samples.size == 0 or coefficients.size == 0:
        raise ValueError("Inputs cannot be empty.")

    if not (
        np.issubdtype(samples.dtype, np.number)
        and np.issubdtype(coefficients.dtype, np.number)
    ):
        raise TypeError("Inputs must contain numeric values.")

    if not np.all(np.isfinite(samples)):
        raise ValueError("Samples contain NaN or Inf.")

    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Coefficients contain NaN or Inf.")

    # 2. FIR convolution
    convolution = np.convolve(samples, coefficients, mode="full")

    # 3. Output validation
    expected_length = samples.size + coefficients.size - 1

    if convolution.size != expected_length:
        raise RuntimeError("FIR output length mismatch.")

    if not np.all(np.isfinite(convolution)):
        raise ArithmeticError("FIR output contains NaN or Inf.")

    return convolution
