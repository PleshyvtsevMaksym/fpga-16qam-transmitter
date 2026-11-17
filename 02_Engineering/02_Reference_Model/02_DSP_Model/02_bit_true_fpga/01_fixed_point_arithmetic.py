"""02-01 Fixed-point arithmetic: bit-true integer operations of the FPGA."""

import numbers

import numpy as np


ROUNDING_MODES = ("round", "truncate")


def signed_range(num_bits: int) -> tuple[int, int]:
    """
    Return the two's complement range of a signed num_bits-wide word.

    min = -2^(N-1),  max = 2^(N-1) - 1

    Parameters
    ----------
    num_bits : int
        Word width N in bits, >= 2.

    Returns
    -------
    tuple of int
        (min_value, max_value).
    """

    if not isinstance(num_bits, numbers.Integral) or num_bits < 2:
        raise ValueError("Word width must be an integer >= 2.")

    return -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1


def required_bits(max_abs: int) -> int:
    """
    Smallest signed word width that holds every value in [-max_abs, max_abs].

    N = ceil(log2(max_abs + 1)) + 1   (one extra bit for the sign)

    Parameters
    ----------
    max_abs : int
        Largest absolute value the word must hold, >= 0.

    Returns
    -------
    int
        Signed word width in bits.
    """

    if max_abs < 0:
        raise ValueError("Maximum absolute value must be non-negative.")

    return int(max_abs).bit_length() + 1


def quantize_coefficients(
    coefficients: np.ndarray,
    num_bits: int
) -> tuple[np.ndarray, float]:
    """
    Convert real filter coefficients to signed integers of num_bits bits.

    The largest coefficient is scaled to the largest positive code, as a
    coefficient file (.coe) for the Vivado FIR Compiler would be:

        scale    = (2^(N-1) - 1) / max(|h|)
        h_int[k] = round(h[k] * scale)

    The quantization error of every coefficient is at most 0.5 LSB.

    Parameters
    ----------
    coefficients : np.ndarray
        Real filter coefficients (for example the RRC from 01-05).
    num_bits : int
        Coefficient word width N in bits.

    Returns
    -------
    coefficients_int : np.ndarray
        Integer coefficients (int64), same length as the input.
    scale : float
        Scale factor: h_int ~ h * scale.
    """

    # 1. Input validation
    coefficients = np.asarray(coefficients)

    if coefficients.ndim != 1 or coefficients.size == 0:
        raise ValueError("Coefficients must be a non-empty 1D array.")

    if np.iscomplexobj(coefficients) or not np.all(np.isfinite(coefficients)):
        raise ValueError("Coefficients must be real and finite.")

    _, max_code = signed_range(num_bits)
    peak = np.max(np.abs(coefficients))

    if peak == 0:
        raise ValueError("Coefficients must not be all zero.")

    # 2. Scale the largest coefficient to the largest code
    scale = max_code / peak

    # 3. Round to integers
    return np.round(coefficients * scale).astype(np.int64), float(scale)


def shift_right(values: np.ndarray, shift: int, rounding: str) -> np.ndarray:
    """
    Divide integers by 2^shift as the FPGA does: drop the shift LSBs.

    truncate:  y = x >> shift                      (floor, free in hardware)
    round:     y = (x + 2^(shift-1)) >> shift      (round half up, one adder)

    Truncation has a mean error of -0.5 LSB (a small DC offset); rounding
    has none.

    Parameters
    ----------
    values : np.ndarray
        Integer values (int64).
    shift : int
        Number of LSBs to drop, >= 0.
    rounding : str
        "round" or "truncate".

    Returns
    -------
    np.ndarray
        Shifted integer values (int64).
    """

    # 1. Input validation
    values = np.asarray(values)

    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("Values must be integers.")

    if not isinstance(shift, numbers.Integral) or shift < 0:
        raise ValueError("Shift must be a non-negative integer.")

    if rounding not in ROUNDING_MODES:
        raise ValueError(f"Rounding must be one of {ROUNDING_MODES}.")

    values = values.astype(np.int64)

    # 2. Nothing to drop
    if shift == 0:
        return values.copy()

    # 3. Arithmetic shift with the chosen rounding
    if rounding == "round":
        values = values + (1 << (shift - 1))

    return values >> shift


def saturate(values: np.ndarray, num_bits: int) -> tuple[np.ndarray, int]:
    """
    Limit integers to a signed num_bits-wide word (saturation logic).

    Values outside -2^(N-1) ... 2^(N-1) - 1 are clipped to the nearest
    limit instead of wrapping around, as two's complement overflow would.

    Parameters
    ----------
    values : np.ndarray
        Integer values.
    num_bits : int
        Output word width N in bits.

    Returns
    -------
    saturated : np.ndarray
        Integer values inside the word range (int64).
    num_clipped : int
        How many values had to be clipped (0 in a correct design).
    """

    min_value, max_value = signed_range(num_bits)
    values = np.asarray(values).astype(np.int64)

    num_clipped = int(np.count_nonzero((values < min_value) | (values > max_value)))

    return np.clip(values, min_value, max_value), num_clipped
