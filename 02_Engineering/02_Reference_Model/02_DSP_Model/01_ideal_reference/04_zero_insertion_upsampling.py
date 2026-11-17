"""01-04 Zero-insertion upsampling: one symbol followed by sps - 1 zeros."""

import numpy as np


def upsample_symbols(symbols: np.ndarray, samples_per_symbol: int) -> np.ndarray:
    """
    Upsample QAM symbols by inserting zeros between symbols.

    Symbol k is placed at sample index k * samples_per_symbol and is
    followed by (samples_per_symbol - 1) zeros.

    Parameters
    ----------
    symbols : np.ndarray
        Array of complex QAM symbols.
    samples_per_symbol : int
        Number of samples per symbol.

    Returns
    -------
    np.ndarray
        Upsampled complex QAM symbols, length len(symbols) * samples_per_symbol.
    """

    # 1. Input validation
    if symbols is None:
        raise ValueError("Input symbols cannot be None.")

    symbols = np.asarray(symbols)

    if symbols.size == 0:
        raise ValueError("Input symbols cannot be empty.")

    if type(samples_per_symbol) is not int or samples_per_symbol <= 0:
        raise ValueError("Samples per symbol must be a positive integer.")

    # 2. Create the zero-filled output array
    upsampled_length = len(symbols) * samples_per_symbol
    upsampled_symbols = np.zeros(upsampled_length, dtype=complex)

    # 3. Place the symbols at every samples_per_symbol-th position
    upsampled_symbols[::samples_per_symbol] = symbols

    return upsampled_symbols
