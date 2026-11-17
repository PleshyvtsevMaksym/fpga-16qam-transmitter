"""Stage 03: Symbol energy normalization."""

import numpy as np


def normalize_symbols(symbols: np.ndarray, average_energy: float) -> np.ndarray:
    """
    Normalize QAM symbols using a specified theoretical average symbol energy.

    For 16-QAM with levels {-3, -1, +1, +3} the theoretical average energy
    is 10, so the symbols are divided by sqrt(10) to obtain Es = 1.

    Parameters
    ----------
    symbols : np.ndarray
        Array of complex QAM symbols.
    average_energy : float
        Theoretical average symbol energy before normalization.

    Returns
    -------
    np.ndarray
        Normalized complex QAM symbols.
    """

    # 1. Input validation
    if symbols is None:
        raise ValueError("Input symbols cannot be None.")

    symbols = np.asarray(symbols)

    if symbols.size == 0:
        raise ValueError("Input symbols cannot be empty.")

    if not np.isfinite(average_energy) or average_energy <= 0:
        raise ValueError("Average energy must be positive and finite.")

    # 2. Scale amplitudes by 1 / sqrt(Es)
    return symbols / np.sqrt(average_energy)
