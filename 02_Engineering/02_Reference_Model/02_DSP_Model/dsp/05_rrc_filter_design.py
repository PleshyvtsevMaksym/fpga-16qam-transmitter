"""Stage 05: Root Raised Cosine (RRC) FIR filter design."""

import numpy as np


def design_rrc(rolloff: float, span: int, sps: int) -> np.ndarray:
    """
    Design a Root Raised Cosine FIR filter.

    The impulse response is evaluated on the normalized time axis
    t = (n - center) / sps (in symbol periods). The points t = 0 and
    t = +/-1/(4 * rolloff), where the general equation is 0/0, use
    their analytical limits.

    Parameters
    ----------
    rolloff : float
        RRC roll-off factor (0 < rolloff <= 1).
    span : int
        Filter duration in symbols.
    sps : int
        Samples per symbol.

    Returns
    -------
    np.ndarray
        Unit-energy RRC FIR coefficients, length span * sps + 1.
    """

    # 1. Input validation
    if not np.isfinite(rolloff) or not (0 < rolloff <= 1):
        raise ValueError("Rolloff must be in the range (0, 1].")

    if type(span) is not int or span <= 0:
        raise ValueError("Span must be a positive integer.")

    if type(sps) is not int or sps <= 0:
        raise ValueError("SPS must be a positive integer.")

    if (span * sps) % 2 != 0:
        raise ValueError("Span * SPS must be even.")

    # 2. Filter parameters
    num_taps = span * sps + 1
    center = (num_taps - 1) // 2

    # 3. Normalized time axis
    x = (np.arange(num_taps) - center) / sps

    # 4. RRC impulse response
    raw = []

    for t in x:

        # Special case: t = 0
        if np.isclose(t, 0.0, atol=1e-12, rtol=0):
            value = 1 + rolloff * (4 / np.pi - 1)

        # Special case: t = +/-1/(4*rolloff)
        elif np.isclose(abs(t), 1 / (4 * rolloff), atol=1e-12, rtol=0):
            value = (rolloff / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))
            )

        # General RRC equation
        else:
            numerator = (
                np.sin(np.pi * t * (1 - rolloff))
                + 4 * rolloff * t
                * np.cos(np.pi * t * (1 + rolloff))
            )
            denominator = (
                np.pi * t * (1 - (4 * rolloff * t) ** 2)
            )
            value = numerator / denominator

        raw.append(value)

    h_raw = np.array(raw, dtype=float)

    # 5. Unit-energy normalization
    energy = np.sum(np.abs(h_raw) ** 2)
    h = h_raw / np.sqrt(energy)

    # 6. Normalization check
    if not np.isclose(np.sum(np.abs(h) ** 2), 1.0):
        raise ValueError("RRC coefficients are not normalized.")

    return h
