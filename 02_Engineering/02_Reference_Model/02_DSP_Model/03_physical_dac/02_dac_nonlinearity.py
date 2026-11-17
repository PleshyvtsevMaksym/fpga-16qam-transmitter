"""03-02 DAC nonlinearity from datasheet SFDR / HD2 / HD3, dBc (to be implemented)."""
"""03-02 DAC nonlinearity: harmonic distortion from datasheet HD2 / HD3."""

import numpy as np


def apply_dac_nonlinearity(
    samples: np.ndarray,
    full_scale: float,
    hd2_dbc: float | None,
    hd3_dbc: float | None,
    test_level_dbfs: float = 0.0
) -> np.ndarray:
    """
    Apply the static nonlinearity of a DAC given by its harmonic distortion.

    The DAC transfer curve is modelled as a memoryless polynomial of the
    signal normalized to full scale, u = x / full_scale:

        y = full_scale * (u + a2 * u^2 + a3 * u^3)

    Datasheets give HD2 and HD3 in dBc for a sine of amplitude A
    (test_level_dbfs, A = 10^(level / 20) of full scale). For
    u = A * cos(wt) the harmonics are a2 * A^2 / 2 and a3 * A^3 / 4,
    so relative to the carrier A:

        a2 = 2 * 10^(HD2 / 20) / A
        a3 = 4 * 10^(HD3 / 20) / A^2

    Both coefficients are taken positive (the sign is not given in
    datasheets; assumption). HD2 also creates a DC term a2 * A^2 / 2,
    HD3 also slightly changes the gain at the carrier, as in a real DAC.

    Parameters
    ----------
    samples : np.ndarray
        Real DAC output samples (after quantization).
    full_scale : float
        Signal value at DAC full scale (same units as the samples).
    hd2_dbc : float or None
        Second harmonic distortion in dBc (negative, e.g. -70).
        None switches the second-order term off.
    hd3_dbc : float or None
        Third harmonic distortion in dBc (negative, e.g. -75).
        None switches the third-order term off.
    test_level_dbfs : float, optional
        Sine level at which the datasheet specifies HD2/HD3, in dBFS
        (0 = full scale, typical values 0 or -1).

    Returns
    -------
    np.ndarray
        Distorted samples, same length and units as the input.
        Both terms None returns a clean copy (linear DAC).
    """