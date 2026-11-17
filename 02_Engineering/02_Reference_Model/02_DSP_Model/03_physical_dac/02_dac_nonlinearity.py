"""03-02 DAC nonlinearity: harmonic distortion from datasheet HD2 / HD3."""

import numpy as np


def apply_dac_nonlinearity(
    samples: np.ndarray,
    full_scale: float,
    hd2_dbc: float | None,
    hd3_dbc: float | None,
    test_level_dbfs: float
) -> np.ndarray:
    """
    Apply the static DAC nonlinearity given by datasheet HD2 / HD3.

    Polynomial y = FS * (u + a2 u^2 + a3 u^3), u = x / FS, with a2 and a3
    chosen so that a sine at the test level gives HD2 and HD3.

    Parameters
    ----------
    samples : np.ndarray
        Real DAC samples.
    full_scale : float
        Signal value at DAC full scale.
    hd2_dbc, hd3_dbc : float or None
        2nd / 3rd harmonic distortion in dBc (negative). None = term off.
    test_level_dbfs : float
        Sine level of the datasheet HD spec in dBFS (<= 0).

    Returns
    -------
    np.ndarray
        Distorted samples, same length and units.
    """

    # 1. Input validation
    if not isinstance(samples, np.ndarray):
        raise TypeError("Samples must be a NumPy array.")

    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Samples must be a non-empty 1D array.")

    if np.iscomplexobj(samples):
        raise TypeError("Samples must be real DAC output samples.")

    for hd_dbc in (hd2_dbc, hd3_dbc):
        if hd_dbc is not None and (not np.isfinite(hd_dbc) or hd_dbc >= 0):
            raise ValueError("Harmonic distortion must be a negative finite value in dBc.")

    # 2. Linear DAC bypass
    if hd2_dbc is None and hd3_dbc is None:
        return samples.copy()

    if not np.isfinite(full_scale) or full_scale <= 0:
        raise ValueError("Full scale must be a positive finite number.")

    if not np.isfinite(test_level_dbfs) or test_level_dbfs > 0:
        raise ValueError("Test level must be a finite value <= 0 dBFS.")

    # 3. Normalize to full scale
    normalized_samples = samples / full_scale

    # 4. Amplitude of the datasheet test sine
    test_amplitude = 10 ** (test_level_dbfs / 20)

    # 5. Polynomial coefficients from HD2 and HD3 (None = term off)
    a2 = 0.0 if hd2_dbc is None else 2 * 10 ** (hd2_dbc / 20) / test_amplitude
    a3 = 0.0 if hd3_dbc is None else 4 * 10 ** (hd3_dbc / 20) / test_amplitude ** 2

    # 6. Apply the transfer curve and scale back
    distorted_samples = full_scale * (
        normalized_samples
        + a2 * normalized_samples ** 2
        + a3 * normalized_samples ** 3
    )

    return distorted_samples
