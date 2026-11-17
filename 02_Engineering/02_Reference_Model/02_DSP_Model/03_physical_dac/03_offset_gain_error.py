"""03-03 DAC offset and gain error from the datasheet, % of full-scale range."""

import numpy as np


def apply_offset_gain_error(
    samples: np.ndarray,
    full_scale: float,
    offset_error_pct_fsr: float | None,
    gain_error_pct_fsr: float | None
) -> np.ndarray:
    """
    Apply the static offset and gain error of the DAC (datasheet, % FSR).

    y = (1 + gain / 100) * x + offset / 100 * FSR,  FSR = 2 * full_scale.

    Parameters
    ----------
    samples : np.ndarray
        Real DAC samples.
    full_scale : float
        Signal value at DAC full scale.
    offset_error_pct_fsr : float or None
        Offset error in % FSR. None = no offset.
    gain_error_pct_fsr : float or None
        Gain error in % FSR. None = nominal gain.

    Returns
    -------
    np.ndarray
        Samples with offset and gain error, same length and units.
    """

    # 1. Input validation
    if not isinstance(samples, np.ndarray):
        raise TypeError("Samples must be a NumPy array.")

    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Samples must be a non-empty 1D array.")

    if np.iscomplexobj(samples):
        raise TypeError("Samples must be real DAC output samples.")

    for error_pct in (offset_error_pct_fsr, gain_error_pct_fsr):
        if error_pct is not None and not np.isfinite(error_pct):
            raise ValueError("Offset and gain error must be finite values in % FSR.")

    # 2. Ideal DAC bypass
    if offset_error_pct_fsr is None and gain_error_pct_fsr is None:
        return samples.copy()

    if not np.isfinite(full_scale) or full_scale <= 0:
        raise ValueError("Full scale must be a positive finite number.")

    # 3. Full-scale range of the bipolar output
    fsr = 2 * full_scale

    # 4. Errors in signal units (None = no error)
    offset_error = 0.0 if offset_error_pct_fsr is None else offset_error_pct_fsr / 100 * fsr
    gain_error = 0.0 if gain_error_pct_fsr is None else gain_error_pct_fsr / 100

    # 5. Scaled and shifted transfer curve
    return (1 + gain_error) * samples + offset_error
