"""
04 Analog output pipeline: DAC output -> signal on the load.

    DAC output
      -> 01 reconstruction filter   type, order, cutoff         (later)
      -> 02 output stage            full-scale current, load    (later)
      -> 03 analog noise            amplifier noise density     (later)
      = transmitter output (what the instruments see)
"""

import numpy as np


def run_analog_output_pipeline(
    dac_output: np.ndarray,
    sampling_rate: float
) -> np.ndarray:
    """
    Pass the DAC output through the analog part of the transmitter.

    The blocks are not modelled yet, so the signal passes unchanged.
    Each block will be added here in the order above, with its datasheet
    parameters from params.

    Parameters
    ----------
    dac_output : np.ndarray
        Real samples at the DAC output (03 physical DAC pipeline).
    sampling_rate : float
        Sample rate of the signal in Hz.

    Returns
    -------
    np.ndarray
        Transmitter output samples, same length as the input.
    """

    # 1. Input validation
    if not isinstance(dac_output, np.ndarray) or dac_output.ndim != 1 or dac_output.size == 0:
        raise ValueError("DAC output must be a non-empty 1D NumPy array.")

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive and finite.")

    # 2. Reconstruction filter, output stage, analog noise - added later
    return dac_output.copy()
