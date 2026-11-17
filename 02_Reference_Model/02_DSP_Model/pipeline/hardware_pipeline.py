"""
Hardware pipeline: what happens to the signal after the FPGA.

The FPGA pipeline (run_fpga_pipeline) ends with the IF samples that the
FPGA sends to the DAC. This module adds what happens to that signal in
real hardware, in the physical order:

    IF samples (FPGA)
      -> DAC impairments (noise 02-06)      added later, one by one
      -> analog reconstruction filter       added later
      -> thermal noise at the output (AWGN, noise 01)
      = transmitter output signal
"""

import numpy as np

from noise import add_awgn


def run_hardware_pipeline(
    if_samples: np.ndarray,
    snr_db: float | None,
    seed: int | None = None
) -> np.ndarray:
    """
    Turn ideal FPGA IF samples into a realistic transmitter output.

    Parameters
    ----------
    if_samples : np.ndarray
        Real IF samples from the ideal pipeline (stage 07 output).
    snr_db : float or None
        SNR per sample of the thermal noise at the output, in dB.
        None means no thermal noise.
    seed : int, optional
        Seed for the noise generator, makes results repeatable.

    Returns
    -------
    np.ndarray
        Transmitter output samples, same length as the input.
    """

    # 1. DAC impairments (02-06) - added later, one by one
    signal = if_samples.copy()

    # 2. Analog reconstruction filter - added later

    # 3. Thermal noise at the transmitter output
    signal = add_awgn(signal, snr_db, seed=seed)

    return signal
