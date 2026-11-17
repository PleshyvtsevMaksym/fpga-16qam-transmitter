"""03-06 Output dynamics: zero-order hold, glitches and settling of the DAC output."""

import numpy as np


def _check_real_1d(samples: np.ndarray) -> None:
    """Raise if samples are not a non-empty real 1D NumPy array."""
    if not isinstance(samples, np.ndarray):
        raise TypeError("Samples must be a NumPy array.")
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Samples must be a non-empty 1D array.")
    if np.iscomplexobj(samples):
        raise TypeError("Samples must be real DAC output samples.")


def apply_zero_order_hold(
    samples: np.ndarray,
    oversampling: int
) -> np.ndarray:
    """
    Hold every DAC sample for one clock period on an M times finer analog grid.

    y[n*M + m] = x[n], m = 0 ... M-1, Fs_a = M * Fs (NRZ staircase).

    Parameters
    ----------
    samples : np.ndarray
        Real DAC samples at Fs.
    oversampling : int
        Analog grid factor M >= 1 (1 = copy, no analog grid).

    Returns
    -------
    np.ndarray
        Staircase at Fs_a, length M * len(samples), same units.
    """

    # 1. Input validation
    _check_real_1d(samples)
    if not isinstance(oversampling, (int, np.integer)) or oversampling < 1:
        raise ValueError("Oversampling must be a positive integer.")

    # 2. Bypass: no analog grid
    if oversampling == 1:
        return samples.copy()

    # 3. Repeat every DAC sample M times
    return np.repeat(samples, int(oversampling))


def add_dac_glitches(
    analog_samples: np.ndarray,
    oversampling: int,
    analog_sampling_rate: float,
    full_scale: float,
    num_bits: int,
    glitch_impulse: float | None,
    glitch_duration: float,
    full_scale_voltage: float
) -> np.ndarray:
    """
    Add a switching glitch at every DAC code transition.

    Area = glitch_impulse * toggled_bits / N (V*s, worst case = major carry),
    converted to signal units; rectangular pulse of L = max(1, round(dur*Fs_a))
    analog samples starting at each transition, positive sign (ASSUMPTION).

    Parameters
    ----------
    analog_samples : np.ndarray
        ZOH staircase on the analog grid.
    oversampling : int
        Analog grid factor M.
    analog_sampling_rate : float
        Fs_a = M * Fs in Hz.
    full_scale : float
        Signal value at DAC full scale.
    num_bits : int
        DAC resolution N.
    glitch_impulse : float or None
        Datasheet glitch impulse in V*s. None = no glitches.
    glitch_duration : float
        Glitch duration in s.
    full_scale_voltage : float
        Output peak voltage at full scale in V.

    Returns
    -------
    np.ndarray
        Analog samples with glitches, same length and units.
    """

    # 1. Input validation
    _check_real_1d(analog_samples)
    if glitch_impulse is not None and (not np.isfinite(glitch_impulse) or glitch_impulse <= 0):
        raise ValueError("Glitch impulse must be positive or None.")

    # 2. Bypass: no glitches
    if glitch_impulse is None:
        return analog_samples.copy()

    if not isinstance(oversampling, (int, np.integer)) or oversampling < 1:
        raise ValueError("Oversampling must be a positive integer.")
    if analog_samples.size % oversampling != 0:
        raise ValueError("Analog length must be a multiple of the oversampling.")
    if not isinstance(num_bits, (int, np.integer)) or num_bits < 1:
        raise ValueError("Number of bits must be a positive integer.")
    for name, value in (("Analog sampling rate", analog_sampling_rate),
                        ("Full scale", full_scale),
                        ("Glitch duration", glitch_duration),
                        ("Full-scale voltage", full_scale_voltage)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")

    # 3. DAC levels (one per clock period) -> offset-binary codes
    dac_levels = analog_samples[::oversampling]
    lsb = 2 * full_scale / 2 ** num_bits
    midscale_code = 2 ** (num_bits - 1)
    codes = np.rint(dac_levels / lsb).astype(np.int64)
    codes = np.clip(codes, -midscale_code, midscale_code - 1) + midscale_code

    # 4. Toggled bits per transition (DAC idles at mid-scale before the first sample)
    previous_codes = np.concatenate(([midscale_code], codes[:-1]))
    changed_bits = np.bitwise_xor(previous_codes, codes)
    toggles = np.zeros(codes.size, dtype=np.int64)
    for bit in range(num_bits):
        toggles += (changed_bits >> bit) & 1

    # 5. Glitch area in signal units * s
    area = glitch_impulse * toggles / num_bits
    area = area / full_scale_voltage * full_scale

    # 6. Rectangular pulse on the analog grid
    glitch_length = max(1, int(round(glitch_duration * analog_sampling_rate)))
    amplitudes = area * analog_sampling_rate / glitch_length

    # 7. Pulses start at every DAC transition (overlapping pulses add)
    impulse_train = np.zeros(analog_samples.size)
    impulse_train[::oversampling] = amplitudes
    glitches = np.convolve(impulse_train, np.ones(glitch_length))[:analog_samples.size]

    return analog_samples + glitches


def apply_settling(
    analog_samples: np.ndarray,
    analog_sampling_rate: float,
    settling_time: float | None,
    settling_accuracy: float
) -> np.ndarray:
    """
    Apply the finite DAC settling as a single-pole response.

    tau = t_s / ln(1/accuracy), alpha = 1 - exp(-1/(tau*Fs_a)),
    y[k] = y[k-1] + alpha*(x[k] - y[k-1]), y[-1] = 0.

    Parameters
    ----------
    analog_samples : np.ndarray
        DAC output on the analog grid.
    analog_sampling_rate : float
        Fs_a in Hz.
    settling_time : float or None
        Datasheet settling time in s. None = instant steps.
    settling_accuracy : float
        Accuracy of the settling spec as a fraction (1e-3 = 0.1 %).

    Returns
    -------
    np.ndarray
        Settled output, same length and units.
    """

    # 1. Input validation
    _check_real_1d(analog_samples)
    if settling_time is not None and (not np.isfinite(settling_time) or settling_time <= 0):
        raise ValueError("Settling time must be positive or None.")

    # 2. Bypass: ideal (instant) settling
    if settling_time is None:
        return analog_samples.copy()

    if not np.isfinite(analog_sampling_rate) or analog_sampling_rate <= 0:
        raise ValueError("Analog sampling rate must be positive and finite.")
    if not 0 < settling_accuracy < 1:
        raise ValueError("Settling accuracy must be between 0 and 1.")

    # 3. Time constant and discrete pole
    tau = settling_time / np.log(1 / settling_accuracy)
    alpha = 1 - np.exp(-1 / (tau * analog_sampling_rate))

    # 4. Impulse response h[k] = alpha * (1 - alpha)^k, cut below 1e-12
    num_taps = int(np.ceil(np.log(1e-12) / np.log(1 - alpha))) + 1 if alpha < 1 else 1
    impulse_response = alpha * (1 - alpha) ** np.arange(num_taps)

    # 5. Same recursion as y[k] = y[k-1] + alpha*(x[k] - y[k-1]), as FIR convolution
    return np.convolve(analog_samples, impulse_response)[:analog_samples.size]
