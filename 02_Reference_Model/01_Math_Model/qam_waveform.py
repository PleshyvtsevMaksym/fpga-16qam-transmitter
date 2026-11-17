
"""Time-domain waveform generation for a basic 16-QAM transmitter."""

import math


def make_time_axis(num_symbols, symbol_rate, sample_rate):
    """Generate a discrete time axis for 16-QAM symbols."""

    if type(num_symbols) is not int or num_symbols <= 0:
        raise ValueError("Number of symbols must be a positive integer.")

    if type(symbol_rate) is not int or symbol_rate <= 0:
        raise ValueError("Symbol rate must be a positive integer.")

    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer.")

    if sample_rate % symbol_rate != 0:
        raise ValueError(
            "Sample rate must be an integer multiple of symbol rate."
        )

    samples_per_symbol = sample_rate // symbol_rate
    total_samples = num_symbols * samples_per_symbol

    time_axis = [n / sample_rate for n in range(total_samples)]

    return time_axis, samples_per_symbol


def hold_iq_symbols(i_symbols, q_symbols, samples_per_symbol):
    """Expand I/Q symbols using rectangular symbol holding."""

    if i_symbols is None or q_symbols is None:
        raise ValueError("I and Q symbols cannot be None.")

    if not i_symbols or len(i_symbols) != len(q_symbols):
        raise ValueError("I and Q must be non-empty and equal in length.")

    if type(samples_per_symbol) is not int or samples_per_symbol <= 0:
        raise ValueError("Samples per symbol must be a positive integer.")

    held_i = []
    held_q = []

    for i, q in zip(i_symbols, q_symbols):
        held_i.extend([i] * samples_per_symbol)
        held_q.extend([q] * samples_per_symbol)

    return held_i, held_q


def generate_carriers(time_axis, carrier_frequency, sample_rate):
    """Generate orthogonal cosine and sine carrier samples."""

    if time_axis is None or len(time_axis) == 0:
        raise ValueError("Time axis cannot be empty.")

    if (
        not isinstance(carrier_frequency, (int, float))
        or not math.isfinite(carrier_frequency)
        or carrier_frequency <= 0
    ):
        raise ValueError("Carrier frequency must be finite and positive.")

    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer.")

    if carrier_frequency >= sample_rate / 2:
        raise ValueError("Carrier frequency must be below Nyquist.")

    cosine_samples = [
        math.cos(2 * math.pi * carrier_frequency * t)
        for t in time_axis
    ]

    sine_samples = [
        math.sin(2 * math.pi * carrier_frequency * t)
        for t in time_axis
    ]

    return cosine_samples, sine_samples


def qam16_waveform(i_samples, q_samples, cosine_carrier, sine_carrier):
    """Generate a real-valued 16-QAM IF waveform."""

    arrays = (i_samples, q_samples, cosine_carrier, sine_carrier)

    if any(array is None for array in arrays):
        raise ValueError("Waveform inputs cannot be None.")

    if len(set(len(array) for array in arrays)) != 1 or not i_samples:
        raise ValueError("Waveform inputs must be non-empty and equal in length.")

    waveform = [
        i * c - q * s
        for i, q, c, s in zip(
            i_samples, q_samples, cosine_carrier, sine_carrier
        )
    ]

    return waveform
