"""02-02 NCO: numerically controlled oscillator (DDS) for the IF carrier."""

import numbers

import numpy as np


def generate_nco(
    num_samples: int,
    sampling_rate: float,
    frequency: float,
    phase_bits: int,
    lut_address_bits: int,
    amplitude_bits: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Generate integer cos/sin carrier samples like an FPGA DDS.

    Structure of the Vivado DDS Compiler (phase accumulator + lookup table):

        phase_increment = round(f / Fs * 2^P)
        phase[n]        = (n * phase_increment) mod 2^P       (P-bit register)
        address[n]      = phase[n] >> (P - A)                 (top A bits)
        cos[n]          = round(cos(2*pi*address / 2^A) * (2^(B-1) - 1))
        sin[n]          = round(sin(2*pi*address / 2^A) * (2^(B-1) - 1))

    Effects of the finite widths:
        - P sets the frequency resolution Fs / 2^P; the real frequency is
          phase_increment * Fs / 2^P (returned).
        - Dropping P - A phase bits (phase truncation) creates spurs.
        - B bits of amplitude add quantization noise (~6 dB per bit).
    Our carrier: f / Fs = 1/20 is not a power-of-two fraction, so a P-bit
    DDS is slightly off frequency and has small phase truncation spurs.
    The alternative is a 20-entry table read by a modulo-20 counter
    (exact frequency, no spurs); not modelled yet.

    Parameters
    ----------
    num_samples : int
        Number of output samples.
    sampling_rate : float
        FPGA output sample rate Fs in Hz.
    frequency : float
        Wanted carrier frequency f in Hz, 0 < f < Fs / 2.
    phase_bits : int
        Phase accumulator width P in bits.
    lut_address_bits : int
        Lookup table address width A in bits (table has 2^A entries), A <= P.
    amplitude_bits : int
        Signed output width B of the cos/sin words in bits.

    Returns
    -------
    cos_int : np.ndarray
        Integer cosine samples (int64), length num_samples.
    sin_int : np.ndarray
        Integer sine samples (int64), length num_samples.
    actual_frequency : float
        Carrier frequency the DDS really produces, in Hz.
    """

    # 1. Input validation
    for name, value in (("num_samples", num_samples), ("phase_bits", phase_bits),
                        ("lut_address_bits", lut_address_bits),
                        ("amplitude_bits", amplitude_bits)):
        if not isinstance(value, numbers.Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    if lut_address_bits > phase_bits:
        raise ValueError("LUT address width cannot exceed the phase accumulator width.")

    if amplitude_bits < 2:
        raise ValueError("Amplitude width must be at least 2 bits.")

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive and finite.")

    if not np.isfinite(frequency) or not (0 < frequency < sampling_rate / 2):
        raise ValueError("Frequency must be between 0 and Nyquist frequency.")

    # 2. Phase increment (tuning word) and the real output frequency
    phase_modulus = 1 << phase_bits
    phase_increment = int(round(frequency / sampling_rate * phase_modulus))
    actual_frequency = phase_increment * sampling_rate / phase_modulus

    # 3. Phase accumulator (wraps around like a P-bit register)
    n = np.arange(num_samples, dtype=np.int64)
    phase = (n * phase_increment) % phase_modulus

    # 4. Phase truncation: only the top A bits address the table
    address = phase >> (phase_bits - lut_address_bits)

    # 5. Lookup tables with B-bit amplitude
    max_code = 2 ** (amplitude_bits - 1) - 1
    table_phase = 2 * np.pi * np.arange(1 << lut_address_bits) / (1 << lut_address_bits)
    cos_table = np.round(np.cos(table_phase) * max_code).astype(np.int64)
    sin_table = np.round(np.sin(table_phase) * max_code).astype(np.int64)

    return cos_table[address], sin_table[address], float(actual_frequency)
