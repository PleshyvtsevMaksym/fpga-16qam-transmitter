"""01-02 16-QAM Gray mapping: bits to symbols and ideal demapping."""


# Gray code per axis: neighbouring levels differ in exactly one bit
AMPLITUDE_TABLE = {
    (0, 0): -3,
    (0, 1): -1,
    (1, 1):  1,
    (1, 0):  3,
}

REVERSE_AMPLITUDE_TABLE = {
    -3: (0, 0),
    -1: (0, 1),
     1: (1, 1),
     3: (1, 0),
}


def qam16_gray_mapping(bits):
    """
    Convert a binary bitstream into integer I/Q symbols.

    Each group of 4 bits forms one symbol: the first two bits select
    the I level, the next two bits select the Q level.

    Parameters
    ----------
    bits : sequence of int
        Bits (0 or 1); the length must be a multiple of 4.

    Returns
    -------
    i_symbols : list of int
        In-phase levels from {-3, -1, +1, +3}.
    q_symbols : list of int
        Quadrature levels from {-3, -1, +1, +3}.
    """

    # 1. Input validation
    if bits is None:
        raise ValueError("Input bits cannot be None.")

    if len(bits) % 4 != 0:
        raise ValueError("Number of bits must be a multiple of 4.")

    if not all(bit in (0, 1) for bit in bits):
        raise ValueError("Bits must be binary (0 or 1).")

    # 2. Map every 4-bit group to one I level and one Q level
    i_symbols = []
    q_symbols = []

    for index in range(0, len(bits), 4):
        chunk = bits[index:index + 4]

        i_symbols.append(AMPLITUDE_TABLE[tuple(chunk[:2])])
        q_symbols.append(AMPLITUDE_TABLE[tuple(chunk[2:])])

    return i_symbols, q_symbols


def qam16_gray_demapping(i_symbols, q_symbols):
    """
    Recover bits from ideal integer I/Q symbols.

    This is an exact table lookup, not a decision device: noisy or
    scaled samples must first be rounded to the nearest valid level.

    Parameters
    ----------
    i_symbols : sequence of int
        In-phase levels from {-3, -1, +1, +3}.
    q_symbols : sequence of int
        Quadrature levels from {-3, -1, +1, +3}.

    Returns
    -------
    list of int
        Recovered bits, 4 per symbol (2 from I, then 2 from Q).
    """

    # 1. Input validation
    if i_symbols is None or q_symbols is None:
        raise ValueError("I and Q symbols cannot be None.")

    if len(i_symbols) != len(q_symbols):
        raise ValueError("I and Q symbols must have the same length.")

    if not all(
        symbol in REVERSE_AMPLITUDE_TABLE
        for symbols in (i_symbols, q_symbols)
        for symbol in symbols
    ):
        raise ValueError("Invalid I/Q amplitude detected.")

    # 2. Look up the bit pairs of every I/Q level
    recovered_bits = []

    for i, q in zip(i_symbols, q_symbols):
        recovered_bits.extend(REVERSE_AMPLITUDE_TABLE[i])
        recovered_bits.extend(REVERSE_AMPLITUDE_TABLE[q])

    return recovered_bits
