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
    """Convert a binary bitstream into integer I/Q symbols."""

    if bits is None:
        raise ValueError("Input bits cannot be None.")

    if len(bits) % 4 != 0:
        raise ValueError("Number of bits must be a multiple of 4.")

    if not all(bit in (0, 1) for bit in bits):
        raise ValueError("Bits must be binary (0 or 1).")

    i_symbols = []
    q_symbols = []

    for index in range(0, len(bits), 4):
        chunk = bits[index:index + 4]

        i_symbols.append(AMPLITUDE_TABLE[tuple(chunk[:2])])
        q_symbols.append(AMPLITUDE_TABLE[tuple(chunk[2:])])

    return i_symbols, q_symbols


def qam16_gray_demapping(i_symbols, q_symbols):
    """Recover bits from ideal integer I/Q symbols."""

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

    recovered_bits = []

    for i, q in zip(i_symbols, q_symbols):
        recovered_bits.extend(REVERSE_AMPLITUDE_TABLE[i])
        recovered_bits.extend(REVERSE_AMPLITUDE_TABLE[q])

    return recovered_bits


def verify_qam16(bits):
    """Verify lossless mapping and demapping."""

    i_symbols, q_symbols = qam16_gray_mapping(bits)
    recovered_bits = qam16_gray_demapping(i_symbols, q_symbols)

    if recovered_bits != bits:
        raise AssertionError("Mapper/Demapper verification failed.")

    return True