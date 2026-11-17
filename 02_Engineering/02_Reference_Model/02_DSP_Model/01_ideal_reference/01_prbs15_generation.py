"""01-01 PRBS15 generation: bit source using the Xilinx XAPP884 architecture."""


def prbs15_generator(num_bits: int, seed: int = 0x7FFF) -> list[int]:
    """
    Generate PRBS15 bits using a 15-bit Fibonacci LFSR.

    Reference: AMD/Xilinx XAPP884, ITU-T O.150.
    Polynomial: P(x) = x^15 + x^14 + 1.

    The feedback is calculated as stage 15 XOR stage 14.
    The feedback is inverted at the output, while the
    non-inverted feedback is shifted into stage 1.

    The register has a maximum period of 32767 bits.

    Parameters
    ----------
    num_bits : int
        Number of output bits.
    seed : int, optional
        Initial non-zero register state, 1 to 32767
        (default 0x7FFF).

    Returns
    -------
    list of int
        Generated bits (0 or 1).
    """

    # 1. Input validation
    if type(num_bits) is not int or num_bits <= 0:
        raise ValueError("Number of bits must be a positive integer.")

    if type(seed) is not int or not (1 <= seed < (1 << 15)):
        raise ValueError("Seed must be an integer from 1 to 32767.")

    # 2. LFSR initialization
    register = seed
    output_bits = []

    for _ in range(num_bits):

        # 3. Read feedback taps (stages 14 and 15)
        bit_14 = (register >> 13) & 1
        bit_15 = (register >> 14) & 1

        # 4. Calculate feedback using XOR
        feedback = bit_14 ^ bit_15

        # 5. Generate inverted output bit
        output_bits.append(feedback ^ 1)

        # 6. Shift left and insert feedback into stage 1
        register = ((register << 1) & 0x7FFF) | feedback

    return output_bits
