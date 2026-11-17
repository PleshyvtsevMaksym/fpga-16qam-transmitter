
"""PRBS15 generator and verification using a 15-bit Fibonacci LFSR."""


def prbs15_generator(num_bits, seed=0x0001):
    """Generate PRBS15 bits using a 15-bit right-shifting LFSR."""

    if type(num_bits) is not int or num_bits <= 0:
        raise ValueError("Number of bits must be a positive integer.")

    if type(seed) is not int or not (1 <= seed < (1 << 15)):
        raise ValueError("Seed must be an integer from 1 to 32767.")

    register = seed
    output_bits = []

    for _ in range(num_bits):

        # Read output bit before shifting
        bit_0 = register & 1
        bit_1 = (register >> 1) & 1

        # Store output
        output_bits.append(bit_0)

        # Calculate feedback
        feedback = bit_0 ^ bit_1

        # Shift right and insert feedback at bit 14
        register = (register >> 1) | (feedback << 14)

    return output_bits


def verify_prbs15(seed=0x0001):
    """Verify PRBS15 sequence, determinism and full LFSR period."""

    if type(seed) is not int or not (1 <= seed < (1 << 15)):
        raise ValueError("Seed must be an integer from 1 to 32767.")

    period = (1 << 15) - 1

    # Generate two periods
    sequence = prbs15_generator(2 * period, seed)

    # 1. Verify output length
    assert len(sequence) == 2 * period, "Incorrect sequence length."

    # 2. Verify binary values
    assert all(bit in (0, 1) for bit in sequence), \
        "Sequence contains non-binary values."

    # 3. Verify deterministic generation
    reference = prbs15_generator(period, seed)

    assert sequence[:period] == reference, \
        "Generator is not deterministic."

    # 4. Verify repetition after exactly one period
    assert sequence[:period] == sequence[period:], \
        "Sequence does not repeat after 32767 bits."

    # 5. Independently verify the bit recurrence:
    #    output[n+15] = output[n] XOR output[n+1]
    for n in range(period):
        expected_bit = sequence[n] ^ sequence[n + 1]
        assert sequence[n + 15] == expected_bit, \
            f"PRBS recurrence mismatch at bit {n + 15}."

    # 6. Reconstruct all 15-bit LFSR states from output bits.
    #    The first 15 output bits represent the initial state,
    #    ordered from LSB to MSB.
    visited_states = set()

    for n in range(period):
        state = 0

        for bit_index in range(15):
            state |= sequence[n + bit_index] << bit_index

        assert state != 0, "Zero state encountered."
        assert state not in visited_states, \
            f"Repeated state before full period at step {n}."

        visited_states.add(state)

    assert len(visited_states) == period, \
        "Incorrect number of unique states."

    # 7. Verify return to initial state
    final_state = 0

    for bit_index in range(15):
        final_state |= sequence[period + bit_index] << bit_index

    assert final_state == seed, \
        "LFSR did not return to the initial seed."

    print("PRBS15 Verification")
    print("-----------------------------")
    print(f"Seed:             0x{seed:04X}")
    print(f"Expected period:  {period}")
    print(f"Unique states:    {len(visited_states)}")
    print(f"Zero state:       Not encountered")
    print(f"Sequence repeat:  PASS")
    print(f"Bit recurrence:   PASS")
    print(f"Full period:      PASS")
    print(f"Verification:     PASS")

    return True


if __name__ == "__main__":

    NUM_SYMBOLS = 4096
    NUM_BITS = NUM_SYMBOLS * 4

    bits = prbs15_generator(NUM_BITS)

    print("First 32 bits:", bits[:32])
    print("Generated bits:", len(bits))
    print()

    verify_prbs15()
