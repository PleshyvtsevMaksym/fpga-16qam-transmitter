"""01-02 tests: Gray-coded 16-QAM mapping and ideal demapping."""

import itertools
import unittest
from importlib import import_module

prbs15_generator = import_module("01_ideal_reference").prbs15_generator
qam16_gray_mapping = import_module("01_ideal_reference").qam16_gray_mapping
qam16_gray_demapping = import_module("01_ideal_reference").qam16_gray_demapping

# All 16 symbols written out by hand: (b0 b1 b2 b3) -> (I, Q)
# Per axis: 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3
EXPECTED_SYMBOLS = {
    (0, 0, 0, 0): (-3, -3), (0, 0, 0, 1): (-3, -1),
    (0, 0, 1, 1): (-3, +1), (0, 0, 1, 0): (-3, +3),
    (0, 1, 0, 0): (-1, -3), (0, 1, 0, 1): (-1, -1),
    (0, 1, 1, 1): (-1, +1), (0, 1, 1, 0): (-1, +3),
    (1, 1, 0, 0): (+1, -3), (1, 1, 0, 1): (+1, -1),
    (1, 1, 1, 1): (+1, +1), (1, 1, 1, 0): (+1, +3),
    (1, 0, 0, 0): (+3, -3), (1, 0, 0, 1): (+3, -1),
    (1, 0, 1, 1): (+3, +1), (1, 0, 1, 0): (+3, +3),
}


def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))


class TestGrayMapping(unittest.TestCase):

    def test_all_16_bit_combinations(self):
        for bits, (i_expected, q_expected) in EXPECTED_SYMBOLS.items():
            i_symbols, q_symbols = qam16_gray_mapping(list(bits))
            self.assertEqual((i_symbols, q_symbols), ([i_expected], [q_expected]))

    def test_levels_and_alphabet(self):
        bits = [b for word in itertools.product((0, 1), repeat=4) for b in word]
        i_symbols, q_symbols = qam16_gray_mapping(bits)
        self.assertEqual(set(i_symbols), {-3, -1, 1, 3})
        self.assertEqual(set(q_symbols), {-3, -1, 1, 3})
        self.assertEqual(len(set(zip(i_symbols, q_symbols))), 16)

    def test_sequence_of_symbols_keeps_order(self):
        # Two symbols: 0010 -> (-3, +3), 1101 -> (+1, -1)
        self.assertEqual(
            qam16_gray_mapping([0, 0, 1, 0, 1, 1, 0, 1]),
            ([-3, 1], [3, -1]),
        )

    def test_gray_adjacency(self):
        # Horizontally or vertically neighbouring points (distance 2)
        # must differ in exactly one bit
        point_to_bits = {point: bits for bits, point in EXPECTED_SYMBOLS.items()}
        pairs = 0
        for (i, q), bits in point_to_bits.items():
            for neighbour in ((i + 2, q), (i, q + 2)):
                if neighbour in point_to_bits:
                    pairs += 1
                    self.assertEqual(hamming_distance(bits, point_to_bits[neighbour]), 1)
        self.assertEqual(pairs, 24)    # 2 directions * 4 rows * 3 neighbours

    def test_mapping_demapping_round_trip(self):
        bits = prbs15_generator(16384)
        i_symbols, q_symbols = qam16_gray_mapping(bits)
        self.assertEqual(qam16_gray_demapping(i_symbols, q_symbols), bits)

    def test_demapping_by_hand(self):
        self.assertEqual(qam16_gray_demapping([3, -1], [1, -3]), [1, 0, 1, 1, 0, 1, 0, 0])

    def test_invalid_mapping_input(self):
        for bits in (None, [0, 1, 1], [0, 1, 2, 0], [0, 1, -1, 0]):
            with self.assertRaises(ValueError):
                qam16_gray_mapping(bits)

    def test_invalid_demapping_input(self):
        for i_symbols, q_symbols in ((None, [1]), ([1, 3], [1]), ([0], [1]), ([1], [2])):
            with self.assertRaises(ValueError):
                qam16_gray_demapping(i_symbols, q_symbols)


class TestGrayMappingExtra(unittest.TestCase):

    def test_numpy_bits_accepted(self):
        import numpy as np
        bits = np.array([1, 0, 0, 1, 0, 0, 1, 1])
        self.assertEqual(qam16_gray_mapping(bits), ([3, -3], [-1, 1]))

    def test_empty_input_gives_empty_output(self):
        self.assertEqual(qam16_gray_mapping([]), ([], []))
        self.assertEqual(qam16_gray_demapping([], []), [])

    def test_output_lengths(self):
        bits = prbs15_generator(400)
        i_symbols, q_symbols = qam16_gray_mapping(bits)
        self.assertEqual(len(i_symbols), 100)
        self.assertEqual(len(q_symbols), 100)

    def test_prbs_over_4_periods_uses_all_16_symbols_equally(self):
        # 4 * (2^15 - 1) bits: every 4-bit word appears 2^11 times, the inverted
        # all-zero state word (1111) once less
        bits = prbs15_generator(4 * (2 ** 15 - 1))
        i_symbols, q_symbols = qam16_gray_mapping(bits)
        counts = {}
        for pair in zip(i_symbols, q_symbols):
            counts[pair] = counts.get(pair, 0) + 1
        self.assertEqual(len(counts), 16)
        self.assertEqual(sorted(set(counts.values())), [2047, 2048])
        self.assertEqual(counts[(1, 1)], 2047)    # bits 11 11

    def test_input_not_modified(self):
        bits = [0, 1, 1, 0]
        qam16_gray_mapping(bits)
        self.assertEqual(bits, [0, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
