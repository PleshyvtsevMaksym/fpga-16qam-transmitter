"""Stage 01 tests: PRBS15 generator."""

import unittest

import numpy as np

from dsp import prbs15_generator

PERIOD = 2 ** 15 - 1    # 32767 = 7 * 31 * 151


class TestPrbs15(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.two_periods = prbs15_generator(2 * PERIOD)

    def test_length_and_binary_values(self):
        for num_bits in (1, 15, 16384):
            bits = prbs15_generator(num_bits)
            self.assertEqual(len(bits), num_bits)
            self.assertTrue(all(bit in (0, 1) for bit in bits))

    def test_first_bits_for_seed_0x0001_by_hand(self):
        # The first 15 outputs are the seed bits, LSB first: 1, 0 x 14.
        # Then b[n+15] = b[n] ^ b[n+1]:
        #   b15 = b0 ^ b1 = 1,  b16..b28 = 0,
        #   b29 = b14 ^ b15 = 1,  b30 = b15 ^ b16 = 1
        expected = [1] + [0] * 14 + [1] + [0] * 13 + [1, 1]
        self.assertEqual(prbs15_generator(31), expected)

    def test_first_15_bits_equal_seed_lsb_first(self):
        for seed in (0x0001, 0x1234, 0x7FFF, 0x4000):
            bits = prbs15_generator(15, seed=seed)
            expected = [(seed >> k) & 1 for k in range(15)]
            self.assertEqual(bits, expected)

    def test_deterministic(self):
        self.assertEqual(prbs15_generator(1000), prbs15_generator(1000))

    def test_recurrence(self):
        b = self.two_periods
        for n in range(len(b) - 15):
            self.assertEqual(b[n + 15], b[n] ^ b[n + 1], msg=f"bit {n + 15}")

    def test_full_period(self):
        b = self.two_periods
        self.assertEqual(b[:PERIOD], b[PERIOD:])

        # No shorter period: every shorter period would divide 32767
        for divisor in (7, 31, 151, 7 * 31, 7 * 151, 31 * 151):
            self.assertNotEqual(b[:divisor], b[divisor:2 * divisor])

    def test_all_nonzero_states_visited_once(self):
        # 15 consecutive output bits (LSB first) are the register state
        b = np.array(self.two_periods)
        weights = 1 << np.arange(15)
        states = {
            int(np.dot(b[n:n + 15], weights)) for n in range(PERIOD)
        }
        self.assertEqual(len(states), PERIOD)
        self.assertNotIn(0, states)

    def test_balance_of_ones_and_zeros(self):
        # m-sequence property: 2^14 ones and 2^14 - 1 zeros per period
        self.assertEqual(sum(self.two_periods[:PERIOD]), 2 ** 14)

    def test_invalid_input(self):
        for num_bits in (0, -1, 1.5, "8", None):
            with self.assertRaises(ValueError):
                prbs15_generator(num_bits)

        for seed in (0, 1 << 15, -1, 1.0):
            with self.assertRaises(ValueError):
                prbs15_generator(10, seed=seed)


if __name__ == "__main__":
    unittest.main()
