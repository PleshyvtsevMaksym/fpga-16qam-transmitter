"""01-01 tests: PRBS15 generator (ITU-T O.150, x^15 + x^14 + 1, inverted output)."""

import unittest
from importlib import import_module

import numpy as np

prbs15_generator = import_module("01_ideal_reference").prbs15_generator

PERIOD = 2 ** 15 - 1    # 32767 = 7 * 31 * 151


def reference_prbs15(num_bits, seed):
    """Independent model: f[n] = f[n-14] ^ f[n-15], stage k holds f[-k], output = not f."""
    f = {-k: (seed >> (k - 1)) & 1 for k in range(1, 16)}
    out = []
    for n in range(num_bits):
        f[n] = f[n - 14] ^ f[n - 15]
        out.append(f[n] ^ 1)
    return out


class TestPrbs15(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.two_periods = np.array(prbs15_generator(2 * PERIOD))

    def test_output_type_length_and_values(self):
        for num_bits in (1, 15, 16, 16384):
            bits = prbs15_generator(num_bits)
            self.assertIsInstance(bits, list)
            self.assertEqual(len(bits), num_bits)
            self.assertTrue(set(bits) <= {0, 1})
            self.assertTrue(all(type(b) is int for b in bits))

    def test_first_bits_for_seed_0x0001_by_hand(self):
        # f[-1] = 1, other stages 0: f[0..12] = 0, f[13] = f[-1] ^ f[-2] = 1,
        # f[14] = f[0] ^ f[-1] = 1; output is inverted
        self.assertEqual(prbs15_generator(15, seed=0x0001), [1] * 13 + [0, 0])

    def test_matches_independent_model_for_several_seeds(self):
        for seed in (0x0001, 0x1234, 0x4000, 0x7FFF, 0x5555):
            self.assertEqual(prbs15_generator(500, seed=seed),
                             reference_prbs15(500, seed), msg=hex(seed))

    def test_default_seed_is_all_ones(self):
        self.assertEqual(prbs15_generator(100), prbs15_generator(100, seed=0x7FFF))

    def test_deterministic(self):
        self.assertEqual(prbs15_generator(1000), prbs15_generator(1000))

    def test_prefix_property(self):
        # A longer sequence starts with the shorter one
        self.assertEqual(prbs15_generator(2000)[:700], prbs15_generator(700))

    def test_inverted_recurrence(self):
        # not f satisfies o[n] = o[n-14] ^ o[n-15] ^ 1
        o = self.two_periods
        np.testing.assert_array_equal(o[15:], o[1:-14] ^ o[:-15] ^ 1)

    def test_full_period(self):
        o = self.two_periods
        np.testing.assert_array_equal(o[:PERIOD], o[PERIOD:])

    def test_no_shorter_period(self):
        o = self.two_periods
        for divisor in (7, 31, 151, 217, 1057, 4681):
            self.assertFalse(np.array_equal(o[:PERIOD], np.roll(o[:PERIOD], divisor)),
                             msg=f"period {divisor}")

    def test_balance_inverted_m_sequence(self):
        # m-sequence has 2^14 ones; inverted output has 2^14 - 1 ones
        ones = int(np.sum(self.two_periods[:PERIOD]))
        self.assertEqual(ones, 2 ** 14 - 1)
        self.assertEqual(PERIOD - ones, 2 ** 14)

    def test_longest_runs(self):
        # m-sequence: longest run 15 ones and 14 zeros; inverted: 15 zeros, 14 ones
        o = np.concatenate([self.two_periods[:PERIOD], self.two_periods[:20]])
        edges = np.flatnonzero(np.diff(o)) + 1
        runs = np.diff(np.concatenate([[0], edges, [o.size]]))
        values = o[np.concatenate([[0], edges])]
        self.assertEqual(int(np.max(runs[values == 0])), 15)
        self.assertEqual(int(np.max(runs[values == 1])), 14)

    def test_all_nonzero_15_bit_windows_once_per_period(self):
        # Every 15-bit window except all-ones (inverted all-zero state) appears once
        o = self.two_periods
        weights = 1 << np.arange(15)
        windows = np.array([int(np.dot(o[n:n + 15], weights)) for n in range(PERIOD)])
        self.assertEqual(np.unique(windows).size, PERIOD)
        self.assertNotIn(2 ** 15 - 1, windows)

    def test_invalid_num_bits(self):
        for num_bits in (0, -1, 1.5, "10", None, True, np.int64(10)):
            with self.assertRaises(ValueError, msg=repr(num_bits)):
                prbs15_generator(num_bits)

    def test_invalid_seed(self):
        for seed in (0, -1, 2 ** 15, 1.0, None, True):
            with self.assertRaises(ValueError, msg=repr(seed)):
                prbs15_generator(10, seed=seed)


if __name__ == "__main__":
    unittest.main()
