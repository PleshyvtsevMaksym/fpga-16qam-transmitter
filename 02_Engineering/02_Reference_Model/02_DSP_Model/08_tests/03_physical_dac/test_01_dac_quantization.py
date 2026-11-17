"""03-01 tests: DAC quantization and full scale."""

import unittest
from importlib import import_module

import numpy as np

_dac = import_module("03_physical_dac")
quantize_dac = _dac.quantize_dac
compute_full_scale = _dac.compute_full_scale


class TestQuantizeDac(unittest.TestCase):

    def test_by_hand_3_bit(self):
        # N = 3, FS = 1: delta = 0.25, codes -4 ... 3
        x = np.array([0.0, 0.1, 0.13, -0.37, 0.99, -1.2, 0.8])
        np.testing.assert_allclose(quantize_dac(x, 3, 1.0),
                                   [0.0, 0.0, 0.25, -0.25, 0.75, -1.0, 0.75])

    def test_none_is_ideal_copy(self):
        x = np.array([0.123, -0.456])
        y = quantize_dac(x, None, 1.0)
        np.testing.assert_array_equal(y, x)
        self.assertIsNot(y, x)

    def test_levels_on_lsb_grid_and_error_half_lsb(self):
        # Inside the code range (top code is FS - delta, asymmetric two's complement)
        for bits in (4, 8, 12, 14):
            delta = 2 / 2 ** bits
            x = np.random.default_rng(bits).uniform(-1.0, 1.0 - 1.5 * delta, 10000)
            y = quantize_dac(x, bits, 1.0)
            np.testing.assert_allclose(y / delta, np.round(y / delta), atol=1e-9)
            self.assertLessEqual(np.max(np.abs(y - x)), delta / 2 + 1e-12)

    def test_clipping_limits(self):
        y = quantize_dac(np.array([-5.0, 5.0]), 12, 1.0)
        np.testing.assert_allclose(y, [-1.0, 1.0 - 2 / 4096])

    def test_number_of_levels(self):
        y = quantize_dac(np.linspace(-1.5, 1.5, 100000), 6, 1.0)
        self.assertEqual(np.unique(y).size, 64)

    def test_sqnr_of_full_scale_sine(self):
        # SQNR = 6.02 N + 1.76 dB for a full-scale sine
        n = np.arange(2 ** 16)
        for bits in (8, 12):
            x = (1 - 2 / 2 ** bits) * np.sin(2 * np.pi * 1031 / 2 ** 16 * n)
            e = quantize_dac(x, bits, 1.0) - x
            sqnr = 10 * np.log10(np.mean(x ** 2) / np.mean(e ** 2))
            self.assertAlmostEqual(sqnr, 6.02 * bits + 1.76, delta=0.3)

    def test_input_not_modified_and_numpy_int_bits(self):
        x = np.array([0.3, -0.3])
        quantize_dac(x, np.int64(4), 1.0)
        np.testing.assert_array_equal(x, [0.3, -0.3])

    def test_invalid(self):
        good = np.ones(4) * 0.1
        for args, error in [(([0.1], 12, 1.0), TypeError),
                            ((np.ones((2, 2)), 12, 1.0), ValueError),
                            ((np.array([]), 12, 1.0), ValueError),
                            ((np.ones(2, dtype=complex), 12, 1.0), TypeError),
                            ((np.array([np.nan]), 12, 1.0), ValueError),
                            ((good, 0, 1.0), ValueError), ((good, -3, 1.0), ValueError),
                            ((good, 2.5, 1.0), ValueError), ((good, 12, 0.0), ValueError),
                            ((good, 12, -1.0), ValueError), ((good, 12, np.inf), ValueError)]:
            with self.assertRaises(error, msg=str(args[1:])):
                quantize_dac(*args)


class TestComputeFullScale(unittest.TestCase):

    def test_peak_and_backoff(self):
        x = np.array([0.2, -0.5, 0.4])
        self.assertAlmostEqual(compute_full_scale(x, 0.0), 0.5)
        self.assertAlmostEqual(compute_full_scale(x, 20 * np.log10(2)), 1.0)
        self.assertAlmostEqual(compute_full_scale(x, 20.0), 5.0)
        self.assertIsInstance(compute_full_scale(x, 0.0), float)

    def test_invalid(self):
        for args in [([1.0], 0.0), (np.ones((2, 2)), 0.0), (np.array([]), 0.0),
                     (np.ones(3), -1.0), (np.ones(3), np.nan), (np.zeros(3), 0.0)]:
            with self.assertRaises(ValueError, msg=str(args)):
                compute_full_scale(*args)


if __name__ == "__main__":
    unittest.main()
