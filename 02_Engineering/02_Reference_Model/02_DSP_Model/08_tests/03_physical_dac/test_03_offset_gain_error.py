"""03-03 tests: DAC offset and gain error in % FSR."""

import unittest
from importlib import import_module

import numpy as np

apply_offset_gain_error = import_module("03_physical_dac").apply_offset_gain_error


class TestOffsetGainError(unittest.TestCase):

    def test_by_hand(self):
        # FS = 1 -> FSR = 2; offset 0.1 % FSR = 0.002; gain 1 % -> x 1.01
        x = np.array([0.0, 0.5, -1.0])
        np.testing.assert_allclose(apply_offset_gain_error(x, 1.0, 0.1, 1.0),
                                   [0.002, 0.507, -1.008])

    def test_offset_only_shifts_mean(self):
        x = np.sin(np.linspace(0, 20 * np.pi, 2000, endpoint=False))
        y = apply_offset_gain_error(x, 2.0, -0.5, None)
        self.assertAlmostEqual(np.mean(y) - np.mean(x), -0.5 / 100 * 4.0)
        np.testing.assert_allclose(y - x, -0.02)

    def test_gain_only_scales(self):
        x = np.array([0.3, -0.7])
        np.testing.assert_allclose(apply_offset_gain_error(x, 1.0, None, -2.0), x * 0.98)

    def test_zero_errors_change_nothing(self):
        x = np.array([0.3, -0.7])
        np.testing.assert_allclose(apply_offset_gain_error(x, 1.0, 0.0, 0.0), x)

    def test_none_is_copy(self):
        x = np.array([0.3, -0.7])
        y = apply_offset_gain_error(x, 1.0, None, None)
        np.testing.assert_array_equal(y, x)
        self.assertIsNot(y, x)

    def test_invalid(self):
        x = np.ones(4)
        for args, error in [(([0.1], 1.0, 0.1, 1.0), TypeError),
                            ((np.ones((2, 2)), 1.0, 0.1, 1.0), ValueError),
                            ((np.array([]), 1.0, 0.1, 1.0), ValueError),
                            ((x + 0j, 1.0, 0.1, 1.0), TypeError),
                            ((x, 1.0, np.nan, 1.0), ValueError),
                            ((x, 1.0, 0.1, np.inf), ValueError),
                            ((x, 0.0, 0.1, None), ValueError),
                            ((x, -1.0, None, 1.0), ValueError)]:
            with self.assertRaises(error, msg=str(args[1:])):
                apply_offset_gain_error(*args)


if __name__ == "__main__":
    unittest.main()
