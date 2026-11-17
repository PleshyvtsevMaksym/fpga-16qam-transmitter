"""03-02 tests: static DAC nonlinearity from datasheet HD2 / HD3."""

import unittest
from importlib import import_module

import numpy as np

apply_dac_nonlinearity = import_module("03_physical_dac").apply_dac_nonlinearity

N = 2 ** 14
BIN = 101                      # coherent tone: 101 periods in N samples


def harmonics_dbc(y):
    """Levels of the 2nd and 3rd harmonic relative to the fundamental, dBc."""
    spectrum = np.abs(np.fft.rfft(y)) / N
    return (20 * np.log10(spectrum[2 * BIN] / spectrum[BIN]),
            20 * np.log10(spectrum[3 * BIN] / spectrum[BIN]))


def sine(level_dbfs, full_scale):
    return full_scale * 10 ** (level_dbfs / 20) * np.cos(2 * np.pi * BIN * np.arange(N) / N)


class TestDacNonlinearity(unittest.TestCase):

    def test_datasheet_hd_reproduced_at_test_level(self):
        for level in (0.0, -6.0):
            y = apply_dac_nonlinearity(sine(level, 2.0), 2.0, -70.0, -75.0, level)
            hd2, hd3 = harmonics_dbc(y)
            self.assertAlmostEqual(hd2, -70.0, delta=0.05, msg=f"{level} dBFS")
            self.assertAlmostEqual(hd3, -75.0, delta=0.05, msg=f"{level} dBFS")

    def test_hd_scales_with_signal_level(self):
        # HD2 grows 1 dB per dB of signal, HD3 2 dB per dB
        y = apply_dac_nonlinearity(sine(-10.0, 1.0), 1.0, -70.0, -75.0, 0.0)
        hd2, hd3 = harmonics_dbc(y)
        self.assertAlmostEqual(hd2, -80.0, delta=0.05)
        self.assertAlmostEqual(hd3, -95.0, delta=0.1)

    def test_only_hd2_or_only_hd3(self):
        x = sine(0.0, 1.0)
        hd2, hd3 = harmonics_dbc(apply_dac_nonlinearity(x, 1.0, -60.0, None, 0.0))
        self.assertAlmostEqual(hd2, -60.0, delta=0.05)
        self.assertLess(hd3, -200)
        hd2, hd3 = harmonics_dbc(apply_dac_nonlinearity(x, 1.0, None, -60.0, 0.0))
        self.assertLess(hd2, -200)
        self.assertAlmostEqual(hd3, -60.0, delta=0.05)

    def test_polynomial_by_hand(self):
        # HD2 = 20 log10(0.005) -> a2 = 0.01 at 0 dBFS; x = 0.5 -> 0.5 + 0.0025
        hd2 = 20 * np.log10(0.005)
        y = apply_dac_nonlinearity(np.array([0.5, -0.5]), 1.0, hd2, None, 0.0)
        np.testing.assert_allclose(y, [0.5025, -0.4975])

    def test_hd3_is_odd_symmetric(self):
        x = np.linspace(-1, 1, 101)
        y = apply_dac_nonlinearity(x, 1.0, None, -50.0, 0.0)
        np.testing.assert_allclose(y, -y[::-1], atol=1e-15)

    def test_none_is_copy(self):
        x = sine(0.0, 1.0)
        y = apply_dac_nonlinearity(x, 1.0, None, None, 0.0)
        np.testing.assert_array_equal(y, x)
        self.assertIsNot(y, x)

    def test_invalid(self):
        x = sine(0.0, 1.0)
        for args, error in [(([0.1], 1.0, -70.0, -75.0, 0.0), TypeError),
                            ((np.ones((2, 2)), 1.0, -70.0, -75.0, 0.0), ValueError),
                            ((np.array([]), 1.0, -70.0, -75.0, 0.0), ValueError),
                            ((x + 0j, 1.0, -70.0, -75.0, 0.0), TypeError),
                            ((x, 1.0, 0.0, -75.0, 0.0), ValueError),
                            ((x, 1.0, -70.0, 5.0, 0.0), ValueError),
                            ((x, 1.0, np.nan, None, 0.0), ValueError),
                            ((x, 0.0, -70.0, None, 0.0), ValueError),
                            ((x, 1.0, -70.0, None, 1.0), ValueError),
                            ((x, 1.0, -70.0, None, np.nan), ValueError)]:
            with self.assertRaises(error, msg=str(args[1:])):
                apply_dac_nonlinearity(*args)


if __name__ == "__main__":
    unittest.main()
