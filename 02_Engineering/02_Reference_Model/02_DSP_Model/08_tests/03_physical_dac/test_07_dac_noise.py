"""03-07 tests: DAC output noise from the datasheet NSD."""

import unittest
from importlib import import_module

import numpy as np

add_dac_noise = import_module("03_physical_dac").add_dac_noise
compute_psd = import_module("05_measurement_instruments").compute_psd

FS = 100e6


class TestDacNoise(unittest.TestCase):

    def test_noise_power_formula(self):
        # P = FS^2/2 * 10^(NSD/10) * Fs/2
        x = np.zeros(2 ** 20)
        y = add_dac_noise(x, -140.0, 2.0, FS, 0)
        expected = 2.0 ** 2 / 2 * 1e-14 * FS / 2
        self.assertAlmostEqual(np.mean(y ** 2) / expected, 1.0, delta=0.01)

    def test_psd_reads_back_nsd(self):
        for nsd, fs in ((-150.0, FS), (-140.0, 8 * FS)):
            y = add_dac_noise(np.zeros(2 ** 20), nsd, 1.0, fs, 1)
            f, psd = compute_psd(y, fs, fs / 1e4, 1.0)
            band = (f > 0.1 * fs / 2) & (f < 0.9 * fs / 2)
            self.assertAlmostEqual(float(np.mean(psd[band])), nsd, delta=0.2, msg=nsd)

    def test_noise_is_zero_mean_white_and_independent_of_signal(self):
        x = np.sin(np.arange(2 ** 16) * 0.1)
        y = add_dac_noise(x, -100.0, 1.0, FS, 2)
        e = y - x
        self.assertLess(abs(np.mean(e)), 4 * np.std(e) / np.sqrt(e.size))
        correlation = np.corrcoef(e[:-1], e[1:])[0, 1]
        self.assertLess(abs(correlation), 0.02)

    def test_seed_and_none(self):
        x = np.zeros(100)
        np.testing.assert_array_equal(add_dac_noise(x, -100.0, 1.0, FS, 3),
                                      add_dac_noise(x, -100.0, 1.0, FS, 3))
        y = add_dac_noise(x, None, 1.0, FS, 3)
        np.testing.assert_array_equal(y, x)
        self.assertIsNot(y, x)

    def test_invalid(self):
        x = np.zeros(4)
        for args, error in [(([0.0], -100.0, 1.0, FS, 0), TypeError),
                            ((np.ones((2, 2)), -100.0, 1.0, FS, 0), ValueError),
                            ((np.array([]), -100.0, 1.0, FS, 0), ValueError),
                            ((x + 0j, -100.0, 1.0, FS, 0), TypeError),
                            ((x, -100.0, 0.0, FS, 0), ValueError),
                            ((x, -100.0, 1.0, 0.0, 0), ValueError),
                            ((x, -100.0, 1.0, np.nan, 0), ValueError)]:
            with self.assertRaises(error, msg=str(args[1:])):
                add_dac_noise(*args)


if __name__ == "__main__":
    unittest.main()
