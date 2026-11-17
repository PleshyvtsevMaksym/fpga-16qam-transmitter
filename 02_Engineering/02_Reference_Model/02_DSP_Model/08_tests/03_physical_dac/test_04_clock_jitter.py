"""03-04 tests: white clock jitter of the DAC update instants."""

import unittest
from importlib import import_module

import numpy as np

add_clock_jitter = import_module("03_physical_dac").add_clock_jitter

FS = 100e6
N = 2 ** 18


def sine(frequency):
    return np.sin(2 * np.pi * frequency * np.arange(N) / FS)


def snr_db(x, y):
    return 10 * np.log10(np.mean(x ** 2) / np.mean((y - x) ** 2))


class TestClockJitter(unittest.TestCase):

    def test_snr_matches_step_model(self):
        # e = -dx * dt * Fs; for a sine: SNR = 1 / (4 sin^2(pi f/Fs) sigma^2 Fs^2)
        f = 5e6
        for sigma in (1e-12, 10e-12, 100e-12):
            y = add_clock_jitter(sine(f), sigma, FS, None, 1)
            theory = -10 * np.log10(4 * np.sin(np.pi * f / FS) ** 2 * sigma ** 2 * FS ** 2)
            self.assertAlmostEqual(snr_db(sine(f), y), theory, delta=0.1, msg=sigma)

    def test_20_db_per_decade_of_jitter(self):
        x = sine(5e6)
        s1 = snr_db(x, add_clock_jitter(x, 1e-12, FS, None, 3))
        s2 = snr_db(x, add_clock_jitter(x, 10e-12, FS, None, 3))
        self.assertAlmostEqual(s1 - s2, 20.0, delta=1e-6)

    def test_clock_and_dac_jitter_add_in_power(self):
        x = sine(5e6)
        combined = add_clock_jitter(x, 3e-12, FS, 4e-12, 7)
        single = add_clock_jitter(x, 5e-12, FS, None, 7)
        np.testing.assert_allclose(combined, single, atol=1e-15)
        np.testing.assert_allclose(add_clock_jitter(x, None, FS, 5e-12, 7), single, atol=1e-15)

    def test_constant_signal_only_first_step(self):
        x = np.full(100, 0.5)
        y = add_clock_jitter(x, 1e-9, FS, None, 0)
        self.assertNotEqual(y[0], x[0])
        np.testing.assert_array_equal(y[1:], x[1:])

    def test_zero_jitter_and_none(self):
        x = sine(5e6)
        np.testing.assert_array_equal(add_clock_jitter(x, 0.0, FS, 0.0, 0), x)
        y = add_clock_jitter(x, None, FS, None, 0)
        np.testing.assert_array_equal(y, x)
        self.assertIsNot(y, x)

    def test_seed_reproducible(self):
        x = sine(5e6)
        np.testing.assert_array_equal(add_clock_jitter(x, 1e-12, FS, None, 5),
                                      add_clock_jitter(x, 1e-12, FS, None, 5))
        self.assertFalse(np.array_equal(add_clock_jitter(x, 1e-12, FS, None, 5),
                                        add_clock_jitter(x, 1e-12, FS, None, 6)))

    def test_invalid(self):
        x = sine(5e6)[:10]
        for args, error in [(([0.1], 1e-12, FS, None, 0), TypeError),
                            ((np.ones((2, 2)), 1e-12, FS, None, 0), ValueError),
                            ((np.array([]), 1e-12, FS, None, 0), ValueError),
                            ((x + 0j, 1e-12, FS, None, 0), TypeError),
                            ((x, -1e-12, FS, None, 0), ValueError),
                            ((x, None, FS, np.nan, 0), ValueError),
                            ((x, 1e-12, 0.0, None, 0), ValueError),
                            ((x, 1e-12, np.inf, None, 0), ValueError)]:
            with self.assertRaises(error, msg=str(args[1:])):
                add_clock_jitter(*args)


if __name__ == "__main__":
    unittest.main()
