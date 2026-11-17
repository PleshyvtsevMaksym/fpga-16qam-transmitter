"""Analysis 01 tests: FFT amplitude spectrum."""

import unittest
from importlib import import_module

import numpy as np

compute_spectrum = import_module("05_measurement_instruments").compute_spectrum

FLOOR_DB = 20 * np.log10(np.finfo(float).tiny)    # about -6153 dB


def direct_dft(x):
    """Reference DFT from its definition: X[k] = sum_n x[n] e^(-j 2 pi k n / N)."""
    N = len(x)
    n = np.arange(N)
    return np.array([np.sum(x * np.exp(-2j * np.pi * k * n / N)) for k in range(N)])


class TestSpectrum(unittest.TestCase):

    def test_frequency_axis_even_length(self):
        # N = 8, Fs = 8 Hz: -4, -3, ..., 3 Hz (Fs/2 belongs to the negative side)
        f, _ = compute_spectrum(np.ones(8), 8.0)
        np.testing.assert_allclose(f, [-4, -3, -2, -1, 0, 1, 2, 3], atol=1e-12)

    def test_frequency_axis_odd_length(self):
        f, _ = compute_spectrum(np.ones(5), 10.0)
        np.testing.assert_allclose(f, [-4, -2, 0, 2, 4], atol=1e-12)

    def test_dc_input(self):
        N, A = 16, 2.5
        f, db = compute_spectrum(np.full(N, A), 16.0)
        zero = N // 2
        self.assertEqual(f[zero], 0.0)
        self.assertAlmostEqual(db[zero], 20 * np.log10(A), places=12)
        others = np.delete(db, zero)
        self.assertTrue(np.all(others < -250))

    def test_cosine_gives_two_bins_of_half_amplitude(self):
        N, k0, A, fs = 64, 5, 3.0, 64.0
        n = np.arange(N)
        f, db = compute_spectrum(A * np.cos(2 * np.pi * k0 * n / N), fs)
        for freq in (k0, -k0):
            index = np.argmin(np.abs(f - freq))
            self.assertEqual(f[index], freq)
            self.assertAlmostEqual(db[index], 20 * np.log10(A / 2), places=10)
        self.assertEqual(sorted(np.argsort(db)[-2:]), [N // 2 - k0, N // 2 + k0])

    def test_complex_exponential_gives_one_bin(self):
        N, k0, A, fs = 64, -7, 1.5, 6.4e6
        n = np.arange(N)
        f, db = compute_spectrum(A * np.exp(2j * np.pi * k0 * n / N), fs)
        index = np.argmax(db)
        self.assertAlmostEqual(f[index], k0 * fs / N, places=6)
        self.assertAlmostEqual(db[index], 20 * np.log10(A), places=10)
        # The mirror bin of a complex tone is empty
        mirror = np.argmin(np.abs(f + k0 * fs / N))
        self.assertLess(db[mirror], -250)

    def test_against_direct_dft_and_shift(self):
        rng = np.random.default_rng(11)
        for N in (16, 15):
            x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            f, db = compute_spectrum(x, 1.0)
            X = direct_dft(x)
            for index, freq in enumerate(f):
                k = int(round(freq * N)) % N
                self.assertAlmostEqual(10 ** (db[index] / 20), abs(X[k]) / N, places=12)

    def test_real_signal_is_conjugate_symmetric(self):
        rng = np.random.default_rng(12)
        N = 128
        f, db = compute_spectrum(rng.standard_normal(N), 1.0)
        zero = N // 2
        for k in range(1, N // 2):
            self.assertEqual(f[zero + k], -f[zero - k])
            self.assertAlmostEqual(db[zero + k], db[zero - k], places=9)

    def test_complex_signal_is_not_symmetric(self):
        N = 64
        n = np.arange(N)
        x = np.exp(2j * np.pi * 4 * n / N) + 0.1 * np.exp(-2j * np.pi * 9 * n / N)
        _, db = compute_spectrum(x, 64.0)
        self.assertAlmostEqual(db[N // 2 + 4], 0.0, places=10)     # +4 Hz, amplitude 1
        self.assertLess(db[N // 2 - 4], -250)                       # -4 Hz is empty
        self.assertAlmostEqual(db[N // 2 - 9], -20.0, places=10)    # -9 Hz, amplitude 0.1

    def test_zero_input_is_finite(self):
        _, db = compute_spectrum(np.zeros(8), 1.0)
        self.assertTrue(np.all(np.isfinite(db)))
        np.testing.assert_allclose(db, FLOOR_DB)

    def test_invalid_input(self):
        for samples, fs, error in [
            ([1.0], 1.0, TypeError), (np.array(["a"]), 1.0, TypeError),
            (np.ones((2, 2)), 1.0, ValueError), (np.array([]), 1.0, ValueError),
            (np.array([np.nan]), 1.0, ValueError), (np.ones(4), 0.0, ValueError),
            (np.ones(4), np.inf, ValueError),
        ]:
            with self.assertRaises(error):
                compute_spectrum(samples, fs)


if __name__ == "__main__":
    unittest.main()
