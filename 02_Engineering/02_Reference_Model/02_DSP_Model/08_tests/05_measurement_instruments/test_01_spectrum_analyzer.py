"""05-01 tests: spectrum analyzer (amplitude spectrum and PSD)."""

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


compute_psd = import_module("05_measurement_instruments").compute_psd


class TestPsd(unittest.TestCase):

    FS = 100e6

    def test_white_noise_floor_equals_nsd(self):
        # Noise with NSD -150 dBFS/Hz, FS = 1: P = 1/2 * 10^(-15) * Fs/2
        rng = np.random.default_rng(0)
        sigma = np.sqrt(0.5 * 10 ** (-15) * self.FS / 2)
        f, psd = compute_psd(sigma * rng.standard_normal(2 ** 20), self.FS, 10e3, 1.0)
        self.assertAlmostEqual(float(np.mean(psd[(f > 5e6) & (f < 45e6)])), -150.0, delta=0.3)

    def test_full_scale_sine_total_power_0_dbfs(self):
        n = np.arange(2 ** 18)
        x = 2.0 * np.cos(2 * np.pi * 5e6 * n / self.FS)
        f, psd = compute_psd(x, self.FS, 10e3, 2.0)
        power_dbfs = 10 * np.log10(np.sum(10 ** (psd / 10)) * (f[1] - f[0]))
        self.assertAlmostEqual(power_dbfs, 0.0, delta=0.05)
        self.assertAlmostEqual(f[np.argmax(psd)], 5e6, delta=2 * (f[1] - f[0]))

    def test_reference_level_shifts_by_20log(self):
        x = np.random.default_rng(1).standard_normal(2 ** 16)
        _, p1 = compute_psd(x, self.FS, 100e3, 1.0)
        _, p2 = compute_psd(x, self.FS, 100e3, 10.0)
        np.testing.assert_allclose(p1 - p2, 20.0, atol=1e-9)

    def test_frequency_axis_and_rbw(self):
        f, psd = compute_psd(np.ones(2 ** 18), self.FS, 10e3, 1.0)
        segment = 2 * (len(f) - 1)
        self.assertEqual(segment, 1 << int(np.ceil(np.log2(1.5 * self.FS / 10e3))))
        self.assertLessEqual(1.5 * self.FS / segment, 10e3)
        self.assertEqual(f[0], 0.0)
        self.assertEqual(f[-1], self.FS / 2)
        self.assertEqual(len(f), len(psd))

    def test_short_signal_uses_whole_record(self):
        f, psd = compute_psd(np.random.default_rng(2).standard_normal(1000), self.FS, 1.0, 1.0)
        self.assertEqual(len(f), 512 // 2 + 1)
        self.assertTrue(np.all(np.isfinite(psd)))

    def test_zero_signal_is_finite(self):
        _, psd = compute_psd(np.zeros(4096), self.FS, 1e6, 1.0)
        self.assertTrue(np.all(np.isfinite(psd)))

    def test_invalid_input(self):
        good = np.ones(64)
        for args, error in [
            (([1.0], self.FS, 1e6, 1.0), ValueError), ((np.ones((4, 4)), self.FS, 1e6, 1.0), ValueError),
            ((np.array([]), self.FS, 1e6, 1.0), ValueError),
            ((np.ones(8, dtype=complex), self.FS, 1e6, 1.0), TypeError),
            ((good, 0.0, 1e6, 1.0), ValueError), ((good, self.FS, 0.0, 1.0), ValueError),
            ((good, self.FS, 1e6, -1.0), ValueError), ((good, np.nan, 1e6, 1.0), ValueError),
        ]:
            with self.assertRaises(error, msg=str(args[1:])):
                compute_psd(*args)


if __name__ == "__main__":
    unittest.main()
