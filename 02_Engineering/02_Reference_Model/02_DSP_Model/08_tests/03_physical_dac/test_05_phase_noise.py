"""03-05 tests: clock phase noise from an L(f) mask."""

import unittest
from importlib import import_module

import numpy as np

add_clock_phase_noise = import_module("03_physical_dac").add_clock_phase_noise

FS = 100e6
N = 2 ** 18
X = np.sin(2 * np.pi * 5e6 * np.arange(N) / FS)


def error_power(y):
    return np.mean((y - X) ** 2)


class TestPhaseNoise(unittest.TestCase):

    def test_flat_mask_error_power(self):
        # S_phi = 2 * 10^(L/10) over 0 ... Fs/2 -> var(phi) = 10^(L/10) * Fs
        # e = -dx * phi / (2 pi)  ->  P_e = mean(dx^2) * var(phi) / (4 pi^2)
        level = -120.0
        y = add_clock_phase_noise(X, FS, [1.0, FS], [level, level], 1)
        dx = np.diff(X, prepend=0.0)
        expected = np.mean(dx ** 2) * 10 ** (level / 10) * FS / (4 * np.pi ** 2)
        self.assertAlmostEqual(error_power(y) / expected, 1.0, delta=0.05)

    def test_10_db_lower_mask_gives_10_db_less_error(self):
        y1 = add_clock_phase_noise(X, FS, [1.0, FS], [-110.0, -110.0], 2)
        y2 = add_clock_phase_noise(X, FS, [1.0, FS], [-120.0, -120.0], 2)
        self.assertAlmostEqual(10 * np.log10(error_power(y1) / error_power(y2)), 10.0, delta=1e-6)

    def test_mask_shape_matters(self):
        # Same level far out, worse close-in: more error
        good = add_clock_phase_noise(X, FS, [1e3, 1e6], [-150.0, -150.0], 3)
        poor = add_clock_phase_noise(X, FS, [1e3, 1e6], [-60.0, -150.0], 3)
        self.assertGreater(error_power(poor), error_power(good))

    def test_none_is_copy(self):
        for offsets, levels in ((None, [-100.0]), ([1e3], None), (None, None)):
            y = add_clock_phase_noise(X, FS, offsets, levels, 0)
            np.testing.assert_array_equal(y, X)
            self.assertIsNot(y, X)

    def test_seed_reproducible_and_finite(self):
        a = add_clock_phase_noise(X, FS, [1e3, 1e6], [-100.0, -150.0], 9)
        b = add_clock_phase_noise(X, FS, [1e3, 1e6], [-100.0, -150.0], 9)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all(np.isfinite(a)))
        self.assertEqual(a.shape, X.shape)

    def test_odd_length(self):
        y = add_clock_phase_noise(X[:1001], FS, [1e3], [-100.0], 0)
        self.assertEqual(len(y), 1001)

    def test_invalid(self):
        x = X[:16]
        for args, error in [(([0.1], FS, [1e3], [-100.0], 0), TypeError),
                            ((np.ones((2, 2)), FS, [1e3], [-100.0], 0), ValueError),
                            ((np.array([]), FS, [1e3], [-100.0], 0), ValueError),
                            ((x + 0j, FS, [1e3], [-100.0], 0), TypeError),
                            ((x, FS, [], [], 0), ValueError),
                            ((x, FS, [1e3, 1e4], [-100.0], 0), ValueError),
                            ((x, FS, [1e4, 1e3], [-100.0, -110.0], 0), ValueError),
                            ((x, FS, [0.0, 1e3], [-100.0, -110.0], 0), ValueError),
                            ((x, FS, [1e3], [np.nan], 0), ValueError),
                            ((x, 0.0, [1e3], [-100.0], 0), ValueError)]:
            with self.assertRaises(error, msg=str(args[1:])):
                add_clock_phase_noise(*args)


if __name__ == "__main__":
    unittest.main()
