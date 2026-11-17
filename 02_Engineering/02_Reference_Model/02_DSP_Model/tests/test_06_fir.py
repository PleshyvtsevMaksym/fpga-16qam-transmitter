"""Stage 06 tests: FIR pulse shaping (full linear convolution)."""

import unittest

import numpy as np

from fpga import apply_fir


def direct_convolution(x, h):
    """Reference: y[n] = sum_k h[k] * x[n - k], written as two loops."""
    y = np.zeros(len(x) + len(h) - 1, dtype=complex)
    for n in range(len(y)):
        for k in range(len(h)):
            if 0 <= n - k < len(x):
                y[n] += h[k] * x[n - k]
    return y


class TestFir(unittest.TestCase):

    def test_example_by_hand(self):
        # x = [1+1j, 2-1j], h = [1, 0.5]
        # y0 = 1+1j, y1 = (2-1j) + 0.5(1+1j) = 2.5-0.5j, y2 = 0.5(2-1j) = 1-0.5j
        y = apply_fir(np.array([1 + 1j, 2 - 1j]), np.array([1.0, 0.5]))
        np.testing.assert_allclose(y, [1 + 1j, 2.5 - 0.5j, 1 - 0.5j], rtol=0, atol=1e-12)

    def test_unit_impulse_returns_coefficients(self):
        h = np.array([0.1, -0.4, 0.8, -0.4, 0.1])
        np.testing.assert_allclose(apply_fir(np.array([1.0]), h), h, rtol=0, atol=1e-15)

    def test_delayed_impulse(self):
        h = np.array([1.0, 2.0, 3.0])
        x = np.array([0, 0, 1j, 0])
        np.testing.assert_allclose(apply_fir(x, h), [0, 0, 1j, 2j, 3j, 0], rtol=0, atol=1e-15)

    def test_against_direct_sum_complex_input(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(40) + 1j * rng.standard_normal(40)
        h = rng.standard_normal(9)
        y = apply_fir(x, h)
        self.assertEqual(len(y), len(x) + len(h) - 1)
        np.testing.assert_allclose(y, direct_convolution(x, h), rtol=0, atol=1e-12)

    def test_real_coefficients_filter_i_and_q_separately(self):
        rng = np.random.default_rng(8)
        x = rng.standard_normal(30) + 1j * rng.standard_normal(30)
        h = rng.standard_normal(6)
        y = apply_fir(x, h)
        np.testing.assert_allclose(y.real, direct_convolution(x.real, h).real, atol=1e-12)
        np.testing.assert_allclose(y.imag, direct_convolution(x.imag, h).real, atol=1e-12)

    def test_linearity(self):
        rng = np.random.default_rng(9)
        x1, x2 = rng.standard_normal(20), rng.standard_normal(20) * 1j
        h = rng.standard_normal(5)
        a, b = 2.5, -0.75 + 1j
        np.testing.assert_allclose(apply_fir(a * x1 + b * x2, h),
                                   a * apply_fir(x1, h) + b * apply_fir(x2, h), atol=1e-12)

    def test_output_is_not_trivially_zero(self):
        y = apply_fir(np.array([1.0, -1.0]), np.array([1.0, 1.0]))
        np.testing.assert_allclose(y, [1.0, 0.0, -1.0], atol=1e-15)

    def test_invalid_input(self):
        h = np.ones(3)
        for samples, coefficients, error in [
            (None, h, ValueError), (np.ones((2, 2)), h, ValueError),
            (np.array([]), h, ValueError), (np.array([np.nan]), h, ValueError),
            (np.ones(3), np.array([np.inf]), ValueError),
            (np.array(["a"]), h, TypeError),
        ]:
            with self.assertRaises(error):
                apply_fir(samples, coefficients)


if __name__ == "__main__":
    unittest.main()
