"""Stage 03 tests: symbol energy normalization."""

import itertools
import unittest

import numpy as np

from dsp import normalize_symbols

LEVELS = (-3, -1, 1, 3)
ALPHABET = np.array([i + 1j * q for i, q in itertools.product(LEVELS, LEVELS)])


class TestNormalization(unittest.TestCase):

    def test_theoretical_16qam_energy_is_10(self):
        # Per axis: (9 + 1 + 1 + 9) / 4 = 5, so Es = 5 + 5 = 10
        # I^2 + Q^2 keeps integers exact (np.abs would round through sqrt)
        self.assertEqual(np.mean(ALPHABET.real ** 2 + ALPHABET.imag ** 2), 10.0)

    def test_normalized_alphabet_has_unit_energy(self):
        normalized = normalize_symbols(ALPHABET, 10.0)
        self.assertAlmostEqual(np.mean(np.abs(normalized) ** 2), 1.0, places=14)

    def test_known_values(self):
        # 1 / sqrt(10) = 0.31622776601683794
        normalized = normalize_symbols(np.array([3 + 3j, -1 + 1j, 1 - 3j]), 10.0)
        expected = 0.31622776601683794 * np.array([3 + 3j, -1 + 1j, 1 - 3j])
        np.testing.assert_allclose(normalized, expected, rtol=0, atol=1e-15)

    def test_scale_factor_is_sqrt_energy(self):
        for energy in (1.0, 2.0, 10.0, 42.0):
            normalized = normalize_symbols(ALPHABET, energy)
            np.testing.assert_allclose(normalized * np.sqrt(energy), ALPHABET, atol=1e-13)

    def test_shape_and_dtype(self):
        normalized = normalize_symbols(ALPHABET, 10.0)
        self.assertEqual(normalized.shape, ALPHABET.shape)
        self.assertEqual(normalized.dtype, np.complex128)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            normalize_symbols(None, 10.0)
        with self.assertRaises(ValueError):
            normalize_symbols(np.array([]), 10.0)
        for energy in (0.0, -1.0, np.nan, np.inf):
            with self.assertRaises(ValueError):
                normalize_symbols(ALPHABET, energy)


if __name__ == "__main__":
    unittest.main()
