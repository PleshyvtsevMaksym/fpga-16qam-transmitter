"""Stage 04 tests: zero-insertion upsampling."""

import unittest

import numpy as np

from fpga import upsample_symbols

SYMBOLS = np.array([1 + 2j, -3 + 0j, 0.5 - 1j, -1 - 1j])


class TestUpsampling(unittest.TestCase):

    def test_example_by_hand(self):
        expected = np.array([1 + 2j, 0, 0, -3, 0, 0, 0.5 - 1j, 0, 0, -1 - 1j, 0, 0])
        np.testing.assert_array_equal(upsample_symbols(SYMBOLS, 3), expected)

    def test_length_positions_and_zeros(self):
        for factor in (1, 2, 3, 7, 100):
            upsampled = upsample_symbols(SYMBOLS, factor)
            self.assertEqual(len(upsampled), len(SYMBOLS) * factor)

            positions = np.arange(len(SYMBOLS)) * factor
            np.testing.assert_array_equal(upsampled[positions], SYMBOLS)

            mask = np.ones(len(upsampled), dtype=bool)
            mask[positions] = False
            self.assertTrue(np.all(upsampled[mask] == 0))

    def test_factor_one_is_identity(self):
        np.testing.assert_array_equal(upsample_symbols(SYMBOLS, 1), SYMBOLS)

    def test_energy_kept_and_power_divided(self):
        energy = np.sum(np.abs(SYMBOLS) ** 2)
        for factor in (2, 5, 100):
            upsampled = upsample_symbols(SYMBOLS, factor)
            self.assertAlmostEqual(np.sum(np.abs(upsampled) ** 2), energy, places=12)
            self.assertAlmostEqual(
                np.mean(np.abs(upsampled) ** 2),
                np.mean(np.abs(SYMBOLS) ** 2) / factor,
                places=12,
            )

    def test_output_dtype_is_complex(self):
        self.assertEqual(upsample_symbols(SYMBOLS, 4).dtype, np.complex128)

    def test_invalid_input(self):
        for symbols, factor in ((None, 2), (np.array([]), 2), (SYMBOLS, 0),
                                (SYMBOLS, -2), (SYMBOLS, 2.0)):
            with self.assertRaises(ValueError):
                upsample_symbols(symbols, factor)


if __name__ == "__main__":
    unittest.main()
