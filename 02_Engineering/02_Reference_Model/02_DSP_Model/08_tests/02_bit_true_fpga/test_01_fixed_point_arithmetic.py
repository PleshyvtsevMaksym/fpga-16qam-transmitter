"""02-01 tests: fixed-point helpers (word widths, coefficients, shifts, saturation)."""

import unittest
from importlib import import_module

import numpy as np

_fpga = import_module("02_bit_true_fpga")
signed_range = _fpga.signed_range
required_bits = _fpga.required_bits
quantize_coefficients = _fpga.quantize_coefficients
shift_right = _fpga.shift_right
saturate = _fpga.saturate


class TestSignedRange(unittest.TestCase):

    def test_values_by_hand(self):
        self.assertEqual(signed_range(2), (-2, 1))
        self.assertEqual(signed_range(8), (-128, 127))
        self.assertEqual(signed_range(12), (-2048, 2047))
        self.assertEqual(signed_range(16), (-32768, 32767))

    def test_numpy_integer_accepted(self):
        self.assertEqual(signed_range(np.int64(12)), (-2048, 2047))

    def test_invalid(self):
        for bits in (1, 0, -4, 2.5, "8", None):
            with self.assertRaises((ValueError, TypeError), msg=repr(bits)):
                signed_range(bits)


class TestRequiredBits(unittest.TestCase):

    def test_values_by_hand(self):
        for value, bits in ((0, 1), (1, 2), (2, 3), (3, 3), (4, 4), (127, 8),
                            (128, 9), (2047, 12), (2048, 13)):
            self.assertEqual(required_bits(value), bits, msg=value)

    def test_minimal_width_for_range(self):
        # +/-x fits signed_range(required_bits(x)) and not one bit less
        for x in range(1, 5000, 7):
            bits = required_bits(x)
            low, high = signed_range(bits)
            self.assertTrue(low <= -x and x <= high)
            if bits > 2:
                low, high = signed_range(bits - 1)
                self.assertFalse(low <= -x and x <= high)

    def test_numpy_integer(self):
        self.assertEqual(required_bits(np.int64(2047)), 12)

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            required_bits(-1)


class TestQuantizeCoefficients(unittest.TestCase):

    def test_peak_maps_to_max_code(self):
        h = np.array([0.1, -0.33, 1.0, -0.33, 0.1])
        h_int, scale = quantize_coefficients(h, 16)
        self.assertEqual(int(np.max(h_int)), 32767)
        self.assertAlmostEqual(scale, 32767.0)
        self.assertTrue(np.issubdtype(h_int.dtype, np.integer))

    def test_negative_peak(self):
        h_int, scale = quantize_coefficients(np.array([0.5, -2.0]), 12)
        np.testing.assert_array_equal(h_int, [512, -2047])
        self.assertAlmostEqual(scale, 2047 / 2)

    def test_error_at_most_half_lsb_and_fits_width(self):
        h = np.random.default_rng(0).uniform(-1, 1, 801)
        for bits in (8, 12, 16, 18):
            h_int, scale = quantize_coefficients(h, bits)
            self.assertLessEqual(np.max(np.abs(h * scale - h_int)), 0.5 + 1e-12)
            low, high = signed_range(bits)
            self.assertTrue(low <= h_int.min() and h_int.max() <= high)

    def test_symmetry_kept(self):
        h = import_module("01_ideal_reference").design_rrc(0.35, 8, 100)
        h_int, _ = quantize_coefficients(h, 16)
        np.testing.assert_array_equal(h_int, h_int[::-1])

    def test_invalid(self):
        for h, bits in ((np.array([]), 16), (np.ones((2, 2)), 16), (np.array([1j]), 16),
                        (np.array([np.nan]), 16), (np.array([np.inf]), 16),
                        (np.zeros(4), 16), (np.ones(4), 1)):
            with self.assertRaises(ValueError):
                quantize_coefficients(h, bits)


class TestShiftRight(unittest.TestCase):

    def test_by_hand(self):
        x = np.array([5, 6, 7, -5, -6, -7])          # / 4 = 1.25 1.5 1.75 -1.25 -1.5 -1.75
        np.testing.assert_array_equal(shift_right(x, 2, "truncate"), [1, 1, 1, -2, -2, -2])
        np.testing.assert_array_equal(shift_right(x, 2, "round"), [1, 2, 2, -1, -1, -2])

    def test_equals_floor_formulas(self):
        x = np.arange(-5000, 5000)
        for shift in (1, 3, 7):
            np.testing.assert_array_equal(shift_right(x, shift, "truncate"),
                                          np.floor(x / 2 ** shift))
            np.testing.assert_array_equal(shift_right(x, shift, "round"),
                                          np.floor(x / 2 ** shift + 0.5))

    def test_zero_shift_returns_copy(self):
        x = np.array([1, -2, 3])
        y = shift_right(x, 0, "round")
        np.testing.assert_array_equal(y, x)
        y[0] = 99
        self.assertEqual(x[0], 1)

    def test_large_values_int64(self):
        x = np.array([2 ** 40 + 3])
        self.assertEqual(int(shift_right(x, 2, "round")[0]), 2 ** 38 + 1)
        self.assertEqual(shift_right(np.array([1], dtype=np.int16), 0, "round").dtype, np.int64)

    def test_invalid(self):
        for args, error in (((np.array([1.0]), 1, "round"), TypeError),
                            ((np.array([1]), -1, "round"), ValueError),
                            ((np.array([1]), 1.5, "round"), ValueError),
                            ((np.array([1]), 1, "nearest"), ValueError)):
            with self.assertRaises(error):
                shift_right(*args)


class TestSaturate(unittest.TestCase):

    def test_clips_and_counts(self):
        y, clipped = saturate(np.array([-3000, 0, 2047, 2048]), 12)
        np.testing.assert_array_equal(y, [-2048, 0, 2047, 2047])
        self.assertEqual(clipped, 2)

    def test_no_clipping_inside_range(self):
        x = np.arange(-2048, 2048)
        y, clipped = saturate(x, 12)
        np.testing.assert_array_equal(y, x)
        self.assertEqual(clipped, 0)

    def test_output_int64_and_count_type(self):
        y, clipped = saturate([1, 2], 8)
        self.assertEqual(y.dtype, np.int64)
        self.assertIsInstance(clipped, int)

    def test_invalid_width(self):
        with self.assertRaises(ValueError):
            saturate(np.array([1]), 1)


if __name__ == "__main__":
    unittest.main()
