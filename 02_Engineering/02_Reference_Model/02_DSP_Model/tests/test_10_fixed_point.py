"""Stage 08-09 tests: bit-true fixed-point FPGA blocks and pipeline."""

import unittest
from importlib import import_module

import numpy as np

signed_range = import_module("02_bit_true_fpga").signed_range
required_bits = import_module("02_bit_true_fpga").required_bits
quantize_coefficients = import_module("02_bit_true_fpga").quantize_coefficients
shift_right = import_module("02_bit_true_fpga").shift_right
saturate = import_module("02_bit_true_fpga").saturate
generate_nco = import_module("02_bit_true_fpga").generate_nco


class TestFixedPointArithmetic(unittest.TestCase):

    def test_signed_range(self):
        self.assertEqual(signed_range(12), (-2048, 2047))

    def test_required_bits(self):
        # 3 needs 2 magnitude bits + sign; 2047 fits 12 bits, 2048 needs 13
        self.assertEqual(required_bits(3), 3)
        self.assertEqual(required_bits(2047), 12)
        self.assertEqual(required_bits(2048), 13)

    def test_shift_right_by_hand(self):
        x = np.array([5, 6, 7, -5, -6, -7])          # / 4 = 1.25 1.5 1.75 -1.25 -1.5 -1.75
        np.testing.assert_array_equal(shift_right(x, 2, "truncate"), [1, 1, 1, -2, -2, -2])
        np.testing.assert_array_equal(shift_right(x, 2, "round"), [1, 2, 2, -1, -1, -2])

    def test_saturate_counts_clipped(self):
        y, clipped = saturate(np.array([-3000, 0, 2047, 2048]), 12)
        np.testing.assert_array_equal(y, [-2048, 0, 2047, 2047])
        self.assertEqual(clipped, 2)

    def test_coefficient_error_below_half_lsb(self):
        h = np.array([0.1, -0.33, 1.0, -0.33, 0.1])
        h_int, scale = quantize_coefficients(h, 16)
        self.assertEqual(int(np.max(h_int)), 32767)
        self.assertLessEqual(np.max(np.abs(h * scale - h_int)), 0.5)


class TestNco(unittest.TestCase):

    def test_quarter_rate_is_exact(self):
        # f = Fs/4: phase steps of 90 degrees, cos = 1 0 -1 0
        c, s, f = generate_nco(8, 100e6, 25e6, 32, 12, 16)
        np.testing.assert_array_equal(c, [32767, 0, -32767, 0] * 2)
        np.testing.assert_array_equal(s, [0, 32767, 0, -32767] * 2)
        self.assertEqual(f, 25e6)

    def test_frequency_resolution(self):
        _, _, f = generate_nco(10, 100e6, 5e6, 32, 12, 16)
        self.assertLess(abs(f - 5e6), 100e6 / 2 ** 32)


class TestFixedPointPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from pipeline import run_ideal_reference_pipeline, run_bit_true_fpga_pipeline
        cls.ideal = run_ideal_reference_pipeline()
        cls.fixed = run_bit_true_fpga_pipeline()

    def test_codes_fit_dac(self):
        import params
        codes = self.fixed["dac_codes"]
        low, high = signed_range(params.DAC_BITS)
        self.assertTrue(np.issubdtype(codes.dtype, np.integer))
        self.assertGreaterEqual(codes.min(), low)
        self.assertLessEqual(codes.max(), high)

    def test_close_to_ideal(self):
        # Same units as the ideal model; error power at least 55 dB below
        x = self.ideal["if_samples"]
        e = self.fixed["if_samples"] - x
        snr = 10 * np.log10(np.mean(x ** 2) / np.mean(e ** 2))
        self.assertGreater(snr, 55)

    def test_evm_like_ideal(self):
        import params
        demodulate_if = import_module("05_measurement_instruments").demodulate_if
        compute_evm = import_module("05_measurement_instruments").compute_evm
        rx, _ = demodulate_if(self.fixed["if_samples"], self.fixed["rrc_coefficients"],
                              params.SAMPLE_RATE, params.IF_FREQUENCY,
                              params.SAMPLES_PER_SYMBOL, self.fixed["normalized_symbols"])
        self.assertLess(compute_evm(rx, self.fixed["normalized_symbols"]), 1.7)


if __name__ == "__main__":
    unittest.main()
