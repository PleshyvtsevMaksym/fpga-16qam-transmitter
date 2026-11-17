"""02-02 tests: DDS / NCO (phase accumulator + cos/sin lookup table)."""

import unittest
from importlib import import_module

import numpy as np

generate_nco = import_module("02_bit_true_fpga").generate_nco


class TestNco(unittest.TestCase):

    def test_quarter_rate_is_exact(self):
        # f = Fs/4: phase steps of 90 degrees, cos = 1 0 -1 0
        c, s, f = generate_nco(8, 100e6, 25e6, 32, 12, 16)
        np.testing.assert_array_equal(c, [32767, 0, -32767, 0] * 2)
        np.testing.assert_array_equal(s, [0, 32767, 0, -32767] * 2)
        self.assertEqual(f, 25e6)

    def test_actual_frequency_on_tuning_grid(self):
        fs, bits = 100e6, 32
        _, _, f = generate_nco(10, fs, 5e6, bits, 12, 16)
        self.assertLessEqual(abs(f - 5e6), fs / 2 ** bits / 2)
        k = f / fs * 2 ** bits
        self.assertAlmostEqual(k, round(k), places=6)

    def test_project_dds_frequency_error(self):
        import params
        _, _, f = generate_nco(4, params.SAMPLE_RATE, params.IF_FREQUENCY,
                               params.FPGA_NCO_PHASE_BITS, params.FPGA_NCO_LUT_ADDR_BITS,
                               params.FPGA_NCO_AMP_BITS)
        self.assertAlmostEqual(f - params.IF_FREQUENCY, 4.657e-3, delta=1e-5)

    def test_output_types_lengths_amplitude(self):
        c, s, f = generate_nco(1000, 100e6, 7.3e6, 32, 12, 16)
        self.assertEqual(len(c), 1000)
        self.assertEqual(len(s), 1000)
        self.assertTrue(np.issubdtype(c.dtype, np.integer))
        self.assertIsInstance(f, float)
        self.assertLessEqual(np.max(np.abs(c)), 32767)
        self.assertLessEqual(np.max(np.abs(s)), 32767)

    def test_cos_and_sin_in_quadrature(self):
        # c^2 + s^2 = A^2 within amplitude rounding
        c, s, _ = generate_nco(5000, 100e6, 3.1e6, 32, 12, 16)
        r = np.sqrt(c.astype(float) ** 2 + s.astype(float) ** 2)
        self.assertLess(np.max(np.abs(r - 32767)), 1.0)

    def test_matches_ideal_cosine_within_phase_truncation(self):
        fs, f0, p, a = 100e6, 3.1e6, 32, 12
        c, _, f = generate_nco(5000, fs, f0, p, a, 16)
        ideal = 32767 * np.cos(2 * np.pi * f * np.arange(5000) / fs)
        # phase error < 2 pi / 2^A  ->  amplitude error < A_max * 2 pi / 2^A (+ rounding)
        self.assertLess(np.max(np.abs(c - ideal)), 32767 * 2 * np.pi / 2 ** a + 1)

    def test_phase_truncation_spurs_about_6db_per_address_bit(self):
        fs, n = 100e6, 2 ** 16
        for a in (8, 10, 12):
            c, _, _ = generate_nco(n, fs, 100e6 * 1031 / n, 32, a, 18)
            spectrum = np.abs(np.fft.rfft(c * np.hanning(n)))
            peak = np.argmax(spectrum)
            spectrum[max(0, peak - 8):peak + 9] = 0
            spur_dbc = 20 * np.log10(np.max(spectrum) / np.abs(np.fft.rfft(c * np.hanning(n)))[peak])
            self.assertLess(spur_dbc, -6.02 * a + 4, msg=f"A = {a}")

    def test_long_run_no_overflow(self):
        c, s, _ = generate_nco(3_000_000, 100e6, 5e6, 32, 12, 16)
        self.assertEqual(c[0], 32767)
        self.assertTrue(np.all(np.abs(c) <= 32767))

    def test_invalid(self):
        good = dict(num_samples=8, sampling_rate=100e6, frequency=5e6,
                    phase_bits=32, lut_address_bits=12, amplitude_bits=16)
        for key, value in (("num_samples", 0), ("num_samples", 2.5), ("phase_bits", 0),
                           ("lut_address_bits", 33), ("amplitude_bits", 1),
                           ("sampling_rate", 0.0), ("sampling_rate", np.nan),
                           ("frequency", 0.0), ("frequency", 50e6), ("frequency", -1e6)):
            with self.assertRaises(ValueError, msg=f"{key} = {value}"):
                generate_nco(**{**good, key: value})


if __name__ == "__main__":
    unittest.main()
