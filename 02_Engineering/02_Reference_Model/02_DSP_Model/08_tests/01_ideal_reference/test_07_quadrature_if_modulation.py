"""01-07 tests: quadrature IF modulation."""

import unittest
from importlib import import_module

import numpy as np

modulate_to_if = import_module("01_ideal_reference").modulate_to_if


class TestIfModulation(unittest.TestCase):

    def test_pure_i_at_quarter_rate(self):
        # fc = Fs/4: cos(pi n / 2) = 1, 0, -1, 0, ...
        x = modulate_to_if(np.ones(8, dtype=complex), 4.0, 1.0)
        np.testing.assert_allclose(x, [1, 0, -1, 0, 1, 0, -1, 0], rtol=0, atol=1e-12)

    def test_pure_q_has_minus_sign(self):
        # -Q sin(pi n / 2) with Q = 1: 0, -1, 0, +1, ...
        x = modulate_to_if(np.full(8, 1j), 4.0, 1.0)
        np.testing.assert_allclose(x, [0, -1, 0, 1, 0, -1, 0, 1], rtol=0, atol=1e-12)

    def test_values_by_hand_at_eighth_rate(self):
        # fc = Fs/8, I = 2, Q = 1:  x[n] = 2 cos(pi n/4) - sin(pi n/4)
        r = np.sqrt(2) / 2
        expected = [2, 2 * r - r, -1, -2 * r - r, -2, -2 * r + r, 1, 2 * r + r]
        x = modulate_to_if(np.full(8, 2 + 1j), 8.0, 1.0)
        np.testing.assert_allclose(x, expected, rtol=0, atol=1e-12)

    def test_carrier_phase_equals_symbol_angle(self):
        # Constant A e^(j phi) gives A cos(2 pi fc n / Fs + phi) for every n,
        # so the phase uses the absolute index and never restarts
        fs, fc, n = 100e6, 5e6, np.arange(1000)
        for amplitude, phase in ((1.0, 0.3), (3 * np.sqrt(2), np.pi / 4), (0.5, -2.0)):
            samples = np.full(len(n), amplitude * np.exp(1j * phase))
            expected = amplitude * np.cos(2 * np.pi * fc * n / fs + phase)
            np.testing.assert_allclose(modulate_to_if(samples, fs, fc), expected, atol=1e-11)

    def test_equals_real_part_of_complex_upconversion(self):
        rng = np.random.default_rng(3)
        samples = rng.standard_normal(500) + 1j * rng.standard_normal(500)
        fs, fc = 100e6, 7e6
        n = np.arange(500)
        expected = np.real(samples * np.exp(2j * np.pi * fc * n / fs))
        np.testing.assert_allclose(modulate_to_if(samples, fs, fc), expected, atol=1e-12)

    def test_output_real_same_length(self):
        x = modulate_to_if(np.ones(37, dtype=complex), 10.0, 1.0)
        self.assertEqual(len(x), 37)
        self.assertTrue(np.isrealobj(x))

    def test_frequency_translation_of_a_tone(self):
        # Baseband tone at f0 -> real tone at fc + f0: FFT peaks at
        # bins +/-(k_c + k_0), each with amplitude 1/2
        N, k_c, k_0, fs = 1024, 100, 12, 1024.0
        n = np.arange(N)
        tone = np.exp(2j * np.pi * k_0 * n / N)
        x = modulate_to_if(tone, fs, k_c * fs / N)
        spectrum = np.abs(np.fft.fft(x)) / N
        peaks = sorted(np.argsort(spectrum)[-2:])
        self.assertEqual(peaks, [k_c + k_0, N - (k_c + k_0)])
        self.assertAlmostEqual(spectrum[k_c + k_0], 0.5, places=12)
        spectrum[peaks] = 0
        self.assertLess(np.max(spectrum), 1e-12)

    def test_invalid_input(self):
        good = np.ones(4, dtype=complex)
        for samples, fs, fc, error in [
            ([1j], 4.0, 1.0, TypeError), (np.ones(4), 4.0, 1.0, TypeError),
            (np.ones((2, 2), dtype=complex), 4.0, 1.0, ValueError),
            (np.array([], dtype=complex), 4.0, 1.0, ValueError),
            (np.array([np.nan + 0j]), 4.0, 1.0, ValueError),
            (good, 0.0, 1.0, ValueError), (good, -4.0, 1.0, ValueError),
            (good, 4.0, 0.0, ValueError), (good, 4.0, 2.0, ValueError),
            (good, 4.0, -1.0, ValueError), (good, 4.0, np.nan, ValueError),
        ]:
            with self.assertRaises(error):
                modulate_to_if(samples, fs, fc)


class TestIfModulationExtra(unittest.TestCase):

    def test_input_not_modified(self):
        x = np.array([1 + 1j, 2 - 1j])
        modulate_to_if(x, 8.0, 1.0)
        np.testing.assert_array_equal(x, [1 + 1j, 2 - 1j])

    def test_amplitude_bounded_by_envelope(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(1000) + 1j * rng.standard_normal(1000)
        y = modulate_to_if(x, 100.0, 7.3)
        self.assertTrue(np.all(np.abs(y) <= np.abs(x) + 1e-12))

    def test_project_carrier_has_integer_samples_per_period(self):
        import params
        ratio = params.SAMPLE_RATE / params.IF_FREQUENCY
        self.assertEqual(ratio, int(ratio))


if __name__ == "__main__":
    unittest.main()
