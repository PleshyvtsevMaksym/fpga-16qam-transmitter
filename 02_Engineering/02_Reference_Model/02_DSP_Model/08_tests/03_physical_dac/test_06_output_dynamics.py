"""03-06 tests: zero-order hold, glitches and settling on the analog grid."""

import unittest
from importlib import import_module

import numpy as np

_dac = import_module("03_physical_dac")
apply_zero_order_hold = _dac.apply_zero_order_hold
add_dac_glitches = _dac.add_dac_glitches
apply_settling = _dac.apply_settling

FS = 100e6


class TestZeroOrderHold(unittest.TestCase):

    def test_by_hand(self):
        np.testing.assert_array_equal(apply_zero_order_hold(np.array([1.0, -2.0]), 3),
                                      [1, 1, 1, -2, -2, -2])

    def test_one_is_copy_and_numpy_int(self):
        x = np.array([0.5, -0.5])
        y = apply_zero_order_hold(x, 1)
        np.testing.assert_array_equal(y, x)
        self.assertIsNot(y, x)
        self.assertEqual(len(apply_zero_order_hold(x, np.int64(4))), 8)

    def test_sinc_droop_and_images(self):
        # Coherent tone at fc: |H(f)| = |sin(pi f/Fs) / (M sin(pi f/(M Fs)))|
        n, m, k = 4096, 8, 205                    # fc = 205/4096 Fs (about 5 MHz)
        x = np.cos(2 * np.pi * k * np.arange(n) / n)
        spectrum = np.abs(np.fft.rfft(apply_zero_order_hold(x, m))) / (n * m)

        def h(bin_analog):
            f = bin_analog / (n * m)              # f / Fs_a
            return abs(np.sin(np.pi * f * m) / (m * np.sin(np.pi * f)))

        for b in (k, n - k, n + k, 2 * n - k, 3 * n + k):
            self.assertAlmostEqual(spectrum[b], 0.5 * h(b), delta=1e-9, msg=f"bin {b}")

    def test_invalid(self):
        for args, error in [(([1.0], 2), TypeError), ((np.ones((2, 2)), 2), ValueError),
                            ((np.array([]), 2), ValueError), ((np.ones(2) + 0j, 2), TypeError),
                            ((np.ones(2), 0), ValueError), ((np.ones(2), -1), ValueError),
                            ((np.ones(2), 2.0), ValueError)]:
            with self.assertRaises(error, msg=str(args[1:])):
                apply_zero_order_hold(*args)


class TestGlitches(unittest.TestCase):

    M = 8
    FS_A = 8 * FS
    BITS = 12
    LSB = 2.0 / 2 ** 12           # FS = 1

    def glitch(self, codes, impulse=2e-12, duration=2e-9, voltage=0.5):
        held = apply_zero_order_hold(np.asarray(codes, dtype=float) * self.LSB, self.M)
        return held, add_dac_glitches(held, self.M, self.FS_A, 1.0, self.BITS,
                                      impulse, duration, voltage)

    def test_major_carry_gives_full_impulse(self):
        # Mid-scale (code 0 = 0x800) -> code -1 (0x7FF): all 12 bits toggle
        held, y = self.glitch([0, -1, -1])
        area = np.sum(y - held) / self.FS_A
        self.assertAlmostEqual(area, 2e-12 / 0.5 * 1.0, delta=1e-20)

    def test_area_proportional_to_toggled_bits(self):
        # 0 -> 1 toggles 1 bit, 1 -> 2 toggles 2 bits, 3 -> 4 toggles 3 bits
        held, y = self.glitch([0, 1, 2, 3, 4])
        g = (y - held).reshape(-1, self.M).sum(axis=1) / self.FS_A
        unit = 2e-12 / 0.5 / self.BITS
        np.testing.assert_allclose(g, np.array([0, 1, 2, 1, 3]) * unit, rtol=1e-12)

    def test_pulse_length_and_position(self):
        # 2 ns at 800 MSa/s = 1.6 -> 2 analog samples, starting at the transition
        held, y = self.glitch([0, 1, 1])
        d = y - held
        nonzero = np.flatnonzero(d)
        np.testing.assert_array_equal(nonzero, [self.M, self.M + 1])
        self.assertAlmostEqual(d[self.M], d[self.M + 1])

    def test_short_glitch_is_one_sample(self):
        held, y = self.glitch([0, 1], duration=1e-15)
        self.assertEqual(np.count_nonzero(y - held), 1)

    def test_long_glitches_overlap_and_add_and_end_is_cut(self):
        # 20 ns = 16 analog samples > 8 per clock: pulses at 8, 16, 24 overlap;
        # the last one is cut at the end of the record (8 of 16 samples)
        held, y = self.glitch([0, 1, 0, 1], duration=20e-9)
        d = y - held
        unit = 2e-12 / 0.5 / self.BITS
        height = unit * self.FS_A / 16
        self.assertAlmostEqual(np.sum(d) / self.FS_A, 2.5 * unit, delta=1e-24)
        self.assertAlmostEqual(d[20], 2 * height, delta=1e-12 * height)
        self.assertAlmostEqual(d[28], 2 * height, delta=1e-12 * height)

    def test_no_transition_no_glitch(self):
        held, y = self.glitch([0, 0, 0, 0])
        np.testing.assert_array_equal(y, held)

    def test_none_is_copy(self):
        held = apply_zero_order_hold(np.array([0.1, 0.2]), self.M)
        y = add_dac_glitches(held, self.M, self.FS_A, 1.0, self.BITS, None, 2e-9, 0.5)
        np.testing.assert_array_equal(y, held)
        self.assertIsNot(y, held)

    def test_invalid(self):
        held = apply_zero_order_hold(np.array([0.1, 0.2]), self.M)
        good = (held, self.M, self.FS_A, 1.0, self.BITS, 2e-12, 2e-9, 0.5)
        for index, value in ((5, 0.0), (5, -1e-12), (5, np.nan), (1, 0), (1, 3),
                             (4, None), (4, 0), (2, 0.0), (3, -1.0), (6, 0.0), (7, 0.0)):
            args = list(good)
            args[index] = value
            with self.assertRaises(ValueError, msg=f"arg {index} = {value}"):
                add_dac_glitches(*args)
        with self.assertRaises(TypeError):
            add_dac_glitches(held + 0j, *good[1:])


class TestSettling(unittest.TestCase):

    FS_A = 800e6

    def test_step_settles_to_accuracy_at_settling_time(self):
        # 10 ns at 800 MSa/s = 8 samples: error after 8 samples = 0.1 %
        y = apply_settling(np.ones(40), self.FS_A, 10e-9, 1e-3)
        self.assertAlmostEqual(1 - y[7], 1e-3, delta=1e-12)
        self.assertGreater(1 - y[6], 1e-3)
        self.assertAlmostEqual(y[-1], 1.0, delta=1e-12)

    def test_first_sample_is_alpha(self):
        tau = 10e-9 / np.log(1e3)
        alpha = 1 - np.exp(-1 / (tau * self.FS_A))
        y = apply_settling(np.ones(5), self.FS_A, 10e-9, 1e-3)
        np.testing.assert_allclose(y, 1 - (1 - alpha) ** np.arange(1, 6), atol=1e-12)

    def test_equals_recursion(self):
        x = np.random.default_rng(0).standard_normal(3000)
        tau = 5e-9 / np.log(1e2)
        alpha = 1 - np.exp(-1 / (tau * self.FS_A))
        expected, previous = np.zeros_like(x), 0.0
        for k, value in enumerate(x):
            previous += alpha * (value - previous)
            expected[k] = previous
        np.testing.assert_allclose(apply_settling(x, self.FS_A, 5e-9, 1e-2), expected, atol=1e-11)

    def test_frequency_response(self):
        # |H| = alpha / |1 - (1 - alpha) e^{-jw}| for a long coherent tone
        n, k = 2 ** 14, 1024
        x = np.cos(2 * np.pi * k * np.arange(n) / n)
        y = apply_settling(np.tile(x, 2), self.FS_A, 10e-9, 1e-3)[n:]
        tau = 10e-9 / np.log(1e3)
        alpha = 1 - np.exp(-1 / (tau * self.FS_A))
        expected = alpha / abs(1 - (1 - alpha) * np.exp(-2j * np.pi * k / n))
        measured = np.abs(np.fft.rfft(y))[k] / (n / 2)
        self.assertAlmostEqual(measured, expected, delta=1e-6)

    def test_slower_settling_more_attenuation(self):
        x = np.cos(2 * np.pi * 0.1 * np.arange(4000))
        fast = apply_settling(x, self.FS_A, 5e-9, 1e-3)[2000:]
        slow = apply_settling(x, self.FS_A, 20e-9, 1e-3)[2000:]
        self.assertLess(np.std(slow), np.std(fast))

    def test_none_is_copy(self):
        x = np.array([1.0, 2.0])
        y = apply_settling(x, self.FS_A, None, 1e-3)
        np.testing.assert_array_equal(y, x)
        self.assertIsNot(y, x)

    def test_invalid(self):
        x = np.ones(4)
        for args, error in [(([1.0], self.FS_A, 1e-9, 1e-3), TypeError),
                            ((np.ones((2, 2)), self.FS_A, 1e-9, 1e-3), ValueError),
                            ((np.array([]), self.FS_A, 1e-9, 1e-3), ValueError),
                            ((x + 0j, self.FS_A, 1e-9, 1e-3), TypeError),
                            ((x, self.FS_A, 0.0, 1e-3), ValueError),
                            ((x, self.FS_A, -1e-9, 1e-3), ValueError),
                            ((x, 0.0, 1e-9, 1e-3), ValueError),
                            ((x, self.FS_A, 1e-9, 0.0), ValueError),
                            ((x, self.FS_A, 1e-9, 1.0), ValueError)]:
            with self.assertRaises(error, msg=str(args[1:])):
                apply_settling(*args)


if __name__ == "__main__":
    unittest.main()
