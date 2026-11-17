"""04-03 tests: additive white Gaussian noise at a given SNR."""

import unittest
from importlib import import_module

import numpy as np

add_awgn = import_module("04_analog_output").add_awgn


def snr_db(x, y):
    return 10 * np.log10(np.mean(np.abs(x) ** 2) / np.mean(np.abs(y - x) ** 2))


class TestAwgn(unittest.TestCase):

    def test_real_snr(self):
        x = np.sin(np.arange(2 ** 18) * 0.01)
        for snr in (0.0, 10.0, 30.0):
            self.assertAlmostEqual(snr_db(x, add_awgn(x, snr, 0)), snr, delta=0.05)

    def test_complex_snr_and_circular_noise(self):
        x = np.exp(1j * np.arange(2 ** 18) * 0.01)
        y = add_awgn(x, 20.0, 1)
        self.assertAlmostEqual(snr_db(x, y), 20.0, delta=0.05)
        e = y - x
        self.assertAlmostEqual(np.var(e.real) / np.var(e.imag), 1.0, delta=0.02)
        self.assertTrue(np.iscomplexobj(y))

    def test_real_stays_real(self):
        self.assertFalse(np.iscomplexobj(add_awgn(np.ones(10), 10.0, 0)))

    def test_seed_and_none(self):
        x = np.ones(100)
        np.testing.assert_array_equal(add_awgn(x, 10.0, 4), add_awgn(x, 10.0, 4))
        self.assertFalse(np.array_equal(add_awgn(x, 10.0, 4), add_awgn(x, 10.0, 5)))
        y = add_awgn(x, None, 4)
        np.testing.assert_array_equal(y, x)
        self.assertIsNot(y, x)

    def test_invalid(self):
        for x in (np.ones((2, 2)), np.array([])):
            with self.assertRaises(ValueError):
                add_awgn(x, 10.0, 0)


class TestAnalogOutputStubs(unittest.TestCase):

    def test_stub_modules_import(self):
        # 04-01 reconstruction filter and 04-02 output stage are not written yet
        for name in ("04_analog_output.01_reconstruction_filter",
                     "04_analog_output.02_output_stage"):
            import_module(name)


if __name__ == "__main__":
    unittest.main()
