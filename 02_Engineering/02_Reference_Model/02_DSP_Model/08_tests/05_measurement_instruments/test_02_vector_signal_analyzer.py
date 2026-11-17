"""05-02 tests: vector signal analyzer (IF demodulation, sync, EVM)."""

import unittest
from importlib import import_module

import numpy as np

_ideal = import_module("01_ideal_reference")
_vsa_module = import_module("05_measurement_instruments.02_vector_signal_analyzer")
demodulate_if = _vsa_module.demodulate_if
compute_evm = _vsa_module.compute_evm
fft_convolve = _vsa_module._fft_convolve

SPS, FS, FC = 16, 16e6, 3e6          # Rs = 1 MBd
NUM_SYMBOLS = 600


def transmitter(phase_rad=0.0, gain=1.0, delay=0, seed=0x7FFF):
    """Small ideal transmitter built from the 01 functions."""
    bits = _ideal.prbs15_generator(4 * NUM_SYMBOLS, seed=seed)
    i, q = _ideal.qam16_gray_mapping(bits)
    symbols = _ideal.normalize_symbols(np.array(i) + 1j * np.array(q), 10.0)
    rrc = _ideal.design_rrc(0.35, 8, SPS)
    baseband = _ideal.apply_fir(_ideal.upsample_symbols(symbols, SPS), rrc)
    baseband = gain * np.exp(1j * phase_rad) * np.concatenate([np.zeros(delay), baseband])
    return _ideal.modulate_to_if(baseband, FS, FC), rrc, symbols


class TestDemodulateIf(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.if_samples, cls.rrc, cls.symbols = transmitter()

    def test_ideal_transmitter(self):
        rx, sync = demodulate_if(self.if_samples, self.rrc, FS, FC, SPS, self.symbols)
        self.assertEqual(rx.shape, self.symbols.shape)
        self.assertEqual(sync["delay_samples"], len(self.rrc) - 1)
        self.assertAlmostEqual(sync["gain"], 1.0, delta=0.01)
        self.assertAlmostEqual(sync["phase_deg"], 0.0, delta=0.1)
        self.assertLess(compute_evm(rx, self.symbols), 2.0)
        self.assertEqual(set(sync), {"delay_samples", "gain", "phase_deg"})

    def test_finds_extra_delay(self):
        for delay in (1, 7, 37):
            x, _, _ = transmitter(delay=delay)
            _, sync = demodulate_if(x, self.rrc, FS, FC, SPS, self.symbols)
            self.assertEqual(sync["delay_samples"], len(self.rrc) - 1 + delay, msg=delay)

    def test_measures_and_corrects_gain_and_phase(self):
        x, _, _ = transmitter(phase_rad=np.radians(30.0), gain=0.5)
        rx, sync = demodulate_if(x, self.rrc, FS, FC, SPS, self.symbols)
        self.assertAlmostEqual(sync["gain"], 0.5, delta=0.005)
        self.assertAlmostEqual(sync["phase_deg"], 30.0, delta=0.1)
        self.assertLess(compute_evm(rx, self.symbols), 2.0)

    def test_evm_follows_awgn(self):
        # Es/N0 at the matched filter: EVM^2 ~ EVM_floor^2 + N0/Es
        add_awgn = import_module("04_analog_output").add_awgn
        rx0, _ = demodulate_if(self.if_samples, self.rrc, FS, FC, SPS, self.symbols)
        floor = compute_evm(rx0, self.symbols)
        noisy = add_awgn(self.if_samples, 20.0, 0)
        rx, _ = demodulate_if(noisy, self.rrc, FS, FC, SPS, self.symbols)
        # SNR 20 dB over Fs/2 = 8 MHz; signal band ~1 MHz -> Es/N0 = 20 + 10log10(8) dB
        expected = 100 * np.sqrt((floor / 100) ** 2 + 10 ** (-(20 + 10 * np.log10(8)) / 10))
        self.assertAlmostEqual(compute_evm(rx, self.symbols), expected, delta=0.15 * expected)

    def test_wrong_reference_gives_large_evm(self):
        _, _, other = transmitter(seed=0x1234)
        rx, _ = demodulate_if(self.if_samples, self.rrc, FS, FC, SPS, other)
        self.assertGreater(compute_evm(rx, other), 50.0)

    def test_list_input_accepted(self):
        rx, _ = demodulate_if(list(self.if_samples), list(self.rrc), FS, FC, SPS, self.symbols)
        self.assertEqual(len(rx), NUM_SYMBOLS)

    def test_invalid(self):
        x, h, s = self.if_samples, self.rrc, self.symbols
        for args in [(np.ones((2, 2)), h, FS, FC, SPS, s), (np.array([]), h, FS, FC, SPS, s),
                     (x + 0j, h, FS, FC, SPS, s), (x, np.array([]), FS, FC, SPS, s),
                     (x, h, FS, FC, SPS, np.array([])), (x, h, 0.0, FC, SPS, s),
                     (x, h, np.nan, FC, SPS, s), (x, h, FS, 0.0, SPS, s),
                     (x, h, FS, FS / 2, SPS, s), (x, h, FS, FC, 0, s),
                     (x[:100], h, FS, FC, SPS, s)]:
            with self.assertRaises(ValueError):
                demodulate_if(*args)


class TestComputeEvm(unittest.TestCase):

    def test_by_hand(self):
        ref = np.array([1 + 0j, -1 + 0j, 1j, -1j])
        self.assertEqual(compute_evm(ref, ref), 0.0)
        self.assertAlmostEqual(compute_evm(ref * 1.1, ref), 10.0)
        self.assertAlmostEqual(compute_evm(ref + 0.05, ref), 5.0)

    def test_relative_to_reference_power(self):
        ref = np.array([3 + 0j, -3 + 0j])
        self.assertAlmostEqual(compute_evm(ref + 0.3, ref), 10.0)

    def test_invalid(self):
        for rx, ref in ((np.array([]), np.array([])), (np.ones(3), np.ones(4))):
            with self.assertRaises(ValueError):
                compute_evm(rx, ref)


class TestFftConvolve(unittest.TestCase):

    def test_equals_numpy_convolve(self):
        rng = np.random.default_rng(0)
        for n, m in ((1, 1), (10, 3), (1000, 801), (37, 64)):
            x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            h = rng.standard_normal(m)
            np.testing.assert_allclose(fft_convolve(x, h), np.convolve(x, h), atol=1e-10)


class TestInstrumentStubs(unittest.TestCase):

    def test_stub_modules_import(self):
        # 05-03 RF metrics and 05-04 oscilloscope are not written yet
        for name in ("05_measurement_instruments.03_rf_metrics",
                     "05_measurement_instruments.04_oscilloscope"):
            import_module(name)


if __name__ == "__main__":
    unittest.main()
