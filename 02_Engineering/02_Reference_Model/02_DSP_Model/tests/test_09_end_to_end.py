"""
Stage 09 tests: the complete transmitter from run_ideal_reference_pipeline().

Symbol recovery uses an independent reference receiver written here with
plain NumPy: coherent downconversion, matched RRC filter, timing at the
total delay of the two filters, gain estimate and nearest-level decisions.
"""

import contextlib
import io
import unittest
from importlib import import_module

import numpy as np

import params
from pipeline import run_ideal_reference_pipeline
compute_spectrum = import_module("05_measurement_instruments").compute_spectrum
qam16_gray_demapping = import_module("01_ideal_reference").qam16_gray_demapping

# Gray table per axis, written independently of the DSP module
AXIS_LEVEL = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}
LEVELS = np.array([-3, -1, 1, 3])


def reference_receiver(if_samples, rrc, fs, fc, sps, num_symbols, timing_offset=0):
    """Return recovered symbols at Es = 1 scale (before gain correction)."""

    # 1. Coherent downconversion; the factor 2 restores the baseband amplitude.
    #    The image at -2 fc (10 MHz) is removed by the matched filter.
    n = np.arange(len(if_samples))
    baseband = 2 * if_samples * np.exp(-2j * np.pi * fc * n / fs)

    # 2. Matched filter: time-reversed transmit RRC
    matched = np.convolve(baseband, rrc[::-1])

    # 3. Symbol timing: each RRC delays by (L - 1) / 2, two filters give L - 1
    total_delay = len(rrc) - 1
    positions = total_delay + timing_offset + sps * np.arange(num_symbols)
    return matched[positions]


def evm_percent(received, reference):
    error = received - reference
    return 100 * np.sqrt(np.mean(np.abs(error) ** 2) / np.mean(np.abs(reference) ** 2))


class TestEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            cls.r = run_ideal_reference_pipeline()
        # Spectra are measurements, computed outside the pipeline
        cls.r["baseband_frequencies"], cls.r["baseband_spectrum_db"] = compute_spectrum(
            cls.r["pulse_shaped_symbols"], params.SAMPLE_RATE)
        cls.r["if_frequencies"], cls.r["if_spectrum_db"] = compute_spectrum(
            cls.r["if_samples"], params.SAMPLE_RATE)
        cls.printed = output.getvalue()
        cls.fs = params.SAMPLE_RATE
        cls.fc = params.IF_FREQUENCY
        cls.sps = params.SAMPLES_PER_SYMBOL
        cls.bandwidth = (1 + params.RRC_ROLLOFF) * params.SYMBOL_RATE / 2   # one side, Hz

    def test_pipeline_prints_nothing(self):
        self.assertEqual(self.printed, "")

    def test_array_lengths(self):
        r, sps = self.r, self.sps
        num_taps = params.RRC_SPAN * sps + 1
        self.assertEqual(self.sps, params.SAMPLE_RATE / params.SYMBOL_RATE)
        self.assertEqual(len(r["bits"]), params.NUM_BITS)
        self.assertEqual(len(r["symbols"]), params.NUM_SYMBOLS)
        self.assertEqual(len(r["normalized_symbols"]), params.NUM_SYMBOLS)
        self.assertEqual(len(r["upsampled_symbols"]), params.NUM_SYMBOLS * sps)
        self.assertEqual(len(r["rrc_coefficients"]), num_taps)
        self.assertEqual(len(r["pulse_shaped_symbols"]), params.NUM_SYMBOLS * sps + num_taps - 1)
        self.assertEqual(len(r["if_samples"]), len(r["pulse_shaped_symbols"]))
        for key in ("baseband_frequencies", "baseband_spectrum_db",
                    "if_frequencies", "if_spectrum_db"):
            self.assertEqual(len(r[key]), len(r["if_samples"]))

    def test_symbols_follow_bits(self):
        bits = self.r["bits"]
        expected = np.array([
            AXIS_LEVEL[tuple(bits[k:k + 2])] + 1j * AXIS_LEVEL[tuple(bits[k + 2:k + 4])]
            for k in range(0, len(bits), 4)
        ])
        np.testing.assert_array_equal(self.r["symbols"], expected)
        np.testing.assert_allclose(self.r["normalized_symbols"] * np.sqrt(10), expected, atol=1e-12)

    def test_pulse_shaping_is_a_sum_of_shifted_pulses(self):
        # y[n] = sum_k s_k h[n - k sps], built without any convolution routine
        s, h = self.r["normalized_symbols"], self.r["rrc_coefficients"]
        y = np.zeros(len(self.r["pulse_shaped_symbols"]), dtype=complex)
        for k, symbol in enumerate(s):
            y[k * self.sps:k * self.sps + len(h)] += symbol * h
        np.testing.assert_allclose(self.r["pulse_shaped_symbols"], y, rtol=0, atol=1e-12)

    def test_if_is_real_part_of_upconverted_baseband(self):
        bb = self.r["pulse_shaped_symbols"]
        n = np.arange(len(bb))
        expected = np.real(bb * np.exp(2j * np.pi * self.fc * n / self.fs))
        # The phase reaches 2*pi*0.05*410400 = 1.3e5 rad; its rounding error
        # (about 1.3e5 * 2.2e-16 = 3e-11 rad) differs between the two
        # formulas, so the tolerance is 1e-10 instead of 1e-12
        np.testing.assert_allclose(self.r["if_samples"], expected, rtol=0, atol=1e-10)

    def test_if_energy_is_half_of_baseband_energy(self):
        # Re{z} = (z + z*)/2 and the two terms do not overlap in frequency
        e_bb = np.sum(np.abs(self.r["pulse_shaped_symbols"]) ** 2)
        e_if = np.sum(self.r["if_samples"] ** 2)
        self.assertAlmostEqual(e_if / e_bb, 0.5, delta=1e-4)

    def test_baseband_bandwidth(self):
        f = self.r["baseband_frequencies"]
        power = 10 ** (self.r["baseband_spectrum_db"] / 10)
        in_band = np.sum(power[np.abs(f) <= self.bandwidth]) / np.sum(power)
        self.assertGreater(in_band, 0.999)

    def test_if_translation_and_bandwidth(self):
        f = self.r["if_frequencies"]
        db = self.r["if_spectrum_db"]
        power = 10 ** (db / 10)

        # 1. Energy lies within (1 + a) Rs / 2 of +fc and -fc
        in_band = np.abs(np.abs(f) - self.fc) <= self.bandwidth
        self.assertGreater(np.sum(power[in_band]) / np.sum(power), 0.999)

        # 2. Both halves carry the same energy (real signal)
        positive = np.sum(power[f > 0])
        negative = np.sum(power[f < 0])
        self.assertAlmostEqual(positive / negative, 1.0, places=9)

        # 3. Almost nothing near DC or near Fs/2
        near_dc = np.abs(f) < self.fc - 2 * self.bandwidth
        self.assertLess(np.sum(power[near_dc]) / np.sum(power), 1e-5)

    def test_if_spectrum_is_shifted_baseband_spectrum(self):
        # Positive IF half = baseband spectrum moved by +fc, scaled by 1/2 (-6.02 dB)
        f_bb, db_bb = self.r["baseband_frequencies"], self.r["baseband_spectrum_db"]
        f_if, db_if = self.r["if_frequencies"], self.r["if_spectrum_db"]
        step = f_bb[1] - f_bb[0]
        band = np.abs(f_bb) <= self.bandwidth
        shift = int(round(self.fc / step))
        moved = np.flatnonzero(band) + shift
        np.testing.assert_allclose(f_if[moved], f_bb[band] + self.fc, atol=1e-3)
        np.testing.assert_allclose(db_if[moved], db_bb[band] - 20 * np.log10(2), atol=0.1)

    def test_reference_receiver_recovers_symbols_and_bits(self):
        s = self.r["normalized_symbols"]
        received = reference_receiver(
            self.r["if_samples"], self.r["rrc_coefficients"],
            self.fs, self.fc, self.sps, len(s)
        )

        # 1. Gain correction (least squares); close to 1 for unit-energy RRC
        gain = np.vdot(s, received) / np.vdot(s, s)
        self.assertAlmostEqual(abs(gain), 1.0, delta=0.01)
        self.assertLess(abs(np.angle(gain)), 1e-3)
        received = received / gain

        # 2. EVM: only residual ISI of the 8-symbol RRC pair (measured 1.6 %)
        self.assertLess(evm_percent(received, s), 3.0)

        # 3. Nearest-level decisions and Gray demapping give the original bits
        scaled = received * np.sqrt(10)
        i_levels = LEVELS[np.argmin(np.abs(scaled.real[:, None] - LEVELS), axis=1)]
        q_levels = LEVELS[np.argmin(np.abs(scaled.imag[:, None] - LEVELS), axis=1)]
        recovered_bits = qam16_gray_demapping(i_levels.tolist(), q_levels.tolist())
        self.assertEqual(recovered_bits, self.r["bits"])

    def test_symbol_timing_at_total_filter_delay(self):
        # EVM is smallest when sampling exactly at the delay of both filters
        s = self.r["normalized_symbols"]
        evm = {
            offset: evm_percent(reference_receiver(
                self.r["if_samples"], self.r["rrc_coefficients"],
                self.fs, self.fc, self.sps, len(s), offset), s)
            for offset in (-2, -1, 0, 1, 2)
        }
        best_offset = min(evm, key=lambda offset: evm[offset])
        self.assertEqual(best_offset, 0)


if __name__ == "__main__":
    unittest.main()
