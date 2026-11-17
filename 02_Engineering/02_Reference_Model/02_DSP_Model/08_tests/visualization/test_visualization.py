"""Visualization tests: every figure is created from real model data, with the right panels."""

from importlib import import_module
import inspect
import unittest
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np

import main
import params
import visualization

_pipeline = import_module("06_pipeline")
run_ideal_reference_pipeline = _pipeline.run_ideal_reference_pipeline
run_bit_true_fpga_pipeline = _pipeline.run_bit_true_fpga_pipeline


class TestVisualization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        visualization.apply_plot_style()
        cls.r = run_ideal_reference_pipeline()
        cls.fpga = run_bit_true_fpga_pipeline()
        cls.dac, cls.tx = main.run_hardware(cls.fpga)
        cls.fs_a = cls.dac["analog_sampling_rate"]
        cls.fs = cls.dac["full_scale"]
        cls.m = main.measure(cls.tx, cls.r, cls.fs_a, cls.fs)
        cls.half_band = (1 + params.RRC_ROLLOFF) * params.SYMBOL_RATE / 2

    def tearDown(self):
        plt.close("all")

    def check(self, fig, number, num_axes):
        """Figure, panel count, title starts with its chain number, all data finite."""
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), num_axes)
        if number is not None:
            self.assertTrue(fig._suptitle.get_text().startswith(str(number)))
        for ax in fig.axes:
            for line in ax.get_lines():
                y = np.asarray(line.get_ydata(), dtype=float)
                self.assertTrue(np.all(np.isfinite(y)), msg=ax.get_title())
            self.assertTrue(ax.get_xlabel(), msg=f"x label missing: {ax.get_title()}")

    def test_1_constellation(self):
        fig = visualization.plot_constellation(self.r["normalized_symbols"])
        self.check(fig, 1, 1)
        labels = [t.get_text() for t in fig.axes[0].texts]
        self.assertEqual(len(labels), 16)
        self.assertEqual(len(set(labels)), 16)
        self.assertTrue(all(len(t) == 4 and set(t) <= {"0", "1"} for t in labels))

    def test_2_pulse_shaping(self):
        fig = visualization.plot_pulse_shaping(
            self.r["normalized_symbols"], self.r["pulse_shaped_symbols"],
            self.r["rrc_coefficients"], params.SAMPLE_RATE, params.SAMPLES_PER_SYMBOL,
            params.SYMBOL_RATE, params.RRC_ROLLOFF, 20, 8)
        self.check(fig, 2, 4)

    def test_2_symbol_markers_sit_on_waveform(self):
        # Symbol x h[0] markers must lie on the RRC output at the symbol instants
        fig = visualization.plot_pulse_shaping(
            self.r["normalized_symbols"], self.r["pulse_shaped_symbols"],
            self.r["rrc_coefficients"], params.SAMPLE_RATE, params.SAMPLES_PER_SYMBOL,
            params.SYMBOL_RATE, params.RRC_ROLLOFF, 20, 8)
        waveform, markers = fig.axes[0].get_lines()[:2]
        t_wave, y_wave = waveform.get_data()
        for t, y in zip(*markers.get_data()):
            self.assertAlmostEqual(np.interp(t, t_wave, y_wave), y, delta=0.05)

    def test_3_fpga_output(self):
        fig = visualization.plot_fpga_output(
            self.r["if_samples"], self.fpga["if_samples"], params.SAMPLE_RATE, self.fs,
            params.SPECTRUM_RBW_HZ, len(self.r["rrc_coefficients"]), 2e-6)
        self.check(fig, 3, 3)
        self.assertIn("SNR 61.4 dB", fig._suptitle.get_text())

    def test_4_dac_transfer(self):
        fig = visualization.plot_dac_transfer(
            self.r["if_samples"], self.dac["quantized"], self.dac["static"],
            params.SAMPLE_RATE, self.fs, params.SPECTRUM_RBW_HZ, params.IF_FREQUENCY,
            self.half_band)
        self.check(fig, 4, 2)

    def test_5_dac_clock(self):
        fig = visualization.plot_dac_clock(
            self.dac["static"], self.dac["clocked"], params.SAMPLE_RATE, self.fs,
            params.SPECTRUM_RBW_HZ, params.IF_FREQUENCY, self.half_band)
        self.check(fig, 5, 2)

    def test_6_dac_output_dynamics(self):
        fig = visualization.plot_dac_output_dynamics(
            self.r["if_samples"], self.dac["held"], self.dac["glitched"],
            self.dac["output"], params.SAMPLE_RATE, params.ANALOG_OVERSAMPLING, self.fs,
            params.SPECTRUM_RBW_HZ, params.IF_FREQUENCY, len(self.r["rrc_coefficients"]), 12)
        self.check(fig, 6, 2)
        # Spectrum axis reaches Fs_a / 2
        self.assertAlmostEqual(fig.axes[1].get_xlim()[1], self.fs_a / 2e6)

    def test_7_measurement(self):
        fig = visualization.plot_measurement(
            self.m["frequencies"], self.m["psd_dbfs_hz"], self.m["symbols"],
            self.r["normalized_symbols"], self.m["evm_percent"], params.IF_FREQUENCY,
            params.SYMBOL_RATE, params.RRC_ROLLOFF, main.hardware_label())
        self.check(fig, 7, 3)
        self.assertIn(f"EVM {self.m['evm_percent']:.2f} %", fig._suptitle.get_text())

    def test_8_evm_sweep(self):
        x = np.array([-150.0, -120.0, -90.0])
        fig = visualization.plot_evm_sweep(x, np.array([1.6, 1.7, 6.0]),
                                           np.array([1.6, 1.7, 6.1]), "NSD (dBFS/Hz)", 12.5)
        self.check(fig, 8, 1)
        self.assertEqual(fig.axes[0].get_yscale(), "log")

    def test_theory_awgn(self):
        fig = visualization.plot_noisy_constellation(self.r["normalized_symbols"], [30, 20, 10], 1)
        self.check(fig, None, 3)

    def test_style_with_and_without_scienceplots(self):
        for style in (None, ["default"]):
            with mock.patch.object(visualization, "_BASE_STYLE", style):
                visualization.apply_plot_style()
                self.assertTrue(plt.rcParams["figure.constrained_layout.use"])
                self.assertEqual(plt.rcParams["savefig.dpi"], 300)

    def test_plot_functions_have_no_default_arguments(self):
        for name, function in inspect.getmembers(visualization, inspect.isfunction):
            if name.startswith("plot_"):
                for p in inspect.signature(function).parameters.values():
                    self.assertIs(p.default, inspect.Parameter.empty, msg=f"{name}({p.name})")

    def test_does_not_modify_inputs(self):
        before = self.dac["output"].copy()
        visualization.plot_dac_output_dynamics(
            self.r["if_samples"], self.dac["held"], self.dac["glitched"],
            self.dac["output"], params.SAMPLE_RATE, params.ANALOG_OVERSAMPLING, self.fs,
            params.SPECTRUM_RBW_HZ, params.IF_FREQUENCY, 900, 12)
        np.testing.assert_array_equal(self.dac["output"], before)


if __name__ == "__main__":
    unittest.main()
