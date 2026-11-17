"""Project tests: main.py helpers (hardware chain, measurement, sweep, label)."""

from importlib import import_module
import contextlib
import io
import unittest
from unittest import mock

import numpy as np
import matplotlib.pyplot as plt

import main
import params

_pipeline = import_module("06_pipeline")
run_ideal_reference_pipeline = _pipeline.run_ideal_reference_pipeline
run_bit_true_fpga_pipeline = _pipeline.run_bit_true_fpga_pipeline


class TestMainHelpers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ideal = run_ideal_reference_pipeline()
        cls.fpga = run_bit_true_fpga_pipeline()
        cls.dac, cls.tx = main.run_hardware(cls.fpga)

    def test_run_hardware_outputs(self):
        m = params.ANALOG_OVERSAMPLING
        self.assertEqual(self.dac["analog_sampling_rate"], m * params.SAMPLE_RATE)
        self.assertEqual(len(self.tx), m * len(self.fpga["if_samples"]))
        self.assertEqual(self.dac["full_scale"], self.fpga["full_scale"])

    def test_overrides_change_single_value(self):
        dac, tx = main.run_hardware(self.fpga, dac_settling_time_s=None, dac_nsd_dbfs_hz=None)
        np.testing.assert_array_equal(tx, dac["glitched"])
        np.testing.assert_array_equal(dac["static"], self.dac["static"])

    def test_measure_baseline_evm(self):
        m = main.measure(self.tx, self.ideal, self.dac["analog_sampling_rate"], self.dac["full_scale"])
        self.assertLess(m["evm_percent"], 2.0)
        self.assertGreater(m["evm_percent"], 1.5)

    def test_evm_sweep_matches_theory(self):
        with mock.patch.object(main, "EVM_NSD_SWEEP", np.array([-150.0, -100.0, -85.0])):
            measured, theory = main.evm_nsd_sweep(self.fpga, self.ideal)
        np.testing.assert_allclose(measured, theory, rtol=0.05)
        self.assertTrue(np.all(np.diff(measured) > 0))

    def test_hardware_label(self):
        label = main.hardware_label()
        for expected in (f"DAC {params.DAC_BITS} bit", "NSD", "glitch", "settling"):
            self.assertIn(expected, label)


class TestMainRun(unittest.TestCase):

    def test_main_runs_and_makes_all_figures(self):
        flags = {name: True for name in dir(main) if name.startswith("SHOW_")}
        flags["SHOW_EVM_VS_NSD"] = False          # covered by the sweep test, slow
        output = io.StringIO()
        with mock.patch.multiple(main, **flags), mock.patch.object(plt, "show"), \
                contextlib.redirect_stdout(output):
            main.main()
        self.assertEqual(len(plt.get_fignums()), sum(flags.values()))
        self.assertIn("EVM at the transmitter output", output.getvalue())
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
