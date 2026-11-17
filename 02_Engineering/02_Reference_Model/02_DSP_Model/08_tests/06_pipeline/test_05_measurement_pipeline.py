"""Pipeline 05 tests: virtual instruments on Fs and on the analog grid."""

import unittest
from importlib import import_module

import numpy as np

import params

_pipeline = import_module("06_pipeline")
run_ideal_reference_pipeline = _pipeline.run_ideal_reference_pipeline
run_measurement_pipeline = _pipeline.run_measurement_pipeline

apply_zero_order_hold = import_module("03_physical_dac").apply_zero_order_hold


def measure(samples, reference, fs):
    return run_measurement_pipeline(samples, reference, fs, params.IF_FREQUENCY,
                                    params.SYMBOL_RATE, params.RRC_ROLLOFF, params.RRC_SPAN,
                                    params.SPECTRUM_RBW_HZ, 0.4)


class TestMeasurementPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.r = run_ideal_reference_pipeline()
        cls.ref = cls.r["normalized_symbols"]
        cls.at_fs = measure(cls.r["if_samples"], cls.ref, params.SAMPLE_RATE)

    def test_keys_and_shapes(self):
        m = self.at_fs
        self.assertEqual(set(m), {"frequencies", "psd_dbfs_hz", "symbols", "sync", "evm_percent"})
        self.assertEqual(len(m["frequencies"]), len(m["psd_dbfs_hz"]))
        self.assertEqual(m["symbols"].shape, self.ref.shape)
        self.assertAlmostEqual(m["frequencies"][-1], params.SAMPLE_RATE / 2)

    def test_ideal_evm_is_rrc_floor(self):
        self.assertAlmostEqual(self.at_fs["evm_percent"], 1.58, delta=0.05)
        self.assertEqual(self.at_fs["sync"]["delay_samples"], len(self.r["rrc_coefficients"]) - 1)

    def test_analog_grid_gives_same_evm(self):
        m = 8
        held = apply_zero_order_hold(self.r["if_samples"], m)
        result = measure(held, self.ref, m * params.SAMPLE_RATE)
        self.assertAlmostEqual(result["evm_percent"], self.at_fs["evm_percent"], delta=0.05)
        self.assertAlmostEqual(result["frequencies"][-1], m * params.SAMPLE_RATE / 2)
        # ZOH delays by half a DAC period = M/2 analog samples
        expected = m * (len(self.r["rrc_coefficients"]) - 1) + m // 2
        self.assertLessEqual(abs(result["sync"]["delay_samples"] - expected), 1)

    def test_psd_peak_at_carrier(self):
        f, psd = self.at_fs["frequencies"], self.at_fs["psd_dbfs_hz"]
        self.assertLess(abs(f[np.argmax(psd)] - params.IF_FREQUENCY),
                        (1 + params.RRC_ROLLOFF) * params.SYMBOL_RATE / 2)

    def test_non_integer_samples_per_symbol(self):
        with self.assertRaises(ValueError):
            measure(self.r["if_samples"], self.ref, 100.5e6)


if __name__ == "__main__":
    unittest.main()
