"""Pipeline 03 tests: physical DAC chain (step order, bypass, analog grid)."""

import unittest
from importlib import import_module

import numpy as np

_pipeline = import_module("06_pipeline")
run_physical_dac_pipeline = _pipeline.run_physical_dac_pipeline

_dac = import_module("03_physical_dac")
FS = 100e6
X = 0.8 * np.sin(2 * np.pi * 5e6 * np.arange(4000) / FS)

OFF = dict(full_scale=1.0, dac_bits=None, dac_backoff_db=0.0, dac_hd2_dbc=None,
           dac_hd3_dbc=None, dac_hd_test_level_dbfs=0.0, dac_offset_error_pct_fsr=None,
           dac_gain_error_pct_fsr=None, clock_jitter_rms=None, dac_jitter_rms=None,
           clock_phase_noise_offsets_hz=None, clock_phase_noise_dbc_hz=None,
           analog_oversampling=4, dac_full_scale_voltage=0.5, dac_glitch_impulse_vs=None,
           dac_glitch_duration_s=2e-9, dac_settling_time_s=None, dac_settling_accuracy=1e-3,
           dac_nsd_dbfs_hz=None, seed=1)

ALL = {**OFF, "dac_bits": 12, "dac_hd2_dbc": -70.0, "dac_hd3_dbc": -75.0,
       "dac_offset_error_pct_fsr": 0.1, "dac_gain_error_pct_fsr": 1.0,
       "clock_jitter_rms": 1e-12, "clock_phase_noise_offsets_hz": [1e3, 1e6],
       "clock_phase_noise_dbc_hz": [-100.0, -150.0], "dac_glitch_impulse_vs": 2e-12,
       "dac_settling_time_s": 10e-9, "dac_nsd_dbfs_hz": -150.0}


class TestPhysicalDacPipeline(unittest.TestCase):

    def test_keys_rates_and_lengths(self):
        r = run_physical_dac_pipeline(X, FS, **ALL)
        self.assertEqual(set(r), {"full_scale", "analog_sampling_rate", "quantized", "static",
                                  "clocked", "held", "glitched", "output"})
        self.assertEqual(r["analog_sampling_rate"], 4 * FS)
        for key in ("quantized", "static", "clocked"):
            self.assertEqual(len(r[key]), len(X), msg=key)
        for key in ("held", "glitched", "output"):
            self.assertEqual(len(r[key]), 4 * len(X), msg=key)
        self.assertTrue(np.all(np.isfinite(r["output"])))

    def test_everything_off_is_pure_zoh(self):
        r = run_physical_dac_pipeline(X, FS, **OFF)
        for key in ("quantized", "static", "clocked"):
            np.testing.assert_array_equal(r[key], X)
        np.testing.assert_array_equal(r["output"], np.repeat(X, 4))

    def test_each_step_equals_its_function(self):
        r = run_physical_dac_pipeline(X, FS, **ALL)
        q = _dac.quantize_dac(X, 12, 1.0)
        np.testing.assert_array_equal(r["quantized"], q)
        s = _dac.apply_offset_gain_error(_dac.apply_dac_nonlinearity(q, 1.0, -70.0, -75.0, 0.0),
                                         1.0, 0.1, 1.0)
        np.testing.assert_array_equal(r["static"], s)
        c = _dac.add_clock_phase_noise(_dac.add_clock_jitter(s, 1e-12, FS, None, 1),
                                       FS, [1e3, 1e6], [-100.0, -150.0], 3)
        np.testing.assert_array_equal(r["clocked"], c)
        h = _dac.apply_zero_order_hold(c, 4)
        np.testing.assert_array_equal(r["held"], h)
        # Glitches are counted on the digital codes (quantized), added to the analog level
        hq = _dac.apply_zero_order_hold(q, 4)
        g = h + (_dac.add_dac_glitches(hq, 4, 4 * FS, 1.0, 12, 2e-12, 2e-9, 0.5) - hq)
        np.testing.assert_array_equal(r["glitched"], g)
        o = _dac.add_dac_noise(_dac.apply_settling(g, 4 * FS, 10e-9, 1e-3), -150.0, 1.0, 4 * FS, 2)
        np.testing.assert_array_equal(r["output"], o)

    def test_full_scale_from_backoff_when_none(self):
        r = run_physical_dac_pipeline(X, FS, **{**OFF, "full_scale": None, "dac_backoff_db": 6.0})
        self.assertAlmostEqual(r["full_scale"], 0.8 * 10 ** (6 / 20), places=12)
        r = run_physical_dac_pipeline(X, FS, **{**OFF, "full_scale": 2.5})
        self.assertEqual(r["full_scale"], 2.5)

    def test_seed_reproducible_and_changes_noise(self):
        a = run_physical_dac_pipeline(X, FS, **ALL)["output"]
        b = run_physical_dac_pipeline(X, FS, **ALL)["output"]
        c = run_physical_dac_pipeline(X, FS, **{**ALL, "seed": 2})["output"]
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_input_not_modified(self):
        x = X.copy()
        run_physical_dac_pipeline(x, FS, **ALL)
        np.testing.assert_array_equal(x, X)

    def test_glitches_follow_codes_not_analog_errors(self):
        # Offset/gain/HD change the analog level but not the code toggles
        base = run_physical_dac_pipeline(X, FS, **{**OFF, "dac_bits": 12,
                                                    "dac_glitch_impulse_vs": 2e-12})
        shifted = run_physical_dac_pipeline(X, FS, **{**OFF, "dac_bits": 12,
                                                       "dac_glitch_impulse_vs": 2e-12,
                                                       "dac_offset_error_pct_fsr": 1.0,
                                                       "dac_gain_error_pct_fsr": 2.0})
        np.testing.assert_allclose(shifted["glitched"] - shifted["held"],
                                   base["glitched"] - base["held"], atol=1e-15)

    def test_glitches_need_dac_bits(self):
        with self.assertRaises(ValueError):
            run_physical_dac_pipeline(X, FS, **{**OFF, "dac_glitch_impulse_vs": 2e-12})

    def test_no_default_arguments(self):
        import inspect
        signature = inspect.signature(run_physical_dac_pipeline)
        self.assertTrue(all(p.default is inspect.Parameter.empty
                            for p in signature.parameters.values()))


if __name__ == "__main__":
    unittest.main()
