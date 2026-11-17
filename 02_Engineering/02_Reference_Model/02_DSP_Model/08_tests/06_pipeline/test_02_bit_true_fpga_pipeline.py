"""Pipeline 02 tests: bit-true FPGA model against the ideal reference."""

import contextlib
import io
import unittest
from importlib import import_module

import numpy as np

import params

_pipeline = import_module("06_pipeline")
run_ideal_reference_pipeline = _pipeline.run_ideal_reference_pipeline
run_bit_true_fpga_pipeline = _pipeline.run_bit_true_fpga_pipeline

_fpga = import_module("02_bit_true_fpga")
_vsa = import_module("05_measurement_instruments")


class TestBitTrueFpgaPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            cls.ideal = run_ideal_reference_pipeline()
            cls.fixed = run_bit_true_fpga_pipeline()
        cls.printed = output.getvalue()

    def test_prints_nothing(self):
        self.assertEqual(self.printed, "")

    def test_same_bits_and_symbols_as_ideal(self):
        self.assertEqual(self.fixed["bits"], self.ideal["bits"])
        np.testing.assert_array_equal(self.fixed["symbols"], self.ideal["symbols"])
        np.testing.assert_allclose(self.fixed["normalized_symbols"], self.ideal["normalized_symbols"])

    def test_lengths_match_ideal(self):
        for key in ("fir_i", "fir_q", "nco_cos", "nco_sin", "dac_codes", "if_samples"):
            self.assertEqual(len(self.fixed[key]), len(self.ideal["if_samples"]), msg=key)

    def test_integer_words_fit_their_widths(self):
        f = self.fixed
        for key, bits in (("rrc_coefficients_int", params.FPGA_COEF_BITS),
                          ("fir_i", params.FPGA_FIR_OUT_BITS), ("fir_q", params.FPGA_FIR_OUT_BITS),
                          ("nco_cos", params.FPGA_NCO_AMP_BITS), ("nco_sin", params.FPGA_NCO_AMP_BITS),
                          ("dac_codes", params.DAC_BITS)):
            low, high = _fpga.signed_range(bits)
            self.assertTrue(np.issubdtype(f[key].dtype, np.integer), msg=key)
            self.assertTrue(low <= f[key].min() and f[key].max() <= high, msg=key)

    def test_fir_is_integer_convolution_shifted(self):
        # Golden reference for the HDL testbench: FIR output from integer math only
        f, sps = self.fixed, params.SAMPLES_PER_SYMBOL
        up = np.zeros(len(f["symbols_int_i"]) * sps, dtype=np.int64)
        up[::sps] = f["symbols_int_i"]
        acc = np.convolve(up, f["rrc_coefficients_int"])
        shift = f["fixed_point"]["fir_shift"]
        expected = (acc + (1 << (shift - 1))) >> shift if params.FPGA_ROUNDING == "round" else acc >> shift
        np.testing.assert_array_equal(f["fir_i"], np.clip(expected, *_fpga.signed_range(params.FPGA_FIR_OUT_BITS)))

    def test_dac_scaling(self):
        f = self.fixed
        lsb = f["full_scale"] / 2 ** (params.DAC_BITS - 1)
        np.testing.assert_allclose(f["if_samples"], f["dac_codes"] * lsb, rtol=1e-12)
        target = 2 ** (params.DAC_BITS - 1) * 10 ** (-params.DAC_BACKOFF_DB / 20)
        self.assertGreater(np.max(np.abs(f["dac_codes"])), 0.98 * min(target, 2 ** (params.DAC_BITS - 1) - 1))

    def test_close_to_ideal(self):
        x = self.ideal["if_samples"]
        e = self.fixed["if_samples"] - x
        self.assertGreater(10 * np.log10(np.mean(x ** 2) / np.mean(e ** 2)), 55.0)

    def test_evm_like_ideal(self):
        rx, _ = _vsa.demodulate_if(self.fixed["if_samples"], self.fixed["rrc_coefficients"],
                                   params.SAMPLE_RATE, params.IF_FREQUENCY,
                                   params.SAMPLES_PER_SYMBOL, self.fixed["normalized_symbols"])
        self.assertLess(_vsa.compute_evm(rx, self.fixed["normalized_symbols"]), 1.7)

    def test_fixed_point_report(self):
        fp = self.fixed["fixed_point"]
        expected_keys = {"symbol_bits", "coef_bits", "coef_max_error_lsb", "accumulator_bits",
                         "fir_shift", "fir_out_bits", "nco_frequency", "nco_frequency_error",
                         "mixed_bits", "gain_int", "gain_shift", "dac_bits", "rounding",
                         "clipped_samples"}
        self.assertEqual(set(fp), expected_keys)
        self.assertEqual(fp["symbol_bits"], 3)
        self.assertLessEqual(fp["coef_max_error_lsb"], 0.5)
        self.assertEqual(fp["fir_shift"], fp["accumulator_bits"] - params.FPGA_FIR_OUT_BITS)
        self.assertLess(abs(fp["nco_frequency_error"]), params.SAMPLE_RATE / 2 ** params.FPGA_NCO_PHASE_BITS)
        self.assertLess(fp["gain_int"], 2 ** (params.FPGA_GAIN_BITS - 1))
        self.assertLessEqual(fp["clipped_samples"], 2)

    def test_dsp48_widths(self):
        # Artix-7 DSP48E1 multiplier 25 x 18: every product fits one slice
        self.assertLessEqual(params.FPGA_COEF_BITS, 18)
        self.assertLessEqual(self.fixed["fixed_point"]["symbol_bits"], 25)
        self.assertLessEqual(params.FPGA_NCO_AMP_BITS, 18)
        self.assertLessEqual(params.FPGA_FIR_OUT_BITS, 25)


if __name__ == "__main__":
    unittest.main()
