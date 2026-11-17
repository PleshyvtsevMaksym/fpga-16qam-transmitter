"""Project tests: params.py is consistent (rates, widths, board, datasheet ranges)."""

import unittest

import numpy as np

import params


class TestSignalParameters(unittest.TestCase):

    def test_derived_values(self):
        self.assertEqual(params.NUM_BITS, params.NUM_SYMBOLS * params.BITS_PER_SYMBOL)
        self.assertEqual(params.BITS_PER_SYMBOL, 4)
        self.assertEqual(params.QAM_AVG_ENERGY, 10.0)
        self.assertEqual(params.SAMPLES_PER_SYMBOL * params.SYMBOL_RATE, params.SAMPLE_RATE)

    def test_rates(self):
        self.assertLess(params.IF_FREQUENCY + (1 + params.RRC_ROLLOFF) * params.SYMBOL_RATE / 2,
                        params.SAMPLE_RATE / 2)
        self.assertGreater(params.IF_FREQUENCY - (1 + params.RRC_ROLLOFF) * params.SYMBOL_RATE / 2, 0)

    def test_rrc(self):
        self.assertTrue(0 < params.RRC_ROLLOFF <= 1)
        self.assertEqual((params.RRC_SPAN * params.SAMPLES_PER_SYMBOL) % 2, 0)


class TestFpgaBoard(unittest.TestCase):

    def test_nexys_a7_clock_is_dac_clock(self):
        self.assertEqual(params.FPGA_BOARD, "Digilent Nexys A7-100T")
        self.assertTrue(params.FPGA_PART.startswith("XC7A100T"))
        self.assertEqual(params.FPGA_CLOCK_HZ, params.SAMPLE_RATE)

    def test_dac_interface_fits_pmods(self):
        self.assertGreaterEqual(params.FPGA_DAC_INTERFACE_PINS, params.DAC_BITS + 1)
        self.assertLessEqual(params.FPGA_DAC_INTERFACE_PINS, 8 * params.FPGA_PMOD_PORTS)

    def test_word_widths(self):
        self.assertIn(params.FPGA_ROUNDING, ("round", "truncate"))
        self.assertLessEqual(params.FPGA_COEF_BITS, 18)        # DSP48E1 B port
        self.assertLessEqual(params.FPGA_NCO_LUT_ADDR_BITS, params.FPGA_NCO_PHASE_BITS)
        self.assertLessEqual(params.FPGA_GAIN_BITS, 25)


class TestHardwareParameters(unittest.TestCase):

    def test_dac(self):
        self.assertIsInstance(params.DAC_BITS, int)
        self.assertGreaterEqual(params.DAC_BACKOFF_DB, 0)
        for hd in (params.DAC_HD2_DBC, params.DAC_HD3_DBC):
            self.assertTrue(hd is None or hd < 0)
        self.assertLessEqual(params.DAC_HD_TEST_LEVEL_DBFS, 0)
        self.assertTrue(params.DAC_NSD_DBFS_HZ is None or params.DAC_NSD_DBFS_HZ < -100)

    def test_clock_mask(self):
        offsets = np.asarray(params.CLOCK_PHASE_NOISE_OFFSETS_HZ)
        levels = np.asarray(params.CLOCK_PHASE_NOISE_DBC_HZ)
        self.assertEqual(offsets.shape, levels.shape)
        self.assertTrue(np.all(np.diff(offsets) > 0))
        self.assertTrue(np.all(np.diff(levels) <= 0))      # phase noise falls with offset
        # Jitter and mask are the same clock: only one may be active
        self.assertIsNone(params.CLOCK_JITTER_RMS)

    def test_output_dynamics(self):
        self.assertIsInstance(params.ANALOG_OVERSAMPLING, int)
        self.assertGreaterEqual(params.ANALOG_OVERSAMPLING, 2)
        self.assertGreater(params.DAC_FULL_SCALE_VOLTAGE, 0)
        self.assertGreater(params.DAC_GLITCH_DURATION_S, 0)
        self.assertTrue(0 < params.DAC_SETTLING_ACCURACY < 1)
        # Settling must finish within one DAC period, else the output never settles
        self.assertLessEqual(params.DAC_SETTLING_TIME_S, 1 / params.SAMPLE_RATE)

    def test_instruments(self):
        self.assertGreater(params.SPECTRUM_RBW_HZ, 0)
        self.assertLess(params.SPECTRUM_RBW_HZ, params.SYMBOL_RATE / 10)
        self.assertEqual(params.VSA_RRC_ROLLOFF, params.RRC_ROLLOFF)
        self.assertEqual(params.VSA_RRC_SPAN, params.RRC_SPAN)


if __name__ == "__main__":
    unittest.main()
