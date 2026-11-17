"""Pipeline report tests: terminal reports of pipelines 01 and 02."""

from importlib import import_module
import contextlib
import io
import unittest

import params

_pipeline = import_module("06_pipeline")
run_ideal_reference_pipeline = _pipeline.run_ideal_reference_pipeline
run_bit_true_fpga_pipeline = _pipeline.run_bit_true_fpga_pipeline
print_pipeline_report = _pipeline.print_pipeline_report
print_fixed_point_report = _pipeline.print_fixed_point_report


def captured(function, argument):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        function(argument)
    return output.getvalue()


class TestPipelineReport(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ideal = run_ideal_reference_pipeline()
        cls.fixed = run_bit_true_fpga_pipeline()

    def test_ideal_report(self):
        text = captured(print_pipeline_report, self.ideal)
        for expected in ("16-QAM", "01 IDEAL REFERENCE PIPELINE", "PIPELINE COMPLETED",
                         str(params.NUM_SYMBOLS), f"{params.IF_FREQUENCY / 1e6:.3f} MHz"):
            self.assertIn(expected, text)
        self.assertNotIn("[FAIL", text)

    def test_fixed_point_report(self):
        fp = self.fixed["fixed_point"]
        text = captured(print_fixed_point_report, fp)
        for expected in ("02 BIT-TRUE FPGA PIPELINE", f"{fp['gain_int']} / 2^{fp['gain_shift']}",
                         f"{params.DAC_BITS} bit", "Clipped samples"):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
