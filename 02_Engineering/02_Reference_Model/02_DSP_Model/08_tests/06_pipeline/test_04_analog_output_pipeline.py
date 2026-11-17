"""Pipeline 04 tests: analog output chain (pass-through until 04-01/02 exist)."""

from importlib import import_module
import unittest

import numpy as np

_pipeline = import_module("06_pipeline")
run_analog_output_pipeline = _pipeline.run_analog_output_pipeline


class TestAnalogOutputPipeline(unittest.TestCase):

    def test_pass_through_copy(self):
        x = np.array([0.1, -0.2, 0.3])
        y = run_analog_output_pipeline(x, 800e6)
        np.testing.assert_array_equal(y, x)
        self.assertIsNot(y, x)

    def test_invalid(self):
        for x, fs in ((np.ones((2, 2)), 1.0), (np.array([]), 1.0), ([1.0], 1.0),
                      (np.ones(3), 0.0), (np.ones(3), np.nan)):
            with self.assertRaises(ValueError):
                run_analog_output_pipeline(x, fs)


if __name__ == "__main__":
    unittest.main()
