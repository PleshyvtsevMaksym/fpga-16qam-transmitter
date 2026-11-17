"""
04 Analog output model: from the DAC pins to the load.

    01 reconstruction filter   02 output stage (current, load, voltage)   03 analog noise
The blocks are connected in pipeline/04_analog_output_pipeline.py.
"""

from importlib import import_module

_analog_noise = import_module("04_analog_output.03_analog_noise")

add_awgn = _analog_noise.add_awgn
