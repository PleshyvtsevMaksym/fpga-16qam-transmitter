from importlib import import_module

_awgn = import_module("noise.01_awgn")

add_awgn = _awgn.add_awgn