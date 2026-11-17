"""Stage 05 tests: RRC filter design."""

import unittest

import numpy as np

from dsp import design_rrc

# (rolloff, span, sps); the last three put t = +/-1/(4*rolloff) on a tap
CONFIGS = [(0.35, 8, 100), (0.35, 8, 25), (0.2, 10, 5),
           (0.25, 8, 4), (0.5, 6, 8), (1.0, 4, 4)]


def rrc_general(t, rolloff):
    """General RRC equation; it is 0/0 at t = 0 and t = +/-1/(4*rolloff)."""
    return (
        (np.sin(np.pi * t * (1 - rolloff)) + 4 * rolloff * t * np.cos(np.pi * t * (1 + rolloff)))
        / (np.pi * t * (1 - (4 * rolloff * t) ** 2))
    )


def rrc_reference(rolloff, span, sps):
    """
    Independent reference: the general equation everywhere. At the two
    singular points the value is taken as the numerical limit
    (f(t - e) + f(t + e)) / 2, not from the analytical special-case formulas.
    """
    t = (np.arange(span * sps + 1) - span * sps // 2) / sps
    h = np.empty(len(t))
    e = 1e-6
    for k, tk in enumerate(t):
        if abs(tk) < 1e-9 or abs(abs(tk) - 1 / (4 * rolloff)) < 1e-9:
            h[k] = (rrc_general(tk - e, rolloff) + rrc_general(tk + e, rolloff)) / 2
        else:
            h[k] = rrc_general(tk, rolloff)
    return h / np.sqrt(np.sum(h ** 2))


def frequency_response(h, f_normalized):
    """|H(f)| by a direct DTFT sum; f_normalized = f / Fs."""
    n = np.arange(len(h))
    return np.array([abs(np.sum(h * np.exp(-2j * np.pi * f * n))) for f in f_normalized])


class TestRrcDesign(unittest.TestCase):

    def test_length_real_finite(self):
        for rolloff, span, sps in CONFIGS:
            h = design_rrc(rolloff, span, sps)
            self.assertEqual(len(h), span * sps + 1)
            self.assertEqual(h.dtype, np.float64)
            self.assertTrue(np.all(np.isfinite(h)))

    def test_symmetric_with_peak_at_center(self):
        for rolloff, span, sps in CONFIGS:
            h = design_rrc(rolloff, span, sps)
            np.testing.assert_allclose(h, h[::-1], rtol=0, atol=1e-15)
            self.assertEqual(np.argmax(h), span * sps // 2)

    def test_unit_energy(self):
        for config in CONFIGS:
            self.assertAlmostEqual(np.sum(design_rrc(*config) ** 2), 1.0, places=12)

    def test_against_independent_reference(self):
        # Includes the singular taps; the limit error with e = 1e-6 is ~1e-9
        for config in CONFIGS:
            np.testing.assert_allclose(design_rrc(*config), rrc_reference(*config),
                                       rtol=0, atol=1e-7, err_msg=str(config))

    def test_center_value_by_hand(self):
        # Before normalization h(0) = 1 + a(4/pi - 1) and, for a = 0.25, sps = 4,
        # h(t = 1) = a/sqrt(2) * ((1 + 2/pi) sin(pi/4a) + (1 - 2/pi) cos(pi/4a))
        #          = 0.25/sqrt(2) * ((1 + 2/pi) * sin(pi) + (1 - 2/pi) * cos(pi))
        a = 0.25
        h0 = 1 + a * (4 / np.pi - 1)
        h1 = a / np.sqrt(2) * (-(1 - 2 / np.pi))
        h = design_rrc(a, 8, 4)
        self.assertAlmostEqual(h[16 + 4] / h[16], h1 / h0, places=12)

    def test_matched_filter_gives_nyquist_pulse(self):
        # RRC * RRC = raised cosine: 1 at the centre and close to 0 at
        # multiples of the symbol period (finite span leaves about 1 % ISI)
        rolloff, span, sps = 0.35, 8, 100
        h = design_rrc(rolloff, span, sps)
        rc = np.convolve(h, h)
        center = len(h) - 1
        self.assertAlmostEqual(rc[center], 1.0, places=12)
        isi = [abs(rc[center + k * sps]) for k in range(1, span)]
        self.assertLess(max(isi), 0.02)

    def test_frequency_response(self):
        # Ideal RRC: flat for |f| < (1-a)Rs/2, -3 dB at Rs/2 and zero for
        # |f| > (1+a)Rs/2. Limits allow for the truncation to 8 symbols.
        rolloff, span, sps = 0.35, 8, 100
        h = design_rrc(rolloff, span, sps)
        h_dc = frequency_response(h, [0.0])[0]

        # DC gain of a unit-energy RRC is about sqrt(sps)
        self.assertAlmostEqual(h_dc / np.sqrt(sps), 1.0, delta=0.02)

        passband = np.linspace(0, (1 - rolloff) / 2, 40) / sps
        ripple_db = 20 * np.log10(frequency_response(h, passband) / h_dc)
        self.assertLess(np.max(np.abs(ripple_db)), 0.25)

        half_rate = frequency_response(h, [0.5 / sps])[0] / h_dc
        self.assertAlmostEqual(half_rate, 1 / np.sqrt(2), delta=0.02)

        stopband = np.linspace(1.5 * (1 + rolloff) / 2 / sps, 0.5, 2000)
        stop_db = 20 * np.log10(frequency_response(h, stopband) / h_dc)
        self.assertLess(np.max(stop_db), -40.0)

    def test_invalid_input(self):
        for args in [(0.0, 8, 4), (-0.1, 8, 4), (1.1, 8, 4), (np.nan, 8, 4),
                     (0.35, 0, 4), (0.35, 2.5, 4), (0.35, 8, 0), (0.35, 8, 4.0),
                     (0.35, 3, 5)]:
            with self.assertRaises(ValueError, msg=str(args)):
                design_rrc(*args)


if __name__ == "__main__":
    unittest.main()
