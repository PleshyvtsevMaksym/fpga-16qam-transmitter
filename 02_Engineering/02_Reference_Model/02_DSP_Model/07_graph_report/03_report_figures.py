"""
Report 03 figures: one page per plot, in signal-chain order.

Every page function takes the run data (04_build_report.collect_data) and
a page factory page(number, title, values, description) -> (fig, ax),
draws one plot and returns the figure. Pages follow the real signal path:
FPGA -> DAC (static errors, clock, hold, glitches, settling) -> analog
output -> instruments. Each impairment page shows the clean signal before
the step (black, thin) over the signal after it (colour), so the change is
visible.
"""

from importlib import import_module

import numpy as np

import params

_instruments = import_module("05_measurement_instruments")
compute_psd = _instruments.compute_psd
compute_spectrum = _instruments.compute_spectrum
qam16_gray_demapping = import_module("01_ideal_reference").qam16_gray_demapping

IDEAL = "#000000"
FPGA = "#0072B2"
DAC = "#56B4E9"
CLOCK = "#CC79A7"
OUTPUT = "#D55E00"
TX = "#009E73"
GRID = "#999999"


# ------------------------------------------------------------------ helpers

def _psd(samples, sampling_rate, d):
    """Spectrum analyzer probe: frequency in MHz, PSD in dBFS/Hz."""
    f, psd = compute_psd(samples, sampling_rate, d["rbw"], d["full_scale"])
    return f / 1e6, psd


def _band_power_dbfs(f_mhz, psd, low_mhz, high_mhz):
    """Integrated power of a PSD between two frequencies, dBFS."""
    band = (f_mhz >= low_mhz) & (f_mhz <= high_mhz)
    df = (f_mhz[1] - f_mhz[0]) * 1e6
    return 10 * np.log10(np.sum(10 ** (psd[band] / 10)) * df + 1e-300)


def _snr_db(reference, signal):
    return 10 * np.log10(np.mean(reference ** 2) / np.mean((signal - reference) ** 2))


def _psd_axes(ax, title, low, high):
    ax.set_title(title)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("PSD (dBFS/Hz)")
    ax.set_ylim(low, high)


def _window(d, num_samples):
    return np.arange(d["start"], d["start"] + num_samples)


def _legend_on_top(ax):
    """Legend in a free band above the traces (time plots fill the whole axes)."""
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.3 * (high - low))
    ax.legend(loc="upper right", ncol=2)


def _decision_lines(ax):
    for b in np.array([-2, 0, 2]) / np.sqrt(10):
        ax.axvline(b, color=GRID, linewidth=0.5, linestyle="--")
        ax.axhline(b, color=GRID, linewidth=0.5, linestyle="--")


# ------------------------------------------------------------------ 01 ideal reference

def page_constellation(d, page):
    s = d["r"]["normalized_symbols"]
    points = np.unique(s)
    fig, ax = page("01-1", "Gray-coded 16-QAM constellation (ideal)", {
        "Points": "16",
        "Es of sent symbols": f"{np.mean(np.abs(s) ** 2):.4f}",
        "Levels per axis": "+-1, +-3 / sqrt(10)",
        "Min. distance": f"{2 / np.sqrt(10):.4f}",
        "Peak / average": f"{10 * np.log10(np.max(np.abs(points) ** 2)):.2f} dB",
        "Symbols sent": f"{len(s)}",
    }, "Alphabet of the transmitter after normalization to Es = 1. Each point carries 4 bits "
       "b0 b1 b2 b3 (b0 b1 select I, b2 b3 select Q); neighbouring points differ in one bit "
       "(Gray code), so a symbol error near a decision boundary costs one bit. Dashed lines are "
       "the decision boundaries used by the analyzer. Depends on: mapping table, QAM_AVG_ENERGY.")
    ax.scatter(points.real, points.imag, s=45, color=FPGA, zorder=3)
    for point in points:
        level = np.rint(point * np.sqrt(10))
        bits = qam16_gray_demapping([int(level.real)], [int(level.imag)])
        ax.annotate("".join(map(str, bits)), (point.real, point.imag),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    _decision_lines(ax)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_xlabel("In-phase I")
    ax.set_ylabel("Quadrature Q")
    ax.set_title("Constellation at Es = 1")
    return fig


def page_rrc_impulse(d, page):
    h = d["r"]["rrc_coefficients"]
    sps = params.SAMPLES_PER_SYMBOL
    center = (len(h) - 1) // 2
    rc = np.convolve(h, h)
    rc_center = len(h) - 1
    isi = np.abs(np.delete(rc[rc_center % sps::sps], rc_center // sps)) / rc[rc_center]
    fig, ax = page("01-2", "RRC pulse-shaping filter: impulse response", {
        "Taps": f"{len(h)}",
        "Span": f"{params.RRC_SPAN} symbols",
        "Roll-off a": f"{params.RRC_ROLLOFF}",
        "Energy": f"{np.sum(h ** 2):.6f}",
        "Peak tap h[0]": f"{h[center]:.5f}",
        "Group delay": f"{center} samples = {center / params.SAMPLE_RATE * 1e6:.1f} us",
        "Worst ISI (RRC*RRC)": f"{20 * np.log10(np.max(isi)):.1f} dB",
    }, "Root-raised-cosine impulse response used both in the transmitter FIR and in the "
       "analyzer matched filter. Red dots: taps at whole symbol periods. The RRC alone is not "
       "zero there; only the pair RRC*RRC (raised cosine) is, up to the truncation to the span. "
       "The worst residual ISI of the pair sets the EVM floor of the ideal model.")
    t = (np.arange(len(h)) - center) / sps
    ax.plot(t, h, color=FPGA)
    ax.plot(t[center % sps::sps], h[center % sps::sps], "o", color=OUTPUT, markersize=4)
    ax.set_xlabel("Time (symbol periods)")
    ax.set_ylabel("h[n]")
    ax.set_title("Impulse response")
    return fig


def page_rrc_magnitude(d, page):
    h = d["r"]["rrc_coefficients"]
    a, rs = params.RRC_ROLLOFF, params.SYMBOL_RATE
    padded = np.zeros(1 << 18)
    padded[:len(h)] = h
    f, db = compute_spectrum(padded, params.SAMPLE_RATE)
    db = db - db[f == 0][0]
    positive = f >= 0
    f, db = f[positive], db[positive]
    stop = db[f > 1.1 * (1 + a) * rs / 2]
    fig, ax = page("01-3", "RRC pulse-shaping filter: magnitude response", {
        "|H| at Rs/2": f"{np.interp(rs / 2, f, db):.2f} dB (ideal -3.01)",
        "Passband ripple": f"{np.ptp(db[f < (1 - a) * rs / 2]):.3f} dB",
        "Stopband peak": f"{np.max(stop):.1f} dB (> 1.1 (1+a)Rs/2)",
        "Occupied band": f"{(1 + a) * rs / 1e6:.3f} MHz",
    }, "Frequency response of the transmit RRC, 0 dB at DC. Lines: (1-a)Rs/2, Rs/2 and "
       "(1+a)Rs/2. The stopband level is limited by truncating the filter to RRC_SPAN symbols; "
       "it sets how much signal leaks into the adjacent channel before any hardware effect.")
    ax.plot(f / 1e6, db, color=FPGA)
    for edge, style in (((1 - a) * rs / 2, ":"), (rs / 2, "--"), ((1 + a) * rs / 2, "-.")):
        ax.axvline(edge / 1e6, color=IDEAL, linestyle=style, linewidth=0.8)
    ax.set_xlim(0, 3 * rs / 1e6)
    ax.set_ylim(-100, 5)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("|H(f)| (dB)")
    ax.set_title("Magnitude response")
    return fig


def _baseband_page(d, page, number, part, name, color):
    r = d["r"]
    h = r["rrc_coefficients"]
    delay = (len(h) - 1) // 2
    sps = params.SAMPLES_PER_SYMBOL
    k = np.arange(20, 36)
    n = np.arange(k[0] * sps, (k[-1] + 1) * sps)
    shaped = part(r["pulse_shaped_symbols"])
    fig, ax = page(number, f"Pulse-shaped baseband {name} (ideal)", {
        f"Peak |{name}|": f"{np.max(np.abs(shaped)):.4f}",
        f"RMS {name}": f"{np.sqrt(np.mean(shaped ** 2)):.4f}",
        "Peak / RMS": f"{20 * np.log10(np.max(np.abs(shaped)) / np.sqrt(np.mean(shaped ** 2))):.2f} dB",
        "Symbols shown": f"{k[0]} ... {k[-1]}",
    }, f"Baseband {name} after the RRC filter, with the group delay removed so that the symbol "
       f"instants line up. Dots: symbol value times h[0], i.e. where an isolated symbol would "
       f"peak; the waveform deviates because neighbouring pulses overlap (intended, removed by "
       f"the matched filter). The peak/RMS ratio sets how much DAC range the signal needs.")
    t_us = n / params.SAMPLE_RATE * 1e6
    ax.plot(t_us, shaped[n + delay], color=color)
    ax.plot(k * sps / params.SAMPLE_RATE * 1e6, part(r["normalized_symbols"][k]) * h[delay],
            "o", color=IDEAL, markersize=4)
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Baseband {name}")
    return fig


def page_baseband_i(d, page):
    return _baseband_page(d, page, "01-4", np.real, "I", FPGA)


def page_baseband_q(d, page):
    return _baseband_page(d, page, "01-5", np.imag, "Q", "#E69F00")


def page_ideal_if_waveform(d, page):
    x = d["r"]["if_samples"]
    n = _window(d, 200)
    env = np.abs(d["r"]["pulse_shaped_symbols"][n])
    fig, ax = page("01-6", "IF waveform (ideal floating point)", {
        "Peak": f"{np.max(np.abs(x)):.4f}",
        "RMS": f"{np.sqrt(np.mean(x ** 2)):.4f}",
        "PAPR": f"{10 * np.log10(np.max(x ** 2) / np.mean(x ** 2)):.2f} dB",
        "Samples per carrier": f"{params.SAMPLE_RATE / params.IF_FREQUENCY:g}",
        "Length": f"{len(x)} samples",
    }, "Real IF signal x[n] = I cos(2 pi fc n/Fs) - Q sin(2 pi fc n/Fs) from the floating-point "
       "reference, with the envelope +-|I + jQ|. This is the target the FPGA and DAC should "
       "reproduce. PAPR = peak-to-average power ratio; with zero back-off the peak sits at DAC "
       "full scale and the RMS level is PAPR below it.")
    t = n / params.SAMPLE_RATE * 1e6
    ax.plot(t, x[n], ".-", color=IDEAL, markersize=3, linewidth=0.8)
    ax.plot(t, env, "--", color=GRID)
    ax.plot(t, -env, "--", color=GRID)
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Amplitude")
    ax.set_title("IF samples and envelope")
    return fig


def page_ideal_if_spectrum(d, page):
    f, psd = _psd(d["r"]["if_samples"], params.SAMPLE_RATE, d)
    fc, hb = d["fc"] / 1e6, d["half_band"] / 1e6
    power = 10 ** (psd / 10)
    cumulative = np.cumsum(power) / np.sum(power)
    obw = np.interp(0.995, cumulative, f) - np.interp(0.005, cumulative, f)
    channel = _band_power_dbfs(f, psd, fc - hb, fc + hb)
    adjacent = max(_band_power_dbfs(f, psd, fc - 3 * hb, fc - hb),
                   _band_power_dbfs(f, psd, fc + hb, fc + 3 * hb))
    fig, ax = page("01-7", "IF spectrum (ideal floating point)", {
        "Channel power": f"{channel:.2f} dBFS",
        "99 % occupied BW": f"{obw:.3f} MHz",
        "Adjacent channel": f"{adjacent - channel:.1f} dBc (worse side)",
        "Peak PSD": f"{np.max(psd):.1f} dBFS/Hz",
        "RBW": f"{params.SPECTRUM_RBW_HZ / 1e3:g} kHz",
    }, "Power spectral density of the ideal IF signal measured like a spectrum analyzer "
       "(Welch, Hann, given RBW), in dBFS/Hz: 0 dBFS = full-scale sine. Shaded: channel "
       "fc +- (1+a)Rs/2. The adjacent-channel level here is the floor of the ideal design "
       "(RRC truncation); hardware can only make it worse.")
    ax.plot(f, psd, color=IDEAL)
    ax.axvspan(fc - hb, fc + hb, color=GRID, alpha=0.15)
    ax.set_xlim(0, params.SAMPLE_RATE / 2e6)
    _psd_axes(ax, "PSD 0 ... Fs/2", -220, np.max(psd) + 10)
    return fig


# ------------------------------------------------------------------ 02 bit-true FPGA

def page_fpga_waveform(d, page):
    codes = d["fpga"]["dac_codes"]
    n = _window(d, 200)
    fig, ax = page("02-1", "FPGA output codes to the DAC (bit-true)", {
        "DAC codes": f"{params.DAC_BITS} bit two's complement",
        "Code range used": f"{codes.min()} ... {codes.max()}",
        "Clipped samples": f"{d['fpga']['fixed_point']['clipped_samples']}",
        "Output gain": f"{d['fpga']['fixed_point']['gain_int']} / 2^{d['fpga']['fixed_point']['gain_shift']}",
        "DDS frequency error": f"{d['fpga']['fixed_point']['nco_frequency_error'] * 1e3:.3f} mHz",
    }, "Integer samples the FPGA sends to the DAC (golden vectors for the HDL testbench): "
       "symbols -> 16-bit RRC FIR -> rounding -> DDS mixer -> output gain -> saturation to the "
       "DAC width. Codes are shown as integers; full scale is +-2^(N-1). Black: the ideal "
       "signal in the same units; the two lie on top of each other, the difference is on 02-2.")
    t = n / params.SAMPLE_RATE * 1e6
    ideal_codes = d["r"]["if_samples"][n] / d["full_scale"] * 2 ** (params.DAC_BITS - 1)
    ax.step(t, codes[n], where="post", color=FPGA, label="FPGA codes (02)")
    ax.plot(t, ideal_codes, color=IDEAL, linewidth=0.6, label="ideal (01) in code units")
    _legend_on_top(ax)
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("DAC code")
    ax.set_title("DAC input codes")
    return fig


def page_fpga_error(d, page):
    ideal, fpga = d["r"]["if_samples"], d["fpga"]["if_samples"]
    lsb = d["full_scale"] / 2 ** (params.DAC_BITS - 1)
    error = (fpga - ideal) / lsb
    n = _window(d, 2000)
    fig, ax = page("02-2", "Fixed-point error: bit-true FPGA - ideal", {
        "SNR vs ideal": f"{_snr_db(ideal, fpga):.1f} dB",
        "RMS error": f"{np.sqrt(np.mean(error ** 2)):.3f} LSB",
        "Max error": f"{np.max(np.abs(error)):.3f} LSB",
        "12-bit ideal SNR": f"{6.02 * params.DAC_BITS + 1.76:.1f} dB (full-scale sine)",
    }, "Difference between the bit-true FPGA output and the floating-point reference, in DAC "
       "LSB. It contains coefficient quantization, FIR rounding, DDS phase truncation (LUT "
       "address width) and the final rounding to DAC codes. An error below about 0.5 LSB RMS "
       "means the FPGA is not the limiting block.")
    ax.plot(n / params.SAMPLE_RATE * 1e6, error[n], color=FPGA, linewidth=0.6)
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Error (LSB)")
    ax.set_title("FPGA - ideal")
    return fig


def page_fpga_spectrum(d, page):
    f, psd = _psd(d["fpga"]["if_samples"], params.SAMPLE_RATE, d)
    _, psd_e = _psd(d["fpga"]["if_samples"] - d["r"]["if_samples"], params.SAMPLE_RATE, d)
    fc, hb = d["fc"] / 1e6, d["half_band"] / 1e6
    outside = (np.abs(f - fc) > 3 * hb)
    fig, ax = page("02-3", "FPGA output spectrum: ideal vs bit-true", {
        "Channel power": f"{_band_power_dbfs(f, psd, fc - hb, fc + hb):.2f} dBFS",
        "Error power": f"{_band_power_dbfs(f, psd_e, 0, f[-1]):.1f} dBFS",
        "Worst spur outside": f"{np.max(psd[outside]):.1f} dBFS/Hz",
    }, "Black: ideal floating-point reference (01). Blue: bit-true FPGA output (02). Where blue "
       "rises above black, the fixed-point arithmetic added error: narrow bands at 15, 25, 35, "
       "45 MHz (DDS) and a flat rounding floor. The error alone is shown on 02-4.")
    _, psd_i = _psd(d["r"]["if_samples"], params.SAMPLE_RATE, d)
    ax.plot(f, psd, color=FPGA, label="bit-true FPGA (02)")
    ax.plot(f, psd_i, color=IDEAL, linewidth=0.6, label="ideal (01)")
    ax.set_xlim(0, params.SAMPLE_RATE / 2e6)
    ax.legend(loc="upper right")
    _psd_axes(ax, "PSD 0 ... Fs/2", -220, np.max(psd) + 10)
    return fig


def page_fpga_error_spectrum(d, page):
    f, psd_e = _psd(d["fpga"]["if_samples"] - d["r"]["if_samples"], params.SAMPLE_RATE, d)
    fc = d["fc"] / 1e6
    spurs = {f"Spur {k * fc:g} MHz": f"{np.max(psd_e[np.abs(f - k * fc) < 0.5]):.1f} dBFS/Hz"
             for k in (3, 5, 7, 9)}
    fig, ax = page("02-4", "FPGA fixed-point error spectrum (FPGA - ideal)", {
        "Error power": f"{_band_power_dbfs(f, psd_e, 0, f[-1]):.1f} dBFS",
        "Error floor (median)": f"{np.median(psd_e):.1f} dBFS/Hz",
        **spurs,
    }, "Spectrum of the fixed-point error alone (bit-true FPGA minus ideal). Narrow bands at "
       "odd harmonics of fc (15, 25, 35, 45 MHz) are DDS phase-truncation products of the "
       "12-bit LUT address; they repeat exactly because Fs/fc is an integer. The flat part is "
       "rounding noise; it includes the final rounding to 12-bit DAC codes, so it sits at the "
       "12-bit quantization level (03-2). The spurs rise 10 ... 20 dB above that floor but stay "
       "about 70 dB below the channel.")
    ax.plot(f, psd_e, color=CLOCK)
    ax.set_xlim(0, params.SAMPLE_RATE / 2e6)
    _psd_axes(ax, "PSD of the error", np.median(psd_e) - 30, np.max(psd_e) + 10)
    return fig


# ------------------------------------------------------------------ 03 physical DAC

def page_dac_transfer(d, page):
    x = d["r"]["if_samples"] / d["full_scale"]
    y = d["dac"]["static"] / d["full_scale"]
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    step = 5
    fig, ax = page("03-1", "DAC static transfer error (quantization + HD + offset/gain)", {
        "Fitted gain error": f"{(slope - 1) * 100:+.3f} % (param {params.DAC_GAIN_ERROR_PCT_FSR})",
        "Fitted offset": f"{intercept / 2 * 100:+.4f} % FSR (param {params.DAC_OFFSET_ERROR_PCT_FSR})",
        "Nonlinear residual": f"{np.max(np.abs(residual)) * 1e3:.3f} mFS peak",
        "LSB": f"{2 / 2 ** params.DAC_BITS * 1e3:.3f} mFS",
    }, "DAC output minus ideal input versus input level, both in units of full scale (FS). "
       "The straight-line trend is the gain and offset error, the curvature is the HD2/HD3 "
       "nonlinearity, the vertical spread is quantization (+-0.5 LSB). The fitted values must "
       "reproduce the datasheet parameters at the top.")
    ax.plot(x[::step], (y[::step] - x[::step]) * 1e3, ".", color=OUTPUT, markersize=1.5, rasterized=True)
    ax.set_xlabel("Input (FS)")
    ax.set_ylabel("Output - input (mFS)")
    ax.set_title("Transfer error")
    return fig


def page_dac_quantized_spectrum(d, page):
    f, psd = _psd(d["dac"]["quantized"], params.SAMPLE_RATE, d)
    e = d["dac"]["quantized"] - d["fpga"]["if_samples"]
    theory = 10 * np.log10(2 / 3 * 4 ** -params.DAC_BITS / (params.SAMPLE_RATE / 2))
    fc, hb = d["fc"] / 1e6, d["half_band"] / 1e6
    floor = np.median(psd[(f > 3 * fc) & (f < params.SAMPLE_RATE / 2e6 - 1)])
    fig, ax = page("03-2", "DAC output after quantization", {
        "Noise floor (3 fc ... Fs/2)": f"{floor:.1f} dBFS/Hz",
        "Theory (LSB^2/12)": f"{theory:.1f} dBFS/Hz",
        "Extra error vs FPGA": f"{np.sqrt(np.mean(e ** 2)) / d['full_scale'] * 2 ** (params.DAC_BITS - 1):.3f} LSB RMS",
        "Channel power": f"{_band_power_dbfs(f, psd, fc - hb, fc + hb):.2f} dBFS",
    }, "Spectrum of the ideal DAC: the FPGA codes reproduced on 2^N levels. The flat floor is "
       "the quantization noise, theory delta^2/12 spread over 0 ... Fs/2 (shown in the values). "
       "With the bit-true FPGA the codes are already integers, so the DAC adds no new error here. "
       "Narrow bands at 15, 25, 35, 45 MHz (odd harmonics of fc) come from the DDS of the FPGA, "
       "see 02-3. Black: ideal floating-point signal; the gap between black and blue is what "
       "a 12-bit DAC costs.")
    _, psd_i = _psd(d["r"]["if_samples"], params.SAMPLE_RATE, d)
    ax.plot(f, psd, color=DAC, label="after quantization (03-01)")
    ax.plot(f, psd_i, color=IDEAL, linewidth=0.6, label="ideal (01)")
    ax.set_xlim(0, params.SAMPLE_RATE / 2e6)
    ax.legend(loc="upper right")
    _psd_axes(ax, "PSD 0 ... Fs/2", -220, np.max(psd) + 10)
    return fig


def page_dac_static_spectrum(d, page):
    f, psd = _psd(d["dac"]["static"], params.SAMPLE_RATE, d)
    fc, hb = d["fc"] / 1e6, d["half_band"] / 1e6
    channel = _band_power_dbfs(f, psd, fc - hb, fc + hb)
    values = {"Channel power": f"{channel:.2f} dBFS",
              "DC (offset)": f"{_band_power_dbfs(f, psd, 0, 0.05):.1f} dBFS"}
    for k in (2, 3):
        band = _band_power_dbfs(f, psd, k * fc - k * hb, k * fc + k * hb)
        values[f"HD{k} band"] = f"{band - channel:.1f} dBc"
    fig, ax = page("03-3", "DAC output with nonlinearity and offset/gain error", values,
                   "Spectrum after the static DAC errors. Shaded: bands where the 2nd and 3rd "
                   "harmonics of the modulated signal fall (k fc, k times wider). The datasheet "
                   "HD2/HD3 is given for a full-scale sine; the modulated signal is mostly far "
                   "below full scale (PAPR), so its harmonic bands are lower than the datasheet "
                   "numbers. Offset appears at DC. Black: the same DAC without static errors.")
    _, psd_q = _psd(d["dac"]["quantized"], params.SAMPLE_RATE, d)
    ax.plot(f, psd, color=OUTPUT, label="with HD2/HD3 + offset/gain (03-02, 03-03)")
    ax.plot(f, psd_q, color=IDEAL, linewidth=0.6, label="before: quantized only (03-2)")
    ax.legend(loc="upper right")
    for k in (2, 3):
        ax.axvspan(k * (fc - hb), k * (fc + hb), color=GRID, alpha=0.2)
        ax.annotate(f"HD{k}", (k * fc, np.max(psd) - 20), ha="center", fontsize=8)
    ax.set_xlim(0, params.SAMPLE_RATE / 2e6)
    _psd_axes(ax, "PSD 0 ... Fs/2", -200, np.max(psd) + 10)
    return fig


def _integrated_jitter():
    """RMS clock jitter from the L(f) mask (same interpolation as 03-05), s."""
    if params.CLOCK_PHASE_NOISE_DBC_HZ is None:
        return None
    offsets = np.asarray(params.CLOCK_PHASE_NOISE_OFFSETS_HZ, dtype=float)
    f = np.logspace(np.log10(offsets[0]), np.log10(params.SAMPLE_RATE / 2), 4000)
    level = np.interp(np.log10(f), np.log10(offsets), params.CLOCK_PHASE_NOISE_DBC_HZ)
    s_phi = 2 * 10 ** (level / 10)
    sigma_phi = np.sqrt(np.sum((s_phi[1:] + s_phi[:-1]) / 2 * np.diff(f)))
    return sigma_phi / (2 * np.pi * params.SAMPLE_RATE)


def page_dac_clock(d, page):
    static, clocked = d["dac"]["static"], d["dac"]["clocked"]
    f, psd_e = _psd(clocked - static, params.SAMPLE_RATE, d)
    jitter = _integrated_jitter()
    _, psd_s = _psd(static, params.SAMPLE_RATE, d)
    _, psd_c = _psd(clocked, params.SAMPLE_RATE, d)
    fig, ax = page("03-4", "DAC output with clock jitter and phase noise", {
        "SNR (clock only)": f"{_snr_db(static, clocked):.1f} dB",
        "Mask jitter (100 Hz..Fs/2)": "off" if jitter is None else f"{jitter * 1e15:.1f} fs RMS",
        "White jitter": "off" if params.CLOCK_JITTER_RMS is None else f"{params.CLOCK_JITTER_RMS * 1e12:g} ps",
        "Error floor (median)": f"{np.median(psd_e):.1f} dBFS/Hz",
    }, "Pink: DAC output sampled with the real clock; black: the same output with an ideal "
       "clock. The error the clock adds (their difference) is given in the values: SNR and its floor. The error is proportional to the sample-to-sample "
       "step, so it follows the signal band around fc; its level against the channel shows "
       "whether the oscillator is good enough. The RMS jitter is integrated from the L(f) mask.")
    ax.plot(f, psd_c, color=CLOCK, label="real clock (03-04, 03-05)")
    ax.plot(f, psd_s, color=IDEAL, linewidth=0.6, label="ideal clock")
    ax.set_xlim(0, params.SAMPLE_RATE / 2e6)
    ax.legend(loc="upper right")
    _psd_axes(ax, "PSD 0 ... Fs/2", -200, np.max(psd_c) + 10)
    return fig


def page_dac_zoh(d, page):
    m, fs_a = d["m"], d["fs_a"]
    n = _window(d, 12)
    k = np.arange(n[0] * m, (n[-1] + 1) * m)
    droop = 20 * np.log10(abs(np.sinc(d["fc"] / params.SAMPLE_RATE)))
    fig, ax = page("03-5", "DAC zero-order hold (staircase on the analog grid)", {
        "Analog grid": f"{fs_a / 1e6:g} MSa/s (M = {m})",
        "Sinc droop at fc": f"{droop:.3f} dB",
        "Sinc at Fs - fc": f"{20 * np.log10(abs(np.sinc((params.SAMPLE_RATE - d['fc']) / params.SAMPLE_RATE))):.1f} dB",
        "Hold time": f"{1 / params.SAMPLE_RATE * 1e9:g} ns",
    }, "The DAC holds each sample for one clock period (NRZ). Black dots: ideal samples at Fs; "
       "blue: held output on an M times finer time grid. The hold acts as a sinc filter: a "
       "small droop at fc and images at k Fs +- fc (see 03-9), which the reconstruction filter "
       "must remove.")
    ax.step(k / fs_a * 1e9, d["dac"]["held"][k] / d["full_scale"], where="post", color=DAC)
    ax.plot(n / params.SAMPLE_RATE * 1e9, d["r"]["if_samples"][n] / d["full_scale"], "o",
            color=IDEAL, markersize=4)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude (FS)")
    ax.set_title("Zero-order hold")
    return fig


def page_dac_glitch(d, page):
    m, fs_a = d["m"], d["fs_a"]
    glitch = d["dac"]["glitched"] - d["dac"]["held"]
    lsb = 2 * d["full_scale"] / 2 ** params.DAC_BITS
    codes = np.rint(d["dac"]["quantized"] / lsb).astype(np.int64) + 2 ** (params.DAC_BITS - 1)
    toggles = np.zeros(codes.size - 1, dtype=np.int64)
    changed = codes[1:] ^ codes[:-1]
    for bit in range(params.DAC_BITS):
        toggles += (changed >> bit) & 1
    n = _window(d, 12)
    k = np.arange(n[0] * m, (n[-1] + 1) * m)
    on = params.DAC_GLITCH_IMPULSE_VS is not None
    fig, ax = page("03-6", "DAC switching glitches (glitch signal alone)", {
        "Mean toggled bits": f"{np.mean(toggles):.2f} of {params.DAC_BITS}",
        "Major-carry area": "off" if not on else f"{params.DAC_GLITCH_IMPULSE_VS * 1e12:g} pV*s",
        "SNR (glitches only)": "off" if not on else f"{_snr_db(d['dac']['held'], d['dac']['glitched']):.1f} dB",
        "Peak glitch": f"{np.max(np.abs(glitch)) / d['full_scale'] * 1e3:.3f} mFS",
    }, "Glitch signal = output with glitches minus the clean staircase. At each code change "
       "the bits switch at slightly different times; the area of the transient grows with the "
       "number of toggled bits (counted on the digital codes) up to the datasheet glitch "
       "impulse at the major carry (mid-scale).")
    ax.plot(k / fs_a * 1e9, glitch[k] / d["full_scale"] * 1e3, color=CLOCK)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Glitch (mFS)")
    ax.set_title("Glitches")
    return fig


def page_dac_settling(d, page):
    m, fs_a = d["m"], d["fs_a"]
    held, glitched = d["dac"]["held"], d["dac"]["glitched"]
    settled = d["dac"]["output"]
    n = _window(d, 12)
    k = np.arange(n[0] * m, (n[-1] + 1) * m)
    ts, acc = params.DAC_SETTLING_TIME_S, params.DAC_SETTLING_ACCURACY
    values = {"Settling": "off" if ts is None else f"{ts * 1e9:g} ns to {acc * 100:g} %"}
    if ts is not None:
        tau = ts / np.log(1 / acc)
        values["Time constant tau"] = f"{tau * 1e9:.3f} ns"
        values["-3 dB bandwidth"] = f"{1 / (2 * np.pi * tau) / 1e6:.1f} MHz"
        values["Phase at fc"] = f"{-np.degrees(np.arctan(2 * np.pi * d['fc'] * tau)):.2f} deg"
        values["Settling / clock period"] = f"{ts * params.SAMPLE_RATE:.2f}"
    fig, ax = page("03-7", "DAC settling (single-pole response) and output noise", values,
                   "Blue: staircase with glitches before settling. Red: final DAC output "
                   "(settling and output noise added). The output approaches each new level "
                   "exponentially and reaches the datasheet accuracy after the settling time; "
                   "if the settling time is close to the clock period, the output never fully "
                   "settles and the signal is low-pass filtered (gain and phase change at fc).")
    ax.step(k / fs_a * 1e9, glitched[k] / d["full_scale"], where="post", color=DAC,
            linewidth=0.8)
    ax.plot(k / fs_a * 1e9, settled[k] / d["full_scale"], color=OUTPUT, linewidth=1.4)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude (FS)")
    ax.set_title("Settling")
    return fig


def page_dac_analog_waveform(d, page):
    m, fs_a, fs = d["m"], d["fs_a"], d["full_scale"]
    ideal, real = d["ideal_held"], d["dac"]["output"]
    n = _window(d, 40)
    k = np.arange(n[0] * m, (n[-1] + 1) * m)
    difference = real - ideal
    tau = (None if params.DAC_SETTLING_TIME_S is None else
           params.DAC_SETTLING_TIME_S / np.log(1 / params.DAC_SETTLING_ACCURACY))
    fig, ax = page("03-8", "Analog DAC output: ideal DAC vs real DAC (all effects)", {
        "RMS difference": f"{np.sqrt(np.mean(difference ** 2)) / fs * 1e3:.2f} mFS",
        "Peak difference": f"{np.max(np.abs(difference)) / fs * 1e3:.2f} mFS",
        "Settling lag (tau)": "off" if tau is None else f"{tau * 1e9:.2f} ns",
        "EVM after VSA": f"{d['meas']['evm_percent']:.3f} %",
        "Analog grid": f"{fs_a / 1e6:g} MSa/s",
    }, "What leaves the DAC pins before any filter. Black: ideal DAC (ideal IF held for one "
       "clock period). Red: real DAC with every effect of 03 on (quantization, HD2/HD3, "
       "offset/gain, clock, glitches, settling, output noise). Most of the difference is the "
       "settling lag and the gain/offset error, which the VSA removes as a delay, gain and "
       "phase; what it cannot remove is the EVM. The staircase and its images are removed by "
       "the reconstruction filter (04).")
    t = k / fs_a * 1e9
    ax.step(t, ideal[k] / fs, where="post", color=IDEAL, linewidth=0.8, label="ideal DAC")
    ax.plot(t, real[k] / fs, color=OUTPUT, linewidth=1.2, label="real DAC (all effects)")
    _legend_on_top(ax)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude (FS)")
    ax.set_title("DAC output, 40 clock periods")
    return fig


def page_dac_output_spectrum(d, page):
    fs, fs_a = params.SAMPLE_RATE, d["fs_a"]
    f, psd = _psd(d["dac"]["output"], fs_a, d)
    _, psd_i = _psd(d["ideal_held"], fs_a, d)
    fc, hb = d["fc"] / 1e6, d["half_band"] / 1e6
    channel = _band_power_dbfs(f, psd, fc - hb, fc + hb)
    values = {"Channel power": f"{channel:.2f} dBFS"}
    for k, sign in ((1, -1), (1, 1), (2, -1), (2, 1)):
        center = k * fs / 1e6 + sign * fc
        image = _band_power_dbfs(f, psd, center - hb, center + hb)
        theory = 20 * np.log10(abs(np.sinc(center * 1e6 / fs)) / abs(np.sinc(fc * 1e6 / fs)))
        values[f"Image {center:g} MHz"] = f"{image - channel:.1f} dBc (sinc {theory:.1f})"
    values["Noise floor"] = f"{np.median(psd[(f > fc + 10) & (f < fs / 1e6 - 10)]):.1f} dBFS/Hz"
    fig, ax = page("03-9", "DAC output spectrum: ideal DAC vs real DAC (all effects)", values,
                   "Full DAC output spectrum up to Fs_a/2. Red: real DAC with every effect on; "
                   "black: ideal DAC (ideal signal, only the hold). Images of the signal appear at "
                   "k Fs +- fc, shaped by the sinc of the hold (theory in the values) and by the settling "
                   "low-pass. Their level relative to the channel tells how much attenuation the "
                   "reconstruction filter (04-01) needs. The flat floor is the DAC NSD.")
    ax.plot(f, psd, color=OUTPUT, label="real DAC output (03-01 ... 03-07)")
    ax.plot(f, psd_i, color=IDEAL, linewidth=0.6, label="ideal DAC (hold only)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, fs_a / 2e6)
    _psd_axes(ax, "PSD 0 ... Fs_a/2", -190, np.max(psd) + 10)
    return fig


# ------------------------------------------------------------------ 04 analog output

def page_analog_output(d, page):
    m, fs_a, fs = d["m"], d["fs_a"], d["full_scale"]
    n = _window(d, 200)
    k = np.arange(n[0] * m, (n[-1] + 1) * m)
    same = np.array_equal(d["tx"], d["dac"]["output"])
    fig, ax = page("04-1", "Transmitter analog output (after 04 analog part)", {
        "Reconstruction filter": "not modelled yet",
        "Output stage (volts)": "not modelled yet",
        "Analog noise": "not modelled yet",
        "Output = DAC output": "yes" if same else "no",
        "Full scale (later)": f"{params.DAC_FULL_SCALE_VOLTAGE:g} V peak",
    }, "Signal at the transmitter output connector, the input of the instruments. Black: ideal "
       "IF samples (01). Red: output of the analog part. The 04 blocks (reconstruction filter, "
       "output stage, amplifier noise) are not written yet, so the output is still the DAC "
       "staircase. With the filter in place the red trace becomes a smooth IF that follows the "
       "black samples with a fixed delay, and the images of 03-9 disappear.")
    ax.plot(k / fs_a * 1e6, d["tx"][k] / fs, color=OUTPUT, linewidth=0.8, label="transmitter output")
    ax.plot(n / params.SAMPLE_RATE * 1e6, d["r"]["if_samples"][n] / fs, ".", color=IDEAL,
            markersize=3, label="ideal IF samples (01)")
    _legend_on_top(ax)
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Amplitude (FS)")
    ax.set_title("Output waveform, 2 symbols")
    return fig


# ------------------------------------------------------------------ 05 instruments

def page_tx_channel(d, page):
    f, psd = d["meas"]["frequencies"] / 1e6, d["meas"]["psd_dbfs_hz"]
    fc, hb = d["fc"] / 1e6, d["half_band"] / 1e6
    channel = _band_power_dbfs(f, psd, fc - hb, fc + hb)
    lower = _band_power_dbfs(f, psd, fc - 3 * hb, fc - hb) - channel
    upper = _band_power_dbfs(f, psd, fc + hb, fc + 3 * hb) - channel
    f_i, psd_i = _psd(d["ideal_held"], d["fs_a"], d)
    ideal_channel = _band_power_dbfs(f_i, psd_i, fc - hb, fc + hb)
    ideal_acpr = max(_band_power_dbfs(f_i, psd_i, fc - 3 * hb, fc - hb),
                     _band_power_dbfs(f_i, psd_i, fc + hb, fc + 3 * hb)) - ideal_channel
    fig, ax = page("05-1", "Transmitter output: channel and adjacent channels", {
        "Channel power": f"{channel:.2f} dBFS",
        "ACPR lower": f"{lower:.1f} dBc",
        "ACPR upper": f"{upper:.1f} dBc",
        "ACPR ideal DAC": f"{ideal_acpr:.1f} dBc (worse side)",
        "Channel BW": f"{2 * hb:.3f} MHz",
        "Target (LTE-based)": "ACLR 45 dB",
    }, "What the spectrum analyzer shows at the transmitter output around fc. Shaded: the "
       "channel and the two adjacent channels of the same bandwidth. ACPR = adjacent channel "
       "power relative to the channel power (preliminary; the RF metrics block 05-03 will "
       "measure ACLR per the rescaled LTE definition). Black: ideal DAC for comparison; when both ACPR values are equal, the "
       "adjacent-channel level is set by the RRC filter (span), not by the hardware.")
    ax.plot(f, psd, color=TX, label="transmitter output")
    ax.plot(f_i, psd_i, color=IDEAL, linewidth=0.6, label="ideal DAC")
    ax.legend(loc="upper right")
    for offset in (-2 * hb, 0, 2 * hb):
        ax.axvspan(fc + offset - hb, fc + offset + hb, color=GRID, alpha=0.15)
    ax.set_xlim(fc - 5 * hb, fc + 5 * hb)
    _psd_axes(ax, "PSD around fc", np.max(psd) - 100, np.max(psd) + 10)
    return fig


def page_vsa_constellation(d, page):
    rx, ref = d["meas"]["symbols"], d["r"]["normalized_symbols"]
    evm = d["meas"]["evm_percent"]
    sync, ideal_sync = d["meas"]["sync"], d["ideal_sync"]
    fig, ax = page("05-2", "Vector signal analyzer: measured constellation", {
        "EVM (RMS)": f"{evm:.3f} %",
        "MER": f"{-20 * np.log10(evm / 100):.1f} dB",
        "Limit (LTE 16-QAM)": "12.5 %",
        "Gain vs ideal": f"{sync['gain'] / ideal_sync['gain']:.4f}",
        "Phase vs ideal": f"{sync['phase_deg'] - ideal_sync['phase_deg']:+.3f} deg",
        "Delay vs ideal": f"{(sync['delay_samples'] - ideal_sync['delay_samples']) / d['fs_a'] * 1e9:+.2f} ns",
    }, "Symbols recovered by the virtual VSA from the transmitter output: downconversion, RRC "
       "matched filter, timing from the known symbols, least-squares gain/phase correction. "
       "Gain, phase and delay are given relative to the ideal signal on the same analog grid, "
       "so they show what the hardware changed (e.g. gain error, settling phase).")
    ax.scatter(rx.real, rx.imag, s=1, alpha=0.3, color=TX, rasterized=True)
    points = np.unique(ref)
    ax.scatter(points.real, points.imag, s=40, marker="+", color=IDEAL, zorder=3)
    _decision_lines(ax)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_xlabel("In-phase I")
    ax.set_ylabel("Quadrature Q")
    ax.set_title("Measured symbols (+ = ideal)")
    return fig


def page_vsa_error(d, page):
    rx, ref = d["meas"]["symbols"], d["r"]["normalized_symbols"]
    error = np.abs(rx - ref) * 100
    fig, ax = page("05-3", "Vector signal analyzer: error vector per symbol", {
        "EVM RMS": f"{d['meas']['evm_percent']:.3f} %",
        "EVM peak": f"{np.max(error):.3f} %",
        "EVM 95th percentile": f"{np.percentile(error, 95):.3f} %",
        "Symbol errors": f"{int(np.sum(np.abs(rx - ref) > 1 / np.sqrt(10)))}",
    }, "Magnitude of the error vector of every symbol in % of Es = 1. A flat cloud means "
       "stationary impairments (noise, ISI); trends or bursts would point to slow effects such "
       "as phase noise or a frequency offset.")
    ax.plot(error, ".", color=TX, markersize=1.5, rasterized=True)
    ax.axhline(d["meas"]["evm_percent"], color=IDEAL, linestyle="--")
    ax.set_xlabel("Symbol index")
    ax.set_ylabel("|error| (% of Es = 1)")
    ax.set_title("Error vector magnitude")
    return fig


def page_evm_sweep(d, page):
    x, measured, theory = d["sweep_x"], d["sweep_measured"], d["sweep_theory"]
    limit = 12.5
    crossing = np.interp(np.log10(limit), np.log10(measured), x) if measured[-1] > limit else None
    fig, ax = page("05-4", "EVM versus DAC noise spectral density", {
        "EVM floor": f"{measured[0]:.2f} %",
        "NSD at 12.5 % EVM": "not reached" if crossing is None else f"{crossing:.1f} dBFS/Hz",
        "Current NSD": f"{params.DAC_NSD_DBFS_HZ} dBFS/Hz",
        "Max model/theory": f"{np.max(np.abs(measured / theory - 1)) * 100:.1f} %",
    }, "EVM from the full hardware chain while the DAC noise density is swept, with the theory "
       "EVM^2 = EVM_floor^2 + N0 Rs / P_signal. The NSD where the EVM reaches the 12.5 % "
       "limit is the loosest acceptable DAC noise; a real design keeps margin below it.")
    ax.semilogy(x, theory, "-", color=IDEAL, label="theory")
    ax.semilogy(x, measured, "o", color=TX, markersize=4, label="model (VSA)")
    ax.axhline(limit, color=OUTPUT, linestyle="--", label="limit 12.5 %")
    ax.set_xlabel("DAC NSD (dBFS/Hz)")
    ax.set_ylabel("EVM (%)")
    ax.set_title("EVM sweep")
    ax.legend(loc="upper left")
    return fig


PAGES = [
    ("01-1_constellation", page_constellation),
    ("01-2_rrc_impulse_response", page_rrc_impulse),
    ("01-3_rrc_magnitude_response", page_rrc_magnitude),
    ("01-4_baseband_I", page_baseband_i),
    ("01-5_baseband_Q", page_baseband_q),
    ("01-6_ideal_if_waveform", page_ideal_if_waveform),
    ("01-7_ideal_if_spectrum", page_ideal_if_spectrum),
    ("02-1_fpga_dac_codes", page_fpga_waveform),
    ("02-2_fpga_error", page_fpga_error),
    ("02-3_fpga_spectrum", page_fpga_spectrum),
    ("02-4_fpga_error_spectrum", page_fpga_error_spectrum),
    ("03-1_dac_transfer_error", page_dac_transfer),
    ("03-2_dac_quantized_spectrum", page_dac_quantized_spectrum),
    ("03-3_dac_static_spectrum", page_dac_static_spectrum),
    ("03-4_dac_clock_error", page_dac_clock),
    ("03-5_dac_zero_order_hold", page_dac_zoh),
    ("03-6_dac_glitches", page_dac_glitch),
    ("03-7_dac_settling", page_dac_settling),
    ("03-8_dac_analog_output_waveform", page_dac_analog_waveform),
    ("03-9_dac_output_spectrum", page_dac_output_spectrum),
    ("04-1_transmitter_analog_output", page_analog_output),
    ("05-1_tx_channel_spectrum", page_tx_channel),
    ("05-2_vsa_constellation", page_vsa_constellation),
    ("05-3_vsa_error_vector", page_vsa_error),
    ("05-4_evm_vs_nsd", page_evm_sweep),
]
