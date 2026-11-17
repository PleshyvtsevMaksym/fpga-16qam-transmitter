"""
Figures of the 16-QAM transmitter model, in the order the signal passes the chain.

    1 Symbol mapping (01)          2 Pulse shaping (01)
    3 FPGA output, bit-true vs ideal (02)
    4 DAC transfer curve (03-01..03)   5 DAC clock (03-04..05)
    6 DAC output dynamics (03-06..07)  7 Instruments at the TX output (05)
    8 EVM sweep (05)                   (theory) AWGN constellation

Every figure shows the ideal reference (black) together with the realistic
signal after that step. Spectra are measured with the virtual spectrum
analyzer (compute_psd) in dBFS/Hz, the same units as the datasheet NSD.
Style: SciencePlots if installed (py -m pip install SciencePlots), else a
built-in equivalent. Call plt.show() once after creating the figures.
"""

import textwrap
from importlib import import_module

import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots  # noqa: F401  (registers the "science" styles)
    _BASE_STYLE = ["science", "grid", "no-latex"]
except ImportError:
    _BASE_STYLE = None

_instruments = import_module("05_measurement_instruments")
compute_spectrum = _instruments.compute_spectrum
compute_psd = _instruments.compute_psd
qam16_gray_demapping = import_module("01_ideal_reference").qam16_gray_demapping
add_awgn = import_module("04_analog_output").add_awgn

# Okabe-Ito colour-blind safe palette, one colour per chain step
IDEAL_COLOR = "#000000"
FPGA_COLOR = "#0072B2"
DAC_COLOR = "#56B4E9"
CLOCK_COLOR = "#CC79A7"
OUTPUT_COLOR = "#D55E00"
TX_COLOR = "#009E73"
Q_COLOR = "#E69F00"
MARK_COLOR = "#999999"


def apply_plot_style() -> None:
    """Common publication style for all figures."""

    if _BASE_STYLE is not None:
        plt.style.use(_BASE_STYLE)
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "figure.constrained_layout.use": True,
        "font.family": "serif" if _BASE_STYLE else "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.titlesize": 11,
        "figure.titleweight": "bold",
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "axes.spines.top": bool(_BASE_STYLE),
        "axes.spines.right": bool(_BASE_STYLE),
        "lines.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })


def _psd(samples, sampling_rate, rbw, full_scale):
    """Spectrum analyzer probe: frequencies in MHz and PSD in dBFS/Hz."""
    frequencies, psd = compute_psd(samples, sampling_rate, rbw, full_scale)
    return frequencies / 1e6, psd


def _psd_axes(ax, title, floor_db=-180, top_db=-40):
    """Common labels and limits of a PSD panel."""
    ax.set_title(title)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("PSD (dBFS/Hz)")
    ax.set_ylim(floor_db, top_db)
    ax.legend(loc="upper right")


def _mark_band(ax, center_mhz, half_band_mhz):
    """Shade the occupied signal band."""
    ax.axvspan(center_mhz - half_band_mhz, center_mhz + half_band_mhz,
               color=MARK_COLOR, alpha=0.15, linewidth=0)


def plot_constellation(normalized_symbols: np.ndarray):
    """
    Plot the Gray-coded 16-QAM alphabet at Es = 1 with its bit labels.

    Parameters
    ----------
    normalized_symbols : np.ndarray
        Symbols normalized to Es = 1.
    """

    # 1. Distinct points and their Gray labels (integer levels = point * sqrt(10))
    points = np.unique(normalized_symbols)
    levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)

    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    fig.suptitle("1  Symbol mapping: Gray-coded 16-QAM (Es = 1)")

    ax.scatter(points.real, points.imag, s=40, color=FPGA_COLOR, zorder=3)
    for point in points:
        integer = np.rint(point * np.sqrt(10))
        bits = qam16_gray_demapping([int(integer.real)], [int(integer.imag)])
        ax.annotate("".join(str(b) for b in bits), (point.real, point.imag),
                    textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)

    # 2. Decision boundaries halfway between levels
    for b in np.array([-2, 0, 2]) / np.sqrt(10):
        ax.axvline(b, color=MARK_COLOR, linewidth=0.6, linestyle="--")
        ax.axhline(b, color=MARK_COLOR, linewidth=0.6, linestyle="--")

    ax.set_xticks(levels)
    ax.set_yticks(levels)
    ax.set_xticklabels([f"{v:.3f}" for v in levels])
    ax.set_yticklabels([f"{v:.3f}" for v in levels])
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_xlabel("In-phase I")
    ax.set_ylabel("Quadrature Q")

    return fig


def plot_pulse_shaping(
    normalized_symbols: np.ndarray,
    pulse_shaped_symbols: np.ndarray,
    rrc_coefficients: np.ndarray,
    sample_rate: float,
    samples_per_symbol: int,
    symbol_rate: float,
    rolloff: float,
    first_symbol: int,
    num_symbols: int
):
    """
    Plot the RRC filter and the shaped I/Q waveform with the symbol values.

    Parameters
    ----------
    normalized_symbols, pulse_shaped_symbols : np.ndarray
        Symbols at Es = 1 and RRC output (01 pipeline).
    rrc_coefficients : np.ndarray
        RRC taps.
    sample_rate, symbol_rate : float
        Fs and Rs in Hz.
    samples_per_symbol : int
        Fs / Rs.
    rolloff : float
        RRC roll-off.
    first_symbol, num_symbols : int
        Symbols shown in the time panels.
    """

    # 1. Time window aligned to the symbol instants (group delay removed)
    delay = (len(rrc_coefficients) - 1) // 2
    k = np.arange(first_symbol, first_symbol + num_symbols)
    n = np.arange(k[0] * samples_per_symbol, (k[-1] + 1) * samples_per_symbol)
    t_us = n / sample_rate * 1e6
    t_sym_us = k * samples_per_symbol / sample_rate * 1e6
    shaped = pulse_shaped_symbols[n + delay]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle(f"2  Pulse shaping: RRC a = {rolloff}, {len(rrc_coefficients)} taps, "
                 f"{samples_per_symbol} samples/symbol")

    # 2. Shaped I and Q with the symbol values at the symbol instants
    for ax, part, name, color in ((axes[0, 0], np.real, "I", FPGA_COLOR),
                                  (axes[0, 1], np.imag, "Q", Q_COLOR)):
        ax.plot(t_us, part(shaped), color=color, label=f"RRC output {name}")
        ax.plot(t_sym_us, part(normalized_symbols[k]) * rrc_coefficients[delay], "o",
                color=IDEAL_COLOR, markersize=4, label="symbol x h[0]")
        ax.set_title(f"Baseband {name} (group delay {delay / sample_rate * 1e6:.1f} us removed)")
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")

    # 3. Impulse response
    ax_h, ax_f = axes[1]
    t_symbols = (np.arange(len(rrc_coefficients)) - delay) / samples_per_symbol
    ax_h.plot(t_symbols, rrc_coefficients, color=FPGA_COLOR)
    ax_h.plot(t_symbols[::samples_per_symbol], rrc_coefficients[::samples_per_symbol],
              "o", color=IDEAL_COLOR, markersize=3, label="symbol-spaced taps")
    ax_h.set_title("Impulse response")
    ax_h.set_xlabel("Time (symbol periods)")
    ax_h.set_ylabel("h[n]")
    ax_h.legend(loc="upper right")

    # 4. Magnitude response, 0 dB at DC, with the band edges
    padded = np.zeros(1 << 16)
    padded[:len(rrc_coefficients)] = rrc_coefficients
    frequencies, response_db = compute_spectrum(padded, sample_rate)
    response_db -= response_db[frequencies == 0][0]
    positive = frequencies >= 0
    ax_f.plot(frequencies[positive] / 1e6, response_db[positive], color=FPGA_COLOR)
    for f_edge, style, label in (((1 - rolloff) * symbol_rate / 2, ":", "(1-a)Rs/2"),
                                 (symbol_rate / 2, "--", "Rs/2"),
                                 ((1 + rolloff) * symbol_rate / 2, "-.", "(1+a)Rs/2")):
        ax_f.axvline(f_edge / 1e6, color=IDEAL_COLOR, linestyle=style, linewidth=0.8,
                     label=f"{label} = {f_edge / 1e6:.3f} MHz")
    ax_f.set_xlim(0, 3 * symbol_rate / 1e6)
    ax_f.set_ylim(-100, 5)
    ax_f.set_title("Magnitude response")
    ax_f.set_xlabel("Frequency (MHz)")
    ax_f.set_ylabel("|H(f)| (dB)")
    ax_f.legend(loc="upper right")

    return fig


def plot_fpga_output(
    ideal_if: np.ndarray,
    fpga_if: np.ndarray,
    sample_rate: float,
    full_scale: float,
    rbw: float,
    start_sample: int,
    duration: float
):
    """
    Compare the bit-true FPGA IF output with the ideal floating-point IF.

    Parameters
    ----------
    ideal_if, fpga_if : np.ndarray
        Ideal and bit-true IF samples, same units and indexing.
    sample_rate : float
        Fs in Hz.
    full_scale : float
        DAC full scale (0 dBFS), signal units.
    rbw : float
        Spectrum analyzer RBW in Hz.
    start_sample : int
        First sample of the time window.
    duration : float
        Time window length in s.
    """

    # 1. Time window and error
    n = np.arange(start_sample, start_sample + int(round(duration * sample_rate)))
    t_us = n / sample_rate * 1e6
    error = fpga_if - ideal_if
    snr_db = 10 * np.log10(np.mean(ideal_if ** 2) / np.mean(error ** 2))

    fig, (ax_t, ax_e, ax_f) = plt.subplots(3, 1, figsize=(10, 7.5),
                                           gridspec_kw={"height_ratios": [2, 1, 2]})
    fig.suptitle(f"3  FPGA output: bit-true vs ideal (SNR {snr_db:.1f} dB)")

    # 2. Waveform: ideal line, FPGA samples
    ax_t.plot(t_us, ideal_if[n] / full_scale, color=IDEAL_COLOR, label="ideal (float)")
    ax_t.plot(t_us, fpga_if[n] / full_scale, ".", color=FPGA_COLOR, markersize=3,
              label="bit-true FPGA")
    ax_t.set_xlabel("Time (us)")
    ax_t.set_ylabel("Amplitude (FS)")
    ax_t.set_title("IF waveform")
    ax_t.legend(loc="upper right", ncol=2)

    # 3. Fixed-point error in LSB-scale units of full scale
    ax_e.plot(t_us, error[n] / full_scale * 1e3, color=FPGA_COLOR)
    ax_e.set_ylabel("Error (mFS)")
    ax_e.set_xlabel("Time (us)")
    ax_e.set_title("FPGA - ideal")

    # 4. Spectrum
    f, psd_ideal = _psd(ideal_if, sample_rate, rbw, full_scale)
    _, psd_fpga = _psd(fpga_if, sample_rate, rbw, full_scale)
    ax_f.plot(f, psd_fpga, color=FPGA_COLOR, label="bit-true FPGA")
    ax_f.plot(f, psd_ideal, color=IDEAL_COLOR, linewidth=0.8, label="ideal (float)")
    ax_f.set_xlim(0, sample_rate / 2e6)
    _psd_axes(ax_f, f"Spectrum, RBW {rbw / 1e3:g} kHz", floor_db=-200)

    return fig


def plot_dac_transfer(
    ideal_if: np.ndarray,
    quantized: np.ndarray,
    static: np.ndarray,
    sample_rate: float,
    full_scale: float,
    rbw: float,
    intermediate_frequency: float,
    half_band: float
):
    """
    Show the static DAC transfer curve: quantization, HD2/HD3, offset/gain.

    Parameters
    ----------
    ideal_if : np.ndarray
        Ideal IF samples.
    quantized, static : np.ndarray
        DAC output after quantization and after nonlinearity + offset/gain.
    sample_rate : float
        Fs in Hz.
    full_scale : float
        DAC full scale, signal units.
    rbw : float
        Spectrum analyzer RBW in Hz.
    intermediate_frequency, half_band : float
        fc and (1 + a) Rs / 2 in Hz.
    """

    fig, (ax_c, ax_f) = plt.subplots(1, 2, figsize=(11, 4.4),
                                     gridspec_kw={"width_ratios": [1, 2]})
    fig.suptitle("4  DAC transfer curve: quantization, nonlinearity, offset/gain")

    # 1. Transfer error vs input level (every 7th sample, enough to see the curve)
    x = ideal_if[::7] / full_scale
    ax_c.plot(x, (quantized[::7] / full_scale - x) * 1e3, ",", color=DAC_COLOR,
              alpha=0.5, label="quantized")
    ax_c.plot(x, (static[::7] / full_scale - x) * 1e3, ",", color=OUTPUT_COLOR,
              label="+ HD + offset/gain")
    ax_c.set_xlabel("Input (FS)")
    ax_c.set_ylabel("Output - input (mFS)")
    ax_c.set_title("Transfer error")
    ax_c.legend(loc="upper left", markerscale=20)

    # 2. Spectrum with harmonic positions
    fc = intermediate_frequency / 1e6
    f, psd_ideal = _psd(ideal_if, sample_rate, rbw, full_scale)
    _, psd_q = _psd(quantized, sample_rate, rbw, full_scale)
    _, psd_s = _psd(static, sample_rate, rbw, full_scale)
    ax_f.plot(f, psd_s, color=OUTPUT_COLOR, label="+ HD + offset/gain")
    ax_f.plot(f, psd_q, color=DAC_COLOR, label="quantized")
    ax_f.plot(f, psd_ideal, color=IDEAL_COLOR, linewidth=0.8, label="ideal")
    for k in (2, 3):
        _mark_band(ax_f, k * fc, k * half_band / 1e6)
        ax_f.annotate(f"HD{k}", (k * fc, -60), ha="center", fontsize=8)
    ax_f.set_xlim(0, sample_rate / 2e6)
    _psd_axes(ax_f, f"Spectrum, RBW {rbw / 1e3:g} kHz", floor_db=-200)

    return fig


def plot_dac_clock(
    static: np.ndarray,
    clocked: np.ndarray,
    sample_rate: float,
    full_scale: float,
    rbw: float,
    intermediate_frequency: float,
    half_band: float
):
    """
    Show the clock impairments (jitter, phase noise) around the carrier.

    Parameters
    ----------
    static, clocked : np.ndarray
        DAC output before and after the clock effects.
    sample_rate : float
        Fs in Hz.
    full_scale : float
        DAC full scale, signal units.
    rbw : float
        Spectrum analyzer RBW in Hz.
    intermediate_frequency, half_band : float
        fc and (1 + a) Rs / 2 in Hz.
    """

    fc = intermediate_frequency / 1e6
    f, psd_before = _psd(static, sample_rate, rbw, full_scale)
    _, psd_after = _psd(clocked, sample_rate, rbw, full_scale)
    _, psd_error = _psd(clocked - static, sample_rate, rbw, full_scale)

    fig, (ax_f, ax_e) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("5  DAC clock: jitter and phase noise")

    # 1. Close-in spectrum
    ax_f.plot(f, psd_after, color=CLOCK_COLOR, label="with clock effects")
    ax_f.plot(f, psd_before, color=IDEAL_COLOR, linewidth=0.8, label="ideal clock")
    _mark_band(ax_f, fc, half_band / 1e6)
    ax_f.set_xlim(fc - 4 * half_band / 1e6, fc + 4 * half_band / 1e6)
    _psd_axes(ax_f, "Around fc", floor_db=-200)

    # 2. Clock-induced error alone over the whole band
    ax_e.plot(f, psd_error, color=CLOCK_COLOR)
    ax_e.set_xlim(0, sample_rate / 2e6)
    ax_e.set_title("Error added by the clock")
    ax_e.set_xlabel("Frequency (MHz)")
    ax_e.set_ylabel("PSD (dBFS/Hz)")

    return fig


def plot_dac_output_dynamics(
    ideal_if: np.ndarray,
    held: np.ndarray,
    glitched: np.ndarray,
    output: np.ndarray,
    sample_rate: float,
    oversampling: int,
    full_scale: float,
    rbw: float,
    intermediate_frequency: float,
    start_sample: int,
    num_dac_samples: int
):
    """
    Show the DAC output on the analog grid: ZOH, glitches, settling, noise.

    Parameters
    ----------
    ideal_if : np.ndarray
        Ideal IF samples at Fs.
    held, glitched, output : np.ndarray
        ZOH staircase, with glitches, final DAC output, at Fs_a = M * Fs.
    sample_rate : float
        Fs in Hz.
    oversampling : int
        Analog grid factor M.
    full_scale : float
        DAC full scale, signal units.
    rbw : float
        Spectrum analyzer RBW in Hz.
    intermediate_frequency : float
        fc in Hz.
    start_sample, num_dac_samples : int
        DAC samples shown in the time panel.
    """

    analog_rate = oversampling * sample_rate
    n = np.arange(start_sample, start_sample + num_dac_samples)
    m = np.arange(start_sample * oversampling, (start_sample + num_dac_samples) * oversampling)

    fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(10, 7.5))
    fig.suptitle(f"6  DAC output: zero-order hold, glitches, settling, noise "
                 f"(analog grid {analog_rate / 1e6:.0f} MSa/s)")

    # 1. Time domain in ns
    ax_t.step(m / analog_rate * 1e9, held[m] / full_scale, where="post",
              color=DAC_COLOR, label="ZOH staircase")
    ax_t.plot(m / analog_rate * 1e9, glitched[m] / full_scale, color=CLOCK_COLOR,
              linewidth=0.8, label="+ glitches")
    ax_t.plot(m / analog_rate * 1e9, output[m] / full_scale, color=OUTPUT_COLOR,
              linewidth=1.4, label="DAC output (settling + noise)")
    ax_t.plot(n / sample_rate * 1e9, ideal_if[n] / full_scale, "o", color=IDEAL_COLOR,
              markersize=3.5, label="ideal samples")
    ax_t.set_xlabel("Time (ns)")
    ax_t.set_ylabel("Amplitude (FS)")
    ax_t.set_title("Waveform")
    ax_t.legend(loc="upper right", ncol=2)

    # 2. Spectrum up to Fs_a/2 with sinc envelope and images
    f, psd_out = _psd(output, analog_rate, rbw, full_scale)
    f_dac, psd_ideal = _psd(ideal_if, sample_rate, rbw, full_scale)
    ax_f.plot(f, psd_out, color=OUTPUT_COLOR, label="DAC output")
    ax_f.plot(f_dac, psd_ideal, color=IDEAL_COLOR, linewidth=0.8, label="ideal (0 ... Fs/2)")
    peak = np.max(psd_ideal)
    sinc_db = 20 * np.log10(np.maximum(np.abs(np.sinc(f * 1e6 / sample_rate)), 1e-6))
    ax_f.plot(f, peak + sinc_db, "--", color=MARK_COLOR, label="sinc(f/Fs) envelope")
    fs_mhz, fc = sample_rate / 1e6, intermediate_frequency / 1e6
    for k in range(1, oversampling // 2 + 1):
        ax_f.axvline(k * fs_mhz, color=MARK_COLOR, linewidth=0.5, linestyle=":")
    ax_f.set_xlim(0, analog_rate / 2e6)
    _psd_axes(ax_f, f"Spectrum, images at k Fs +/- fc = k {fs_mhz:g} +/- {fc:g} MHz",
              floor_db=-180, top_db=peak + 10)

    return fig


def plot_measurement(
    frequencies: np.ndarray,
    psd_dbfs_hz: np.ndarray,
    received_symbols: np.ndarray,
    reference_symbols: np.ndarray,
    evm_percent: float,
    intermediate_frequency: float,
    symbol_rate: float,
    rolloff: float,
    hardware_label: str
):
    """
    Show what the lab instruments report at the transmitter output.

    Parameters
    ----------
    frequencies, psd_dbfs_hz : np.ndarray
        Spectrum analyzer result (05 pipeline).
    received_symbols, reference_symbols : np.ndarray
        VSA symbols and ideal symbols at Es = 1.
    evm_percent : float
        RMS EVM in percent.
    intermediate_frequency, symbol_rate : float
        fc and Rs in Hz.
    rolloff : float
        RRC roll-off.
    hardware_label : str
        Short description of the hardware parameters.
    """

    fc = intermediate_frequency / 1e6
    half_band = (1 + rolloff) * symbol_rate / 2e6
    error = np.abs(received_symbols - reference_symbols) * 100

    fig = plt.figure(figsize=(12, 4.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.4, 1, 1.2])
    ax_f, ax_c, ax_e = (fig.add_subplot(grid[0, i]) for i in range(3))
    fig.suptitle(f"7  Transmitter output, virtual instruments: EVM {evm_percent:.2f} %\n"
                 + textwrap.fill(hardware_label, 150), fontsize=9)

    # 1. Spectrum analyzer: channel and adjacent channels
    ax_f.plot(frequencies / 1e6, psd_dbfs_hz, color=TX_COLOR)
    for offset in (-2 * half_band, 0, 2 * half_band):
        _mark_band(ax_f, fc + offset, half_band)
    ax_f.set_xlim(fc - 5 * half_band, fc + 5 * half_band)
    ax_f.set_title("Spectrum analyzer (channel and adjacent)")
    ax_f.set_xlabel("Frequency (MHz)")
    ax_f.set_ylabel("PSD (dBFS/Hz)")

    # 2. VSA constellation
    ideal = np.unique(reference_symbols)
    ax_c.scatter(received_symbols.real, received_symbols.imag, s=1, alpha=0.3,
                 color=TX_COLOR, rasterized=True, label="measured")
    ax_c.scatter(ideal.real, ideal.imag, s=25, marker="+", color=IDEAL_COLOR,
                 zorder=3, label="ideal")
    for b in np.array([-2, 0, 2]) / np.sqrt(10):
        ax_c.axvline(b, color=MARK_COLOR, linewidth=0.5, linestyle="--")
        ax_c.axhline(b, color=MARK_COLOR, linewidth=0.5, linestyle="--")
    ax_c.set_aspect("equal")
    ax_c.set_xlim(-1.3, 1.3)
    ax_c.set_ylim(-1.3, 1.3)
    ax_c.set_xlabel("In-phase I")
    ax_c.set_ylabel("Quadrature Q")
    ax_c.set_title("VSA constellation")
    ax_c.legend(loc="upper right", markerscale=3)

    # 3. Error vector magnitude per symbol
    ax_e.plot(error, ",", color=TX_COLOR, rasterized=True)
    ax_e.axhline(evm_percent, color=IDEAL_COLOR, linestyle="--",
                 label=f"RMS EVM {evm_percent:.2f} %")
    ax_e.set_xlabel("Symbol index")
    ax_e.set_ylabel("|error| (% of Es = 1)")
    ax_e.set_title("Error vector per symbol")
    ax_e.legend(loc="upper right")

    return fig


def plot_evm_sweep(
    x_values: np.ndarray,
    evm_measured: np.ndarray,
    evm_theory: np.ndarray,
    x_label: str,
    evm_limit: float
):
    """
    Plot measured EVM against one swept hardware parameter, with theory.

    Parameters
    ----------
    x_values : np.ndarray
        Swept parameter values.
    evm_measured, evm_theory : np.ndarray
        Model (VSA) and theoretical EVM in percent.
    x_label : str
        Name and unit of the swept parameter.
    evm_limit : float
        Requirement line in percent.
    """

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    fig.suptitle(f"8  EVM vs {x_label.split(' (')[0]}")

    ax.semilogy(x_values, evm_theory, "-", color=IDEAL_COLOR, label="theory")
    ax.semilogy(x_values, evm_measured, "o", color=TX_COLOR, markersize=4,
                label="model (VSA)")
    ax.axhline(evm_limit, color=OUTPUT_COLOR, linestyle="--",
               label=f"requirement {evm_limit} %")
    ax.set_xlabel(x_label)
    ax.set_ylabel("EVM (%)")
    ax.legend(loc="upper left")

    return fig


def plot_noisy_constellation(
    normalized_symbols: np.ndarray,
    snr_list: list,
    seed: int | None
):
    """
    Theory: 16-QAM symbols with AWGN at several SNR values (up to 4 panels).

    Parameters
    ----------
    normalized_symbols : np.ndarray
        Symbols at Es = 1.
    snr_list : list
        SNR values in dB.
    seed : int or None
        Noise seed.
    """

    ideal = np.unique(normalized_symbols)
    fig, axes = plt.subplots(1, len(snr_list), figsize=(3 * len(snr_list), 3.3))
    fig.suptitle("Theory: 16-QAM with AWGN")

    for ax, snr_db in zip(np.atleast_1d(axes), snr_list):
        noisy = add_awgn(normalized_symbols, snr_db, seed=seed)
        ax.scatter(noisy.real, noisy.imag, s=1, alpha=0.3, color=FPGA_COLOR, rasterized=True)
        ax.scatter(ideal.real, ideal.imag, s=20, marker="+", color=IDEAL_COLOR, zorder=3)
        for b in np.array([-2, 0, 2]) / np.sqrt(10):
            ax.axvline(b, color=MARK_COLOR, linewidth=0.5, linestyle="--")
            ax.axhline(b, color=MARK_COLOR, linewidth=0.5, linestyle="--")
        ax.set_title(f"SNR = {snr_db} dB")
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel("I")
    np.atleast_1d(axes)[0].set_ylabel("Q")

    return fig
