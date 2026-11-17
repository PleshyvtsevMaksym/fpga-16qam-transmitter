"""
Figures of the 16-QAM transmitter model.

Every function takes already calculated arrays, creates one figure and
returns it. Nothing here repeats the DSP processing; the RRC frequency
response uses the project's own compute_spectrum().

Where the signal exists both inside the FPGA and at the transmitter
output (IF waveform, IF spectrum), the figure shows the clean FPGA signal
together with the noisy transmitter output.
Call plt.show() once after creating the figures.
"""

import numpy as np
import matplotlib.pyplot as plt

from analysis import compute_spectrum
from fpga import qam16_gray_demapping
from hardware import add_awgn

I_COLOR = "tab:blue"
Q_COLOR = "tab:orange"
IF_COLOR = "tab:green"
MARK_COLOR = "tab:red"
NOISY_COLOR = "tab:purple"


def apply_plot_style() -> None:
    """Common appearance for all figures (readable at thesis page size)."""

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
        "lines.linewidth": 1.0,
        "legend.fontsize": 9,
        "figure.constrained_layout.use": True,
    })


def plot_constellation(symbols: np.ndarray, normalized_symbols: np.ndarray):
    """
    Plot the 16-QAM alphabet before and after energy normalization.

    Each distinct point is drawn once with its Gray bit label b0 b1 b2 b3
    (b0 b1 select I, b2 b3 select Q). The plot shows the alphabet,
    not how often each point occurs.

    Parameters
    ----------
    symbols : np.ndarray
        Complex symbols with integer levels {-3, -1, +1, +3}.
    normalized_symbols : np.ndarray
        The same symbols after normalization to Es = 1.
    """

    # 1. Distinct constellation points
    points = np.unique(symbols)
    points_normalized = np.unique(normalized_symbols)
    scale = np.max(np.abs(points_normalized.real)) / np.max(np.abs(points.real))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    fig.suptitle("Gray-Coded 16-QAM Constellation")

    titles = (
        "Integer levels (average Es = 10)",
        f"Normalized (Es = 1, levels / sqrt(10), scale {scale:.4f})",
    )

    # 2. Same layout for both panels; only the scale differs
    for ax, data, factor, title in zip(axes, (points, points_normalized), (1.0, scale), titles):
        ax.scatter(data.real, data.imag, s=45, color=I_COLOR, zorder=3)

        for point in points:
            bits = qam16_gray_demapping([int(point.real)], [int(point.imag)])
            label = "".join(str(bit) for bit in bits)
            ax.annotate(label, (point.real * factor, point.imag * factor),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8)

        levels = np.array([-3, -1, 1, 3]) * factor
        ax.set_xticks(levels)
        ax.set_yticks(levels)
        ax.set_xticklabels([f"{v:.3g}" for v in levels])
        ax.set_yticklabels([f"{v:.3g}" for v in levels])
        ax.axhline(0, color="black", linewidth=0.6)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_xlim(-4 * factor, 4 * factor)
        ax.set_ylim(-4 * factor, 4 * factor)
        ax.set_aspect("equal")
        ax.set_xlabel("In-phase (I)")
        ax.set_ylabel("Quadrature (Q)")
        ax.set_title(title)

    return fig


def plot_symbol_processing(
    normalized_symbols: np.ndarray,
    upsampled_symbols: np.ndarray,
    pulse_shaped_symbols: np.ndarray,
    rrc_length: int,
    sample_rate: float,
    samples_per_symbol: int,
    first_symbol: int = 0,
    num_symbols: int = 8,
):
    """
    Show how a few symbols become a sampled pulse-shaped waveform.

    Rows: symbol values, zero-insertion upsampled samples and the RRC
    output. The RRC output is shifted left by the filter group delay
    (rrc_length - 1) / 2 samples, so that every symbol instant lines up
    in all three rows.

    Parameters
    ----------
    normalized_symbols, upsampled_symbols, pulse_shaped_symbols : np.ndarray
        Pipeline arrays.
    rrc_length : int
        Number of RRC coefficients.
    sample_rate : float
        Sampling frequency in Hz.
    samples_per_symbol : int
        Upsampling factor.
    first_symbol : int, optional
        Index of the first symbol shown (default 0).
    num_symbols : int, optional
        Number of symbols shown (default 8).
    """

    # 1. Sample ranges and time axes in microseconds
    delay = (rrc_length - 1) // 2
    k = np.arange(first_symbol, first_symbol + num_symbols)
    n = np.arange(first_symbol * samples_per_symbol, (first_symbol + num_symbols) * samples_per_symbol)
    t_symbols = k * samples_per_symbol / sample_rate * 1e6
    t_samples = n / sample_rate * 1e6

    symbols = normalized_symbols[k]
    upsampled = upsampled_symbols[n]
    shaped = pulse_shaped_symbols[n + delay]
    shaped_at_symbols = pulse_shaped_symbols[k * samples_per_symbol + delay]

    fig, axes = plt.subplots(3, 2, figsize=(12, 7.5), sharex=True)
    fig.suptitle(f"Symbol Processing: symbols {first_symbol} to {first_symbol + num_symbols - 1}, "
                 f"{samples_per_symbol} samples per symbol")

    # 2. One column for I and one for Q
    for col, (part, name, color) in enumerate(
        ((np.real, "I", I_COLOR), (np.imag, "Q", Q_COLOR))
    ):
        ax_sym, ax_up, ax_shaped = axes[:, col]

        ax_sym.stem(t_symbols, part(symbols), linefmt=color, markerfmt="o", basefmt="k-")
        ax_sym.set_title(f"Normalized symbols ({name})")
        ax_sym.set_ylabel("Amplitude")

        ax_up.plot(t_samples, part(upsampled), ".", color="gray", markersize=2,
                   label="inserted zeros")
        ax_up.stem(t_symbols, part(upsampled[::samples_per_symbol]),
                   linefmt=color, markerfmt="o", basefmt="k-", label="symbol samples")
        ax_up.set_title(f"Zero-insertion upsampling ({name})")
        ax_up.set_ylabel("Amplitude")
        ax_up.legend(loc="best")

        ax_shaped.plot(t_samples, part(shaped), color=color, label="RRC output")
        ax_shaped.plot(t_symbols, part(shaped_at_symbols), "o", color=MARK_COLOR,
                       markersize=4, label="symbol instants")
        ax_shaped.set_title(f"Pulse-shaped ({name}), group delay {delay} samples "
                            f"= {delay / sample_rate * 1e6:.2f} us removed")
        ax_shaped.set_ylabel("Amplitude")
        ax_shaped.set_xlabel("Time (us)")
        ax_shaped.legend(loc="best")

    return fig


def plot_rrc_response(
    rrc_coefficients: np.ndarray,
    samples_per_symbol: int,
    symbol_rate: float,
    sample_rate: float,
    rolloff: float,
    fft_length: int = 2 ** 16,
):
    """
    Plot the RRC impulse response and its magnitude frequency response.

    The frequency response is the FFT of the zero-padded coefficients,
    normalized to 0 dB at DC. Vertical lines mark (1 - a) Rs / 2,
    Rs / 2 and (1 + a) Rs / 2.

    Parameters
    ----------
    rrc_coefficients : np.ndarray
        Actual filter coefficients from the pipeline.
    samples_per_symbol : int
        Samples per symbol.
    symbol_rate, sample_rate : float
        Rs and Fs in Hz.
    rolloff : float
        Roll-off factor a.
    fft_length : int, optional
        Zero-padded FFT length for a smooth response (default 65536).
    """

    # 1. Impulse response on a time axis in symbol periods
    center = (len(rrc_coefficients) - 1) // 2
    t = (np.arange(len(rrc_coefficients)) - center) / samples_per_symbol
    symbol_taps = np.arange(0, len(rrc_coefficients), samples_per_symbol)

    # 2. Frequency response from the zero-padded coefficients
    padded = np.zeros(fft_length)
    padded[:len(rrc_coefficients)] = rrc_coefficients
    frequencies, response_db = compute_spectrum(padded, sample_rate)
    response_db = response_db - response_db[frequencies == 0][0]

    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"RRC Filter: roll-off {rolloff}, {len(rrc_coefficients)} taps, "
                 f"{samples_per_symbol} samples per symbol, unit energy")

    ax_t.plot(t, rrc_coefficients, color=I_COLOR)
    ax_t.plot(t[symbol_taps], rrc_coefficients[symbol_taps], "o", color=MARK_COLOR,
              markersize=4, label="symbol-spaced taps")
    ax_t.set_title("Impulse response")
    ax_t.set_xlabel("Time (symbol periods)")
    ax_t.set_ylabel("h[n]")
    ax_t.legend(loc="upper right")

    # 3. One-sided frequency response with the band edges
    positive = frequencies >= 0
    ax_f.plot(frequencies[positive] / 1e6, response_db[positive], color=I_COLOR)

    edges = (
        ((1 - rolloff) * symbol_rate / 2, ":", "(1 - a) Rs/2"),
        (symbol_rate / 2, "--", "Rs/2 (-3 dB ideal)"),
        ((1 + rolloff) * symbol_rate / 2, "-.", "(1 + a) Rs/2"),
    )
    for f_edge, style, label in edges:
        ax_f.axvline(f_edge / 1e6, color=MARK_COLOR, linestyle=style,
                     label=f"{label} = {f_edge / 1e6:.3f} MHz")

    ax_f.set_xlim(0, 3 * symbol_rate / 1e6)
    ax_f.set_ylim(-90, 5)
    ax_f.set_title("Magnitude response (0 dB at DC)")
    ax_f.set_xlabel("Frequency (MHz)")
    ax_f.set_ylabel("|H(f)| (dB)")
    ax_f.legend(loc="upper right")

    return fig


def plot_if_waveform(
    pulse_shaped_symbols: np.ndarray,
    if_samples: np.ndarray,
    sample_rate: float,
    if_frequency: float,
    start_sample: int,
    duration: float = 2e-6,
    tx_output: np.ndarray | None = None,
):
    """
    Plot baseband I, baseband Q and the real IF waveform in one time window.

    If tx_output is given, the IF panel also shows the transmitter output
    (after the hardware pipeline) behind the clean FPGA signal.

    The IF panel also shows the envelope +/-|I + jQ|. Time is the absolute
    sample time n / Fs, the same index used by the IF modulator.

    Parameters
    ----------
    pulse_shaped_symbols : np.ndarray
        Complex baseband after the RRC filter.
    if_samples : np.ndarray
        Real IF samples.
    sample_rate, if_frequency : float
        Fs and fc in Hz.
    start_sample : int
        First sample of the window.
    duration : float, optional
        Window length in seconds (default 2 us = 10 carrier periods at 5 MHz).
    tx_output : np.ndarray, optional
        Transmitter output samples, same indexing as if_samples.
    """

    # 1. Time window
    num_samples = int(round(duration * sample_rate))
    n = np.arange(start_sample, start_sample + num_samples)
    t = n / sample_rate * 1e6
    baseband = pulse_shaped_symbols[n]

    fig, (ax_i, ax_q, ax_if) = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    fig.suptitle(f"Quadrature IF Modulation: fc = {if_frequency / 1e6:.3f} MHz, "
                 f"Fs = {sample_rate / 1e6:.1f} MSPS "
                 f"({sample_rate / if_frequency:.1f} samples per carrier period)")

    # 2. Baseband components
    ax_i.plot(t, baseband.real, color=I_COLOR)
    ax_i.set_title("Baseband I[n]")
    ax_i.set_ylabel("Amplitude")

    ax_q.plot(t, baseband.imag, color=Q_COLOR)
    ax_q.set_title("Baseband Q[n]")
    ax_q.set_ylabel("Amplitude")

    # 3. IF waveform with its envelope
    envelope = np.abs(baseband)
    if tx_output is not None:
        ax_if.plot(t, tx_output[n], color=NOISY_COLOR, alpha=0.6,
                   label="Transmitter output (with noise)")
    ax_if.plot(t, if_samples[n], ".-", color=IF_COLOR, markersize=3,
               label="x[n] = I cos(2 pi fc n/Fs) - Q sin(2 pi fc n/Fs)")
    ax_if.plot(t, envelope, "--", color="gray", label="+/-|I + jQ|")
    ax_if.plot(t, -envelope, "--", color="gray")
    ax_if.set_title("Real IF signal")
    ax_if.set_ylabel("Amplitude")
    ax_if.set_xlabel("Time (us)")
    ax_if.margins(y=0.4)    # room for the legend above the waveform
    ax_if.legend(loc="upper right", ncol=2)

    return fig


def plot_spectra(
    baseband_frequencies: np.ndarray,
    baseband_spectrum_db: np.ndarray,
    if_frequencies: np.ndarray,
    if_spectrum_db: np.ndarray,
    if_frequency: float,
    symbol_rate: float,
    rolloff: float,
    dynamic_range_db: float = 100.0,
    tx_spectrum_db: np.ndarray | None = None,
):
    """
    Plot the complex baseband spectrum and the real IF spectrum.

    If tx_spectrum_db is given, both IF panels also show the transmitter
    output spectrum behind the clean FPGA spectrum (noise floor, spurs).

    All panels show |X[k]| / N in dB re 1 from one FFT over the whole
    signal (rectangular window). This is an amplitude spectrum, not a
    calibrated PSD. The real IF signal has half the baseband amplitude
    in each of its two bands (-6 dB).

    Parameters
    ----------
    baseband_frequencies, baseband_spectrum_db : np.ndarray
        compute_spectrum() output for the complex baseband.
    if_frequencies, if_spectrum_db : np.ndarray
        compute_spectrum() output for the real IF signal.
    if_frequency, symbol_rate : float
        fc and Rs in Hz.
    rolloff : float
        Roll-off factor a; bands are marked at +/-(1 + a) Rs / 2.
    dynamic_range_db : float, optional
        Displayed range below the highest value (default 100 dB).
    tx_spectrum_db : np.ndarray, optional
        compute_spectrum() output for the transmitter output (same bins as IF).
    """

    # 1. Frequencies in MHz and common display range
    f_bb = baseband_frequencies / 1e6
    f_if = if_frequencies / 1e6
    fc = if_frequency / 1e6
    half_band = (1 + rolloff) * symbol_rate / 2 / 1e6
    top = max(np.max(baseband_spectrum_db), np.max(if_spectrum_db)) + 5

    fig, (ax_bb, ax_if, ax_zoom) = plt.subplots(3, 1, figsize=(11, 9.5))
    fig.suptitle(f"Amplitude Spectra: |X[k]|/N of one {len(f_if)}-point FFT, "
                 "rectangular window (not a PSD)")

    # 2. Complex baseband
    ax_bb.plot(f_bb, baseband_spectrum_db, color=I_COLOR)
    for edge in (-half_band, half_band):
        ax_bb.axvline(edge, color=MARK_COLOR, linestyle="--")
    ax_bb.set_xlim(-3 * half_band, 3 * half_band)
    ax_bb.set_title(f"Complex baseband (dashed: +/-(1 + a) Rs/2 = +/-{half_band:.3f} MHz)")

    # 3. Full two-sided IF spectrum (transmitter output behind the clean signal)
    if tx_spectrum_db is not None:
        for ax in (ax_if, ax_zoom):
            ax.plot(f_if, tx_spectrum_db, color=NOISY_COLOR, alpha=0.6,
                    label="Transmitter output (with noise)")
    ax_if.plot(f_if, if_spectrum_db, color=IF_COLOR, label="FPGA output (clean)")
    ax_if.set_xlim(f_if[0], -f_if[0])
    ax_if.set_title("Real IF signal, full range -Fs/2 to +Fs/2")
    ax_if.legend(loc="upper right")

    # 4. Positive IF band around fc
    ax_zoom.plot(f_if, if_spectrum_db, color=IF_COLOR)
    ax_zoom.axvline(fc, color="black", linestyle=":", label=f"fc = {fc:.3f} MHz")
    for edge in (fc - half_band, fc + half_band):
        ax_zoom.axvline(edge, color=MARK_COLOR, linestyle="--")
    ax_zoom.set_xlim(fc - 3 * half_band, fc + 3 * half_band)
    ax_zoom.set_title("Real IF signal, positive band around fc "
                      "(dashed: fc +/- (1 + a) Rs/2)")
    ax_zoom.legend(loc="upper right")

    for ax in (ax_bb, ax_if, ax_zoom):
        ax.set_ylim(top - dynamic_range_db, top)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Magnitude (dB re 1)")

    return fig


def plot_noisy_constellation(
    normalized_symbols: np.ndarray,
    snr_list: list,
    seed: int | None = None
):
    """
    Plot the normalized 16-QAM constellation with AWGN at several SNR values.

    Theory illustration: noise is added directly to the symbols to show
    how much noise 16-QAM tolerates. In real hardware the noise appears
    after the DAC (see the hardware pipeline).

    Each panel shows the noisy symbols, the ideal constellation points and
    the decision boundaries at -2/sqrt(10), 0 and +2/sqrt(10), which lie
    halfway between neighbouring levels -3, -1, +1, +3 (divided by sqrt(10)).

    Parameters
    ----------
    normalized_symbols : np.ndarray
        Complex symbols normalized to Es = 1.
    snr_list : list
        SNR values in dB, one panel per value (up to 4).
    seed : int, optional
        Seed for the noise generator, makes the figure repeatable.
    """

    # 1. One panel per SNR value
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("16-QAM Constellation with AWGN")

    # 2. Ideal points and decision boundaries
    ideal = np.unique(normalized_symbols)
    boundaries = np.array([-2, 0, 2]) / np.sqrt(10)

    for ax, snr_db in zip(axes.flat, snr_list):

        # 3. Add noise at this SNR
        noisy = add_awgn(normalized_symbols, snr_db, seed=seed)

        # 4. Noisy cloud and ideal points
        ax.scatter(noisy.real, noisy.imag, s=2, alpha=0.3,
                   color=I_COLOR, label="Noisy symbols")
        ax.scatter(ideal.real, ideal.imag, s=30,
                   color="black", zorder=3, label="Ideal points")

        # 5. Decision boundaries
        for b in boundaries:
            ax.axvline(b, color=MARK_COLOR, linewidth=0.8)
            ax.axhline(b, color=MARK_COLOR, linewidth=0.8)

        ax.set_title(f"SNR = {snr_db} dB")
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel("In-phase (I)")
        ax.set_ylabel("Quadrature (Q)")

    axes.flat[0].legend(loc="upper right", markerscale=3)

    return fig


def plot_received_constellation(
    received_symbols: np.ndarray,
    normalized_symbols: np.ndarray,
    evm_percent: float,
    hardware_label: str | None = None,
):
    """
    Plot the constellation measured at the transmitter output.

    The symbols come from the virtual analyzer (demodulate_if) applied to
    the output of the hardware pipeline, so they contain every modelled
    impairment. Red lines are the decision boundaries.

    Parameters
    ----------
    received_symbols : np.ndarray
        Complex symbols recovered by demodulate_if().
    normalized_symbols : np.ndarray
        Ideal transmitted symbols at Es = 1 (reference points).
    evm_percent : float
        RMS EVM of the received symbols, shown in the title.
    hardware_label : str, optional
        Short description of the hardware parameters, shown in the title.
    """

    # 1. Ideal points and decision boundaries
    ideal = np.unique(normalized_symbols)
    boundaries = np.array([-2, 0, 2]) / np.sqrt(10)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    title = f"Transmitter Output Constellation: EVM = {evm_percent:.2f} %"
    if hardware_label:
        title += f"\n{hardware_label}"
    fig.suptitle(title)

    # 2. Received cloud, ideal points, boundaries
    ax.scatter(received_symbols.real, received_symbols.imag, s=3, alpha=0.4,
               color=I_COLOR, label="Received symbols")
    ax.scatter(ideal.real, ideal.imag, s=30, color="black", zorder=3,
               label="Ideal points")
    for b in boundaries:
        ax.axvline(b, color=MARK_COLOR, linewidth=0.8, zorder=4)
        ax.axhline(b, color=MARK_COLOR, linewidth=0.8, zorder=4)

    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.legend(loc="upper right", markerscale=3)

    return fig


def plot_evm_sweep(
    x_values: np.ndarray,
    evm_measured: np.ndarray,
    evm_theory: np.ndarray,
    x_label: str,
    evm_limit: float | None = None,
):
    """
    Plot measured EVM against one swept hardware parameter, with theory.

    Parameters
    ----------
    x_values : np.ndarray
        Values of the swept parameter (for example DAC NSD in dBFS/Hz).
    evm_measured : np.ndarray
        EVM from the virtual analyzer, in percent.
    evm_theory : np.ndarray
        Expected EVM in percent, same length as x_values.
    x_label : str
        Name and unit of the swept parameter (x axis label).
    evm_limit : float, optional
        Requirement line in percent (for example 12.5 % for LTE 16-QAM).
    """

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.suptitle(f"EVM vs {x_label.split(' (')[0]} at the transmitter output")

    ax.semilogy(x_values, evm_theory, "-", color="gray", label="Theory")
    ax.semilogy(x_values, evm_measured, "o", color=I_COLOR, label="Model (analyzer)")
    if evm_limit is not None:
        ax.axhline(evm_limit, color=MARK_COLOR, linestyle="--",
                   label=f"Limit {evm_limit} %")

    ax.set_xlabel(x_label)
    ax.set_ylabel("EVM (%)")
    ax.legend(loc="upper left")

    return fig
