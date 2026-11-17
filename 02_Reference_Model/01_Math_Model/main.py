
"""Basic 16-QAM transmitter reference model and visualization."""

import matplotlib.pyplot as plt

from PRBS15 import prbs15_generator
from gray_mapping import qam16_gray_mapping, qam16_gray_demapping

from qam_waveform import (
    make_time_axis,
    hold_iq_symbols,
    generate_carriers,
    qam16_waveform,
)

# --------------------------------------------------
# Ideal Constellation
# --------------------------------------------------

def plot_qam16(i_symbols, q_symbols):
    """Plot the ideal 16-QAM constellation diagram."""

    if i_symbols is None or q_symbols is None:
        raise ValueError("I and Q symbols cannot be None.")

    if not i_symbols or len(i_symbols) != len(q_symbols):
        raise ValueError("I and Q must be non-empty and equal in length.")

    unique_points = sorted(set(zip(i_symbols, q_symbols)))

    unique_i = [point[0] for point in unique_points]
    unique_q = [point[1] for point in unique_points]

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        unique_i,
        unique_q,
        color="royalblue",
        s=90,
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
    )

    ax.set_title("Ideal 16-QAM Constellation")
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xticks([-3, -1, 0, 1, 3])
    ax.set_yticks([-3, -1, 0, 1, 3])

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.grid(True, linestyle="--", alpha=0.35)

    ax.text(
        0.02, 0.98,
        f"Total symbols: {len(i_symbols)}\n"
        f"Unique points: {len(unique_points)}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
    )

    fig.tight_layout()
    return fig


# --------------------------------------------------
# Time-Domain Visualization
# --------------------------------------------------

def plot_qam_waveform(
    time_axis,
    i_samples,
    q_samples,
    cosine_carrier,
    sine_carrier,
    waveform,
    samples_per_symbol,
    num_symbols_to_plot=4,
):
    """Visualize the time-domain quadrature modulation process."""

    if type(samples_per_symbol) is not int or samples_per_symbol <= 0:
        raise ValueError("Samples per symbol must be a positive integer.")

    if type(num_symbols_to_plot) is not int or num_symbols_to_plot <= 0:
        raise ValueError("Number of plotted symbols must be positive.")

    arrays = (
        time_axis,
        i_samples,
        q_samples,
        cosine_carrier,
        sine_carrier,
        waveform,
    )

    if any(array is None for array in arrays):
        raise ValueError("Plot inputs cannot be None.")

    if len(set(len(array) for array in arrays)) != 1 or not time_axis:
        raise ValueError("Plot arrays must be non-empty and equal in length.")

    if len(time_axis) % samples_per_symbol != 0:
        raise ValueError("Incomplete symbol samples.")

    num_samples = min(
        num_symbols_to_plot * samples_per_symbol,
        len(time_axis),
    )

    # Select samples and convert time to microseconds
    t = [value * 1e6 for value in time_axis[:num_samples]]

    i = i_samples[:num_samples]
    q = q_samples[:num_samples]

    cosine = cosine_carrier[:num_samples]
    sine = sine_carrier[:num_samples]

    signal = waveform[:num_samples]

    # Quadrature contributions
    i_component = [a * c for a, c in zip(i, cosine)]
    q_component = [-b * s for b, s in zip(q, sine)]

    fig, axes = plt.subplots(
        4, 1,
        figsize=(12, 11),
        sharex=True,
    )

    # I/Q symbol holding
    axes[0].step(t, i, where="post", label="I[n]", color="royalblue")
    axes[0].step(t, q, where="post", label="Q[n]", color="darkorange")
    axes[0].set_title("1. I/Q Symbol Holding")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_yticks([-3, -1, 0, 1, 3])
    axes[0].legend()

    # Orthogonal carriers
    axes[1].plot(t, cosine, label="Cosine", color="royalblue")
    axes[1].plot(t, sine, label="Sine", color="darkorange")
    axes[1].set_title("2. Orthogonal Carriers")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()

    # Modulated components
    axes[2].plot(t, i_component, label="I × cos", color="royalblue")
    axes[2].plot(t, q_component, label="−Q × sin", color="darkorange")
    axes[2].set_title("3. Quadrature Components")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend()

    # Combined IF signal
    axes[3].plot(
        t,
        signal,
        label="16-QAM IF",
        color="seagreen",
        linewidth=1.8,
    )
    axes[3].set_title("4. Combined 16-QAM Waveform")
    axes[3].set_xlabel("Time (µs)")
    axes[3].set_ylabel("Amplitude")
    axes[3].legend()

    # Symbol boundaries
    symbol_duration_us = (
        (time_axis[1] - time_axis[0])
        * samples_per_symbol
        * 1e6
    ) if len(time_axis) > 1 else None

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.35)

        if symbol_duration_us is not None:
            for k in range(1, (num_samples - 1) // samples_per_symbol + 1):
                ax.axvline(
                    k * symbol_duration_us,
                    color="gray",
                    linestyle=":",
                    linewidth=1,
                )

    fig.suptitle("16-QAM Time-Domain Modulation", fontsize=15)
    fig.tight_layout()

    return fig


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    """Run the basic 16-QAM transmitter model."""

    num_symbols = 4096
    symbol_rate = 1_000_000
    sample_rate = 100_000_000
    carrier_frequency = 5_000_000

    # Binary data source
    bits = prbs15_generator(num_symbols * 4)

    # Gray mapping
    i_symbols, q_symbols = qam16_gray_mapping(bits)

    # Time axis
    time_axis, sps = make_time_axis(
        num_symbols,
        symbol_rate,
        sample_rate,
    )

    # Rectangular I/Q symbol holding
    i_samples, q_samples = hold_iq_symbols(
        i_symbols,
        q_symbols,
        sps,
    )

    # Orthogonal carriers
    cosine_carrier, sine_carrier = generate_carriers(
        time_axis,
        carrier_frequency,
        sample_rate,
    )

    # Quadrature modulation
    waveform = qam16_waveform(
        i_samples,
        q_samples,
        cosine_carrier,
        sine_carrier,
    )

    # Basic verification
    recovered_bits = qam16_gray_demapping(i_symbols, q_symbols)

    assert recovered_bits == bits, "Mapper/Demapper mismatch."
    assert len(i_symbols) == num_symbols
    assert len(waveform) == len(time_axis) == num_symbols * sps

    print("16-QAM Reference Model")
    print("-----------------------------")
    print(f"Generated bits:     {len(bits)}")
    print(f"Mapped symbols:     {len(i_symbols)}")
    print(f"Symbol rate:        {symbol_rate / 1e6:g} MBd")
    print(f"Sample rate:        {sample_rate / 1e6:g} MSPS")
    print(f"Carrier frequency:  {carrier_frequency / 1e6:g} MHz")
    print(f"Samples per symbol: {sps}")
    print(f"Waveform samples:   {len(waveform)}")
    print("Verification:       PASS")

    # Visualization
    plot_qam16(i_symbols, q_symbols)

    plot_qam_waveform(
        time_axis,
        i_samples,
        q_samples,
        cosine_carrier,
        sine_carrier,
        waveform,
        sps,
        num_symbols_to_plot=4,
    )

    plt.show()


if __name__ == "__main__":
    main()
