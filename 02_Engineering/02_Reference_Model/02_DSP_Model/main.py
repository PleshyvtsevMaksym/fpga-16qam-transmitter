"""
Entry point: run the DSP pipeline, print the report and show the figures.

From the 02_DSP_Model folder:
    py main.py
"""

import matplotlib.pyplot as plt

import params
from analysis import compute_spectrum
from pipeline import run_fpga_pipeline, run_hardware_pipeline, print_pipeline_report
import visualization


# Set a figure to False to skip it
SHOW_CONSTELLATION = True
SHOW_SYMBOL_PROCESSING = True
SHOW_RRC_RESPONSE = True
SHOW_IF_WAVEFORM = True
SHOW_SPECTRA = True
SHOW_NOISY_CONSTELLATION = True    # theory illustration: AWGN on symbols

# SNR values (dB) for the noisy constellation figure
NOISE_SNR_LIST = [30, 20, 10, 5]


def main():

    # 1. FPGA pipeline (ideal transmitter)
    results = run_fpga_pipeline()

    # 2. Hardware pipeline after the FPGA (DAC, filter, thermal noise)
    tx_output = run_hardware_pipeline(
        results["if_samples"],
        params.AWGN_SNR_DB,
        seed=params.NOISE_SEED,
    )

    # 3. Measurements (virtual spectrum analyzer)
    bb_frequencies, bb_spectrum_db = compute_spectrum(
        results["pulse_shaped_symbols"], params.SAMPLE_RATE
    )
    if_frequencies, if_spectrum_db = compute_spectrum(
        results["if_samples"], params.SAMPLE_RATE
    )
    _, tx_spectrum_db = compute_spectrum(tx_output, params.SAMPLE_RATE)

    # 4. Terminal report
    print_pipeline_report(results)

    # 5. Figures
    visualization.apply_plot_style()

    if SHOW_CONSTELLATION:
        visualization.plot_constellation(
            results["symbols"],
            results["normalized_symbols"],
        )

    if SHOW_SYMBOL_PROCESSING:
        visualization.plot_symbol_processing(
            results["normalized_symbols"],
            results["upsampled_symbols"],
            results["pulse_shaped_symbols"],
            len(results["rrc_coefficients"]),
            params.SAMPLE_RATE,
            params.SAMPLES_PER_SYMBOL,
            first_symbol=20,    # seed 0x0001 starts with a run of zeros
        )

    if SHOW_RRC_RESPONSE:
        visualization.plot_rrc_response(
            results["rrc_coefficients"],
            params.SAMPLES_PER_SYMBOL,
            params.SYMBOL_RATE,
            params.SAMPLE_RATE,
            params.RRC_ROLLOFF,
        )

    if SHOW_IF_WAVEFORM:
        # Start one filter length in, where the waveform is fully built up
        visualization.plot_if_waveform(
            results["pulse_shaped_symbols"],
            results["if_samples"],
            params.SAMPLE_RATE,
            params.IF_FREQUENCY,
            start_sample=len(results["rrc_coefficients"]),
            tx_output=tx_output,
        )

    if SHOW_SPECTRA:
        visualization.plot_spectra(
            bb_frequencies,
            bb_spectrum_db,
            if_frequencies,
            if_spectrum_db,
            params.IF_FREQUENCY,
            params.SYMBOL_RATE,
            params.RRC_ROLLOFF,
            tx_spectrum_db=tx_spectrum_db,
        )

    if SHOW_NOISY_CONSTELLATION:
        visualization.plot_noisy_constellation(
            results["normalized_symbols"],
            NOISE_SNR_LIST,
            params.NOISE_SEED,
        )

    plt.show()


if __name__ == "__main__":
    main()
