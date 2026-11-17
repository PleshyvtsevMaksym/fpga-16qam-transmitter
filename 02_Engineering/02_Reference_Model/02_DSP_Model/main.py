"""
Entry point: run the DSP pipeline, print the report and show the figures.

From the 02_DSP_Model folder:
    py main.py
"""

import matplotlib.pyplot as plt
import numpy as np

import params
from analysis import compute_spectrum, demodulate_if, compute_evm
from noise import compute_full_scale
from pipeline import run_fpga_pipeline, run_hardware_pipeline, print_pipeline_report
import visualization


# Set a figure to False to skip it
SHOW_CONSTELLATION = True
SHOW_SYMBOL_PROCESSING = True
SHOW_RRC_RESPONSE = True
SHOW_IF_WAVEFORM = True
SHOW_SPECTRA = True
SHOW_NOISY_CONSTELLATION = True    # theory illustration: AWGN on symbols
SHOW_RECEIVED_CONSTELLATION = True
SHOW_EVM_VS_NSD = True

# SNR values (dB) for the noisy constellation figure
NOISE_SNR_LIST = [30, 20, 10, 5]

# DAC noise density sweep (dBFS/Hz) for the EVM vs NSD figure
EVM_NSD_SWEEP = np.arange(-160, -75, 5)
EVM_LIMIT_PERCENT = 12.5          # LTE 16-QAM transmitter limit (3GPP TS 36.104)


def main():

    # 1. FPGA pipeline (ideal transmitter)
    results = run_fpga_pipeline()

    # 2. Hardware pipeline after the FPGA (DAC, filter, thermal noise)
    tx_output = run_hardware(results)

    # 3. Measurements (virtual spectrum analyzer)
    bb_frequencies, bb_spectrum_db = compute_spectrum(
        results["pulse_shaped_symbols"], params.SAMPLE_RATE
    )
    if_frequencies, if_spectrum_db = compute_spectrum(
        results["if_samples"], params.SAMPLE_RATE
    )
    _, tx_spectrum_db = compute_spectrum(tx_output, params.SAMPLE_RATE)

    # 4. Measurements (virtual vector signal analyzer)
    received_symbols, sync = recover_symbols(tx_output, results)
    evm = compute_evm(received_symbols, results["normalized_symbols"])

    # 5. Terminal report
    print_pipeline_report(results)

    # 6. Figures
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

    if SHOW_RECEIVED_CONSTELLATION:
        visualization.plot_received_constellation(
            received_symbols,
            results["normalized_symbols"],
            evm,
            hardware_label(),
        )

    if SHOW_EVM_VS_NSD:
        evm_measured, evm_theory = evm_nsd_sweep(results)
        visualization.plot_evm_sweep(
            EVM_NSD_SWEEP,
            evm_measured,
            evm_theory,
            "DAC noise spectral density (dBFS/Hz)",
            EVM_LIMIT_PERCENT,
        )

    print(f"EVM at the transmitter output: {evm:.2f} % ({hardware_label()})")
    print(f"Analyzer sync: delay {sync['delay_samples']} samples, "
          f"gain {sync['gain']:.4f}, phase {sync['phase_deg']:.3f} deg")

    plt.show()


def run_hardware(results, **overrides):
    """Hardware pipeline with the datasheet values from params; overrides change single values."""

    settings = dict(
        dac_bits=params.DAC_BITS,
        dac_backoff_db=params.DAC_BACKOFF_DB,
        dac_nsd_dbfs_hz=params.DAC_NSD_DBFS_HZ,
        clock_jitter_rms=params.CLOCK_JITTER_RMS,
        dac_jitter_rms=params.DAC_JITTER_RMS,
        seed=params.NOISE_SEED,
    )
    settings.update(overrides)

    return run_hardware_pipeline(results["if_samples"], params.SAMPLE_RATE, **settings)


def hardware_label():
    """Short description of the hardware parameters for titles and the report."""

    jitter_ps = None if params.CLOCK_JITTER_RMS is None else params.CLOCK_JITTER_RMS * 1e12
    return (f"DAC {params.DAC_BITS} bit, back-off {params.DAC_BACKOFF_DB} dB, "
            f"NSD {params.DAC_NSD_DBFS_HZ} dBFS/Hz, clock jitter {jitter_ps} ps")


def recover_symbols(tx_samples, results):
    """Run the virtual analyzer on transmitter output samples; returns (symbols, sync)."""

    return demodulate_if(
        tx_samples,
        results["rrc_coefficients"],
        params.SAMPLE_RATE,
        params.IF_FREQUENCY,
        params.SAMPLES_PER_SYMBOL,
        results["normalized_symbols"],
    )


def evm_nsd_sweep(results):
    """Measured and theoretical EVM (percent) for every NSD in EVM_NSD_SWEEP."""

    reference = results["normalized_symbols"]

    # 1. EVM floor: all hardware effects except the DAC output noise
    floor_tx = run_hardware(results, dac_nsd_dbfs_hz=None)
    evm_0 = compute_evm(recover_symbols(floor_tx, results)[0], reference) / 100

    # 2. Measured EVM through the full hardware pipeline
    evm_measured = []
    for nsd in EVM_NSD_SWEEP:
        tx = run_hardware(results, dac_nsd_dbfs_hz=nsd)
        evm_measured.append(compute_evm(recover_symbols(tx, results)[0], reference))

    # 3. Theory: N0 = P_FS * 10^(NSD/10), Es/N0 = P_signal / (N0 * Rs)
    full_scale = compute_full_scale(results["if_samples"], params.DAC_BACKOFF_DB)
    p_signal = np.mean(results["if_samples"] ** 2)
    n0 = full_scale ** 2 / 2 * 10 ** (EVM_NSD_SWEEP / 10)
    es_n0 = p_signal / (n0 * params.SYMBOL_RATE)
    evm_theory = 100 * np.sqrt(evm_0 ** 2 + 1 / es_n0)

    return np.array(evm_measured), evm_theory


if __name__ == "__main__":
    main()
