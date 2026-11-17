"""
Entry point: run the five model pipelines, print the reports, show the figures.

    01 ideal reference -> 02 bit-true FPGA -> 03 physical DAC
    -> 04 analog output -> 05 virtual measurement instruments

From the 02_DSP_Model folder:
    py main.py
"""

import matplotlib.pyplot as plt
import numpy as np

from importlib import import_module

import params
from pipeline import (
    run_ideal_reference_pipeline,
    run_bit_true_fpga_pipeline,
    run_physical_dac_pipeline,
    run_analog_output_pipeline,
    run_measurement_pipeline,
    print_pipeline_report,
    print_fixed_point_report,
)
import visualization

compute_spectrum = import_module("05_measurement_instruments").compute_spectrum
compute_full_scale = import_module("03_physical_dac").compute_full_scale


# FPGA model that feeds the hardware: True = bit-true fixed-point FPGA,
# False = ideal floating-point reference
USE_FIXED_POINT_FPGA = True

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

    # 01 Ideal reference (figures, comparison) and 02 bit-true FPGA (feeds the DAC)
    results = run_ideal_reference_pipeline()
    fpga = run_bit_true_fpga_pipeline() if USE_FIXED_POINT_FPGA else results

    # 03 Physical DAC and 04 analog output
    tx_output = run_hardware(fpga)

    # 05 Virtual instruments on the transmitter output
    measurement = measure(tx_output, results)
    received_symbols = measurement["symbols"]
    sync = measurement["sync"]
    evm = measurement["evm_percent"]
    tx_spectrum_db = measurement["spectrum_db"]

    # Spectra of the ideal signals for the comparison figure
    bb_frequencies, bb_spectrum_db = compute_spectrum(
        results["pulse_shaped_symbols"], params.SAMPLE_RATE
    )
    if_frequencies, if_spectrum_db = compute_spectrum(
        results["if_samples"], params.SAMPLE_RATE
    )

    # Terminal report
    print_pipeline_report(results)
    if USE_FIXED_POINT_FPGA:
        print_fixed_point_report(fpga["fixed_point"])
        fpga_snr = 10 * np.log10(np.mean(results["if_samples"] ** 2)
                                 / np.mean((fpga["if_samples"] - results["if_samples"]) ** 2))
        print(f"{'Fixed-point vs ideal SNR':<28}: {fpga_snr:.1f} dB")

    # Figures
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
        evm_measured, evm_theory = evm_nsd_sweep(fpga, results)
        visualization.plot_evm_sweep(
            EVM_NSD_SWEEP,
            evm_measured,
            evm_theory,
            "DAC noise spectral density (dBFS/Hz)",
            EVM_LIMIT_PERCENT,
        )

    fpga_label = "fixed-point FPGA" if USE_FIXED_POINT_FPGA else "ideal FPGA"
    print(f"EVM at the transmitter output: {evm:.2f} % ({fpga_label}, {hardware_label()})")
    print(f"Analyzer sync: delay {sync['delay_samples']} samples, "
          f"gain {sync['gain']:.4f}, phase {sync['phase_deg']:.3f} deg")

    plt.show()


def run_hardware(fpga, **overrides):
    """03 physical DAC + 04 analog output with the datasheet values from params; overrides change single values."""

    settings = dict(
        dac_bits=params.DAC_BITS,
        dac_backoff_db=params.DAC_BACKOFF_DB,
        dac_nsd_dbfs_hz=params.DAC_NSD_DBFS_HZ,
        clock_jitter_rms=params.CLOCK_JITTER_RMS,
        dac_jitter_rms=params.DAC_JITTER_RMS,
        seed=params.NOISE_SEED,
        full_scale=fpga.get("full_scale"),
    )
    settings.update(overrides)

    dac_output = run_physical_dac_pipeline(fpga["if_samples"], params.SAMPLE_RATE, **settings)
    return run_analog_output_pipeline(dac_output, params.SAMPLE_RATE)


def hardware_label():
    """Short description of the hardware parameters for titles and the report."""

    jitter_ps = None if params.CLOCK_JITTER_RMS is None else params.CLOCK_JITTER_RMS * 1e12
    return (f"DAC {params.DAC_BITS} bit, back-off {params.DAC_BACKOFF_DB} dB, "
            f"NSD {params.DAC_NSD_DBFS_HZ} dBFS/Hz, clock jitter {jitter_ps} ps")


def measure(tx_samples, results):
    """05 virtual instruments on transmitter output samples (reference from 01)."""

    return run_measurement_pipeline(
        tx_samples,
        results["rrc_coefficients"],
        results["normalized_symbols"],
        params.SAMPLE_RATE,
        params.IF_FREQUENCY,
        params.SAMPLES_PER_SYMBOL,
    )


def evm_nsd_sweep(fpga, results):
    """Measured and theoretical EVM (percent) for every NSD in EVM_NSD_SWEEP."""

    # 1. EVM floor: all hardware effects except the DAC output noise
    floor_tx = run_hardware(fpga, dac_nsd_dbfs_hz=None)
    evm_0 = measure(floor_tx, results)["evm_percent"] / 100

    # 2. Measured EVM through the full hardware pipeline
    evm_measured = []
    for nsd in EVM_NSD_SWEEP:
        tx = run_hardware(fpga, dac_nsd_dbfs_hz=nsd)
        evm_measured.append(measure(tx, results)["evm_percent"])

    # 3. Theory: N0 = P_FS * 10^(NSD/10), Es/N0 = P_signal / (N0 * Rs)
    full_scale = fpga.get("full_scale") or compute_full_scale(
        fpga["if_samples"], params.DAC_BACKOFF_DB
    )
    p_signal = np.mean(fpga["if_samples"] ** 2)
    n0 = full_scale ** 2 / 2 * 10 ** (EVM_NSD_SWEEP / 10)
    es_n0 = p_signal / (n0 * params.SYMBOL_RATE)
    evm_theory = 100 * np.sqrt(evm_0 ** 2 + 1 / es_n0)

    return np.array(evm_measured), evm_theory


if __name__ == "__main__":
    main()
