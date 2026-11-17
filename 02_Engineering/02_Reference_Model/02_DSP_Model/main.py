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
import visualization

_pipeline = import_module("06_pipeline")
run_ideal_reference_pipeline = _pipeline.run_ideal_reference_pipeline
run_bit_true_fpga_pipeline = _pipeline.run_bit_true_fpga_pipeline
run_physical_dac_pipeline = _pipeline.run_physical_dac_pipeline
run_analog_output_pipeline = _pipeline.run_analog_output_pipeline
run_measurement_pipeline = _pipeline.run_measurement_pipeline
print_pipeline_report = _pipeline.print_pipeline_report
print_fixed_point_report = _pipeline.print_fixed_point_report

apply_zero_order_hold = import_module("03_physical_dac").apply_zero_order_hold



# FPGA model that feeds the hardware: True = bit-true fixed-point FPGA,
# False = ideal floating-point reference
USE_FIXED_POINT_FPGA = True

# Figures in signal-chain order; set one to False to skip it
SHOW_SYMBOL_MAPPING = True        # 1  01 mapping
SHOW_PULSE_SHAPING = True         # 2  01 RRC
SHOW_FPGA_OUTPUT = True           # 3  02 bit-true vs ideal
SHOW_DAC_TRANSFER = True          # 4  03-01..03
SHOW_DAC_CLOCK = True             # 5  03-04..05
SHOW_DAC_OUTPUT = True            # 6  03-06..07
SHOW_MEASUREMENT = True           # 7  05 instruments
SHOW_EVM_VS_NSD = True            # 8  05 sweep
SHOW_THEORY_AWGN = False          # theory illustration

# SNR values (dB) for the theory figure
NOISE_SNR_LIST = [30, 20, 10, 5]

# DAC noise density sweep (dBFS/Hz) for the EVM vs NSD figure
EVM_NSD_SWEEP = np.arange(-160, -75, 5)
EVM_LIMIT_PERCENT = 12.5          # LTE 16-QAM transmitter limit (3GPP TS 36.104)


def main():

    # 01 Ideal reference and 02 bit-true FPGA (feeds the DAC)
    results = run_ideal_reference_pipeline()
    fpga = run_bit_true_fpga_pipeline() if USE_FIXED_POINT_FPGA else results

    # 03 Physical DAC and 04 analog output
    dac, tx_output = run_hardware(fpga)
    full_scale = dac["full_scale"]

    # 05 Virtual instruments on the transmitter output
    measurement = measure(tx_output, results, dac["analog_sampling_rate"], full_scale)
    evm = measurement["evm_percent"]
    sync = measurement["sync"]

    # Terminal report
    print_pipeline_report(results)
    if USE_FIXED_POINT_FPGA:
        print_fixed_point_report(fpga["fixed_point"])
        fpga_snr = 10 * np.log10(np.mean(results["if_samples"] ** 2)
                                 / np.mean((fpga["if_samples"] - results["if_samples"]) ** 2))
        print(f"{'Fixed-point vs ideal SNR':<28}: {fpga_snr:.1f} dB")

    # Figures in signal-chain order
    visualization.apply_plot_style()
    ideal_if = results["if_samples"]
    half_band = (1 + params.RRC_ROLLOFF) * params.SYMBOL_RATE / 2
    start = len(results["rrc_coefficients"])    # waveform fully built up

    if SHOW_SYMBOL_MAPPING:
        visualization.plot_constellation(results["normalized_symbols"])

    if SHOW_PULSE_SHAPING:
        visualization.plot_pulse_shaping(
            results["normalized_symbols"], results["pulse_shaped_symbols"],
            results["rrc_coefficients"], params.SAMPLE_RATE,
            params.SAMPLES_PER_SYMBOL, params.SYMBOL_RATE, params.RRC_ROLLOFF,
            first_symbol=20, num_symbols=8,    # seed 0x0001 starts with zeros
        )

    if SHOW_FPGA_OUTPUT:
        visualization.plot_fpga_output(
            ideal_if, fpga["if_samples"], params.SAMPLE_RATE, full_scale,
            params.SPECTRUM_RBW_HZ, start_sample=start, duration=2e-6,
        )

    if SHOW_DAC_TRANSFER:
        visualization.plot_dac_transfer(
            ideal_if, dac["quantized"], dac["static"], params.SAMPLE_RATE,
            full_scale, params.SPECTRUM_RBW_HZ, params.IF_FREQUENCY, half_band,
        )

    if SHOW_DAC_CLOCK:
        visualization.plot_dac_clock(
            dac["static"], dac["clocked"], params.SAMPLE_RATE, full_scale,
            params.SPECTRUM_RBW_HZ, params.IF_FREQUENCY, half_band,
        )

    if SHOW_DAC_OUTPUT:
        visualization.plot_dac_output_dynamics(
            ideal_if, dac["held"], dac["glitched"], dac["output"],
            params.SAMPLE_RATE, params.ANALOG_OVERSAMPLING, full_scale,
            params.SPECTRUM_RBW_HZ, params.IF_FREQUENCY,
            start_sample=start, num_dac_samples=12,
        )

    if SHOW_MEASUREMENT:
        visualization.plot_measurement(
            measurement["frequencies"], measurement["psd_dbfs_hz"],
            measurement["symbols"], results["normalized_symbols"], evm,
            params.IF_FREQUENCY, params.SYMBOL_RATE, params.RRC_ROLLOFF,
            hardware_label(),
        )

    if SHOW_EVM_VS_NSD:
        print(f"EVM sweep: {len(EVM_NSD_SWEEP) + 1} runs of the hardware chain "
              "(figures open when it is done) ...")
        evm_measured, evm_theory = evm_nsd_sweep(fpga, results)
        visualization.plot_evm_sweep(
            EVM_NSD_SWEEP, evm_measured, evm_theory,
            "DAC noise spectral density (dBFS/Hz)", EVM_LIMIT_PERCENT,
        )

    if SHOW_THEORY_AWGN:
        visualization.plot_noisy_constellation(
            results["normalized_symbols"], NOISE_SNR_LIST, params.NOISE_SEED,
        )

    fpga_label = "fixed-point FPGA" if USE_FIXED_POINT_FPGA else "ideal FPGA"
    print(f"EVM at the transmitter output: {evm:.2f} % ({fpga_label}, {hardware_label()})")
    # VSA gain/delay relative to the ideal signal on the same analog grid
    ideal_sync = measure(apply_zero_order_hold(ideal_if, params.ANALOG_OVERSAMPLING),
                         results, dac["analog_sampling_rate"], full_scale)["sync"]
    delay_ns = ((sync["delay_samples"] - ideal_sync["delay_samples"])
                / dac["analog_sampling_rate"] * 1e9)
    print(f"Analyzer vs ideal: delay {delay_ns:+.2f} ns, "
          f"gain {sync['gain'] / ideal_sync['gain']:.4f}, "
          f"phase {sync['phase_deg'] - ideal_sync['phase_deg']:+.3f} deg")

    plt.show()


def run_hardware(fpga, **overrides):
    """03 physical DAC + 04 analog output with params values; overrides change single values. Returns (dac dict, tx output at Fs_a)."""

    settings = dict(
        full_scale=fpga.get("full_scale"),
        dac_bits=params.DAC_BITS,
        dac_backoff_db=params.DAC_BACKOFF_DB,
        dac_hd2_dbc=params.DAC_HD2_DBC,
        dac_hd3_dbc=params.DAC_HD3_DBC,
        dac_hd_test_level_dbfs=params.DAC_HD_TEST_LEVEL_DBFS,
        dac_offset_error_pct_fsr=params.DAC_OFFSET_ERROR_PCT_FSR,
        dac_gain_error_pct_fsr=params.DAC_GAIN_ERROR_PCT_FSR,
        clock_jitter_rms=params.CLOCK_JITTER_RMS,
        dac_jitter_rms=params.DAC_JITTER_RMS,
        clock_phase_noise_offsets_hz=params.CLOCK_PHASE_NOISE_OFFSETS_HZ,
        clock_phase_noise_dbc_hz=params.CLOCK_PHASE_NOISE_DBC_HZ,
        analog_oversampling=params.ANALOG_OVERSAMPLING,
        dac_full_scale_voltage=params.DAC_FULL_SCALE_VOLTAGE,
        dac_glitch_impulse_vs=params.DAC_GLITCH_IMPULSE_VS,
        dac_glitch_duration_s=params.DAC_GLITCH_DURATION_S,
        dac_settling_time_s=params.DAC_SETTLING_TIME_S,
        dac_settling_accuracy=params.DAC_SETTLING_ACCURACY,
        dac_nsd_dbfs_hz=params.DAC_NSD_DBFS_HZ,
        seed=params.NOISE_SEED,
    )
    settings.update(overrides)

    dac = run_physical_dac_pipeline(fpga["if_samples"], params.SAMPLE_RATE, **settings)
    tx_output = run_analog_output_pipeline(dac["output"], dac["analog_sampling_rate"])
    return dac, tx_output


def hardware_label():
    """Short description of the hardware parameters for titles and the report."""

    if params.CLOCK_PHASE_NOISE_DBC_HZ is not None:
        clock = "clock L(f) mask " + "/".join(str(v) for v in params.CLOCK_PHASE_NOISE_DBC_HZ) + " dBc/Hz"
    elif params.CLOCK_JITTER_RMS is not None:
        clock = f"clock jitter {params.CLOCK_JITTER_RMS * 1e12} ps"
    else:
        clock = "ideal clock"
    return (f"DAC {params.DAC_BITS} bit, back-off {params.DAC_BACKOFF_DB} dB, "
            f"HD2/HD3 {params.DAC_HD2_DBC}/{params.DAC_HD3_DBC} dBc, "
            f"offset/gain {params.DAC_OFFSET_ERROR_PCT_FSR}/{params.DAC_GAIN_ERROR_PCT_FSR} % FSR, "
            f"NSD {params.DAC_NSD_DBFS_HZ} dBFS/Hz, {clock}, "
            f"glitch {params.DAC_GLITCH_IMPULSE_VS * 1e12} pV*s, "
            f"settling {params.DAC_SETTLING_TIME_S * 1e9} ns")


def measure(tx_samples, results, sampling_rate, full_scale):
    """05 virtual instruments on transmitter output samples (reference from 01)."""

    return run_measurement_pipeline(
        tx_samples,
        results["normalized_symbols"],
        sampling_rate,
        params.IF_FREQUENCY,
        params.SYMBOL_RATE,
        params.VSA_RRC_ROLLOFF,
        params.VSA_RRC_SPAN,
        params.SPECTRUM_RBW_HZ,
        full_scale,
    )


def evm_nsd_sweep(fpga, results):
    """Measured and theoretical EVM (percent) for every NSD in EVM_NSD_SWEEP."""

    # 1. EVM floor: all hardware effects except the DAC output noise
    floor_dac, floor_tx = run_hardware(fpga, dac_nsd_dbfs_hz=None)
    fs_a, full_scale = floor_dac["analog_sampling_rate"], floor_dac["full_scale"]
    evm_0 = measure(floor_tx, results, fs_a, full_scale)["evm_percent"] / 100

    # 2. Measured EVM through the full hardware pipeline
    evm_measured = []
    for nsd in EVM_NSD_SWEEP:
        _, tx = run_hardware(fpga, dac_nsd_dbfs_hz=nsd)
        evm_measured.append(measure(tx, results, fs_a, full_scale)["evm_percent"])

    # 3. Theory: N0 = P_FS * 10^(NSD/10), Es/N0 = P_signal / (N0 * Rs)
    p_signal = np.mean(fpga["if_samples"] ** 2)
    n0 = full_scale ** 2 / 2 * 10 ** (EVM_NSD_SWEEP / 10)
    es_n0 = p_signal / (n0 * params.SYMBOL_RATE)
    evm_theory = 100 * np.sqrt(evm_0 ** 2 + 1 / es_n0)

    return np.array(evm_measured), evm_theory


if __name__ == "__main__":
    main()
