"""Terminal reports: transmitter configuration and stages of 01 ideal reference, word widths of 02 bit-true FPGA."""

from params import (
    NUM_SYMBOLS,
    SYMBOL_RATE,
    SAMPLE_RATE,
    IF_FREQUENCY,
    SAMPLES_PER_SYMBOL,
    RRC_ROLLOFF,
    RRC_SPAN,
)

WIDTH = 60


def print_pipeline_report(results: dict) -> None:
    """
    Print the configuration and the size of every pipeline output.

    [DONE] only means that the stage was executed. Mathematical
    verification is done separately by the test suite (tests/test.py).

    Parameters
    ----------
    results : dict
        Output of run_ideal_reference_pipeline().
    """

    # 1. Header and configuration
    print()
    print("=" * WIDTH)
    print("16-QAM DSP REFERENCE MODEL".center(WIDTH))
    print("=" * WIDTH)
    print()
    print("TRANSMITTER CONFIGURATION")
    print("-" * WIDTH)
    print(f"{'Modulation':<20}: Gray-coded 16-QAM")
    print(f"{'Number of Symbols':<20}: {NUM_SYMBOLS}")
    print(f"{'Symbol Rate':<20}: {SYMBOL_RATE / 1e6:.3f} MBd")
    print(f"{'Sampling Rate':<20}: {SAMPLE_RATE / 1e6:.3f} MSPS")
    print(f"{'Samples per Symbol':<20}: {SAMPLES_PER_SYMBOL}")
    print(f"{'IF Frequency':<20}: {IF_FREQUENCY / 1e6:.3f} MHz")
    print(f"{'RRC Roll-off / Span':<20}: {RRC_ROLLOFF} / {RRC_SPAN} symbols")

    # 2. Executed stages with sizes taken from the returned arrays
    stages = [
        ("PRBS15 Generation", f"{len(results['bits'])} bits"),
        ("16-QAM Gray Mapping", f"{len(results['symbols'])} symbols"),
        ("Symbol Normalization", f"{len(results['normalized_symbols'])} samples"),
        ("Zero-Insertion Upsampling", f"{len(results['upsampled_symbols'])} samples"),
        ("RRC Filter Design", f"{len(results['rrc_coefficients'])} coefficients"),
        ("FIR Pulse Shaping", f"{len(results['pulse_shaped_symbols'])} samples"),
        ("Quadrature IF Modulation", f"{len(results['if_samples'])} samples"),
    ]

    print()
    print("01 IDEAL REFERENCE PIPELINE")
    print("-" * WIDTH)

    for index, (name, details) in enumerate(stages, start=1):
        print(f"{index:02d}. {name:<30}{details:<20}[DONE]")

    # 3. Footer
    print()
    print("=" * WIDTH)
    print("PIPELINE COMPLETED".center(WIDTH))
    print("=" * WIDTH)


def print_fixed_point_report(fixed_point: dict) -> None:
    """Print the word widths and checks of the bit-true FPGA pipeline."""

    fp = fixed_point

    print()
    print("02 BIT-TRUE FPGA PIPELINE")
    print("-" * WIDTH)
    print(f"{'Symbols I/Q':<28}: {fp['symbol_bits']} bit (levels +-1, +-3)")
    print(f"{'RRC coefficients':<28}: {fp['coef_bits']} bit, "
          f"max error {fp['coef_max_error_lsb']:.3f} LSB")
    print(f"{'FIR accumulator':<28}: {fp['accumulator_bits']} bit (worst case)")
    print(f"{'FIR output':<28}: {fp['fir_out_bits']} bit "
          f"(drop {fp['fir_shift']} LSB, {fp['rounding']})")
    print(f"{'DDS frequency':<28}: {fp['nco_frequency'] / 1e6:.9f} MHz "
          f"(error {fp['nco_frequency_error'] * 1e3:.3f} mHz)")
    print(f"{'I*cos - Q*sin':<28}: {fp['mixed_bits']} bit")
    print(f"{'Output gain':<28}: {fp['gain_int']} / 2^{fp['gain_shift']}")
    print(f"{'Output to DAC':<28}: {fp['dac_bits']} bit")
    print(f"{'Clipped samples':<28}: {fp['clipped_samples']}")
