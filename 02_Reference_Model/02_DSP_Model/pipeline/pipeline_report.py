"""Terminal report of the transmitter configuration and executed DSP stages."""

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
        Output of run_fpga_pipeline().
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
    print("DSP PROCESSING PIPELINE")
    print("-" * WIDTH)

    for index, (name, details) in enumerate(stages, start=1):
        print(f"{index:02d}. {name:<30}{details:<20}[DONE]")

    # 3. Footer
    print()
    print("=" * WIDTH)
    print("PIPELINE COMPLETED".center(WIDTH))
    print("=" * WIDTH)
