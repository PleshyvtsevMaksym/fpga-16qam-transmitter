"""
FPGA pipeline: the ideal transmitter, DSP stages 01-07.

Everything here is what the FPGA computes. It ends with the IF samples
that are sent to the DAC. What happens after the FPGA is in
hardware_pipeline.py.
"""

import numpy as np

from dsp import (
    prbs15_generator,
    qam16_gray_mapping,
    normalize_symbols,
    upsample_symbols,
    design_rrc,
    apply_fir,
    modulate_to_if,
)
from params import (
    NUM_BITS,
    QAM_AVG_ENERGY,
    SAMPLES_PER_SYMBOL,
    SAMPLE_RATE,
    IF_FREQUENCY,
    RRC_ROLLOFF,
    RRC_SPAN,
)


def run_fpga_pipeline():
    """
    Run the ideal FPGA transmitter chain (stages 01-07) with the parameters above.

    Measurements such as the spectrum are not part of the chain; they are
    computed from the returned signals with the analysis package.

    Returns
    -------
    dict
        Every intermediate signal, keyed by name, in processing order:
        bits, symbols, normalized_symbols, upsampled_symbols,
        rrc_coefficients, pulse_shaped_symbols, if_samples.
    """

    # 1. PRBS15 generation
    bits = prbs15_generator(NUM_BITS)

    # 2. Gray-coded 16-QAM mapping
    map_i, map_q = qam16_gray_mapping(bits)
    symbols = np.asarray(map_i) + 1j * np.asarray(map_q)

    # 3. Symbol energy normalization
    normalized_symbols = normalize_symbols(
        symbols,
        QAM_AVG_ENERGY
    )

    # 4. Zero-insertion upsampling
    upsampled_symbols = upsample_symbols(
        normalized_symbols,
        SAMPLES_PER_SYMBOL
    )

    # 5. RRC filter design
    rrc_coefficients = design_rrc(
        RRC_ROLLOFF,
        RRC_SPAN,
        SAMPLES_PER_SYMBOL
    )

    # 6. FIR pulse shaping
    pulse_shaped_symbols = apply_fir(
        upsampled_symbols,
        rrc_coefficients
    )

    # 7. Quadrature IF modulation
    if_samples = modulate_to_if(
        pulse_shaped_symbols,
        SAMPLE_RATE,
        IF_FREQUENCY
    )

    return {
        "bits": bits,
        "symbols": symbols,
        "normalized_symbols": normalized_symbols,
        "upsampled_symbols": upsampled_symbols,
        "rrc_coefficients": rrc_coefficients,
        "pulse_shaped_symbols": pulse_shaped_symbols,
        "if_samples": if_samples,
    }
