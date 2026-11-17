"""
FPGA pipeline: the transmitter inside the FPGA.

    run_fpga_pipeline              ideal floating-point reference, stages 01-07
    run_fpga_fixed_point_pipeline  bit-true model with the FPGA word widths

Both end with the IF samples that are sent to the DAC. What happens after
the FPGA is in hardware_pipeline.py.
"""

import numpy as np

from fpga import (
    prbs15_generator,
    qam16_gray_mapping,
    normalize_symbols,
    upsample_symbols,
    design_rrc,
    apply_fir,
    modulate_to_if,
    quantize_coefficients,
    required_bits,
    shift_right,
    saturate,
    generate_nco,
)
from params import (
    NUM_BITS,
    QAM_AVG_ENERGY,
    SAMPLES_PER_SYMBOL,
    SAMPLE_RATE,
    IF_FREQUENCY,
    RRC_ROLLOFF,
    RRC_SPAN,
    FPGA_COEF_BITS,
    FPGA_FIR_OUT_BITS,
    FPGA_NCO_PHASE_BITS,
    FPGA_NCO_LUT_ADDR_BITS,
    FPGA_NCO_AMP_BITS,
    FPGA_GAIN_BITS,
    FPGA_ROUNDING,
    DAC_BITS,
    DAC_BACKOFF_DB,
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


def run_fpga_fixed_point_pipeline():
    """
    Run the bit-true FPGA transmitter: integers of fixed width, as in HDL.

    Same algorithm as run_fpga_pipeline, but every value is an integer of
    the width set in params (section 4), exactly as the HDL will compute it.
    The output is the stream of DAC codes; it later serves as the golden
    reference for the HDL testbench (bit-by-bit comparison).

        bits -> Gray levels +-1, +-3 (3-bit integers)
        -> zero insertion x sps
        -> FIR with FPGA_COEF_BITS coefficients, full-precision accumulator
        -> drop LSBs to FPGA_FIR_OUT_BITS (rounding, saturation)
        -> DDS cos/sin (FPGA_NCO_* widths)
        -> I*cos - Q*sin (full precision)
        -> constant gain (FPGA_GAIN_BITS) to DAC codes, DAC_BITS wide, with
           the peak DAC_BACKOFF_DB below full scale

    Stage 03 (division by sqrt(10)) has no block of its own: it is a
    constant factor and is absorbed in the output gain.

    Returns
    -------
    dict
        Integer signals: bits, symbols_int_i, symbols_int_q,
        rrc_coefficients_int, fir_i, fir_q, nco_cos, nco_sin, dac_codes.
        For the hardware model and the analyzer, in the units of the ideal
        model: if_samples (= dac_codes * LSB), full_scale, symbols,
        normalized_symbols, rrc_coefficients (ideal, for the analyzer).
        "fixed_point": word widths, shifts and checks for the report.
    """

    sps = SAMPLES_PER_SYMBOL

    # 1. Bits and Gray mapping: levels +-1, +-3 are already integers
    bits = prbs15_generator(NUM_BITS)
    map_i, map_q = qam16_gray_mapping(bits)
    symbols_i = np.asarray(map_i).astype(np.int64)
    symbols_q = np.asarray(map_q).astype(np.int64)
    symbol_bits = required_bits(int(np.max(np.abs(np.concatenate([symbols_i, symbols_q])))))

    # 2. Zero insertion (upsampling) on integers
    upsampled_i = np.zeros(symbols_i.size * sps, dtype=np.int64)
    upsampled_q = np.zeros(symbols_q.size * sps, dtype=np.int64)
    upsampled_i[::sps] = symbols_i
    upsampled_q[::sps] = symbols_q

    # 3. RRC coefficients as FPGA_COEF_BITS integers
    rrc = design_rrc(RRC_ROLLOFF, RRC_SPAN, sps)
    rrc_int, coef_scale = quantize_coefficients(rrc, FPGA_COEF_BITS)

    # 4. FIR with a full-precision accumulator (exact integer convolution).
    #    Polyphase: each output uses one tap per symbol, so the worst case
    #    is 3 * the largest sum of |h| over one phase.
    acc_i = np.convolve(upsampled_i, rrc_int)
    acc_q = np.convolve(upsampled_q, rrc_int)
    worst_case = 3 * max(int(np.sum(np.abs(rrc_int[p::sps]))) for p in range(sps))
    accumulator_bits = required_bits(worst_case)

    # 5. Drop LSBs to the FIR output width (worst case never overflows)
    fir_shift = max(0, accumulator_bits - FPGA_FIR_OUT_BITS)
    fir_i, clipped_i = saturate(shift_right(acc_i, fir_shift, FPGA_ROUNDING), FPGA_FIR_OUT_BITS)
    fir_q, clipped_q = saturate(shift_right(acc_q, fir_shift, FPGA_ROUNDING), FPGA_FIR_OUT_BITS)

    # 6. DDS carrier
    nco_cos, nco_sin, nco_frequency = generate_nco(
        fir_i.size, SAMPLE_RATE, IF_FREQUENCY,
        FPGA_NCO_PHASE_BITS, FPGA_NCO_LUT_ADDR_BITS, FPGA_NCO_AMP_BITS,
    )

    # 7. Quadrature modulation in full precision
    mixed = fir_i * nco_cos - fir_q * nco_sin
    mixed_bits = required_bits(int(np.max(np.abs(mixed))))

    # 8. Output gain: peak at DAC_BACKOFF_DB below full scale 2^(N-1).
    #    gain = gain_int / 2^gain_shift, gain_int fits FPGA_GAIN_BITS (unsigned part)
    target_peak = 2 ** (DAC_BITS - 1) * 10 ** (-DAC_BACKOFF_DB / 20)
    gain = target_peak / np.max(np.abs(mixed))
    gain_shift = int(np.floor(np.log2((2 ** (FPGA_GAIN_BITS - 1) - 1) / gain)))
    if gain_shift < 0:
        raise ValueError("Output gain does not fit FPGA_GAIN_BITS; increase the width.")
    gain_int = int(round(gain * 2 ** gain_shift))

    dac_codes, clipped_out = saturate(
        shift_right(mixed * gain_int, gain_shift, FPGA_ROUNDING), DAC_BITS
    )

    # 9. Same units as the ideal model: 1 integer step of "mixed" equals
    #    2^fir_shift / (coef_scale * sqrt(10) * max cos code) ideal units
    max_cos = 2 ** (FPGA_NCO_AMP_BITS - 1) - 1
    mixed_unit = 2 ** fir_shift / (coef_scale * np.sqrt(QAM_AVG_ENERGY) * max_cos)
    dac_lsb = mixed_unit * 2 ** gain_shift / gain_int
    if_samples = dac_codes * dac_lsb
    full_scale = 2 ** (DAC_BITS - 1) * dac_lsb

    symbols = symbols_i + 1j * symbols_q

    return {
        "bits": bits,
        "symbols_int_i": symbols_i,
        "symbols_int_q": symbols_q,
        "rrc_coefficients_int": rrc_int,
        "fir_i": fir_i,
        "fir_q": fir_q,
        "nco_cos": nco_cos,
        "nco_sin": nco_sin,
        "dac_codes": dac_codes,
        "if_samples": if_samples,
        "full_scale": float(full_scale),
        "symbols": symbols,
        "normalized_symbols": normalize_symbols(symbols.astype(complex), QAM_AVG_ENERGY),
        "rrc_coefficients": rrc,
        "fixed_point": {
            "symbol_bits": symbol_bits,
            "coef_bits": FPGA_COEF_BITS,
            "coef_max_error_lsb": float(np.max(np.abs(rrc * coef_scale - rrc_int))),
            "accumulator_bits": accumulator_bits,
            "fir_shift": fir_shift,
            "fir_out_bits": FPGA_FIR_OUT_BITS,
            "nco_frequency": nco_frequency,
            "nco_frequency_error": nco_frequency - IF_FREQUENCY,
            "mixed_bits": mixed_bits,
            "gain_int": gain_int,
            "gain_shift": gain_shift,
            "dac_bits": DAC_BITS,
            "rounding": FPGA_ROUNDING,
            "clipped_samples": clipped_i + clipped_q + clipped_out,
        },
    }
