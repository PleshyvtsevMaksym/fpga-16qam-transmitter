"""Transmitter, FPGA fixed-point and hardware parameters of the 16-QAM model (values only)."""


# ============================================================
# 1. MODULATION PARAMETERS (01 ideal reference)
# ============================================================

BITS_PER_SYMBOL = 4
NUM_SYMBOLS = 4096
NUM_BITS = NUM_SYMBOLS * BITS_PER_SYMBOL

QAM_AVG_ENERGY = 10.0


# ============================================================
# 2. SIGNAL PARAMETERS (01 ideal reference)
# ============================================================

SYMBOL_RATE = 1_000_000       # 1 MBd
SAMPLE_RATE = 100_000_000     # 100 MSPS
IF_FREQUENCY = 5_000_000      # 5 MHz

SAMPLES_PER_SYMBOL = SAMPLE_RATE // SYMBOL_RATE


# ============================================================
# 3. RRC FILTER PARAMETERS (01 ideal reference)
# ============================================================

RRC_ROLLOFF = 0.35
RRC_SPAN = 8                  # Symbols


# ============================================================
# 4. FPGA FIXED-POINT PARAMETERS (02 bit-true FPGA)
# ============================================================
# Word widths of the FPGA design. Artix-7 DSP48E1 multiplies 25 x 18 bit,
# so widths up to 18 (one operand) and 25 (other operand) cost one slice.
# The FPGA output width is DAC_BITS (section 5).

FPGA_COEF_BITS = 16                # RRC coefficient width (FIR Compiler .coe)
FPGA_FIR_OUT_BITS = 16             # I and Q width after the FIR (after rounding)
FPGA_NCO_PHASE_BITS = 32           # DDS phase accumulator width P
FPGA_NCO_LUT_ADDR_BITS = 12        # DDS lookup table address width A (2^A entries)
FPGA_NCO_AMP_BITS = 16             # DDS cos/sin amplitude width B
FPGA_GAIN_BITS = 18                # Output scaling constant width
FPGA_ROUNDING = "round"            # "round" or "truncate" when LSBs are dropped


# ============================================================
# 5. DAC PARAMETERS (03 physical DAC, datasheet values)
# ============================================================
# Every value comes from a datasheet (cite it here) or is marked
# ASSUMPTION until the component is chosen. None = effect switched off.

NOISE_SEED = 1                     # Same random effects on every run

# 5.1 DAC resolution and scaling
DAC_BITS = 12                      # Resolution N; ASSUMPTION (no DAC chosen yet)
DAC_BACKOFF_DB = 0.0               # Signal peak below DAC full scale, dB (design choice)

# 5.2 DAC output noise
DAC_NSD_DBFS_HZ = -150.0           # Noise spectral density, dBFS/Hz; ASSUMPTION

# 5.3 Clock
CLOCK_JITTER_RMS = 1e-12           # Oscillator RMS phase jitter (12 kHz-20 MHz), s; ASSUMPTION
DAC_JITTER_RMS = None              # DAC internal (aperture) jitter, s; None = not given


# ============================================================
# 6. ANALOG OUTPUT PARAMETERS (04 analog output)
# ============================================================
# Reconstruction filter, output stage and load: added with 04-01 ... 04-03.
