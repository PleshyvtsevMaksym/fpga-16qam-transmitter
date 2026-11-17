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
SAMPLE_RATE = 100_000_000     # 100 MSPS = Nexys A7 100 MHz clock (section 4.1)
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

# 4.1 FPGA board (fixed): Digilent Nexys A7-100T (reference manual, DS180)
FPGA_BOARD = "Digilent Nexys A7-100T"
FPGA_PART = "XC7A100T-1CSG324C"     # Artix-7, speed grade -1
FPGA_CLOCK_HZ = 100_000_000         # On-board 100 MHz oscillator = DAC clock (SAMPLE_RATE)
FPGA_DSP48_SLICES = 240             # DSP48E1 (25 x 18 multiplier) available
FPGA_BRAM_KBIT = 4_860              # Block RAM, kbit (135 x 36 kbit)
FPGA_LOGIC_CELLS = 101_440
FPGA_IO_STANDARD = "LVCMOS33"       # Pmod I/O, 3.3 V CMOS
FPGA_IO_VOLTAGE = 3.3               # V
FPGA_PMOD_PORTS = 4                 # JA ... JD, 8 signals each; DAC board connects here
FPGA_DAC_INTERFACE_PINS = 13        # 12 parallel data + 1 clock = 2 Pmod ports; ASSUMPTION (own board)

# 4.2 Word widths

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

# 5.2 DAC nonlinearity (static, from harmonic distortion)
DAC_HD2_DBC = -70.0                # 2nd harmonic, dBc (datasheet: "HD2"); ASSUMPTION
DAC_HD3_DBC = -75.0                # 3rd harmonic, dBc (datasheet: "HD3"); ASSUMPTION
DAC_HD_TEST_LEVEL_DBFS = 0.0       # Sine level of the HD2/HD3 spec, dBFS; ASSUMPTION

# 5.3 DAC offset and gain error (static)
DAC_OFFSET_ERROR_PCT_FSR = 0.1     # Offset error, % FSR (datasheet: "offset error"); ASSUMPTION
DAC_GAIN_ERROR_PCT_FSR = 1.0       # Gain error, % FSR (datasheet: "gain error"); ASSUMPTION

# 5.4 DAC output noise
DAC_NSD_DBFS_HZ = -150.0           # Noise spectral density, dBFS/Hz; ASSUMPTION

# 5.5 Clock
CLOCK_PHASE_NOISE_OFFSETS_HZ = [100, 1e3, 1e4, 1e5, 1e6, 1e7]               # L(f) mask offsets, Hz
CLOCK_PHASE_NOISE_DBC_HZ = [-90, -120, -140, -150, -155, -155]          # L(f) at a 100 MHz clock, dBc/Hz; ASSUMPTION (good XO)
CLOCK_JITTER_RMS = None            # White RMS jitter, s: only when no L(f) mask is used (else counted twice)
DAC_JITTER_RMS = None              # DAC internal (aperture) jitter, s; None = not given

# 5.6 DAC output dynamics (on the analog time grid)
ANALOG_OVERSAMPLING = 8            # Analog grid Fs_a = M * Fs (simulation, not hardware)
DAC_FULL_SCALE_VOLTAGE = 0.5       # Output peak voltage at full scale, V; ASSUMPTION (e.g. 20 mA into 25 ohm)
DAC_GLITCH_IMPULSE_VS = 2e-12      # Glitch impulse, V*s (datasheet: "glitch impulse", pV*s); ASSUMPTION
DAC_GLITCH_DURATION_S = 2e-9       # Glitch duration, s; ASSUMPTION
DAC_SETTLING_TIME_S = 10e-9        # Settling time, s (datasheet: "settling time"); ASSUMPTION
DAC_SETTLING_ACCURACY = 1e-3       # Accuracy of the settling spec (0.1 %); ASSUMPTION


# ============================================================
# 6. ANALOG OUTPUT PARAMETERS (04 analog output)
# ============================================================
# Reconstruction filter, output stage and load: added with 04-01 ... 04-03.


# ============================================================
# 7. MEASUREMENT INSTRUMENT SETTINGS (05 virtual instruments)
# ============================================================
# Settings of the lab instruments, the same values are used in the lab.

SPECTRUM_RBW_HZ = 10e3             # Spectrum analyzer resolution bandwidth, Hz
VSA_RRC_ROLLOFF = RRC_ROLLOFF      # VSA measurement filter = transmit RRC
VSA_RRC_SPAN = RRC_SPAN            # Symbols
