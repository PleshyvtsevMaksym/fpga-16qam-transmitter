"""Transmitter and noise parameters of the 16-QAM model (values only)."""


# ============================================================
# 1. MODULATION PARAMETERS
# ============================================================

BITS_PER_SYMBOL = 4
NUM_SYMBOLS = 4096
NUM_BITS = NUM_SYMBOLS * BITS_PER_SYMBOL

QAM_AVG_ENERGY = 10.0


# ============================================================
# 2. SIGNAL PARAMETERS
# ============================================================

SYMBOL_RATE = 1_000_000       # 1 MBd
SAMPLE_RATE = 100_000_000     # 100 MSPS
IF_FREQUENCY = 5_000_000      # 5 MHz

SAMPLES_PER_SYMBOL = SAMPLE_RATE // SYMBOL_RATE


# ============================================================
# 3. RRC FILTER PARAMETERS
# ============================================================

RRC_ROLLOFF = 0.35
RRC_SPAN = 8                  # Symbols


# ============================================================
# 4. HARDWARE PARAMETERS (datasheet values)
# ============================================================
# Every value comes from a datasheet (cite it here) or is marked
# ASSUMPTION until the component is chosen. None = effect switched off.

NOISE_SEED = 1                     # Same random effects on every run

# 4.1 DAC resolution and scaling
DAC_BITS = 12                      # Resolution N; ASSUMPTION (no DAC chosen yet)
DAC_BACKOFF_DB = 0.0               # Signal peak below DAC full scale, dB (design choice)

# 4.2 DAC output noise
DAC_NSD_DBFS_HZ = -150.0           # Noise spectral density, dBFS/Hz; ASSUMPTION

# 4.3 Clock
CLOCK_JITTER_RMS = 1e-12           # Oscillator RMS phase jitter (12 kHz-20 MHz), s; ASSUMPTION
DAC_JITTER_RMS = None              # DAC internal (aperture) jitter, s; None = not given
