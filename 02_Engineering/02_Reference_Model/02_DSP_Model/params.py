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
# 4. NOISE PARAMETERS
# ============================================================

AWGN_SNR_DB = 20              # None = clean signal
NOISE_SEED = 1                # Same noise on every run