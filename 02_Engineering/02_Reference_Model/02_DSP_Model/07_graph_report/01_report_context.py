"""Report 01 context: output folder versions, git state and the parameter block."""

import re
import subprocess
from datetime import datetime
from pathlib import Path

import params

MODEL_DIR = Path(__file__).resolve().parent.parent
GRAPHS_DIR = MODEL_DIR.parent / "results" / "DSP_Graphs"


def next_version_dir(base_dir: Path, new_major: bool) -> Path:
    """
    Create the next version folder vMAJOR.MINOR (v0.0, v0.1, ...).

    Parameters
    ----------
    base_dir : Path
        Folder that holds the version folders.
    new_major : bool
        True starts the next major version (v1.0 after v0.7).

    Returns
    -------
    Path
        The new, empty version folder.
    """

    # 1. Existing versions
    base_dir.mkdir(parents=True, exist_ok=True)
    versions = []
    for folder in base_dir.iterdir():
        match = re.fullmatch(r"v(\d+)\.(\d+)", folder.name)
        if folder.is_dir() and match:
            versions.append((int(match.group(1)), int(match.group(2))))

    # 2. Next version number
    if not versions:
        major, minor = 0, 0
    else:
        last_major, last_minor = max(versions)
        major, minor = (last_major + 1, 0) if new_major else (last_major, last_minor + 1)

    # 3. Create the folder
    folder = base_dir / f"v{major}.{minor}"
    folder.mkdir()
    return folder


def git_state() -> str:
    """
    Short commit hash of the model, with "+changes" when files are not committed.

    Returns
    -------
    str
        For example "7185a93+changes", or "no git" outside a repository.
    """

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=MODEL_DIR,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "."], cwd=MODEL_DIR,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "no git"
    return commit + ("+changes" if dirty else "")


def _value(value, scale, unit):
    """Format a parameter that may be None (effect switched off)."""
    return "off" if value is None else f"{value * scale:g} {unit}"


def _hz(frequency):
    """Short frequency text: 100, 1k, 10k, 1M."""

    for scale, unit in ((1e6, "M"), (1e3, "k")):
        if frequency >= scale:
            return f"{frequency / scale:g}{unit}"
    return f"{frequency:g}"


def parameter_columns() -> list[str]:
    """
    Parameters of the run, as four text columns for the top of every page.

    Returns
    -------
    list of str
        Transmitter, FPGA, DAC and clock/output/instrument columns.
    """

    p = params
    mask = ["off"] if p.CLOCK_PHASE_NOISE_DBC_HZ is None else [
        f"{_hz(f)} {l:g}" for f, l in zip(p.CLOCK_PHASE_NOISE_OFFSETS_HZ, p.CLOCK_PHASE_NOISE_DBC_HZ)]
    mask_lines = [", ".join(mask[i:i + 3]) for i in range(0, len(mask), 3)]
    transmitter = "\n".join([
        "TRANSMITTER (01)",
        f"Modulation      Gray 16-QAM, PRBS15",
        f"Symbols         {p.NUM_SYMBOLS}",
        f"Symbol rate     {p.SYMBOL_RATE / 1e6:g} MBd",
        f"Sample rate     {p.SAMPLE_RATE / 1e6:g} MSa/s",
        f"Samples/symbol  {p.SAMPLES_PER_SYMBOL}",
        f"IF              {p.IF_FREQUENCY / 1e6:g} MHz",
        f"RRC             a = {p.RRC_ROLLOFF}, span {p.RRC_SPAN}",
    ])
    fpga = "\n".join([
        "FPGA (02)",
        f"Board           {p.FPGA_BOARD}",
        f"Part / clock    {p.FPGA_PART}, {p.FPGA_CLOCK_HZ / 1e6:g} MHz",
        f"Coefficients    {p.FPGA_COEF_BITS} bit",
        f"FIR output      {p.FPGA_FIR_OUT_BITS} bit, {p.FPGA_ROUNDING}",
        f"DDS P/A/B       {p.FPGA_NCO_PHASE_BITS}/{p.FPGA_NCO_LUT_ADDR_BITS}/{p.FPGA_NCO_AMP_BITS} bit",
        f"Output gain     {p.FPGA_GAIN_BITS} bit",
        f"Interface       {p.FPGA_DAC_INTERFACE_PINS} pins, {p.FPGA_IO_STANDARD}",
    ])
    dac = "\n".join([
        "DAC (03)",
        f"Resolution      {p.DAC_BITS} bit, back-off {p.DAC_BACKOFF_DB:g} dB",
        f"HD2 / HD3       {_value(p.DAC_HD2_DBC, 1, 'dBc')} / {_value(p.DAC_HD3_DBC, 1, 'dBc')}",
        f"HD test level   {p.DAC_HD_TEST_LEVEL_DBFS:g} dBFS",
        f"Offset / gain   {_value(p.DAC_OFFSET_ERROR_PCT_FSR, 1, '%')} / "
        f"{_value(p.DAC_GAIN_ERROR_PCT_FSR, 1, '%')} FSR",
        f"NSD             {_value(p.DAC_NSD_DBFS_HZ, 1, 'dBFS/Hz')}",
        f"Glitch          {_value(p.DAC_GLITCH_IMPULSE_VS, 1e12, 'pV*s')}, "
        f"{p.DAC_GLITCH_DURATION_S * 1e9:g} ns",
        f"Settling        {_value(p.DAC_SETTLING_TIME_S, 1e9, 'ns')} to {p.DAC_SETTLING_ACCURACY * 100:g} %",
    ])
    other = "\n".join([
        "CLOCK / OUTPUT / INSTRUMENTS",
        f"Jitter clk/DAC  {_value(p.CLOCK_JITTER_RMS, 1e12, 'ps')} / {_value(p.DAC_JITTER_RMS, 1e12, 'ps')}",
        f"L(f) dBc/Hz     {mask_lines[0]}",
        *[f"                {line}" for line in mask_lines[1:]],
        f"Full scale      {p.DAC_FULL_SCALE_VOLTAGE:g} V peak",
        f"Analog grid     M = {p.ANALOG_OVERSAMPLING} ({p.ANALOG_OVERSAMPLING * p.SAMPLE_RATE / 1e6:g} MSa/s)",
        f"Analyzer RBW    {p.SPECTRUM_RBW_HZ / 1e3:g} kHz",
        f"VSA filter      RRC a = {p.VSA_RRC_ROLLOFF}, span {p.VSA_RRC_SPAN}",
    ])
    return [transmitter, fpga, dac, other]


def run_stamp(version: str) -> str:
    """Version, date and git state for the page header."""
    return f"{version}  |  {datetime.now():%Y-%m-%d %H:%M}  |  commit {git_state()}"
