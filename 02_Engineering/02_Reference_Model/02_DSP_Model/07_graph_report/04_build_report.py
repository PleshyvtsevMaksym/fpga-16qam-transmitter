"""Report 04 build: run the whole model once and save every page as PDF."""

import contextlib
import io
import shutil
from importlib import import_module
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import main
import params
import visualization

_pipeline = import_module("06_pipeline")
run_ideal_reference_pipeline = _pipeline.run_ideal_reference_pipeline
run_bit_true_fpga_pipeline = _pipeline.run_bit_true_fpga_pipeline
print_pipeline_report = _pipeline.print_pipeline_report
print_fixed_point_report = _pipeline.print_fixed_point_report

_context = import_module("07_graph_report.01_report_context")
_page = import_module("07_graph_report.02_report_page")
_figures = import_module("07_graph_report.03_report_figures")
apply_zero_order_hold = import_module("03_physical_dac").apply_zero_order_hold


def collect_data() -> dict:
    """
    Run 01 -> 05 and the EVM sweep with the values from params.

    Returns
    -------
    dict
        Every signal and measurement the report pages need.
    """

    # 1. Model chain (same calls as main.py)
    r = run_ideal_reference_pipeline()
    fpga = run_bit_true_fpga_pipeline() if main.USE_FIXED_POINT_FPGA else r
    dac, tx = main.run_hardware(fpga)
    fs_a, full_scale = dac["analog_sampling_rate"], dac["full_scale"]
    meas = main.measure(tx, r, fs_a, full_scale)

    # 2. Ideal signal on the same analog grid (reference for gain/phase/delay)
    ideal_sync = main.measure(apply_zero_order_hold(r["if_samples"], params.ANALOG_OVERSAMPLING),
                              r, fs_a, full_scale)["sync"]

    # 3. Ideal DAC: the ideal IF held on the analog grid (no DAC errors)
    ideal_held = apply_zero_order_hold(r["if_samples"], params.ANALOG_OVERSAMPLING)

    # 4. EVM sweep
    measured, theory = main.evm_nsd_sweep(fpga, r)

    return {
        "r": r, "fpga": fpga, "dac": dac, "tx": tx, "meas": meas, "ideal_sync": ideal_sync, "ideal_held": ideal_held,
        "sweep_x": main.EVM_NSD_SWEEP, "sweep_measured": measured, "sweep_theory": theory,
        "fs_a": fs_a, "m": params.ANALOG_OVERSAMPLING, "full_scale": full_scale,
        "rbw": params.SPECTRUM_RBW_HZ, "fc": params.IF_FREQUENCY,
        "half_band": (1 + params.RRC_ROLLOFF) * params.SYMBOL_RATE / 2,
        "start": len(r["rrc_coefficients"]),
    }


def terminal_report(data: dict) -> str:
    """
    Text of the terminal report (01, 02 and the transmitter output).

    Parameters
    ----------
    data : dict
        Output of collect_data().

    Returns
    -------
    str
        The report as printed by main.py.
    """

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        print_pipeline_report(data["r"])
        if main.USE_FIXED_POINT_FPGA:
            print_fixed_point_report(data["fpga"]["fixed_point"])
        sync, ideal = data["meas"]["sync"], data["ideal_sync"]
        print(f"EVM at the transmitter output: {data['meas']['evm_percent']:.2f} %")
        print(f"Hardware: {main.hardware_label()}")
        print(f"Analyzer vs ideal: gain {sync['gain'] / ideal['gain']:.4f}, "
              f"phase {sync['phase_deg'] - ideal['phase_deg']:+.3f} deg")
    return output.getvalue()


def build_report(output_dir: Path, new_major: bool) -> Path:
    """
    Save one PDF per page, the combined report, the terminal report and params.

    Parameters
    ----------
    output_dir : Path
        Folder that holds the version folders (results/DSP_Graphs).
    new_major : bool
        True starts a new major version (v1.0, v2.0, ...).

    Returns
    -------
    Path
        The version folder that was written.
    """

    # 1. Run the model
    print("Running the model (01 ... 05 and the EVM sweep) ...")
    data = collect_data()

    # 2. Version folder and page factory
    folder = _context.next_version_dir(output_dir, new_major)
    version = folder.name
    stamp = _context.run_stamp(version)
    columns = _context.parameter_columns()

    def page(number, title, values, description):
        return _page.new_page(number, title, stamp, columns, values, description)

    # 3. Pages: one PDF each and one combined PDF
    visualization.apply_plot_style()
    with matplotlib.rc_context({"figure.constrained_layout.use": False}), \
            PdfPages(folder / f"DSP_Report_{version}.pdf") as combined:
        for name, make_page in _figures.PAGES:
            fig = make_page(data, page)
            fig.savefig(folder / f"{name}.pdf")
            combined.savefig(fig)
            plt.close(fig)
            print(f"  {name}.pdf")

    # 4. Traceability: terminal report and the exact params.py of this run
    (folder / "terminal_report.txt").write_text(
        f"{stamp}\n\n{terminal_report(data)}", encoding="utf-8")
    shutil.copy(Path(params.__file__), folder / "params_snapshot.py")

    print(f"Report saved: {folder}")
    return folder
