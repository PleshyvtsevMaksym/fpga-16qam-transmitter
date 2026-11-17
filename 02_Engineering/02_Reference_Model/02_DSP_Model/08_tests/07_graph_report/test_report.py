"""Report tests: version folders, parameter block, page template and a full PDF build."""

import contextlib
import io
import re
import tempfile
import unittest
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt

import params

context = import_module("07_graph_report.01_report_context")
page_module = import_module("07_graph_report.02_report_page")
figures = import_module("07_graph_report.03_report_figures")
build = import_module("07_graph_report.04_build_report")


class TestVersionFolders(unittest.TestCase):

    def test_first_version_is_v0_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(context.next_version_dir(Path(tmp) / "graphs", False).name, "v0.0")

    def test_minor_and_major_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            names = [context.next_version_dir(base, major).name
                     for major in (False, False, True, False)]
            self.assertEqual(names, ["v0.0", "v0.1", "v1.0", "v1.1"])

    def test_numeric_not_text_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "v0.9").mkdir()
            (base / "v0.10").mkdir()
            self.assertEqual(context.next_version_dir(base, False).name, "v0.11")

    def test_other_folders_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "old").mkdir()
            (base / "v2.3.pdf").write_text("x")
            self.assertEqual(context.next_version_dir(base, False).name, "v0.0")


class TestParameterBlock(unittest.TestCase):

    def test_hz_format(self):
        self.assertEqual([context._hz(f) for f in (100, 1e3, 10e3, 1e6)], ["100", "1k", "10k", "1M"])

    def test_four_columns_fit_the_page(self):
        columns = context.parameter_columns()
        self.assertEqual(len(columns), 4)
        for column in columns:
            self.assertLessEqual(max(len(line) for line in column.splitlines()), 46)
            self.assertLessEqual(len(column.splitlines()), 10)

    def test_key_values_present(self):
        text = "\n".join(context.parameter_columns())
        for expected in (params.FPGA_BOARD, f"{params.DAC_BITS} bit",
                         f"{params.SAMPLE_RATE / 1e6:g} MSa/s", f"{params.IF_FREQUENCY / 1e6:g} MHz"):
            self.assertIn(expected, text)

    def test_run_stamp(self):
        stamp = context.run_stamp("v3.4")
        self.assertIn("v3.4", stamp)
        self.assertRegex(stamp, r"\d{4}-\d{2}-\d{2}")


class TestPageTemplate(unittest.TestCase):

    def test_page_is_a4_landscape_with_texts(self):
        fig, ax = page_module.new_page("03-2", "Title", "v0.0 | stamp", context.parameter_columns(),
                                       {"EVM": "1.6 %"}, "Description text.")
        try:
            width, height = fig.get_size_inches()
            self.assertAlmostEqual(width, 11.69, places=2)
            self.assertAlmostEqual(height, 8.27, places=2)
            texts = " ".join(t.get_text() for t in fig.texts)
            for expected in ("03-2", "Title", "v0.0 | stamp", "EVM", "1.6 %", "Description text."):
                self.assertIn(expected, texts)
            self.assertIn(ax, fig.axes)
        finally:
            plt.close(fig)

    def test_page_list_in_chain_order(self):
        names = [name for name, _ in figures.PAGES]
        self.assertEqual(len(names), len(set(names)))
        numbers = [tuple(int(x) for x in re.match(r"(\d+)-(\d+)_", n).groups()) for n in names]
        self.assertEqual(numbers, sorted(numbers))


class TestFullBuild(unittest.TestCase):
    """One real build into a temporary folder (runs the whole model, about 30 s)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        with contextlib.redirect_stdout(io.StringIO()):
            cls.folder = build.build_report(Path(cls.tmp.name), False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_page_saved(self):
        for name, _ in figures.PAGES:
            path = self.folder / f"{name}.pdf"
            self.assertTrue(path.is_file(), name)
            self.assertTrue(path.read_bytes().startswith(b"%PDF"))

    def test_combined_report_and_traceability(self):
        self.assertEqual(self.folder.name, "v0.0")
        self.assertTrue((self.folder / "DSP_Report_v0.0.pdf").is_file())
        self.assertIn("EVM", (self.folder / "terminal_report.txt").read_text(encoding="utf-8"))
        self.assertEqual((self.folder / "params_snapshot.py").read_text(encoding="utf-8"),
                         Path(params.__file__).read_text(encoding="utf-8"))

    def test_no_figures_left_open(self):
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
