"""
Run the verification suite, one folder per model part.

From the 02_DSP_Model folder:
    py -m 08_tests.test                     every folder
    py -m 08_tests.test 03_physical_dac     one folder (several allowed)
"""

import sys
import unittest
from collections import defaultdict
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
MODEL_DIR = TESTS_DIR.parent
FOLDERS = ("00_project", "01_ideal_reference", "02_bit_true_fpga", "03_physical_dac",
           "04_analog_output", "05_measurement_instruments", "06_pipeline", "07_graph_report",
           "visualization")


class _CountingResult(unittest.TextTestResult):
    """Text result that also counts passed tests per folder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed_by_folder = defaultdict(int)

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed_by_folder[test.id().split(".")[1]] += 1


def _folder(test_id: str) -> str:
    return test_id.split(".")[1]


def main(selected: list[str]) -> int:

    # 1. Collect tests/<folder>/test_*.py
    folders = selected or FOLDERS
    suite = unittest.TestSuite()
    for folder in folders:
        if folder not in FOLDERS:
            print(f"Unknown test folder: {folder}. Choose from {FOLDERS}.")
            return 2
        suite.addTests(unittest.defaultTestLoader.discover(
            start_dir=str(TESTS_DIR / folder), pattern="test_*.py",
            top_level_dir=str(MODEL_DIR),
        ))

    # 2. Run with one line per test
    runner = unittest.TextTestRunner(verbosity=2, resultclass=_CountingResult)
    result = runner.run(suite)

    # 3. Summary per folder
    failed = defaultdict(int)
    for test, _ in result.failures + result.errors:
        failed[_folder(test.id())] += 1

    print("=" * 60)
    print(f"{'Folder':<30}{'PASS':>8}{'FAIL':>8}")
    for folder in folders:
        print(f"{folder:<30}{result.passed_by_folder[folder]:>8}{failed[folder]:>8}")
    print("-" * 60)
    total_failed = len(result.failures) + len(result.errors)
    print(f"{'Total':<30}{sum(result.passed_by_folder.values()):>8}{total_failed:>8}")
    print("RESULT:", "PASS" if result.wasSuccessful() else "FAIL")
    print("=" * 60)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
