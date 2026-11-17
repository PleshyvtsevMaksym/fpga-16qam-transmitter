"""
Run the complete verification suite.

From the 02_DSP_Model folder:
    py -m tests.test
"""

import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
MODEL_DIR = TESTS_DIR.parent


def main():

    # 1. Collect every tests/test_*.py module
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(TESTS_DIR),
        pattern="test_*.py",
        top_level_dir=str(MODEL_DIR),
    )

    # 2. Run with one line per test case
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    # 3. Summary
    failed = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = result.testsRun - failed - errors - skipped

    print("=" * 60)
    print(f"PASS: {passed}   FAIL: {failed}   ERROR: {errors}   SKIPPED: {skipped}")
    print("RESULT:", "PASS" if result.wasSuccessful() else "FAIL")
    print("=" * 60)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
