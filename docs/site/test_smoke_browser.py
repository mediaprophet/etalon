"""Completion parsing must fail closed before real Chrome execution in CI."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("etalon_browser_smoke", Path(__file__).with_name("smoke_browser.py"))
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


class BrowserCompletionTests(unittest.TestCase):
    def test_complete_portable_cases_preserve_unsupported_count(self):
        result = SMOKE.check_result('<body data-etalon-state="complete" data-etalon-passed="7" data-etalon-failed="0" data-etalon-unsupported="6"></body>')
        self.assertEqual(result, {"passed": 7, "failed": 0, "unsupported": 6})

    def test_nonterminal_results_cannot_pass(self):
        with self.assertRaises(ValueError):
            SMOKE.check_result('<body data-etalon-state="running" data-etalon-passed="7" data-etalon-failed="0" data-etalon-unsupported="0"></body>')

    def test_failed_case_cannot_pass(self):
        with self.assertRaises(ValueError):
            SMOKE.check_result('<body data-etalon-state="complete" data-etalon-passed="7" data-etalon-failed="1" data-etalon-unsupported="0"></body>')

    def test_missing_counters_cannot_pass(self):
        with self.assertRaises(ValueError):
            SMOKE.check_result('<body data-etalon-state="complete"></body>')


if __name__ == "__main__":
    unittest.main()
