"""Completion parsing must fail closed before real Chrome execution in CI."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("etalon_browser_smoke", Path(__file__).with_name("smoke_browser.py"))
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


class BrowserCompletionTests(unittest.TestCase):
    def test_complete_portable_cases_preserve_unsupported_count(self):
        result = SMOKE.check_result('<body data-etalon-state="complete" data-etalon-qualification="failed" data-etalon-passed="7" data-etalon-failed="0" data-etalon-unsupported="13" data-etalon-not-run="0"></body>')
        self.assertEqual(result, {"passed": 7, "failed": 0, "unsupported": 13, "not-run": 0})

    def test_gpu_qualification_rejects_unsupported_cases(self):
        with self.assertRaises(ValueError):
            SMOKE.check_result('<body data-etalon-state="complete" data-etalon-qualification="failed" data-etalon-passed="2" data-etalon-failed="0" data-etalon-unsupported="18" data-etalon-not-run="0"></body>', require_gpu=True)

    def test_gpu_qualification_requires_all_cases(self):
        result = SMOKE.check_result('<body data-etalon-state="complete" data-etalon-qualification="passed" data-etalon-passed="20" data-etalon-failed="0" data-etalon-unsupported="0" data-etalon-not-run="0"></body>', require_gpu=True)
        self.assertEqual(result["passed"], 20)

    def test_incomplete_case_count_cannot_pass(self):
        with self.assertRaises(ValueError):
            SMOKE.check_result('<body data-etalon-state="complete" data-etalon-qualification="failed" data-etalon-passed="2" data-etalon-failed="0" data-etalon-unsupported="0" data-etalon-not-run="0"></body>')

    def test_not_run_cases_cannot_pass(self):
        with self.assertRaises(ValueError):
            SMOKE.check_result('<body data-etalon-state="complete" data-etalon-qualification="failed" data-etalon-passed="2" data-etalon-failed="0" data-etalon-unsupported="10" data-etalon-not-run="1"></body>')

    def test_synthetic_rejection_requires_explicit_boundary_mode(self):
        document = '<body data-etalon-state="complete" data-etalon-qualification="failed" data-etalon-passed="2" data-etalon-failed="18" data-etalon-unsupported="0" data-etalon-not-run="0"></body>'
        self.assertEqual(SMOKE.check_result(document, boundary_mode="adapter-rejected")["failed"], 18)
        with self.assertRaises(ValueError):
            SMOKE.check_result(document)
        with self.assertRaises(ValueError):
            SMOKE.check_result(document, boundary_mode="adapter-null")
        with self.assertRaises(ValueError):
            SMOKE.check_result(document, require_gpu=True, boundary_mode="adapter-rejected")

    def test_synthetic_null_cannot_count_as_gpu_qualification(self):
        document = '<body data-etalon-state="complete" data-etalon-qualification="failed" data-etalon-passed="2" data-etalon-failed="0" data-etalon-unsupported="18" data-etalon-not-run="0"></body>'
        self.assertEqual(SMOKE.check_result(document, boundary_mode="adapter-null")["unsupported"], 18)
        with self.assertRaises(ValueError):
            SMOKE.check_result(document, require_gpu=True)
        with self.assertRaises(ValueError):
            SMOKE.check_result(document, boundary_mode="adapter-rejected")

    def test_overall_pass_cannot_hide_unavailable_gpu(self):
        document = '<body data-etalon-state="complete" data-etalon-qualification="passed" data-etalon-passed="2" data-etalon-failed="0" data-etalon-unsupported="18" data-etalon-not-run="0"></body>'
        for options in ({}, {"boundary_mode": "adapter-null"}, {"require_gpu": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SMOKE.check_result(document, **options)

    def test_overall_pass_cannot_hide_adapter_rejection(self):
        document = '<body data-etalon-state="complete" data-etalon-qualification="passed" data-etalon-passed="2" data-etalon-failed="18" data-etalon-unsupported="0" data-etalon-not-run="0"></body>'
        with self.assertRaises(ValueError):
            SMOKE.check_result(document, boundary_mode="adapter-rejected")

    def test_gpu_pass_requires_explicit_matching_overall_qualification(self):
        document = '<body data-etalon-state="complete" data-etalon-passed="20" data-etalon-failed="0" data-etalon-unsupported="0" data-etalon-not-run="0"></body>'
        for marker in ("", ' data-etalon-qualification="failed"', ' data-etalon-qualification="running"'):
            with self.subTest(marker=marker), self.assertRaises(ValueError):
                SMOKE.check_result(document.replace('<body', '<body' + marker), require_gpu=True)

    def test_nonterminal_results_cannot_pass(self):
        with self.assertRaises(ValueError):
            SMOKE.check_result('<body data-etalon-state="running" data-etalon-passed="7" data-etalon-failed="0" data-etalon-unsupported="0" data-etalon-not-run="0"></body>')

    def test_failed_case_cannot_pass(self):
        with self.assertRaises(ValueError):
            SMOKE.check_result('<body data-etalon-state="complete" data-etalon-qualification="failed" data-etalon-passed="7" data-etalon-failed="1" data-etalon-unsupported="0" data-etalon-not-run="0"></body>')

    def test_missing_counters_cannot_pass(self):
        with self.assertRaises(ValueError):
            SMOKE.check_result('<body data-etalon-state="complete" data-etalon-qualification="failed" data-etalon-not-run="0"></body>')


if __name__ == "__main__":
    unittest.main()
