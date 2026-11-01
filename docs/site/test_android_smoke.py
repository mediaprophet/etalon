"""Device ownership and terminal reports; all ADB/CDP interactions are mocked."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("android_smoke", Path(__file__).with_name("android_smoke.py"))
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)
REVISION = "a" * 40


def report():
    return {
        "source_revision": REVISION,
        "runtime": {"source_revision": REVISION, "backend": "BrowserWebGpu",
                    "gpu_status": "available", "device_type": "Other",
                    "requested_feature_level": "core", "bounds": {"cases": 2}},
        "results": [{"id": "cpu", "status": "pass", "requires_gpu": False},
                    {"id": "gpu", "status": "pass", "requires_gpu": True}],
        "counts": {"pass": 2, "fail": 0, "unsupported": 0, "not_run": 0},
        "qualification": {"status": "passed", "expected_cases": 2, "expected_gpu_cases": 1,
                          "gpu_cases": [{"id": "gpu", "status": "pass"}]},
    }


class ReportTests(unittest.TestCase):
    def test_blank_browser_identity_is_not_misclassified_as_software(self):
        SMOKE.check_report(report(), REVISION)

    def test_source_catalog_profile_and_software_fail_closed(self):
        changes = [
            lambda r: r.update(source_revision="b" * 40),
            lambda r: r["runtime"].update(source_revision="b" * 40),
            lambda r: r["runtime"].update(requested_feature_level="compatibility"),
            lambda r: r["runtime"].update(device_type="Cpu"),
            lambda r: r["runtime"].update(adapter="Google SwiftShader"),
            lambda r: r["runtime"].update(backend="Empty"),
            lambda r: r["runtime"]["bounds"].update(cases=13),
            lambda r: r["results"][1].update(status="unsupported"),
            lambda r: r["results"][1].update(requires_gpu=False),
            lambda r: r["qualification"].update(expected_gpu_cases=0),
            lambda r: r["qualification"]["gpu_cases"][0].update(id="cpu"),
            lambda r: r["counts"].update(fail=1),
        ]
        for change in changes:
            candidate = report()
            change(candidate)
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                SMOKE.check_report(candidate, REVISION)
        for invalid in ([], {}, {"runtime": []}):
            with self.assertRaises(ValueError):
                SMOKE.check_report(invalid, REVISION)

    def test_receipt_cannot_escape_selected_repository_root_or_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="etalon-android-test-") as scratch:
            root = Path(scratch)
            with patch.object(SMOKE, "REPO", root):
                output = root / "target/qualification/test.json"
                self.assertEqual(SMOKE.receipt_path(output), output)
                with self.assertRaises(ValueError):
                    SMOKE.receipt_path(root / "outside.json")
                output.parent.mkdir(parents=True)
                output.write_text("existing", encoding="utf-8")
                with self.assertRaises(ValueError):
                    SMOKE.receipt_path(output)


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.session = SMOKE.DeviceSession("phone-serial", 43210, "unique", timeout=1)
        self.commands = []
        self.old = {"id": "existing", "type": "page", "url": "https://example.org"}
        self.own = None

    def command(self, args, timeout=20):
        self.commands.append(args)
        self.assertEqual(args[:3], ["adb", "-s", "phone-serial"])
        tail = args[3:]
        if tail == ["get-state"]:
            return "device\n"
        if tail == ["reverse", "--list"]:
            return "phone-serial tcp:8080 tcp:7777\n"
        if tail == ["forward", "tcp:0", "localabstract:chrome_devtools_remote"]:
            return "45678\n"
        if tail[:3] == ["shell", "am", "start"]:
            self.own = {"id": "owned", "type": "page", "url": self.session.url,
                        "webSocketDebuggerUrl": "ws://localhost:9222/devtools/page/owned"}
        return ""

    def pages(self, port):
        self.assertEqual(port, 45678)
        return [self.old] + ([self.own] if self.own else [])

    def test_success_uses_unused_reverse_port_and_closes_only_owned_tab(self):
        with patch.object(SMOKE, "command", self.command), patch.object(SMOKE, "pages", self.pages), patch.object(SMOKE, "http_get") as http:
            endpoint = self.session.open("compatibility")
            self.assertEqual(endpoint, "ws://127.0.0.1:45678/devtools/page/owned")
            self.assertIn("feature-level=compatibility", self.session.url)
            self.assertIn("autorun=0", self.session.url)
            self.session.close()
        tails = [args[3:] for args in self.commands]
        self.assertIn(["reverse", "--no-rebind", "tcp:8081", "tcp:43210"], tails)
        self.assertIn(["reverse", "--remove", "tcp:8081"], tails)
        self.assertIn(["forward", "--remove", "tcp:45678"], tails)
        http.assert_called_once_with(45678, "/json/close/owned")

    def test_forward_failure_cleans_reverse_without_touching_tabs(self):
        def fail_forward(args, timeout=20):
            if args[3:5] == ["forward", "tcp:0"]:
                raise ValueError("forward unavailable")
            return self.command(args, timeout)
        with patch.object(SMOKE, "command", fail_forward), patch.object(SMOKE, "pages") as pages:
            with self.assertRaisesRegex(ValueError, "forward unavailable"):
                try:
                    self.session.open("core")
                finally:
                    self.session.close()
        pages.assert_not_called()
        self.assertEqual(self.commands[-1][3:], ["reverse", "--remove", "tcp:8081"])

    def test_launch_timeout_cleans_ports_and_preserves_existing_page(self):
        with patch.object(SMOKE, "command", self.command), patch.object(SMOKE, "pages", return_value=[self.old]), patch.object(SMOKE.time, "monotonic", side_effect=[0, 2]), patch.object(SMOKE, "http_get") as http:
            with self.assertRaises(TimeoutError):
                try:
                    self.session.open("core")
                finally:
                    self.session.close()
        http.assert_not_called()
        self.assertEqual(self.commands[-1][3:], ["reverse", "--remove", "tcp:8081"])

    def test_tab_cleanup_failure_still_removes_both_owned_ports(self):
        with patch.object(SMOKE, "command", self.command), patch.object(SMOKE, "pages", self.pages), patch.object(SMOKE, "http_get", side_effect=OSError("socket gone")):
            self.session.open("core")
            with self.assertRaisesRegex(ValueError, "tab cleanup"):
                self.session.close()
        self.assertEqual([args[3:] for args in self.commands[-2:]],
                         [["forward", "--remove", "tcp:45678"], ["reverse", "--remove", "tcp:8081"]])

    def test_error_after_launch_finds_own_nonce_tab_without_target_id(self):
        with patch.object(SMOKE, "command", self.command), patch.object(SMOKE, "pages", self.pages), patch.object(SMOKE, "http_get") as http:
            self.session.open("core")
            self.session.target_id = None
            self.session.close()
        http.assert_called_once_with(45678, "/json/close/owned")


class EnvironmentTests(unittest.TestCase):
    def test_missing_protection_is_unknown_and_only_read_commands_are_used(self):
        with patch.object(SMOKE, "command", side_effect=["null\n", "fingerprint", "gpu detail"]) as call:
            result = SMOKE.device_environment("selected-phone")
        fields = result["fields"]
        self.assertEqual(fields["advanced_protection_mode"]["interpretation"], "unknown")
        self.assertEqual(fields["advanced_protection_mode"]["status"], "unavailable")
        self.assertEqual([c.args[0] for c in call.call_args_list], [
            ["adb", "-s", "selected-phone", "shell", "settings", "get", "secure", "advanced_protection_mode"],
            ["adb", "-s", "selected-phone", "shell", "getprop", "ro.build.fingerprint"],
            ["adb", "-s", "selected-phone", "shell", "dumpsys", "gpu"],
        ])

    def test_errors_and_oversized_unicode_are_bounded_and_not_inferred(self):
        with patch.object(SMOKE, "command", side_effect=[ValueError("permission denied"), "🛰" * 1000, TimeoutError("deadline")]):
            fields = SMOKE.device_environment("phone")["fields"]
        self.assertEqual(fields["advanced_protection_mode"]["status"], "unavailable")
        self.assertTrue(fields["build_fingerprint"]["truncated"])
        self.assertLessEqual(len(fields["build_fingerprint"]["value"].encode("utf-8")), 1024)
        self.assertEqual(fields["gpu_service"]["status"], "unavailable")


class OrchestrationTests(unittest.TestCase):
    def test_receipt_retains_full_report_and_cleanup_runs_on_pass_or_failure(self):
        for passes in (True, False):
            candidate = report()
            candidate["results"][1]["details"] = {"evidence": [1, 2, 3]}
            if not passes:
                candidate["results"][1]["status"] = "unsupported"
            with tempfile.TemporaryDirectory(prefix="etalon-android-test-") as scratch, patch.object(SMOKE, "DeviceSession") as cls, patch.object(SMOKE, "command", return_value=json.dumps(candidate)):
                cls.return_value.open.return_value = "ws://127.0.0.1:1234/devtools/page/own"
                cls.return_value.url = "http://127.0.0.1:8080/testing.html?etalon-run=own"
                path = Path(scratch) / "receipt.json"
                if passes:
                    SMOKE.qualify_device("phone", 12345, REVISION, path)
                else:
                    with self.assertRaises(ValueError):
                        SMOKE.qualify_device("phone", 12345, REVISION, path)
                cls.return_value.close.assert_called_once()
                receipt = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(receipt["report"], candidate)
                self.assertEqual(receipt["qualification"], "passed" if passes else "failed")

    def test_timeout_and_cleanup_failure_preserve_both_errors(self):
        with tempfile.TemporaryDirectory(prefix="etalon-android-test-") as scratch, patch.object(SMOKE, "DeviceSession") as cls, patch.object(SMOKE, "command", side_effect=TimeoutError("protocol timeout")):
            cls.return_value.close.side_effect = ValueError("reverse cleanup failed")
            path = Path(scratch) / "failed.json"
            with self.assertRaisesRegex(ValueError, "protocol timeout; cleanup also failed: reverse cleanup failed"):
                SMOKE.qualify_device("phone", 12345, REVISION, path)
            cls.return_value.close.assert_called_once()
            receipt = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["qualification"], "failed")
            self.assertIsNone(receipt["report"])

    def test_command_budget_and_deadline_are_enforced(self):
        with patch.object(SMOKE, "MAX_REPORT_BYTES", 1024):
            with self.assertRaisesRegex(ValueError, "budget"):
                SMOKE.command([sys.executable, "-c", "import sys;sys.stdout.write('x'*8192)"])
        with self.assertRaises(subprocess.TimeoutExpired):
            SMOKE.command([sys.executable, "-c", "import time;time.sleep(10)"], timeout=0.05)


@unittest.skipUnless(shutil.which("node"), "Node is required for mocked CDP tests")
class ProtocolTests(unittest.TestCase):
    def test_browser_diagnostic_bounds_and_rendering_events(self):
        result = subprocess.run(["node", "--test", str(SMOKE.REPO / "tests/wasm/runner/android-diagnostics.test.js")], capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout.decode("utf-8", "replace") + result.stderr.decode("utf-8", "replace"))

    def test_page_protocol_completion_timeout_errors_and_disconnect(self):
        module = (SMOKE.REPO / "tests/wasm/runner/android-smoke.js").as_uri()
        code = """
import assert from 'node:assert/strict';
const { CdpClient, collectReport, MAX_REPORT_BYTES } = await import(MODULE);
class Socket extends EventTarget {
  closed = false;
  requests = [];
  send(data) {
    const request = JSON.parse(data); this.requests.push(request);
    queueMicrotask(() => this.dispatchEvent(new MessageEvent('message', {
      data: JSON.stringify({ id: request.id, result: { result: { value: { state: 'complete', report: '{"ok":true}' } } } })
    })));
  }
  close() { this.closed = true; this.dispatchEvent(new Event('close')); }
}
const socket = new Socket();
const client = new CdpClient(socket);
socket.dispatchEvent(new MessageEvent('message', {data: JSON.stringify({method: 'Log.entryAdded', params: {entry: {source: 'rendering', level: 'warning', text: 'No available adapters.'}}})}));
const report = JSON.parse(await collectReport(client, 'http://127.0.0.1:8080/testing.html?etalon-run=own', 1000));
assert.equal(report.ok, true);
assert.equal(report.browser_diagnostics.automatic_upload, false);
assert.equal(report.browser_diagnostics.events[0].text, 'No available adapters.');
assert.deepEqual(socket.requests.slice(0, 2).map(r => r.method), ['Log.enable', 'Runtime.enable']);
assert.equal(socket.requests[2].method, 'Runtime.evaluate');
assert.ok(socket.requests[2].params.expression.includes('location.href !=='));
client.close(); assert.ok(socket.closed);
assert.ok(socket.requests.every(r => r.method !== 'Browser.close' && r.method !== 'Target.closeTarget'));
let now = 0;
await assert.rejects(collectReport({ call: async () => ({ result: { value: { state: 'running' } } }) }, 'own', 2, async () => { now++; }, () => now), /did not complete/);
await assert.rejects(collectReport({ call: async () => ({ exceptionDetails: { text: 'page failed' } }) }, 'own', 20), /page failed/);
await assert.rejects(collectReport({ call: async () => ({ result: { value: { state: 'complete', report: 'x'.repeat(MAX_REPORT_BYTES + 1) } } }) }, 'own', 20), /oversized/);
const silent = new Socket(); silent.send = () => {};
const pending = new CdpClient(silent);
await assert.rejects(pending.call('Runtime.evaluate', {}, 1), /timed out/);
const interrupted = pending.call('Runtime.evaluate', {}, 1000);
pending.close(); await assert.rejects(interrupted, /session ended/);
await assert.rejects(CdpClient.connect('ws://127.0.0.1:9222/devtools/browser/other'), /page endpoint/);
class Refused extends Socket { constructor() { super(); queueMicrotask(() => this.dispatchEvent(new Event('error'))); } }
await assert.rejects(CdpClient.connect('ws://127.0.0.1:9222/devtools/page/own', 100, Refused), /connection failed/);
""".replace("MODULE", json.dumps(module))
        result = subprocess.run(["node", "--input-type=module", "-e", code], capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))


if __name__ == "__main__":
    unittest.main()
