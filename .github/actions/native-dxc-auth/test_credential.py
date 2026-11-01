"""Validate opaque synthetic credentials with the real curl netrc parser."""
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest


@unittest.skipUnless(shutil.which("pwsh"), "PowerShell host required")
class CredentialTests(unittest.TestCase):
    def serialize(self, values):
        env = dict(os.environ, DXC_SERIALIZER=str(Path(__file__).with_name("credential.ps1")))
        env.pop("DXC_GITHUB_TOKEN", None)
        command = r"""
$ErrorActionPreference = 'Stop'
. $env:DXC_SERIALIZER
$values = [Console]::In.ReadToEnd() | ConvertFrom-Json
$results = @(foreach ($value in $values) {
    try { @{ content = (ConvertTo-DxcNetrc -Token $value); rejected = $false } }
    catch { @{ content = $null; rejected = $true } }
})
ConvertTo-Json -InputObject $results -Compress
"""
        result = subprocess.run(
            [shutil.which("pwsh"), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
            input=json.dumps(values), env=env, capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_punctuation_base64_jwt_and_quotes_have_exact_serialization(self):
        cases = [
            ("ghs_synthetic", '"ghs_synthetic"'),
            ("+/base64==", '"+/base64=="'),
            ("eyJ.test-payload.signature_123", '"eyJ.test-payload.signature_123"'),
            ('space # : ; "quote" \\ slash', '"space # : ; \\"quote\\" \\\\ slash"'),
            ("référence", '"référence"'),
        ]
        results = self.serialize([value for value, _ in cases])
        for (_, expected), result in zip(cases, results):
            self.assertFalse(result["rejected"])
            self.assertEqual(result["content"], "machine api.github.com\nlogin x-access-token\npassword " + expected + "\n")

    def test_controls_line_separators_empty_and_byte_overflow_are_rejected(self):
        controls = list(range(32)) + list(range(127, 160)) + [0x2028, 0x2029]
        values = [None, "", "a" * 8193, "é" * 4097]
        values += ["before" + chr(value) + "after" for value in controls]
        self.assertTrue(all(result["rejected"] for result in self.serialize(values)))
        self.assertFalse(self.serialize(["a" * 8192])[0]["rejected"])
        self.assertFalse(self.serialize(["é" * 4096])[0]["rejected"])

    @unittest.skipUnless(shutil.which("curl.exe") or shutil.which("curl"), "curl required")
    def test_curl_preserves_opaque_password_and_does_not_authenticate_another_host(self):
        values = ["+/base64==", "eyJ.test-payload.signature_123", 'a" default login injected password fake \\ end']
        contents = self.serialize(values)
        curl = shutil.which("curl.exe") or shutil.which("curl")
        for token, result in zip(values, contents):
            self.assertFalse(result["rejected"])
            for host in ("api.github.com", "unmatched.invalid"):
                with self.subTest(host=host), tempfile.TemporaryDirectory(prefix="etalon-dxc-netrc-test-") as scratch:
                    netrc = Path(scratch) / "synthetic.netrc"
                    netrc.write_text(result["content"], encoding="utf-8")
                    received = []

                    class Capture(BaseHTTPRequestHandler):
                        def do_GET(self):
                            received.append(self.headers.get("Authorization"))
                            self.send_response(200)
                            self.send_header("Content-Length", "0")
                            self.end_headers()

                        def log_message(self, *_):
                            pass

                    with HTTPServer(("127.0.0.1", 0), Capture) as server:
                        server.timeout = 5
                        port = server.server_address[1]
                        worker = threading.Thread(target=server.handle_request, daemon=True)
                        worker.start()
                        response = subprocess.run(
                            [curl, "-q", "--silent", "--show-error", "--fail", "--max-time", "4",
                             "--noproxy", "*", "--netrc-optional", "--netrc-file", str(netrc),
                             "--connect-to", f"{host}:{port}:127.0.0.1:{port}", f"http://{host}:{port}/"],
                            capture_output=True, text=True, encoding="utf-8", timeout=8,
                        )
                        worker.join(timeout=6)
                    self.assertFalse(worker.is_alive())
                    self.assertEqual(response.returncode, 0, response.stderr)
                    expected = None
                    if host == "api.github.com":
                        expected = "Basic " + base64.b64encode(("x-access-token:" + token).encode()).decode()
                    self.assertEqual(received, [expected])


if __name__ == "__main__":
    unittest.main()
