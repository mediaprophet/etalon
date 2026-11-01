#!/usr/bin/env python3
"""Execute the assembled first-party WASM page in local headless Chrome."""

import argparse
from functools import partial
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

MAX_DOM_BYTES = 2 * 1024 * 1024
MAX_PROFILE_BYTES = 64 * 1024 * 1024


class TerminalState(HTMLParser):
    def __init__(self):
        super().__init__()
        self.state = {}

    def handle_starttag(self, tag, attrs):
        if tag == "body":
            self.state = dict(attrs)


def check_result(document, require_gpu=False, boundary_mode=None):
    parser = TerminalState()
    parser.feed(document)
    state = parser.state
    if state.get("data-etalon-state") != "complete":
        raise ValueError("browser tests never reached the terminal completion marker")
    counts = {}
    for name in ("passed", "failed", "unsupported", "not-run"):
        value = state.get("data-etalon-" + name, "")
        if not value.isascii() or not value.isdecimal():
            raise ValueError("browser completion counters are missing or invalid")
        counts[name] = int(value)
    expected_qualification = "passed" if counts == {"passed": 20, "failed": 0, "unsupported": 0, "not-run": 0} else "failed"
    if state.get("data-etalon-qualification") != expected_qualification:
        raise ValueError("overall qualification is missing or contradicts the actual case results")
    if boundary_mode is not None:
        if require_gpu or boundary_mode not in ("adapter-null", "adapter-rejected"):
            raise ValueError("invalid synthetic boundary qualification mode")
        expected = {"passed": 2, "failed": 0, "unsupported": 18, "not-run": 0}
        if boundary_mode == "adapter-rejected":
            expected.update(failed=18, unsupported=0)
        if counts != expected:
            raise ValueError("synthetic adapter boundary counts differ: " + json.dumps(counts))
        return counts
    if counts["passed"] < 2 or counts["failed"] != 0 or counts["not-run"] != 0 or sum(counts.values()) != 20:
        raise ValueError("WASM tests failed or did not execute: " + json.dumps(counts) + "; cases=" + state.get("data-etalon-errors", "")[:4000])
    if require_gpu and (counts["passed"] != 20 or counts["unsupported"] != 0):
        raise ValueError("GPU qualification requires all 20 cases: " + json.dumps(counts))
    return counts


class LocalHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def run(site):
    site = site.resolve(strict=True)
    if not (site / "testing.html").is_file():
        raise ValueError("assembled browser test page is missing")
    chrome = os.environ.get("CHROME_BIN") or next(
        (path for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")
         if (path := shutil.which(name))), None
    )
    if not chrome:
        raise ValueError("headless Chrome is required for real WASM browser qualification")
    handler = partial(LocalHandler, directory=str(site))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            for mode, feature_level in (("default-availability", "core"),
                                        ("software-webgpu", "core"),
                                        ("software-webgpu", "compatibility"),
                                        ("adapter-null", "core"),
                                        ("adapter-null", "compatibility"),
                                        ("adapter-rejected", "core")):
                with tempfile.TemporaryDirectory(prefix="etalon-browser-smoke-") as scratch:
                    profile = Path(scratch) / "profile"
                    command = [
                        "node", str(Path(__file__).resolve().parents[2] / "tests/wasm/runner/docs-smoke.js"),
                        "--chrome", chrome, "--profile", str(profile),
                        "--mode", mode, "--feature-level", feature_level,
                        "--url", f"http://127.0.0.1:{server.server_port}/testing.html",
                    ]
                    result = subprocess.run(command, capture_output=True, timeout=110)
                    if result.returncode:
                        raise ValueError("headless Chrome failed: " + result.stderr[-2000:].decode("utf-8", "replace"))
                    if len(result.stdout) > MAX_DOM_BYTES:
                        raise ValueError("browser DOM budget exceeded")
                    profile_size = sum(p.stat().st_size for p in profile.rglob("*") if p.is_file())
                    if profile_size > MAX_PROFILE_BYTES:
                        raise ValueError("browser profile budget exceeded")
                    boundary = mode if mode in ("adapter-null", "adapter-rejected") else None
                    counts = check_result(result.stdout.decode("utf-8"), require_gpu=(mode == "software-webgpu"), boundary_mode=boundary)
                    print(json.dumps({"mode": mode, "requested_feature_level": feature_level,
                                      "qualification": "passed" if counts["passed"] == 20 else "failed",
                                      "execution": "synthetic adapter boundary through Rust/WASM" if boundary else "Rust/WASM over browser WebGPU", **counts}))
                    print(result.stderr.decode("utf-8", "replace")[-16000:])
        finally:
            server.shutdown()
            worker.join(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True, type=Path)
    parser.add_argument("--adb-device", help="Qualify an explicitly selected physical Android device")
    parser.add_argument("--expected-revision")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--feature-level", choices=("core", "compatibility"), default="core")
    parser.add_argument("--timeout", type=int, default=120)
    arguments = parser.parse_args()
    try:
        if arguments.adb_device:
            if not arguments.expected_revision or not arguments.receipt:
                raise ValueError("Android qualification requires --expected-revision and --receipt")
            from android_smoke import run as run_android
            run_android(arguments.site, arguments.adb_device, arguments.expected_revision,
                        arguments.receipt, arguments.feature_level, arguments.timeout)
        else:
            run(arguments.site)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Assembled browser qualification failed: {error}\n")


if __name__ == "__main__":
    main()
