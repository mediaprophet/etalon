#!/usr/bin/env python3
"""Qualify an assembled WASM page on an explicitly selected Android device.

Requires Node 22+, adb on PATH, an unlocked USB-debugging-authorized phone,
and Chrome already open with its DevTools socket available. No browser flags,
protection settings, profiles, or existing tabs are changed. Reports are retained
only at the requested new path below this repository's target/qualification.
"""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import urlopen
import uuid

REPO = Path(__file__).resolve().parents[2]
MAX_REPORT_BYTES = 2 * 1024 * 1024


def command(args, timeout=20):
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        outputs = [bytearray(), bytearray()]
        overflow = threading.Event()
        lock = threading.Lock()

        def read(pipe, index):
            while chunk := pipe.read(4096):
                with lock:
                    if sum(map(len, outputs)) + len(chunk) > MAX_REPORT_BYTES:
                        overflow.set()
                        process.kill()
                        return
                    outputs[index].extend(chunk)

        readers = [threading.Thread(target=read, args=(pipe, i), daemon=True)
                   for i, pipe in enumerate((process.stdout, process.stderr))]
        for reader in readers:
            reader.start()
        try:
            process.wait(timeout=timeout)
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            for reader in readers:
                reader.join(timeout=5)
        if overflow.is_set():
            raise ValueError("Command output exceeds qualification budget")
        if any(reader.is_alive() for reader in readers):
            raise ValueError("Command output did not close")
        if process.returncode:
            raise ValueError("Command failed: " + outputs[1][-2000:].decode("utf-8", "replace"))
        return outputs[0].decode("utf-8")


def device_environment(serial):
    """Collect bounded raw evidence; an absent setting never means protection is off."""
    fields = {}
    probes = (
        ("advanced_protection_mode", ["settings", "get", "secure", "advanced_protection_mode"], 128),
        ("build_fingerprint", ["getprop", "ro.build.fingerprint"], 1024),
        ("gpu_service", ["dumpsys", "gpu"], 16384),
    )
    for name, args, budget in probes:
        try:
            raw = command(["adb", "-s", serial, "shell", *args], timeout=5).strip()
            encoded = raw.encode("utf-8")
            value = encoded[:budget].decode("utf-8", "ignore")
            fields[name] = {"status": "unavailable" if raw in ("", "null") else "reported",
                            "value": value, "truncated": len(encoded) > budget}
            if name == "advanced_protection_mode":
                fields[name]["interpretation"] = {"1": "reported-enabled", "0": "reported-disabled"}.get(raw, "unknown")
                fields[name]["user_scope"] = "ADB shell default settings user; not proof of browser policy"
        except (OSError, ValueError, subprocess.SubprocessError, TimeoutError) as error:
            fields[name] = {"status": "unavailable", "error": str(error)[:512]}
    return {"schema": "etalon.android-environment.v1", "read_only": True,
            "automatic_upload": False, "fields": fields}


def http_get(port, path):
    with urlopen(f"http://127.0.0.1:{port}{path}", timeout=3) as response:
        data = response.read(MAX_REPORT_BYTES + 1)
    if len(data) > MAX_REPORT_BYTES:
        raise ValueError("DevTools response budget exceeded")
    return data


def pages(port):
    result = json.loads(http_get(port, "/json/list"))
    if not isinstance(result, list):
        raise ValueError("Invalid DevTools target list")
    return result


class DeviceSession:
    def __init__(self, serial, host_port, token, timeout=120):
        self.serial, self.host_port, self.token = serial, host_port, token
        self.timeout = timeout
        self.reverse_port = self.forward_port = self.target_id = None
        self.before = set()
        self.url = None

    def adb(self, *args):
        return command(["adb", "-s", self.serial, *args])

    def open(self, feature_level):
        if self.adb("get-state").strip() != "device":
            raise ValueError("Selected Android device is not authorized and ready")
        existing = self.adb("reverse", "--list")
        occupied = {part for line in existing.splitlines() for part in line.split()[1:2]}
        for port in range(8080, 8180):
            if f"tcp:{port}" not in occupied:
                self.adb("reverse", "--no-rebind", f"tcp:{port}", f"tcp:{self.host_port}")
                self.reverse_port = port
                break
        if self.reverse_port is None:
            raise ValueError("No unused Android reverse port in 8080..8179")
        port = self.adb("forward", "tcp:0", "localabstract:chrome_devtools_remote").strip()
        if not port.isdecimal() or not 0 < int(port) < 65536:
            raise ValueError("ADB did not return an allocated DevTools port")
        self.forward_port = int(port)
        self.before = {item["id"] for item in pages(self.forward_port)}
        query = urlencode({"autorun": "0", "feature-level": feature_level, "etalon-run": self.token})
        self.url = f"http://127.0.0.1:{self.reverse_port}/testing.html?{query}"
        self.adb("shell", "am", "start", "-a", "android.intent.action.VIEW", "-d",
                 shlex.quote(self.url), "--ez", "create_new_tab", "true", "-p", "com.android.chrome")
        deadline = time.monotonic() + min(self.timeout, 20)
        while time.monotonic() < deadline:
            found = [item for item in pages(self.forward_port)
                     if item.get("type") == "page" and item.get("url") == self.url
                     and item.get("id") not in self.before]
            if len(found) == 1:
                target = found[0]
                self.target_id = target["id"]
                path = urlsplit(target["webSocketDebuggerUrl"]).path
                if not path.startswith("/devtools/page/"):
                    raise ValueError("DevTools did not expose a page-only endpoint")
                return f"ws://127.0.0.1:{self.forward_port}{path}"
            if len(found) > 1:
                raise ValueError("Android launch created multiple qualification tabs")
            time.sleep(0.2)
        raise TimeoutError("Chrome did not expose the new qualification tab")

    def close(self):
        errors = []
        if self.forward_port:
            try:
                targets = pages(self.forward_port)
                for target in targets:
                    owned = target.get("id") == self.target_id or (
                        self.url and target.get("url") == self.url and target.get("id") not in self.before)
                    if owned:
                        http_get(self.forward_port, "/json/close/" + quote(target["id"], safe=""))
            except Exception as error:
                errors.append(f"tab cleanup: {error}")
            try:
                self.adb("forward", "--remove", f"tcp:{self.forward_port}")
            except Exception as error:
                errors.append(f"forward cleanup: {error}")
        if self.reverse_port:
            try:
                self.adb("reverse", "--remove", f"tcp:{self.reverse_port}")
            except Exception as error:
                errors.append(f"reverse cleanup: {error}")
        if errors:
            raise ValueError("; ".join(errors))


def check_report(report, revision, feature_level="core"):
    if (not isinstance(report, dict) or not isinstance(report.get("runtime"), dict) or
            not isinstance(report.get("qualification"), dict)):
        raise ValueError("Invalid qualification report structure")
    if report.get("source_revision") != revision or report.get("runtime", {}).get("source_revision") != revision:
        raise ValueError("Page/WASM source revision differs from the requested revision")
    qualification = report.get("qualification", {})
    results, gpu_cases = report.get("results", []), qualification.get("gpu_cases", [])
    if (not isinstance(results, list) or not isinstance(gpu_cases, list) or
            any(not isinstance(item, dict) for item in results + gpu_cases)):
        raise ValueError("Invalid qualification case structure")
    expected, expected_gpu = qualification.get("expected_cases"), qualification.get("expected_gpu_cases")
    if (type(expected) is not int or type(expected_gpu) is not int or
            not 0 < expected_gpu <= expected <= 64 or len(results) != expected or len(gpu_cases) != expected_gpu):
        raise ValueError("Qualification catalog or case counts are incomplete")
    ids = [item.get("id") for item in results]
    gpu_ids = [item.get("id") for item in gpu_cases]
    if (any(not isinstance(value, str) or not value for value in ids + gpu_ids) or
            len(set(ids)) != expected or len(set(gpu_ids)) != expected_gpu or not set(gpu_ids) <= set(ids)):
        raise ValueError("Qualification case identities are invalid")
    if (qualification.get("status") != "passed" or
            any(item.get("status") != "pass" for item in results + gpu_cases)):
        raise ValueError("Physical Android qualification did not pass every Rust/WASM GPU case")
    runtime = report.get("runtime", {})
    if (any(type(item.get("requires_gpu")) is not bool for item in results) or
            {item["id"] for item in results if item["requires_gpu"]} != set(gpu_ids)):
        raise ValueError("GPU case catalog contradicts execution results")
    if runtime.get("requested_feature_level") != feature_level:
        raise ValueError("Executed feature level differs from the requested profile")
    if runtime.get("bounds", {}).get("cases") != expected:
        raise ValueError("WASM catalog bounds differ from the qualification catalog")
    if runtime.get("backend") != "BrowserWebGpu" or runtime.get("gpu_status") != "available":
        raise ValueError("No executing browser WebGPU backend was reported")
    identity = " ".join(str(runtime.get(key, "")) for key in ("adapter", "driver", "driver_info")).lower()
    if runtime.get("device_type") == "Cpu" or any(name in identity for name in ("swiftshader", "llvmpipe", "lavapipe", "software adapter")):
        raise ValueError("A software adapter cannot qualify the physical Android GPU path")
    if report.get("counts") != {"pass": expected, "fail": 0, "unsupported": 0, "not_run": 0}:
        raise ValueError("Qualification counts contradict case results")
    return qualification


def receipt_path(path):
    path = path.resolve()
    root = (REPO / "target" / "qualification").resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Receipt must be a new file below target/qualification")
    if path.exists():
        raise ValueError("Receipt already exists; select a new output filename")
    return path


def qualify_device(serial, host_port, revision, receipt, feature_level="core", timeout=120):
    session = DeviceSession(serial, host_port, uuid.uuid4().hex, timeout)
    report = None
    android_environment = device_environment(serial)
    failure = None
    try:
        endpoint = session.open(feature_level)
        output = command(["node", str(REPO / "tests/wasm/runner/android-smoke.js"),
                          "--endpoint", endpoint, "--timeout", str(timeout),
                          "--url", session.url], timeout=timeout + 10)
        report = json.loads(output)
        check_report(report, revision, feature_level)
    except BaseException as error:
        failure = error
    finally:
        try:
            session.close()
        except Exception as cleanup_error:
            failure = ValueError(f"{failure}; cleanup also failed: {cleanup_error}"
                                 if failure else f"Cleanup failed: {cleanup_error}")
    envelope = {"schema": "etalon.android-qualification.v1", "device_serial": serial,
                "requested_feature_level": feature_level, "expected_revision": revision,
                "qualification": "failed" if failure else "passed", "report": report,
                "android_environment": android_environment}
    if failure:
        envelope["failure"] = str(failure)[:4000]
    payload = (json.dumps(envelope, indent=2) + "\n").encode("utf-8")
    if len(payload) > MAX_REPORT_BYTES:
        raise ValueError("Receipt byte budget exceeded")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("xb") as destination:
        destination.write(payload)
    if failure:
        raise failure
    return envelope


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def run(site, serial, revision, receipt, feature_level="core", timeout=120):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Expected revision must be a full lowercase Git SHA")
    if not serial or serial.startswith("-") or any(char.isspace() for char in serial):
        raise ValueError("An explicit ADB device serial is required")
    if feature_level not in ("core", "compatibility") or not 1 <= timeout <= 600:
        raise ValueError("Invalid feature level or timeout")
    site = site.resolve(strict=True)
    if not (site / "testing.html").is_file():
        raise ValueError("Assembled testing.html is missing")
    receipt = receipt_path(receipt)
    with ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(site))) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            result = qualify_device(serial, server.server_port, revision, receipt, feature_level, timeout)
            print(json.dumps({"qualification": result["qualification"], "receipt": str(receipt)}))
        finally:
            server.shutdown()
            worker.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--adb-device", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--feature-level", choices=("core", "compatibility"), default="core")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    try:
        run(args.site, args.adb_device, args.expected_revision, args.receipt, args.feature_level, args.timeout)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Android qualification failed: {error}\n")


if __name__ == "__main__":
    main()
