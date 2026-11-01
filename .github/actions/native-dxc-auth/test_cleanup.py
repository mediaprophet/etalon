"""Exercise credential cleanup ownership and non-destructive refusal."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid


@unittest.skipUnless(sys.platform == "win32" and shutil.which("pwsh"), "Windows PowerShell host required")
class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="etalon-dxc-cleanup-test-")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.runner = self.root / "runner"
        self.runner.mkdir()
        self.target = self.runner / ("etalon-static-dxc-" + uuid.uuid4().hex)
        self.target.mkdir()
        (self.target / "owner.txt").write_text(self.target.name, encoding="ascii")
        self.script = Path(__file__).with_name("cleanup.ps1")

    def cleanup(self, target=None):
        env = dict(os.environ, RUNNER_TEMP=str(self.runner))
        return subprocess.run(
            [shutil.which("pwsh"), "-NoLogo", "-NoProfile", "-NonInteractive",
             "-File", str(self.script), "-Directory", str(target or self.target)],
            env=env, capture_output=True, text=True, timeout=30,
        )

    def test_owned_credentials_are_removed_and_cleanup_is_idempotent(self):
        (self.target / "credentials.netrc").write_text("synthetic credential", encoding="ascii")
        self.assertEqual(self.cleanup().returncode, 0)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.cleanup().returncode, 0)

    def test_outside_parent_is_preserved(self):
        outside = self.root / ("etalon-static-dxc-" + uuid.uuid4().hex)
        outside.mkdir()
        marker = outside / "owner.txt"
        marker.write_text(outside.name, encoding="ascii")
        self.assertNotEqual(self.cleanup(outside).returncode, 0)
        self.assertEqual(marker.read_text(encoding="ascii"), outside.name)

    def test_unknown_content_prevents_all_deletion(self):
        sentinel = self.target / "caller-owned"
        sentinel.write_bytes(b"preserve")
        self.assertNotEqual(self.cleanup().returncode, 0)
        self.assertEqual(sentinel.read_bytes(), b"preserve")
        self.assertTrue((self.target / "owner.txt").exists())

    def test_wrong_marker_prevents_deletion(self):
        marker = self.target / "owner.txt"
        marker.write_text("different-owner", encoding="ascii")
        self.assertNotEqual(self.cleanup().returncode, 0)
        self.assertEqual(marker.read_text(encoding="ascii"), "different-owner")


if __name__ == "__main__":
    unittest.main()
