import hashlib
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import regenerate_dense_cpu_golden as helper


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="dense-helper-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data = Path(helper.__file__).with_name("dense_cpu_golden.rs").read_bytes()

    def test_success(self):
        target = self.root / "trace.rs"
        helper.promote(self.data, target)
        self.assertEqual(target.read_bytes(), self.data)

    def test_existing_destination_survives(self):
        target = self.root / "trace.rs"
        target.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            helper.promote(self.data, target)
        self.assertEqual(target.read_bytes(), b"existing")

    def test_partial_write_failure_removes_only_owned_file(self):
        target = self.root / "trace.rs"
        unrelated = self.root / "keep"
        unrelated.write_bytes(b"keep")
        class Failing:
            def __enter__(self):
                self.file = open(target, "xb")
                return self
            def write(self, data):
                self.file.write(data[:17])
                raise OSError("forced partial write")
            def __exit__(self, *unused):
                self.file.close()
        with self.assertRaises(OSError):
            helper.promote(self.data, target, lambda *args: Failing())
        self.assertFalse(target.exists())
        self.assertEqual(unrelated.read_bytes(), b"keep")

    def test_short_write_is_failure_and_cleans_output(self):
        target = self.root / "trace.rs"
        class Short:
            def __enter__(self):
                self.file = open(target, "xb")
                return self
            def write(self, data):
                return self.file.write(data[:7])
            def __exit__(self, *unused):
                self.file.close()
        with self.assertRaises(OSError):
            helper.promote(self.data, target, lambda *args: Short())
        self.assertFalse(target.exists())

    def test_hash_and_byte_budget_fail_before_output(self):
        target = self.root / "trace.rs"
        for invalid in [b"bad", bytes(65537)]:
            with self.assertRaises(ValueError):
                helper.promote(invalid, target)
            self.assertFalse(target.exists())

    def test_source_read_failure_cleans_temporary_directory(self):
        target = self.root / "trace.rs"
        args = SimpleNamespace(qualia_root=self.root, dependencies=self.root,
                               output=target, optimized=False)
        with patch.object(subprocess, "check_output", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                helper.generate(args)
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
