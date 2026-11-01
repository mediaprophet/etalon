import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from check_sensing_provenance import verify


class ReferenceIntegrityTests(unittest.TestCase):
    def fixture(self, root, payload=b"upstream\n"):
        (root / "reference.txt").write_bytes(payload)
        row = {"path": "reference.txt", "bytes": len(payload),
               "sha256": hashlib.sha256(payload).hexdigest()}
        (root / "provenance.json").write_text(json.dumps({"files": [row]}), encoding="utf-8")
        return row

    def test_line_endings_are_not_silently_normalized(self):
        with tempfile.TemporaryDirectory(prefix="etalon-reference-") as temporary:
            root = Path(temporary)
            self.fixture(root)
            self.assertEqual(verify(root), (1, 9))
            (root / "reference.txt").write_bytes(b"upstream\r\n")
            with self.assertRaisesRegex(ValueError, "length mismatch"):
                verify(root)

    def test_same_size_corruption_and_duplicate_declarations_fail(self):
        with tempfile.TemporaryDirectory(prefix="etalon-reference-") as temporary:
            root = Path(temporary)
            row = self.fixture(root)
            (root / "reference.txt").write_bytes(b"tampered\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify(root)
            self.fixture(root)
            (root / "provenance.json").write_text(json.dumps({"files": [row, row]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                verify(root)

    def test_escaping_manifest_path_fails_before_reading_external_bytes(self):
        with tempfile.TemporaryDirectory(prefix="etalon-reference-") as temporary:
            parent = Path(temporary)
            root = parent / "references"
            root.mkdir()
            row = self.fixture(root)
            (parent / "outside").write_bytes(b"upstream\n")
            row["path"] = "../outside"
            (root / "provenance.json").write_text(json.dumps({"files": [row]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes root"):
                verify(root)


if __name__ == "__main__":
    unittest.main()
