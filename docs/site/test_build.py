"""Offline checks for site assembly, links and non-destructive failure."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("etalon_site_build", Path(__file__).with_name("build.py"))
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
REVISION = "a" * 40


class SiteAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="etalon-pages-test-")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.rustdoc = self.root / "rustdoc"
        for relative in BUILDER.REQUIRED_DOCS + ("etalon/struct.PreparedCompute.html",):
            path = self.rustdoc / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("<!doctype html><title>Fixture API</title>", encoding="utf-8")
        self.output = self.root / "pages"

    def test_complete_site_records_revision_and_resolves_local_links(self):
        BUILDER.build(self.rustdoc, self.output, REVISION)
        index = (self.output / "index.html").read_text(encoding="utf-8")
        self.assertIn(REVISION, index)
        self.assertNotIn("{{REVISION}}", index)
        self.assertTrue((self.output / ".nojekyll").is_file())
        self.assertTrue((self.output / "doc/etalon_sense/index.html").is_file())
        BUILDER.validate_links(self.output)

    def test_existing_artifact_is_never_replaced(self):
        self.output.mkdir()
        sentinel = self.output / "caller-owned"
        sentinel.write_bytes(b"preserve")
        with self.assertRaises(ValueError):
            BUILDER.build(self.rustdoc, self.output, REVISION)
        self.assertEqual(sentinel.read_bytes(), b"preserve")

    def test_missing_api_prevents_partial_artifact(self):
        (self.rustdoc / "etalon_sense/index.html").unlink()
        with self.assertRaises(ValueError):
            BUILDER.build(self.rustdoc, self.output, REVISION)
        self.assertFalse(self.output.exists())

    def test_invalid_source_revision_prevents_artifact(self):
        with self.assertRaises(ValueError):
            BUILDER.build(self.rustdoc, self.output, '"><script>')
        self.assertFalse(self.output.exists())

    def test_budget_failure_prevents_artifact(self):
        original = BUILDER.MAX_BYTES
        try:
            BUILDER.MAX_BYTES = 1
            with self.assertRaises(ValueError):
                BUILDER.build(self.rustdoc, self.output, REVISION)
        finally:
            BUILDER.MAX_BYTES = original
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
