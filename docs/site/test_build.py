"""Offline checks for site assembly, links and non-destructive failure."""
import importlib.util
import hashlib
import json
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
        self.browser = self.root / "browser-build"
        self.browser.mkdir()
        (self.browser / BUILDER.BROWSER_FILES[0]).write_text(
            'export const binary = new URL("etalon_browser_demo_bg.wasm", import.meta.url);',
            encoding="utf-8",
        )
        (self.browser / BUILDER.BROWSER_FILES[1]).write_bytes(b"\0asm\x01\0\0\0")
        self.output = self.root / "pages"

    def test_complete_site_records_revision_and_resolves_local_links(self):
        BUILDER.build(self.rustdoc, self.output, REVISION, self.browser)
        index = (self.output / "index.html").read_text(encoding="utf-8")
        self.assertIn(REVISION, index)
        self.assertNotIn("{{REVISION}}", index)
        self.assertTrue((self.output / ".nojekyll").is_file())
        self.assertTrue((self.output / "doc/etalon_sense/index.html").is_file())
        self.assertEqual(
            (self.output / "browser" / BUILDER.BROWSER_FILES[1]).read_bytes(),
            b"\0asm\x01\0\0\0",
        )
        BUILDER.validate_links(self.output)

    def test_offline_manifest_is_revision_bound_and_hashes_actual_assets(self):
        BUILDER.build(self.rustdoc, self.output, REVISION, self.browser)
        manifest = json.loads((self.output / "offline-assets.json").read_text())
        self.assertEqual(manifest["revision"], REVISION)
        self.assertLessEqual(len(manifest["assets"]), 160)
        for asset in manifest["assets"]:
            actual = (self.output / asset["path"]).read_bytes()
            self.assertEqual(asset["bytes"], len(actual))
            self.assertEqual(asset["sha256"], hashlib.sha256(actual).hexdigest())
        names = {asset["path"] for asset in manifest["assets"]}
        self.assertIn("browser/etalon_browser_demo_bg.wasm", names)
        self.assertIn("manifest.webmanifest", names)
        self.assertNotIn("service-worker.js", names)
        app = json.loads((self.output / "manifest.webmanifest").read_text())
        self.assertEqual(app["display"], "standalone")
        for icon in app["icons"]:
            self.assertTrue((self.output / icon["src"]).is_file())

    def test_existing_artifact_is_never_replaced(self):
        self.output.mkdir()
        sentinel = self.output / "caller-owned"
        sentinel.write_bytes(b"preserve")
        with self.assertRaises(ValueError):
            BUILDER.build(self.rustdoc, self.output, REVISION, self.browser)
        self.assertEqual(sentinel.read_bytes(), b"preserve")

    def test_missing_api_prevents_partial_artifact(self):
        (self.rustdoc / "etalon_sense/index.html").unlink()
        with self.assertRaises(ValueError):
            BUILDER.build(self.rustdoc, self.output, REVISION, self.browser)
        self.assertFalse(self.output.exists())

    def test_invalid_source_revision_prevents_artifact(self):
        with self.assertRaises(ValueError):
            BUILDER.build(self.rustdoc, self.output, '"><script>', self.browser)
        self.assertFalse(self.output.exists())

    def test_budget_failure_prevents_artifact(self):
        original = BUILDER.MAX_BYTES
        try:
            BUILDER.MAX_BYTES = 1
            with self.assertRaises(ValueError):
                BUILDER.build(self.rustdoc, self.output, REVISION, self.browser)
        finally:
            BUILDER.MAX_BYTES = original
        self.assertFalse(self.output.exists())

    def test_missing_compiled_module_prevents_artifact(self):
        (self.browser / BUILDER.BROWSER_FILES[1]).unlink()
        with self.assertRaises(ValueError):
            BUILDER.build(self.rustdoc, self.output, REVISION, self.browser)
        self.assertFalse(self.output.exists())

    def test_invalid_wasm_header_prevents_artifact(self):
        (self.browser / BUILDER.BROWSER_FILES[1]).write_bytes(b"not wasm")
        with self.assertRaises(ValueError):
            BUILDER.build(self.rustdoc, self.output, REVISION, self.browser)
        self.assertFalse(self.output.exists())

    def test_browser_budget_prevents_artifact(self):
        original = BUILDER.MAX_BROWSER_BYTES
        try:
            BUILDER.MAX_BROWSER_BYTES = 1
            with self.assertRaises(ValueError):
                BUILDER.build(self.rustdoc, self.output, REVISION, self.browser)
        finally:
            BUILDER.MAX_BROWSER_BYTES = original
        self.assertFalse(self.output.exists())

    def test_remote_module_is_rejected_and_staging_is_cleaned(self):
        (self.browser / BUILDER.BROWSER_FILES[0]).write_text(
            'import "https://cdn.example.invalid/library.js";', encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "remote dependency"):
            BUILDER.build(self.rustdoc, self.output, REVISION, self.browser)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".etalon-pages-*")), [])

    def test_missing_snippet_is_rejected(self):
        (self.browser / BUILDER.BROWSER_FILES[0]).write_text(
            'import "./snippets/missing.js";', encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            BUILDER.build(self.rustdoc, self.output, REVISION, self.browser)
        self.assertFalse(self.output.exists())

    def test_compiler_generated_snippet_remains_same_origin(self):
        snippet = self.browser / "snippets" / "generated" / "inline.js"
        snippet.parent.mkdir(parents=True)
        snippet.write_text("export const value = 1;", encoding="utf-8")
        (self.browser / BUILDER.BROWSER_FILES[0]).write_text(
            'export { value } from "./snippets/generated/inline.js";', encoding="utf-8"
        )
        BUILDER.build(self.rustdoc, self.output, REVISION, self.browser)
        self.assertTrue((self.output / "browser/snippets/generated/inline.js").is_file())

    def test_module_cannot_escape_artifact(self):
        (self.browser / BUILDER.BROWSER_FILES[0]).write_text(
            'import "../../outside.js";', encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            BUILDER.build(self.rustdoc, self.output, REVISION, self.browser)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
