#!/usr/bin/env python3
"""Assemble the public Pages artifact from explicit static assets and rustdoc."""

import argparse
import html
import hashlib
import json
from html.parser import HTMLParser
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import unquote, urlsplit

MAX_BYTES = 512 * 1024 * 1024
MAX_FILES = 50_000
MAX_BROWSER_BYTES = 32 * 1024 * 1024
ASSETS = (
    "index.html", "getting-started.html", "licensing.html", "capabilities.html",
    "testing.html", "browser-tests.js", "adapter-diagnostic.js", "styles.css",
    "app-install.js", "app-environment.js", "app-storage.js", "wasm-runtime.js", "service-worker.js",
    "manifest.webmanifest", "app-icon.svg",
)
BINARY_ASSETS = ("app-icon-192.png", "app-icon-512.png")
BROWSER_FILES = ("etalon_browser_demo.js", "etalon_browser_demo_bg.wasm")
JS_REFERENCES = re.compile(
    r"(?:\b(?:import|export)\s+(?:[^;]*?\s+from\s+)?|"
    r"\bimport\s*\(\s*|\bnew\s+URL\s*\(\s*)[\"']([^\"']+)[\"']"
)
REQUIRED_DOCS = (
    "etalon/index.html",
    "etalon_wgsl_forge/index.html",
    "etalon_nquin/index.html",
    "etalon_tensor/index.html",
    "etalon_render/index.html",
    "etalon_wasm/index.html",
    "etalon_agent/index.html",
    "etalon_sense/index.html",
    "etalon_sense_core/index.html",
    "etalon_sense_runtime/index.html",
    "etalon_signal/index.html",
    "etalon_sense_cpu/index.html",
    "etalon_sense_sdr/index.html",
    "etalon_sense_host/index.html",
    "etalon_sense_nfc/index.html",
    "etalon_sense_camera/index.html",
    "etalon_sense_gpu/index.html",
)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("href", "src") and value:
                self.links.append(value)


def inspect_tree(root):
    files = []
    size = 0
    for current, directories, names in os.walk(root, followlinks=False):
        current = Path(current)
        for name in directories:
            if (current / name).is_symlink():
                raise ValueError("input directory contains a symbolic link")
        for name in sorted(names):
            path = current / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("input is not an ordinary file")
            size += path.stat().st_size
            files.append(path.relative_to(root))
            if size > MAX_BYTES or len(files) > MAX_FILES:
                raise ValueError("documentation artifact budget exceeded")
    return files, size


def validate_reference(site, source, link, same_origin=False):
    parsed = urlsplit(link)
    if parsed.scheme or parsed.netloc:
        if same_origin:
            raise ValueError(f"browser module requires a remote dependency: {link}")
        return
    if not parsed.path:
        return
    if parsed.path.startswith("/"):
        raise ValueError(f"project Pages asset requires a relative link: {link}")
    path = (source.parent / unquote(parsed.path)).resolve()
    if not path.is_relative_to(site) or not path.is_file():
        raise ValueError(f"missing or unsafe local link in {source.name}: {link}")


def validate_links(site):
    site = site.resolve()
    for asset in ASSETS:
        source = site / asset
        if asset.endswith(".html"):
            parser = Links()
            parser.feed(source.read_text(encoding="utf-8"))
            for link in parser.links:
                validate_reference(site, source, link)
    modules = [site / asset for asset in ASSETS if asset.endswith(".js")]
    modules.extend((site / "browser").rglob("*.js"))
    for source in modules:
        for reference in JS_REFERENCES.findall(source.read_text(encoding="utf-8")):
            validate_reference(site, source, reference, same_origin=True)


def inspect_browser(root):
    files, size = inspect_tree(root)
    if size > MAX_BROWSER_BYTES or len(files) > 128:
        raise ValueError("browser artifact budget exceeded")
    for filename in BROWSER_FILES:
        if Path(filename) not in files:
            raise ValueError(f"required compiled browser asset is missing: {filename}")
    for relative in files:
        if relative.as_posix() not in BROWSER_FILES and not (
            relative.parts[0] == "snippets" and relative.suffix == ".js"
        ):
            raise ValueError(f"unexpected browser build artifact: {relative}")
    with (root / BROWSER_FILES[1]).open("rb") as module:
        if module.read(8) != b"\0asm\x01\0\0\0":
            raise ValueError("browser artifact is not a WebAssembly version 1 module")
    return files, size


def write_offline_manifest(site, revision):
    paths = [name for name in ASSETS if name != "service-worker.js"] + list(BINARY_ASSETS)
    paths += [path.relative_to(site).as_posix() for path in sorted((site / "browser").rglob("*")) if path.is_file()]
    if len(paths) > 160:
        raise ValueError("offline asset count exceeded")
    assets, total = [], 0
    for name in paths:
        path = site / name
        length = path.stat().st_size
        total += length
        if total > 40 * 1024 * 1024:
            raise ValueError("offline asset byte budget exceeded")
        assets.append({"path": name, "bytes": length, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    payload = json.dumps({"schema": "etalon.offline-assets.v1", "revision": revision, "assets": assets}, separators=(",", ":"))
    if len(payload.encode()) > 65536:
        raise ValueError("offline manifest byte budget exceeded")
    (site / "offline-assets.json").write_text(payload, encoding="utf-8")


def build(rustdoc, output, revision, browser_assets):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("revision must be a full Git commit SHA")
    rustdoc = rustdoc.resolve(strict=True)
    browser_assets = browser_assets.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise ValueError("refusing to replace an existing artifact directory")
    if output.is_relative_to(rustdoc) or output.is_relative_to(browser_assets):
        raise ValueError("output must be outside the build inputs")
    for path in REQUIRED_DOCS:
        if not (rustdoc / path).is_file():
            raise ValueError(f"required public documentation is missing: {path}")
    files, size = inspect_tree(rustdoc)
    browser_files, browser_size = inspect_browser(browser_assets)
    size += browser_size
    assets = Path(__file__).resolve().parent
    rendered = {}
    for asset in ASSETS:
        source = assets / asset
        if source.is_symlink() or source.stat().st_size > 256 * 1024:
            raise ValueError("static asset budget exceeded")
        value = source.read_text(encoding="utf-8")
        rendered[asset] = value.replace("{{REVISION}}", html.escape(revision.lower()))
    size += sum(len(value.encode("utf-8")) for value in rendered.values())
    for name in BINARY_ASSETS:
        source = assets / name
        if source.is_symlink() or source.stat().st_size > 256 * 1024:
            raise ValueError("icon budget exceeded")
        size += source.stat().st_size
    if size > MAX_BYTES or len(files) + len(browser_files) + len(ASSETS) + 1 > MAX_FILES:
        raise ValueError("documentation artifact budget exceeded")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".etalon-pages-", dir=output.parent) as scratch:
        site = Path(scratch) / "site"
        (site / "doc").mkdir(parents=True)
        for relative in files:
            destination = site / "doc" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(rustdoc / relative, destination)
        for relative in browser_files:
            destination = site / "browser" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(browser_assets / relative, destination)
        for asset in ASSETS:
            (site / asset).write_text(rendered[asset], encoding="utf-8", newline="\n")
        for name in BINARY_ASSETS:
            shutil.copyfile(assets / name, site / name)
        write_offline_manifest(site, revision.lower())
        (site / ".nojekyll").write_text("", encoding="ascii")
        validate_links(site.resolve())
        os.rename(site, output)
    print(f"Pages artifact ready: {len(files)} rustdoc files, {size} bytes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rustdoc", required=True, type=Path)
    parser.add_argument("--browser-assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    arguments = parser.parse_args()
    try:
        build(arguments.rustdoc, arguments.output, arguments.revision, arguments.browser_assets)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Documentation assembly failed: {error}\n")


if __name__ == "__main__":
    main()
