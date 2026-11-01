#!/usr/bin/env python3
"""Assemble the public Pages artifact from explicit static assets and rustdoc."""

import argparse
import html
from html.parser import HTMLParser
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import unquote, urlsplit

MAX_BYTES = 512 * 1024 * 1024
MAX_FILES = 50_000
ASSETS = ("index.html", "getting-started.html", "licensing.html", "styles.css")
REQUIRED_DOCS = (
    "etalon/index.html",
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
                raise ValueError("rustdoc directory contains a symbolic link")
        for name in sorted(names):
            path = current / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("rustdoc input is not an ordinary file")
            size += path.stat().st_size
            files.append(path.relative_to(root))
            if size > MAX_BYTES or len(files) > MAX_FILES:
                raise ValueError("documentation artifact budget exceeded")
    return files, size


def validate_links(site):
    for asset in ASSETS:
        if not asset.endswith(".html"):
            continue
        parser = Links()
        parser.feed((site / asset).read_text(encoding="utf-8"))
        for link in parser.links:
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            path = (site / unquote(parsed.path)).resolve()
            if not path.is_relative_to(site) or not path.is_file():
                raise ValueError(f"missing or unsafe local link in {asset}: {link}")


def build(rustdoc, output, revision):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("revision must be a full Git commit SHA")
    rustdoc = rustdoc.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise ValueError("refusing to replace an existing artifact directory")
    if output.is_relative_to(rustdoc):
        raise ValueError("output must be outside the rustdoc input")
    for path in REQUIRED_DOCS:
        if not (rustdoc / path).is_file():
            raise ValueError(f"required public documentation is missing: {path}")
    files, size = inspect_tree(rustdoc)
    assets = Path(__file__).resolve().parent
    rendered = {}
    for asset in ASSETS:
        source = assets / asset
        if source.is_symlink() or source.stat().st_size > 256 * 1024:
            raise ValueError("static asset budget exceeded")
        value = source.read_text(encoding="utf-8")
        rendered[asset] = value.replace("{{REVISION}}", html.escape(revision.lower()))
    size += sum(len(value.encode("utf-8")) for value in rendered.values())
    if size > MAX_BYTES or len(files) + len(ASSETS) + 1 > MAX_FILES:
        raise ValueError("documentation artifact budget exceeded")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".etalon-pages-", dir=output.parent) as scratch:
        site = Path(scratch) / "site"
        (site / "doc").mkdir(parents=True)
        for relative in files:
            destination = site / "doc" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(rustdoc / relative, destination)
        for asset in ASSETS:
            (site / asset).write_text(rendered[asset], encoding="utf-8", newline="\n")
        (site / ".nojekyll").write_text("", encoding="ascii")
        validate_links(site.resolve())
        os.rename(site, output)
    print(f"Pages artifact ready: {len(files)} rustdoc files, {size} bytes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rustdoc", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    arguments = parser.parse_args()
    try:
        build(arguments.rustdoc, arguments.output, arguments.revision)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Documentation assembly failed: {error}\n")


if __name__ == "__main__":
    main()
