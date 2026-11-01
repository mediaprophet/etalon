"""Verify retained SDK reference bytes without network or native execution."""
import argparse
import hashlib
import json
from pathlib import Path

MAX_FILE_BYTES = 512 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024


def verify(root):
    root = Path(root).resolve(strict=True)
    seen = set()
    total = 0
    records = sorted(root.rglob("*provenance*.json"))
    if not records:
        raise ValueError("No provenance records found")
    for manifest in records:
        if manifest.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError("Provenance record exceeds budget")
        document = json.loads(manifest.read_text(encoding="utf-8-sig"))
        rows = document.get("files")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Missing reference file declarations")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Invalid reference declaration")
            base = manifest.parent if "local_file" in row else root
            relative = row.get("local_file", row.get("path"))
            if not isinstance(relative, str) or not relative:
                raise ValueError("Missing reference path")
            candidate = base / relative
            path = candidate.resolve(strict=True)
            if root not in path.parents or not path.is_file() or candidate.is_symlink():
                raise ValueError("Reference escapes root or is not a regular file")
            if path in seen:
                raise ValueError("Duplicate reference declaration")
            size = row.get("bytes")
            digest = row.get("sha256")
            if type(size) is not int or size <= 0 or size > MAX_FILE_BYTES:
                raise ValueError("Invalid reference byte budget")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("Invalid reference digest")
            try:
                expected = bytes.fromhex(digest)
            except ValueError as error:
                raise ValueError("Invalid reference digest") from error
            if len(expected) != 32:
                raise ValueError("Invalid reference digest")
            total += size
            if total > MAX_TOTAL_BYTES:
                raise ValueError("Reference collection exceeds budget")
            if path.stat().st_size != size:
                raise ValueError(f"Reference length mismatch: {path.relative_to(root)}")
            with path.open("rb") as source:
                payload = source.read(size + 1)
            if len(payload) != size or hashlib.sha256(payload).digest() != expected:
                raise ValueError(f"Reference hash mismatch: {path.relative_to(root)}")
            seen.add(path)
    return len(seen), total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1]
                        / "sensing-vendor-adapters" / "sdk-reference")
    count, total = verify(parser.parse_args().root)
    print(f"Verified {count} exact reference files ({total} bytes)")


if __name__ == "__main__":
    main()
