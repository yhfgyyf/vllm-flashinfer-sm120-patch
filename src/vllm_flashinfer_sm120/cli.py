from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from importlib import resources


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> dict:
    manifest_path = resources.files("vllm_flashinfer_sm120").joinpath("patch_manifest.json")
    return json.loads(manifest_path.read_text())


def _flashinfer_root() -> Path:
    try:
        import flashinfer  # type: ignore
    except Exception as e:
        raise RuntimeError("flashinfer is not installed in current environment") from e
    return Path(flashinfer.__file__).resolve().parent


def _patch_file_source(rel_path: str) -> Path:
    return resources.files("vllm_flashinfer_sm120").joinpath("patches", "flashinfer", rel_path)


def _status_rows() -> list[tuple[str, str, str]]:
    manifest = _load_manifest()
    root = _flashinfer_root()
    rows: list[tuple[str, str, str]] = []
    for item in manifest["files"]:
        rel = item["path"]
        dst = root / rel
        if not dst.exists():
            rows.append((rel, "missing", ""))
            continue
        cur = _sha256(dst)
        if cur == item["patched_sha256"]:
            rows.append((rel, "patched", cur))
        elif cur == item["base_sha256"]:
            rows.append((rel, "base", cur))
        else:
            rows.append((rel, "unknown", cur))
    return rows


def status_main() -> None:
    manifest = _load_manifest()
    print(f"flashinfer target version: {manifest['flashinfer_version']}")
    print(f"flashinfer install root: {_flashinfer_root()}")
    for rel, st, digest in _status_rows():
        print(f"[{st:7}] {rel}")
        if digest:
            print(f"         sha256={digest}")


def apply_main() -> None:
    ap = argparse.ArgumentParser(description="Apply SM120 flashinfer patch files")
    ap.add_argument("--force", action="store_true", help="patch even when current file hash is unknown")
    args = ap.parse_args()

    manifest = _load_manifest()
    root = _flashinfer_root()

    applied = 0
    skipped = 0
    already = 0

    for item in manifest["files"]:
        rel = item["path"]
        dst = root / rel
        src = _patch_file_source(rel)

        if not dst.exists():
            raise FileNotFoundError(f"target file missing: {dst}")

        cur = _sha256(dst)
        if cur == item["patched_sha256"]:
            print(f"[already] {rel}")
            already += 1
            continue

        if cur != item["base_sha256"] and not args.force:
            print(f"[skip] {rel}: current file hash is unknown (use --force to override)")
            skipped += 1
            continue

        backup = dst.with_suffix(dst.suffix + ".bak.sm120patch")
        if not backup.exists():
            shutil.copy2(dst, backup)

        dst.write_bytes(src.read_bytes())
        new_hash = _sha256(dst)
        if new_hash != item["patched_sha256"]:
            raise RuntimeError(f"hash check failed after patch: {rel}")

        print(f"[patched] {rel}")
        applied += 1

    print(f"done: applied={applied}, already={already}, skipped={skipped}")


if __name__ == "__main__":
    # default entry for manual debug
    status_main()
