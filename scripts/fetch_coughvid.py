"""Download the COUGHVID corpus from Zenodo.

Resumable, retrying and checksum-verified.

Two failure modes have already been seen on this link, and both looked like
success from the outside: a full disk silently truncated every write, and the
connection stalled at 70% until the socket timed out. So the rule here is that
a download is finished when the digest matches, never because bytes stopped
arriving. Anything short of the digest is a resume point, not an outcome.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import time
import urllib.request
from pathlib import Path

URL = "https://zenodo.org/api/records/4048312/files/public_dataset.zip/content"
EXPECTED_MD5 = "5c30a8b00c8bb7783a2c15a48cb8ea9e"
EXPECTED_BYTES = 951442487
DEST = Path("data/raw/coughvid")
CHUNK = 1 << 20
MAX_ATTEMPTS = 40


def _digest(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _check_space(target: Path, have: int) -> None:
    free = shutil.disk_usage(target.parent).free
    needed = EXPECTED_BYTES - have
    if free < needed + (1 << 30):
        raise SystemExit(
            f"need ~{(needed + (1 << 30)) / 1e9:.1f} GB free, have {free / 1e9:.1f} GB"
        )


def _pull(target: Path) -> None:
    """One attempt. Appends whatever it can, then returns."""
    have = _size(target)
    req = urllib.request.Request(URL)
    if have:
        req.add_header("Range", f"bytes={have}-")

    with urllib.request.urlopen(req, timeout=60) as resp:
        if have and resp.status != 206:
            print("server ignored the range request, restarting from zero")
            target.unlink(missing_ok=True)
            have = 0
        mode = "ab" if have else "wb"
        done = have
        with target.open(mode) as fh:
            while True:
                block = resp.read(CHUNK)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                print(f"\r{done / 1e6:8.1f} / {EXPECTED_BYTES / 1e6:.1f} MB "
                      f"({100 * done / EXPECTED_BYTES:5.1f}%)", end="", flush=True)


def download(target: Path) -> None:
    if _size(target) > EXPECTED_BYTES:
        print("local file is larger than expected, restarting")
        target.unlink()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        have = _size(target)
        if have == EXPECTED_BYTES:
            return
        _check_space(target, have)
        if have:
            print(f"\nattempt {attempt}: resuming at {have / 1e6:.1f} MB", flush=True)
        try:
            _pull(target)
        except Exception as exc:
            grew = _size(target) - have
            print(f"\n  interrupted by {type(exc).__name__} after {grew / 1e6:.1f} MB",
                  flush=True)
            time.sleep(min(5 * attempt, 30))
            continue
        if _size(target) == EXPECTED_BYTES:
            return

    raise SystemExit(
        f"still short after {MAX_ATTEMPTS} attempts "
        f"({_size(target)} of {EXPECTED_BYTES}). Re-run to continue."
    )


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    archive = DEST / "public_dataset.zip"

    download(archive)
    print("\nverifying checksum ...", flush=True)
    actual = _digest(archive)
    if actual != EXPECTED_MD5:
        raise SystemExit(f"checksum mismatch: {actual} != {EXPECTED_MD5}")

    print(f"md5 ok: {actual}")
    print(f"{archive} ready ({_size(archive) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
