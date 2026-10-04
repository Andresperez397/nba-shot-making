"""Download NBA Stats play-by-play and schedules (sportsdataverse releases) into data/raw/, pinned by SHA-256.

The release files are updated in place, so the first run writes data/manifest.csv with each file's
size and SHA-256. Later runs verify every file and stop if anything differs. Use --repin to accept
new upstream versions (and log it in DEVIATIONS.md).

Seasons are named by their ending year (2025 = the 2024-25 season), as in the source.
"""
from __future__ import annotations

import csv
import hashlib
import ssl
import sys
import urllib.request
from datetime import date
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.csv"
BASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
SEASONS = range(2016, 2027)  # 2015-16 through 2025-26 (2015-16 and 2016-17 only for the audit check)
FILES = ([("nba_stats_pbp", f"nba_play_by_play_{s}.parquet") for s in SEASONS]
         + [("nba_stats_schedules", f"nba_schedule_{s}.parquet") for s in SEASONS])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    ctx = ssl.create_default_context(cafile=certifi.where())
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, context=ctx, timeout=300) as r, open(tmp, "wb") as f:
        f.write(r.read())
    with open(tmp, "rb") as f:
        head = f.read(4)
    if head != b"PAR1":  # an HTML error page is not a parquet file
        tmp.unlink()
        raise RuntimeError(f"not a parquet file: {url}")
    tmp.rename(dest)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    for release, name in FILES:
        dest = RAW / name
        url = f"{BASE}/{release}/{name}"
        if not dest.exists():
            print("downloading", name, flush=True)
            download(url, dest)
        rows.append({"release": release, "name": name, "url": url, "bytes": dest.stat().st_size,
                     "sha256": sha256(dest)})
    if not MANIFEST.exists() or "--repin" in sys.argv:
        for r in rows:
            r["pinned_on"] = date.today().isoformat()
        with open(MANIFEST, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {MANIFEST.relative_to(ROOT)} ({len(rows)} files)")
        return
    with open(MANIFEST) as f:
        pinned = {r["name"]: r["sha256"] for r in csv.DictReader(f)}
    bad = [r["name"] for r in rows if pinned.get(r["name"]) != r["sha256"]]
    if bad:
        raise SystemExit(f"files differ from data/manifest.csv: {bad}. Upstream data changed; delete them "
                         "to re-download, or rerun with --repin and log it in DEVIATIONS.md.")
    print(f"all {len(rows)} files match data/manifest.csv")


if __name__ == "__main__":
    main()
