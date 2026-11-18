"""Fetch named Edinburgh clips (color.mp4) by HTTP byte range from the annotated archive on Google Drive.

Same method as fetch_data_probe.py: the archive index (data/raw/edinburgh/archive_index.json)
gives each file's offset and size inside the 3.29 GB archive, so one clip is one bounded range
request. The annotation JSON and times.txt for every clip are already local.
Usage: python scripts/fetch_clips.py 2019_11_11/000028 2019_11_22/000010 2019_12_10/000078
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "edinburgh"
VIDEO_URL = "https://drive.usercontent.google.com/download?id=1awlVJav7nRWnxHVNSCdnQXinQgbxGEdd&export=download&confirm=t"
ARCHIVE_SIZE = 3292047360


def fetch(clip: str, entries: list[dict]) -> None:
    entry = next(e for e in entries if e["name"] == f"annotated/{clip}/color.mp4")
    path = RAW / "annotated" / clip / "color.mp4"
    if path.exists() and path.stat().st_size == entry["bytes"]:
        print(clip, "already present", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    start, end = entry["offset"], entry["offset"] + entry["bytes"] - 1
    tmp = path.with_suffix(".mp4.part")
    digest, got, t0 = hashlib.sha256(), 0, time.time()
    with requests.get(VIDEO_URL, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(20, 120)) as r:
        r.raise_for_status()
        if r.status_code != 206 or r.headers.get("Content-Range") != f"bytes {start}-{end}/{ARCHIVE_SIZE}":
            raise RuntimeError(f"server did not honour the byte range: {r.status_code} {r.headers.get('Content-Range')}")
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                got += len(chunk)
                if got > entry["bytes"]:
                    raise RuntimeError("response exceeded expected size")
                fh.write(chunk); digest.update(chunk)
                if got // 50_000_000 != (got - len(chunk)) // 50_000_000:
                    print(f"  {clip}: {got / 1e6:.0f} of {entry['bytes'] / 1e6:.0f} MB", flush=True)
    if got != entry["bytes"]:
        raise RuntimeError("incomplete clip")
    tmp.replace(path)
    prov = RAW / "clip_provenance.json"
    log = json.loads(prov.read_text()) if prov.exists() else {}
    log[clip] = {**entry, "sha256": digest.hexdigest(), "source_url": VIDEO_URL, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S")}
    prov.write_text(json.dumps(log, indent=2))
    print(f"{clip}: done, {got / 1e6:.0f} MB in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    entries = json.loads((RAW / "archive_index.json").read_text())
    for clip in sys.argv[1:]:
        fetch(clip, entries)
    print("ALL CLIPS DONE")
