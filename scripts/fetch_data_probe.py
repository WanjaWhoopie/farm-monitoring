"""Fetch public dataset metadata and bounded samples, preserving source files.

Uses HTTP byte ranges instead of downloading the multi-GB video archive.
Run with a Python environment containing requests. No model training occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import time
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
AUDIO_URL = "https://zenodo.org/api/records/8252482/files/Soundwel%20Dataset%20-%20Audio%20and%20Spectrograms.zip/content"
VIDEO_URL = "https://drive.usercontent.google.com/download?id=1awlVJav7nRWnxHVNSCdnQXinQgbxGEdd&export=download&confirm=t"


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True), encoding="utf-8")


class RangeFile(io.RawIOBase):
    """Read-only seekable HTTP file with strict range and download-budget checks."""

    def __init__(self, url, size, budget=160_000_000):
        self.url, self.size, self.budget = url, size, budget
        self.pos, self.transferred, self.requests = 0, 0, 0
        self.session = requests.Session()
        self.cache_start, self.cache = -1, b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos + offset if whence == 1 else self.size + offset)
        if self.pos < 0:
            raise ValueError("Negative seek")
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, max(0, self.size - self.pos))
        if n == 0:
            return b""
        if self.cache_start <= self.pos and self.pos + n <= self.cache_start + len(self.cache):
            start = self.pos - self.cache_start
            result = self.cache[start:start+n]
            self.pos += len(result)
            return result
        end = min(self.size - 1, self.pos + max(n, 65536) - 1)
        count = end - self.pos + 1
        if self.transferred + count > self.budget:
            raise RuntimeError("Bounded sample download budget exceeded")
        expected = f"bytes {self.pos}-{end}/{self.size}"
        # Explicitly distinct URLs prevent proxies from reusing a different range.
        sep = "&" if "?" in self.url else "?"
        url = self.url + sep + f"probe_range={self.pos}-{end}"
        last = None
        for attempt in range(3):
            try:
                with self.session.get(url, headers={"Range": f"bytes={self.pos}-{end}"}, stream=True, timeout=(20, 60)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError(f"Server did not honor bounded range: {response.status_code}, {response.headers.get('Content-Range')}")
                    data = bytearray()
                    for chunk in response.iter_content(262144):
                        data.extend(chunk)
                        self.transferred += len(chunk)
                        if len(data) > count or self.transferred > self.budget:
                            raise RuntimeError("Server exceeded download budget")
                    if len(data) != count:
                        raise RuntimeError("Incomplete byte range")
                break
            except requests.RequestException as exc:
                last = exc
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        self.requests += 1
        self.cache_start, self.cache = self.pos, bytes(data)
        result = self.cache[:n]
        self.pos += len(result)
        return result


def fetch_metadata():
    response = requests.get("https://zenodo.org/api/records/8252482", timeout=40)
    response.raise_for_status()
    metadata = response.json()
    save_json(RAW / "soundwel" / "record.json", metadata)
    entry = next(f for f in metadata["files"] if f["key"] == "SoundwelDatasetKey.xlsx")
    response = requests.get(entry["links"]["self"], timeout=60)
    response.raise_for_status()
    if len(response.content) != entry["size"] or hashlib.md5(response.content).hexdigest() != entry["checksum"].split(":")[1]:
        raise RuntimeError("Soundwel metadata checksum mismatch")
    path = RAW / "soundwel" / entry["key"]
    path.write_bytes(response.content)
    audio_entry = next(f for f in metadata["files"] if f["key"].endswith(".zip"))
    remote = RangeFile(AUDIO_URL, audio_entry["size"], budget=15_000_000)
    with zipfile.ZipFile(remote) as archive:
        entries = [{"name": f.filename, "bytes": f.file_size, "compressed_bytes": f.compress_size, "crc32": f.CRC} for f in archive.infolist() if not f.is_dir()]
    save_json(RAW / "soundwel" / "archive_index.json", entries)
    print("Audio archive indexed:", len(entries), "entries; bytes transferred:", remote.transferred, flush=True)


def fetch_vision():
    remote = RangeFile(VIDEO_URL, 3292047360)
    entries = []
    video_saved = False
    with tarfile.open(fileobj=remote, mode="r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            if ".." in parts or PurePosixPath(member.name).is_absolute():
                raise RuntimeError("Unsafe archive path")
            entries.append({"name": member.name, "bytes": member.size, "offset": member.offset_data})
            basename = parts[-1]
            wanted = basename in {"output.json", "times.txt", "background.png"}
            if basename == "color.mp4" and not video_saved and member.size < 100_000_000:
                wanted = True
                video_saved = True
            if wanted:
                path = RAW / "edinburgh" / Path(*parts)
                if not path.exists() or path.stat().st_size != member.size:
                    data = archive.extractfile(member).read()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                print("Video sample:", member.name, member.size, flush=True)
    save_json(RAW / "edinburgh" / "archive_index.json", entries)
    save_json(RAW / "edinburgh" / "download_stats.json", {"url": VIDEO_URL, "archive_bytes": remote.size, "bytes_transferred": remote.transferred, "range_requests": remote.requests})
    print("Video archive indexed:", len(entries), "entries; bytes transferred:", remote.transferred, flush=True)


def fetch_audio(names):
    metadata = json.loads((RAW / "soundwel" / "record.json").read_text())
    size = next(f["size"] for f in metadata["files"] if f["key"].endswith(".zip"))
    remote = RangeFile(AUDIO_URL, size, budget=40_000_000)
    wanted = set(json.loads(Path(names).read_text(encoding="utf-8")))
    found = set()
    manifest = []
    with zipfile.ZipFile(remote) as archive:
        for info in archive.infolist():
            name = PurePosixPath(info.filename).name
            if name in wanted:
                if name in found:
                    raise RuntimeError("Duplicate sample basename")
                data = archive.read(info)  # ZipFile validates CRC.
                path = RAW / "soundwel" / "audio" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                manifest.append({"file": name, "archive_member": info.filename, "sha256": hashlib.sha256(data).hexdigest()})
                found.add(name)
                print("Audio sample:", name, len(data), flush=True)
    if found != wanted:
        raise RuntimeError(f"Missing selected audio: {sorted(wanted-found)}")
    save_json(RAW / "soundwel" / "sample_provenance.json", {"files": manifest, "bytes_transferred": remote.transferred, "range_requests": remote.requests, "url": AUDIO_URL})


def fetch_video_clip():
    """Download only the smallest of the twelve RGB clips (about 213 MB)."""
    entries = json.loads((RAW / "edinburgh" / "archive_index.json").read_text())
    entry = min((e for e in entries if e["name"].endswith("/color.mp4")), key=lambda e: e["bytes"])
    if entry["bytes"] > 240_000_000:
        raise RuntimeError("Single-clip download cap exceeded")
    path = RAW / "edinburgh" / Path(*PurePosixPath(entry["name"]).parts)
    if path.exists() and path.stat().st_size == entry["bytes"]:
        print("Video clip already present:", path, flush=True)
        return
    start, end = entry["offset"], entry["offset"] + entry["bytes"] - 1
    temporary = path.with_suffix(".mp4.part")
    digest, transferred = hashlib.sha256(), 0
    with requests.get(VIDEO_URL + f"&probe_range={start}-{end}", headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(20, 60)) as response:
        response.raise_for_status()
        if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/3292047360":
            raise RuntimeError("Server did not honor single-clip byte range")
        with temporary.open("wb") as output:
            for chunk in response.iter_content(1048576):
                transferred += len(chunk)
                if transferred > entry["bytes"]:
                    raise RuntimeError("Clip response exceeded expected size")
                output.write(chunk)
                digest.update(chunk)
                if transferred // 25000000 != (transferred-len(chunk)) // 25000000:
                    print("RGB bytes downloaded:", transferred, flush=True)
    if transferred != entry["bytes"]:
        raise RuntimeError("Incomplete RGB clip")
    temporary.replace(path)
    save_json(RAW / "edinburgh" / "rgb_clip_provenance.json", {**entry, "sha256": digest.hexdigest(), "source_url": VIDEO_URL})
    print("RGB clip downloaded:", path, transferred, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["metadata", "vision", "audio", "video-clip"])
    parser.add_argument("--names", help="JSON array of selected WAV basenames")
    args = parser.parse_args()
    if args.stage == "metadata":
        fetch_metadata()
    elif args.stage == "vision":
        fetch_vision()
    elif args.stage == "video-clip":
        fetch_video_clip()
    elif args.names:
        fetch_audio(args.names)
    else:
        parser.error("audio stage requires --names")
