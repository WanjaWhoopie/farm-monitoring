"""Event summaries and bytes per day: what a pen monitor would send instead of video (systems track).

The vision branch turns five minutes of video into one small record per pen: how many pigs were
seen, the share of time in each behaviour, how much the pigs moved, and any flags. This script
1. defines that record (one per time window) and fills it from the real outputs in reports/
   (predicted pen budgets from the geometry model, boxes per frame and movement from the
   detector's tracks);
2. encodes it four ways (readable JSON, compact JSON, compressed JSON, a fixed 24-byte binary
   record) and measures the size of each;
3. works out bytes per day for every way of getting the same information off the farm, from
   sending the raw video down to one binary summary per hour, and how long each takes on a
   2G link, and how many SMS messages it would need.

Window = one Edinburgh clip (five minutes). A deployed system would use the same record at
5 minutes, 15 minutes or 1 hour; all three are costed.

Binary record (24 bytes, little-endian):
  version u8 | pen u8 | window start u32 (unix s) | window length u8 (min) | pigs seen x10 u8
  | 7 behaviour shares u8 each (0-255 = 0-100%) | motion u16 (px/s x10) | calls u8
  | negative-valence share u8 | flags u8 (bits 0-5 one per rule, bits 6-7 severity) | CRC-8 u8
  | reserved u8 x3
Audio fields are present in the record but empty here: there is no paired audio for these pens.
The flags and severity come from the fusion rules (scripts/fusion_rules.py, deploy/rules.json);
decode_binary() is the inverse of encode_binary() and deploy/check_handover.py tests the round trip.

Usage: python scripts/event_summary.py [--out DIR]
Outputs: event_summary_example.json, summaries_all_clips.jsonl, bytes_per_day.csv,
         bytes_per_day_figure.png, run_info.json
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import json
import platform
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from activity_budget import CLASSES7, ROOT, capture_seconds  # noqa: E402

BUDGETS = ROOT / "reports" / "activity_budget_2026-11-16" / "budgets_pen_level.csv"
TRACKS = ROOT / "reports" / "detector_2026-11-18"
ARCHIVE_INDEX = ROOT / "data" / "raw" / "edinburgh" / "archive_index.json"
FULL_RATE_HZ = 6.04
WINDOW_S = 300  # one clip
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#5a5955", "#e1e0d9"

# Link assumptions (stated in the README): GPRS 2G in practice ~40 kbit/s = 5,000 bytes/s;
# one SMS carries 140 bytes of binary payload (or 160 7-bit characters).
LINK_2G_BPS = 5_000
SMS_BYTES = 140
RULE_BITS = ["movement", "rest", "call", "combined", "count", "feeding"]  # bit 0 .. bit 5 of the flags byte
BINARY_FORMAT = "<BBIBB7BHBBB"  # body, then CRC-8 and three reserved bytes
# Audio assumptions: 16 kHz, 16-bit mono PCM; Opus speech coding at 24 kbit/s.
AUDIO_PCM_BPS = 16_000 * 2
AUDIO_OPUS_BPS = 24_000 / 8


def crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def summary_for_clip(clip: str, budgets: pd.DataFrame, pen: int) -> dict:
    """One record per clip from the model's predicted pen budget and the detector's track (if any)."""
    row = budgets[(budgets["method"] == "geometry") & (budgets["clip_id"] == clip)].iloc[0]
    secs = capture_seconds(clip) if (ROOT / "data" / "raw" / "edinburgh" / "annotated" / clip / "times.txt").exists() else None
    t_start = int(datetime.datetime.strptime(clip[:10], "%Y_%m_%d").replace(hour=12).timestamp())
    pigs_seen, motion = None, None
    tf = TRACKS / f"track_{clip.replace('/', '_')}.csv"
    if tf.exists():
        tr = pd.read_csv(tf)
        pigs_seen = round(float(len(tr) / tr["frame"].nunique()), 1)
        tr = tr.sort_values(["track_id", "frame"])
        cx, cy = tr["x"] + tr["width"] / 2, tr["y"] + tr["height"] / 2
        same = tr["track_id"].eq(tr["track_id"].shift()) & tr["frame"].eq(tr["frame"].shift() + 1)
        step = np.hypot(cx.diff(), cy.diff())[same] * FULL_RATE_HZ
        motion = round(float(step.mean()), 1)
    return {"v": 1, "pen": pen, "t": t_start, "win_min": WINDOW_S // 60,
            "pigs_seen": pigs_seen,
            "budget": {c: round(float(row[c]), 3) for c in CLASSES7},
            "motion_px_s": motion,
            "audio": {"calls": None, "neg_valence": None},
            "flags": [],
            "severity": 0,
            "src": "clip " + clip}


def encode_binary(s: dict) -> bytes:
    shares = [int(round(255 * s["budget"][c])) for c in CLASSES7]
    flags = sum(1 << RULE_BITS.index(f) for f in s.get("flags", []) if f in RULE_BITS)
    flags |= (int(s.get("severity", 0)) & 0b11) << 6
    body = struct.pack(BINARY_FORMAT, s["v"], s["pen"], s["t"], s["win_min"],
                       int(round(10 * (s["pigs_seen"] or 0))), *shares,
                       int(round(10 * (s["motion_px_s"] or 0))), s["audio"]["calls"] or 0,
                       int(round(255 * (s["audio"]["neg_valence"] or 0))), flags)
    return body + bytes([crc8(body)]) + bytes(3)


def decode_binary(b: bytes) -> dict:
    """Inverse of encode_binary. Raises ValueError on a bad length or CRC."""
    if len(b) != 24:
        raise ValueError(f"expected 24 bytes, got {len(b)}")
    body = b[:20]
    if crc8(body) != b[20]:
        raise ValueError("CRC mismatch")
    v, pen, t, win_min, pigs10, *rest = struct.unpack(BINARY_FORMAT, body)
    shares, motion10, calls, neg255, flags = rest[:7], rest[7], rest[8], rest[9], rest[10]
    return {"v": v, "pen": pen, "t": t, "win_min": win_min, "pigs_seen": pigs10 / 10,
            "budget": {c: round(x / 255, 3) for c, x in zip(CLASSES7, shares)},
            "motion_px_s": motion10 / 10,
            "audio": {"calls": calls, "neg_valence": round(neg255 / 255, 3)},
            "flags": [r for i, r in enumerate(RULE_BITS) if flags >> i & 1],
            "severity": flags >> 6}


def human(n: float) -> str:
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= div:
            return f"{n / div:.3g} {unit}"
    return f"{n:.0f} B"


def draw_figure(table: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = table.iloc[::-1]
    colours = {"raw": PALETTE[1], "boxes": PALETTE[3], "summary": PALETTE[0]}
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    bars = ax.barh(t["stream"], t["bytes_per_day"], color=[colours[k] for k in t["kind"]], height=0.62)
    ax.set_xscale("log")
    for b, v in zip(bars, t["bytes_per_day"]):
        ax.text(v * 1.25, b.get_y() + b.get_height() / 2, human(v), va="center", fontsize=8.5, color=INK)
    for x, label in ((SMS_BYTES * 288, "one SMS every 5 min"), (LINK_2G_BPS * 86400, "2G link busy all day")):
        ax.axvline(x, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(x, len(t) - 0.35, "  " + label, fontsize=8, color=INK2, va="top")
    ax.set_xlim(100, 5e11)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel("bytes per day for one pen (log scale)", color=INK2, fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colours["raw"], label="raw signal"), Patch(color=colours["boxes"], label="box stream"),
                       Patch(color=colours["summary"], label="event summary")], frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("What leaves the farm: bytes per day for one pen, by what is sent", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ROOT / "reports" / f"deployability_{datetime.date.today().isoformat()}"
    out.mkdir(parents=True, exist_ok=True)

    budgets = pd.read_csv(BUDGETS)
    clips = sorted(budgets["clip_id"].unique())
    summaries = [summary_for_clip(c, budgets, pen=1) for c in clips]
    example = next(s for s in summaries if s["pigs_seen"] is not None)
    (out / "event_summary_example.json").write_text(json.dumps(example, indent=2), encoding="utf-8")
    with (out / "summaries_all_clips.jsonl").open("w", encoding="utf-8") as fh:
        for s in summaries:
            fh.write(json.dumps(s, separators=(",", ":")) + "\n")

    # sizes of one record
    pretty = json.dumps(example, indent=2).encode()
    compact = json.dumps(example, separators=(",", ":")).encode()
    gz = gzip.compress(compact, 9)
    binary = encode_binary(example)
    sizes = {"json_pretty": len(pretty), "json_compact": len(compact), "json_compact_gzip": len(gz), "binary": len(binary)}
    print("one record:", sizes)

    # raw streams from real file sizes
    idx = json.loads(ARCHIVE_INDEX.read_text())
    mp4 = [e["bytes"] for e in idx if e["name"].endswith("color.mp4")]
    video_per_clip = float(np.mean(mp4))
    track_sizes = [f.stat().st_size for f in TRACKS.glob("track_*.csv")]
    boxes_csv_per_clip = float(np.mean(track_sizes))
    boxes_per_frame = float(np.mean([s["pigs_seen"] for s in summaries if s["pigs_seen"] is not None]))
    per_day = 86400 / WINDOW_S  # 288 windows
    rows = [
        ("raw video, as recorded (mean of the 12 clips)", "raw", video_per_clip * per_day),
        ("raw audio, 16 kHz 16-bit PCM", "raw", AUDIO_PCM_BPS * 86400),
        ("audio compressed (Opus 24 kbit/s)", "raw", AUDIO_OPUS_BPS * 86400),
        ("box stream, CSV at 6 per s (detector output as written)", "boxes", boxes_csv_per_clip * per_day),
        ("box stream, binary 8 bytes per box at 2 per s", "boxes", boxes_per_frame * 2 * 8 * 86400),
        ("event summary, compact JSON every 5 min", "summary", len(compact) * per_day),
        ("event summary, compressed JSON every 5 min", "summary", len(gz) * per_day),
        ("event summary, 24-byte binary every 5 min", "summary", len(binary) * per_day),
        ("event summary, 24-byte binary every 15 min", "summary", len(binary) * 96),
        ("event summary, 24-byte binary every hour", "summary", len(binary) * 24),
    ]
    table = pd.DataFrame(rows, columns=["stream", "kind", "bytes_per_day"])
    table["bytes_per_day"] = table["bytes_per_day"].round(0)
    table["readable"] = table["bytes_per_day"].map(human)
    table["hours_on_2g_link"] = (table["bytes_per_day"] / LINK_2G_BPS / 3600).round(2)
    table["sms_per_day"] = np.ceil(table["bytes_per_day"] / SMS_BYTES).astype(int)
    table["share_of_raw_video"] = (table["bytes_per_day"] / table["bytes_per_day"].iloc[0]).map(lambda x: f"{x:.2e}")
    table.to_csv(out / "bytes_per_day.csv", index=False)
    print("\n" + table[["stream", "readable", "hours_on_2g_link", "sms_per_day"]].to_string(index=False))
    draw_figure(table, out / "bytes_per_day_figure.png")

    info = {"saved_at": datetime.datetime.now().isoformat(timespec="seconds"), "window_s": WINDOW_S,
            "record_bytes": sizes, "video_bytes_per_clip_mean": round(video_per_clip), "box_csv_bytes_per_clip_mean": round(boxes_csv_per_clip),
            "boxes_per_frame_mean": round(boxes_per_frame, 2), "assumptions": {"link_2g_bytes_per_s": LINK_2G_BPS, "sms_payload_bytes": SMS_BYTES,
            "audio_pcm_bytes_per_s": AUDIO_PCM_BPS, "audio_opus_bytes_per_s": AUDIO_OPUS_BPS},
            "inputs": {"budgets": str(BUDGETS.relative_to(ROOT)), "tracks": str(TRACKS.relative_to(ROOT)), "archive_index": str(ARCHIVE_INDEX.relative_to(ROOT))},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("\nSaved to", out)


if __name__ == "__main__":
    main()
