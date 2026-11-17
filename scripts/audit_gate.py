"""Audit sheet for the label gate, built so the eye can actually judge each row.

For every sampled gated row the sheet shows three zoomed crops side by side:
the last human-stored frame before it, the gated frame itself, and the next
human-stored frame after it. The crop window is the same for all three, so any
movement of the pig is visible, and the red box on the middle frame is the
interpolated box being audited. Two questions per row:

  box_ok   : does the red box in the middle frame sit on the same pig, covering it?
  label_ok : could that pig be doing the labelled behaviour between the two human frames?

Rows are drawn from clips whose color.mp4 is present under data/raw/edinburgh/annotated.
Usage: python scripts/audit_gate.py [--n 24] [--seed 42] [--out DIR]
Outputs: audit_sheet.png (and audit_sheet_2.png ... when more than 12 rows), audit_rows.csv to fill in.
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from activity_budget import DATA_ROOT, ROOT, load_labels  # noqa: E402

RED, WHITE, GREY = (0, 0, 255), (255, 255, 255), (170, 170, 170)


def bracketing_stored(series: pd.DataFrame, idx: int) -> tuple[pd.Series | None, pd.Series | None]:
    st = series[series["annotation_source"] == "stored"]
    before = st[st["annotation_index"] < idx]
    after = st[st["annotation_index"] > idx]
    return (before.iloc[-1] if len(before) else None, after.iloc[0] if len(after) else None)


def read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"could not read frame {idx}")
    return frame


def crop_window(box: pd.Series, frame_w: int, frame_h: int, scale: float = 3.0, min_side: int = 260) -> tuple[int, int, int, int]:
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    side = max(min_side, scale * max(box["width"], box["height"]))
    x0 = int(np.clip(cx - side / 2, 0, max(frame_w - side, 0)))
    y0 = int(np.clip(cy - side / 2, 0, max(frame_h - side, 0)))
    return x0, y0, int(min(side, frame_w - x0)), int(min(side, frame_h - y0))


def draw_panel(frame: np.ndarray, win: tuple[int, int, int, int], box: pd.Series | None, caption: str, size: int, colour) -> np.ndarray:
    x0, y0, w, h = win
    img = frame[y0:y0 + h, x0:x0 + w].copy()
    if box is not None:
        bx, by = int(box["x"] - x0), int(box["y"] - y0)
        cv2.rectangle(img, (bx, by), (bx + int(box["width"]), by + int(box["height"])), colour, 2)
    img = cv2.resize(img, (size, int(size * h / w)))
    bar = np.full((22, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, caption, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--panel", type=int, default=300, help="width of each crop in pixels")
    ap.add_argument("--per-sheet", type=int, default=12)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ROOT / "reports" / f"gate_audit_{datetime.date.today().isoformat()}"
    out.mkdir(parents=True, exist_ok=True)

    labels = load_labels()
    have_video = [c for c in labels["clip_id"].unique() if (DATA_ROOT / "annotated" / c / "color.mp4").exists()]
    print("clips with video on this machine:", have_video)
    pool = labels[(labels["tier"] == "gated") & labels["clip_id"].isin(have_video)]
    sample = pool.sample(min(args.n, len(pool)), random_state=args.seed).sort_values(["clip_id", "pig_id", "annotation_index"])
    print(f"sampled {len(sample)} gated rows from {len(pool)} available")

    strips, rows = [], []
    caps: dict[str, cv2.VideoCapture] = {}
    for k, (_, r) in enumerate(sample.iterrows(), 1):
        clip = r["clip_id"]
        if clip not in caps:
            caps[clip] = cv2.VideoCapture(str(DATA_ROOT / "annotated" / clip / "color.mp4"))
        cap = caps[clip]
        fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        series = labels[(labels["clip_id"] == clip) & (labels["pig_id"] == r["pig_id"])]
        before, after = bracketing_stored(series, r["annotation_index"])
        win = crop_window(r, fw, fh)
        panels = []
        for tag, rec, colour in (("human, before", before, GREY), ("GATED (audit this)", r, RED), ("human, after", after, GREY)):
            if rec is None:
                blank = np.full((win[3], win[2], 3), 60, np.uint8)
                panels.append(draw_panel(blank, (0, 0, win[2], win[3]), None, f"{tag}: none", args.panel, colour))
                continue
            frame = read_frame(cap, rec["annotation_index"])
            cap_txt = f"{tag}  frame {int(rec['annotation_index'])}  {rec['behavior']}"
            panels.append(draw_panel(frame, win, rec, cap_txt, args.panel, colour))
        hmax = max(p.shape[0] for p in panels)
        panels = [np.vstack([p, np.full((hmax - p.shape[0], p.shape[1], 3), 30, np.uint8)]) for p in panels]
        strip = np.hstack([panels[0], np.full((hmax, 4, 3), 255, np.uint8), panels[1], np.full((hmax, 4, 3), 255, np.uint8), panels[2]])
        head = np.full((26, strip.shape[1], 3), 255, np.uint8)
        gap = "" if before is None or after is None else f"   gap {int(after['annotation_index'] - before['annotation_index'])} frames (about {(after['elapsed_seconds'] - before['elapsed_seconds']):.1f} s)"
        cv2.putText(head, f"#{k}  {clip}  pig {r['pig_id']}  frame {int(r['annotation_index'])}  label: {r['behavior']}{gap}",
                    (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        strips.append(np.vstack([head, strip]))
        rows.append({"n": k, "clip_id": clip, "pig_id": r["pig_id"], "annotation_index": int(r["annotation_index"]),
                     "behavior": r["behavior"], "annotation_source": r["annotation_source"],
                     "before_frame": None if before is None else int(before["annotation_index"]),
                     "after_frame": None if after is None else int(after["annotation_index"]),
                     "box_ok": "", "label_ok": "", "note": ""})
    for cap in caps.values():
        cap.release()

    # Sheets: two strips per row, `per_sheet` strips per sheet.
    wmax = max(s.shape[1] for s in strips)
    strips = [np.hstack([s, np.full((s.shape[0], wmax - s.shape[1], 3), 255, np.uint8)]) for s in strips]
    for si in range(0, len(strips), args.per_sheet):
        chunk = strips[si:si + args.per_sheet]
        lines = []
        for a in range(0, len(chunk), 2):
            pair = chunk[a:a + 2]
            if len(pair) == 1:
                pair.append(np.full_like(pair[0], 255))
            hmax = max(p.shape[0] for p in pair)
            pair = [np.vstack([p, np.full((hmax - p.shape[0], p.shape[1], 3), 255, np.uint8)]) for p in pair]
            lines.append(np.hstack([pair[0], np.full((hmax, 16, 3), 255, np.uint8), pair[1]]))
            lines.append(np.full((14, lines[-1].shape[1], 3), 255, np.uint8))
        sheet = np.vstack(lines)
        name = "audit_sheet.png" if si == 0 else f"audit_sheet_{si // args.per_sheet + 1}.png"
        cv2.imwrite(str(out / name), sheet)
        print("wrote", name, sheet.shape[1], "x", sheet.shape[0])
    pd.DataFrame(rows).to_csv(out / "audit_rows.csv", index=False)
    print("wrote audit_rows.csv; fill in box_ok and label_ok with y/n (and ? for cannot tell)")


if __name__ == "__main__":
    main()
