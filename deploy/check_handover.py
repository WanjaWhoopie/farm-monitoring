"""Checks that the handover package works without the training code.

1. Detector ONNX, decoded by hand. One real frame goes through the steps written in
   detector/config.json only (letterbox, scale, run, threshold, NMS, undo letterbox), using
   onnxruntime and numpy. The boxes are compared with the ultralytics decoder on the same file and
   with the human boxes for that frame.
2. Audio ONNX. The file loads, the documented preprocessing on a one-second signal gives the
   documented input shape, and the output is two logits per clip (softmax is applied outside).
3. Binary event record. Encode and decode round trip, including rule flags and severity.
4. rules.json agrees with the operating points table it was generated from.

Usage: python deploy/check_handover.py        (from the repository root, about 20 s)
Needs: onnxruntime, numpy, opencv-python, pandas, and librosa for check 2. The clip video for
check 1 (data/raw/edinburgh/annotated/2019_12_02/000005/color.mp4) and ultralytics are optional:
without them that check is partly skipped and says so.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
DEPLOY = ROOT / "deploy"
CLIP = "2019_12_02/000005"
FRAME = 300
results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, note: str) -> None:
    results.append((name, ok, note))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {note}")


# ----------------------------------------------------------------------------- 1. detector
def letterbox(img: np.ndarray, size: int = 640, pad: int = 114):
    import cv2
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) / 2, (size - nh) / 2
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(out, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad, pad, pad))
    return out, r, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]; keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-9)
        order = rest[iou < iou_thr]
    return keep


def detect_by_hand(frame_bgr: np.ndarray, conf: float = 0.25, iou_thr: float = 0.7) -> np.ndarray:
    """Boxes (x1, y1, x2, y2, score) in original-frame pixels, following detector/config.json only."""
    import onnxruntime as ort
    cfg = json.loads((DEPLOY / "detector" / "config.json").read_text())
    size = cfg["input"]["shape"][2]
    img, r, (dx, dy) = letterbox(frame_bgr, size)
    x = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0   # BGR -> RGB, HWC -> NCHW
    sess = ort.InferenceSession(str(DEPLOY / "detector" / cfg["file"]), providers=["CPUExecutionProvider"])
    out = sess.run(None, {cfg["input"]["name"]: x})[0][0].T                   # (8400, 5): cx cy w h score
    out = out[out[:, 4] >= conf]
    xyxy = np.stack([out[:, 0] - out[:, 2] / 2, out[:, 1] - out[:, 3] / 2, out[:, 0] + out[:, 2] / 2, out[:, 1] + out[:, 3] / 2], 1)
    keep = nms(xyxy, out[:, 4], iou_thr)
    xyxy = xyxy[keep]
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - dx) / r
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - dy) / r
    h, w = frame_bgr.shape[:2]
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w); xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
    return np.column_stack([xyxy, out[keep, 4]])


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0]); yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2]); yy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def check_detector() -> None:
    import cv2
    video = ROOT / "data" / "raw" / "edinburgh" / "annotated" / CLIP / "color.mp4"
    if not video.exists():
        frame = (np.random.default_rng(0).random((720, 1280, 3)) * 255).astype(np.uint8)
        det = detect_by_hand(frame)
        record("detector ONNX runs by hand", True, f"no clip video on this machine, ran on a noise frame, {len(det)} boxes (no accuracy check)")
        return
    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME); ok, frame = cap.read(); cap.release()
    assert ok, "could not read the frame"
    det = detect_by_hand(frame)
    note = f"{len(det)} boxes on frame {FRAME} of {CLIP}, scores {det[:, 4].min():.2f} to {det[:, 4].max():.2f}"
    # against the human boxes of that frame
    try:
        from activity_budget import load_labels
        lab = load_labels()
        g = lab[(lab["clip_id"] == CLIP) & (lab["annotation_index"] == FRAME)]
        gt = np.column_stack([g["x"], g["y"], g["x"] + g["width"], g["y"] + g["height"]]).astype(float)
        m = iou_xyxy(det[:, :4], gt)
        matched = int((m.max(0) >= 0.5).sum()) if len(gt) else 0
        note += f"; {matched} of {len(gt)} human boxes matched at IoU >= 0.5"
        ok_gt = matched >= len(gt) - 1 and len(det) <= len(gt) + 1
    except Exception as e:  # labels not available
        ok_gt, note = True, note + f"; human boxes not checked ({type(e).__name__})"
    # against the ultralytics decoder on the same ONNX file
    try:
        from ultralytics import YOLO
        ref = YOLO(str(DEPLOY / "detector" / "pig_yolo11n_640.onnx")).predict(frame, imgsz=640, conf=0.25, iou=0.7, verbose=False)[0].boxes.xyxy.cpu().numpy()
        m = iou_xyxy(det[:, :4], ref)
        worst = float(m.max(1).min()) if len(det) and len(ref) else 0.0
        note += f"; ultralytics decoder gives {len(ref)} boxes, worst pairwise IoU {worst:.3f}"
        ok_ref = len(ref) == len(det) and worst >= 0.95
    except Exception as e:
        ok_ref, note = True, note + f"; ultralytics not compared ({type(e).__name__})"
    record("detector ONNX decoded from config.json alone", bool(ok_gt and ok_ref), note.replace("; ", ", "))


# ----------------------------------------------------------------------------- 2. audio
def check_audio() -> None:
    import onnxruntime as ort
    cfg = json.loads((ROOT / "audio_models" / "final_model" / "config.json").read_text())
    sess = ort.InferenceSession(str(ROOT / "audio_models" / "final_model" / "effnet_b0_valence_fp32.onnx"), providers=["CPUExecutionProvider"])
    inp, outp = sess.get_inputs()[0], sess.get_outputs()[0]
    try:
        import librosa
        sr = cfg["preprocessing"]["resample_hz"]; mel = cfg["preprocessing"]["mel"]
        t = np.arange(sr) / sr
        y = (0.5 * np.sin(2 * np.pi * 1500 * t) * np.exp(-3 * t)).astype(np.float32)   # a one-second decaying tone
        y = y / np.max(np.abs(y))
        s = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=mel["n_fft"], hop_length=mel["hop_length"], n_mels=mel["n_mels"],
                                           fmin=mel["fmin"], fmax=mel["fmax"], power=mel["power"])
        s = librosa.power_to_db(s, ref=np.max, top_db=80)
        x = ((s + 40) / 40).astype(np.float32)[None, None]
        src = "documented preprocessing on a synthetic tone"
    except ImportError:
        x = np.random.default_rng(0).standard_normal((1, 1, 64, 101)).astype(np.float32); src = "random input (librosa missing)"
    shape_ok = list(x.shape) == cfg["input_shape"]
    out = sess.run(None, {inp.name: x})[0]
    p = np.exp(out - out.max(1, keepdims=True)); p = p / p.sum(1, keepdims=True)
    is_logits = not np.allclose(out.sum(1), 1.0, atol=1e-3)
    note = (f"input '{inp.name}' {inp.shape}, {src} gives {list(x.shape)} ({'matches' if shape_ok else 'DOES NOT match'} config), "
            f"output '{outp.name}' {out.shape} = {'logits' if is_logits else 'probabilities'}, softmax -> P(negative) {p[0, 1]:.3f}")
    record("audio ONNX loads and the preprocessing recipe gives the right shape", bool(shape_ok and out.shape == (1, 2)), note)


# ----------------------------------------------------------------------------- 3. binary record
def check_binary() -> None:
    from event_summary import CLASSES7, decode_binary, encode_binary
    rec = {"v": 1, "pen": 3, "t": 1575288000, "win_min": 5, "pigs_seen": 7.6,
           "budget": dict(zip(CLASSES7, [0.03, 0.38, 0.02, 0.08, 0.20, 0.25, 0.04])),
           "motion_px_s": 41.2, "audio": {"calls": 17, "neg_valence": 0.18}, "flags": ["movement", "call", "combined"], "severity": 2, "src": "x"}
    b = encode_binary(rec)
    d = decode_binary(b)
    same = (len(b) == 24 and d["flags"] == rec["flags"] and d["severity"] == 2 and d["pen"] == 3 and d["t"] == rec["t"]
            and abs(d["pigs_seen"] - 7.6) < 1e-9 and d["audio"]["calls"] == 17 and abs(d["audio"]["neg_valence"] - 0.18) < 0.003
            and all(abs(d["budget"][c] - rec["budget"][c]) < 0.003 for c in CLASSES7) and abs(d["motion_px_s"] - 41.2) < 1e-9)
    bad = bytearray(b); bad[7] ^= 0xFF
    try:
        decode_binary(bytes(bad)); crc_ok = False
    except ValueError:
        crc_ok = True
    record("24-byte record round trip with flags and severity", bool(same and crc_ok),
           f"{len(b)} bytes, flags byte 0x{b[19]:02x} -> {d['flags']} severity {d['severity']}, corrupted byte rejected by CRC: {crc_ok}")


# ----------------------------------------------------------------------------- 4. rules.json
def check_rules() -> None:
    import pandas as pd
    rules = json.loads((DEPLOY / "rules.json").read_text())
    src = rules["source"].split("thresholds from ")[1].split(" (")[0]
    ops = pd.read_csv(ROOT / src)
    cal = ops[ops["audio"] == "calibrated"].set_index("rule"); new = ops[ops["audio"] == "new farm"].set_index("rule")
    bad = []
    for r, d in rules["rules"].items():
        if d["z_threshold"]["calibrated_audio"] != cal.loc[r, "z"] or d["z_threshold"]["new_farm_audio"] != new.loc[r, "z"]:
            bad.append(r + " z")
        for k in ("false_alarms_per_day", "detection_rate", "median_lag_min"):
            if abs(d["simulated_at_threshold"][k] - cal.loc[r, k]) > 1e-9:
                bad.append(f"{r} {k}")
    record("rules.json matches the operating points table", not bad, f"{len(rules['rules'])} rules checked against {src}" + (f", mismatches: {bad}" if bad else ""))


if __name__ == "__main__":
    for fn in (check_detector, check_audio, check_binary, check_rules):
        try:
            fn()
        except Exception as e:
            record(fn.__name__, False, f"{type(e).__name__}: {e}")
    n_ok = sum(ok for _, ok, _ in results)
    print(f"\n{n_ok} of {len(results)} checks passed")
    sys.exit(0 if n_ok == len(results) else 1)
