"""Expand sparse Edinburgh annotations without mixing clips or inventing labels.

Original JSON annotations are change points. Boxes are linearly interpolated,
behavior is forward-filled, and annotation slot f maps to RGB frame 3*f.
Outputs retain provenance flags; interpolated boxes are not independent labels.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "edinburgh"
OUT = ROOT / "data" / "processed" / "vision"
REPORTS = ROOT / "reports"
ALIASES = {"playwithtoy": "play-with-toy", "jumpontopof": "jump-on-top-of"}
GROUPS = {
    "sleep": "resting", "lying": "resting", "sitting": "stationary", "standing": "stationary",
    "walk": "locomotion", "run": "locomotion", "eat": "feeding", "drink": "drinking",
    "investigating": "exploring", "nose-poke-elsewhere": "exploring", "nose-to-nose": "social_contact",
    "play-with-toy": "toy_play", "fight": "interaction_candidate", "chase": "interaction_candidate",
    "jump-on-top-of": "interaction_candidate", "mount": "interaction_candidate", "tail-biting": "interaction_candidate",
    "other": "unknown",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    dense_rows, sparse_rows, clips = [], [], []
    duplicates, boundary_clips = 0, 0
    all_images = []
    for source in sorted(RAW.rglob("output.json")):
        obj = json.loads(source.read_text(encoding="utf-8"))
        folder = source.parent
        clip_id = folder.parent.name + "/" + folder.name
        image = Image.open(folder / "background.png").convert("RGB")
        width, height = image.size
        timestamps = (folder / "times.txt").read_text().splitlines()
        if len(timestamps) != 1800:
            raise ValueError(f"Unexpected timestamp count in {clip_id}")
        capture_times = pd.to_datetime(pd.Series(timestamps))
        capture_intervals = capture_times.diff().dt.total_seconds().dropna()
        if (capture_intervals <= 0).any():
            raise ValueError("Non-monotonic capture timestamps")
        slots = np.arange(0, len(timestamps), 3) // 3
        if len(obj["objects"]) != 8:
            raise ValueError(f"Unexpected track count in {clip_id}")
        clip_sparse = 0
        for track in obj["objects"]:
            unique = {}
            for event in track["frames"]:
                f = event["frameNumber"]
                if f != int(f) or not 0 <= f < len(slots):
                    raise ValueError("Annotation frame outside expected domain")
                duplicates += int(f in unique)
                unique[f] = event
            events = [unique[f] for f in sorted(unique)]
            frame_numbers = np.array(sorted(unique))
            if frame_numbers[0] != 0:
                raise ValueError("Cannot forward-fill an unannotated track start")
            values = np.array([[e["bbox"][k] for k in ("x", "y", "width", "height")] for e in events], dtype=float)
            if not np.isfinite(values).all() or (values[:,2:] <= 0).any():
                raise ValueError("Invalid source bounding box")
            for event in events:
                sparse_rows.append({"clip_id": clip_id, "pig_id_within_clip": track["id"], "annotation_frame": event["frameNumber"], "behavior_raw": event["behaviour"], **event["bbox"]})
            clip_sparse += len(events)
            interpolated = np.column_stack([np.interp(slots, frame_numbers, values[:,j]) for j in range(4)])
            left = np.searchsorted(frame_numbers, slots, side="right") - 1
            exact = set(frame_numbers)
            for slot, box, event_index in zip(slots, interpolated, left):
                event = events[event_index]
                behavior = ALIASES.get(event["behaviour"], event["behaviour"])
                if behavior not in GROUPS:
                    raise ValueError(f"Unmapped observed behavior: {behavior}")
                x, y, w, h = box
                x1, y1, x2, y2 = np.clip([x,y,x+w,y+h], [0,0,0,0], [width,height,width,height])
                changed = not np.allclose([x,y,x+w,y+h], [x1,y1,x2,y2])
                boundary_clips += int(changed)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError("Degenerate box after clipping")
                raw_frame = int(slot)*3
                dense_rows.append({"clip_id": clip_id, "day_group": folder.parent.name, "pig_id_within_clip": track["id"], "annotation_frame": int(slot), "raw_video_frame": raw_frame, "timestamp_source": timestamps[raw_frame], "behavior_raw": event["behaviour"], "behavior": behavior, "behavior_group_draft": GROUPS[behavior], "x1": x1, "y1": y1, "x2": x2, "y2": y2, "yolo_xc": (x1+x2)/(2*width), "yolo_yc": (y1+y2)/(2*height), "yolo_w": (x2-x1)/width, "yolo_h": (y2-y1)/height, "visible": event.get("visible",False), "source_ground_truth_flag": event.get("isGroundTruth",False), "exact_box_keyframe": int(slot) in exact, "bbox_clipped": changed})
        clips.append({"clip_id": clip_id, "tracks": len(obj["objects"]), "raw_frames": len(timestamps), "annotation_slots": len(slots), "source_change_points": clip_sparse, "width": width, "height": height, "first_timestamp": timestamps[0], "last_timestamp": timestamps[-1], "capture_span_s": float((capture_times.iloc[-1]-capture_times.iloc[0]).total_seconds()), "median_capture_fps": float(1/capture_intervals.median()), "max_capture_gap_s": float(capture_intervals.max()), "has_local_rgb": (folder/"color.mp4").exists()})
        all_images.append((clip_id, image))
    dense = pd.DataFrame(dense_rows)
    if len(clips) != 12:
        raise ValueError("Expected all 12 annotated clips before preprocessing")
    if dense.duplicated(["clip_id","pig_id_within_clip","annotation_frame"]).any():
        raise ValueError("Duplicate dense annotation")
    dense.to_csv(OUT/"annotations_2hz.csv",index=False)
    pd.DataFrame(sparse_rows).to_csv(OUT/"annotation_change_points.csv",index=False)
    pd.DataFrame(clips).to_csv(OUT/"clip_manifest.csv",index=False)
    dense.groupby(["behavior","clip_id"]).size().rename("pig_annotation_slots").reset_index().to_csv(OUT/"behavior_by_clip.csv",index=False)
    counts = dense.groupby("behavior").agg(pig_annotation_slots=("behavior","size"), clips=("clip_id","nunique"), days=("day_group","nunique")).sort_values("pig_annotation_slots",ascending=False)
    counts.to_csv(OUT/"behavior_counts.csv")
    # All days from one pen and mostly the same pigs. These are day holdouts only.
    folds = []
    days = sorted(dense["day_group"].unique())
    for fold, day in enumerate(days):
        folds.append({"fold": fold, "held_out_day": day, "test_clips": sorted(dense.loc[dense["day_group"]==day,"clip_id"].unique()), "train_clips": sorted(dense.loc[dense["day_group"]!=day,"clip_id"].unique())})
    (OUT/"leave_one_day_out.json").write_text(json.dumps(folds,indent=2),encoding="utf-8")
    (OUT/"behavior_mapping_draft.json").write_text(json.dumps({"aliases":ALIASES,"draft_groups":GROUPS,"note":"Draft grouping requires domain review. interaction_candidate is not a confirmed welfare anomaly."},indent=2),encoding="utf-8")
    # Visual check of frame-zero alignment for every clip, without full video download.
    canvas=Image.new("RGB",(1280, 4*265),"white")
    draw=ImageDraw.Draw(canvas)
    for j,(clip_id,im) in enumerate(all_images):
        preview=im.copy(); d=ImageDraw.Draw(preview)
        for row in dense[(dense["clip_id"]==clip_id)&(dense["annotation_frame"]==0)].to_dict("records"):
            d.rectangle((row['x1'],row['y1'],row['x2'],row['y2']),outline="red",width=4)
            d.text((row['x1']+3,row['y1']+3),str(row['pig_id_within_clip'])+': '+row['behavior'],fill="yellow",stroke_width=1,stroke_fill="black")
        preview.thumbnail((420,237)); x=j%3*426; y=j//3*265
        draw.text((x+3,y+3),clip_id,fill="black");canvas.paste(preview,(x,y+25))
    canvas.save(REPORTS/"vision_annotation_preview.jpg",quality=92)
    decoded = []
    for video in sorted(RAW.rglob("color.mp4")):
        cap=cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError("RGB video failed to open")
        info={"file":str(video.relative_to(ROOT)),"fps_container":cap.get(cv2.CAP_PROP_FPS),"frames_container":int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),"width":int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),"height":int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        folder=video.parent;clip_id=folder.parent.name+'/'+folder.name
        frame_dir=OUT/"sample_frames"; label_dir=OUT/"sample_yolo_labels"; crop_dir=OUT/"sample_crops"
        for directory in (frame_dir,label_dir,crop_dir):directory.mkdir(exist_ok=True)
        exported=0
        for f in [0,3,30,300,900,1797]:
            cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read()
            if not ok:raise ValueError(f"Could not decode actual RGB frame {f}")
            stem=folder.parent.name+'_'+folder.name+'_'+str(f).zfill(6)
            if not cv2.imwrite(str(frame_dir/(stem+'.jpg')),bgr):raise IOError("Frame write failed")
            records=dense[(dense['clip_id']==clip_id)&(dense['raw_video_frame']==f)]
            labels=[]
            for row in records.to_dict('records'):
                labels.append('0 '+' '.join(f'{row[k]:.8f}' for k in ['yolo_xc','yolo_yc','yolo_w','yolo_h']))
                x1,y1,x2,y2=[int(round(row[k])) for k in ['x1','y1','x2','y2']]
                crop=bgr[y1:y2,x1:x2]
                if crop.size==0:raise ValueError('Empty crop')
                name=stem+'_pig'+str(row['pig_id_within_clip'])+'_'+row['behavior']+'.jpg'
                if not cv2.imwrite(str(crop_dir/name),cv2.resize(crop,(128,128))):raise IOError('Crop write failed')
            (label_dir/(stem+'.txt')).write_text('\n'.join(labels)+'\n',encoding='utf-8')
            exported+=1
        cap.release();info['exported_real_frames']=exported;decoded.append(info)
    summary={"annotated_clips":len(clips),"days":len(days),"unique_labeled_video_frames":int(dense[['clip_id','annotation_frame']].drop_duplicates().shape[0]),"pig_annotation_slots":len(dense),"sparse_change_points":len(sparse_rows),"duplicates_resolved_last_wins":duplicates,"boxes_clipped_to_image":boundary_clips,"raw_label_counts_change_points":dict(Counter(r['behavior_raw'] for r in sparse_rows)),"expanded_label_counts":counts.to_dict(orient='index'),"decoded_rgb_clips":decoded,"important":["Raw frame = 3 * annotation slot. Labeled frequency is 2Hz, raw video 6Hz.","Box interpolation and forward-filled behavior are derived annotations, not new independent samples.","Pig identifiers are retained within clips only. No claim of unseen-pig generalization.","All twelve clips cover one hour, not continuous annotated 23-day activity.","Do not use the per-clip mask.png; the source documents it as incorrect.","Preserve fine labels; coarse grouping is a draft requiring review."]}
    (REPORTS/"vision_probe_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()
