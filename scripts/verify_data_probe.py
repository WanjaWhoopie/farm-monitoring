"""Check processed outputs directly against untouched source observations."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "data" / "processed"
raw = ROOT / "data" / "raw"
checks = {}
meta = pd.read_csv(out / "soundwel_metadata.csv")
assert len(meta) == 6887 and meta["archive_present"].all()
assert meta["audio_filename"].is_unique
assert (meta.groupby("split_group_proxy")["candidate_group_fold"].nunique() == 1).all()
checks["audio_metadata_rows_and_group_exclusivity"] = len(meta)
manifest = pd.read_csv(out / "audio_probe_manifest.csv")
assert len(manifest) == 68 and manifest["context_category"].nunique() == 17
assert (manifest.groupby("context_category").size() == 4).all()
provenance = json.loads((raw / "soundwel" / "sample_provenance.json").read_text())
hashes = {r["file"]:r["sha256"] for r in provenance["files"]}
for record in manifest.to_dict("records"):
    name = record["audio_filename"]
    digest = hashlib.sha256((raw / "soundwel" / "audio" / name).read_bytes()).hexdigest()
    assert digest == record["sha256_raw"] == hashes[name]
    src, src_rate = sf.read(raw / "soundwel" / "audio" / name)
    dst, dst_rate = sf.read(out / "audio_16khz" / name)
    assert dst_rate == 16000 and dst.ndim == 1 and np.isfinite(dst).all()
    assert abs(len(dst) / dst_rate - len(src) / src_rate) <= 1 / dst_rate
    feature = np.load(out / "audio_16khz" / (Path(name).stem + ".logmel.npy"), allow_pickle=False)
    assert feature.shape == (1,64,197) and np.isfinite(feature).all()
checks["decoded_audio_duration_hash_and_feature_checks"] = len(manifest)
vision = pd.read_csv(out / "vision" / "annotations_2hz.csv",dtype={"pig_id_within_clip":str})
assert len(vision) == 57600
assert (vision.groupby("clip_id").size() == 4800).all()
assert not vision.duplicated(["clip_id","pig_id_within_clip","annotation_frame"]).any()
assert (vision["raw_video_frame"] == 3*vision["annotation_frame"]).all()
assert vision[["yolo_xc","yolo_yc","yolo_w","yolo_h"]].apply(lambda x:x.between(0,1)).all().all()
exact_count = 0
interpolated_count = 0
aliases = {"playwithtoy":"play-with-toy","jumpontopof":"jump-on-top-of"}
for path in sorted((raw / "edinburgh").rglob("output.json")):
    obj = json.loads(path.read_text())
    clip = path.parent.parent.name + "/" + path.parent.name
    times = (path.parent / "times.txt").read_text().splitlines()
    for track in obj["objects"]:
        events = sorted(track["frames"], key=lambda f:f["frameNumber"])
        rows = vision[(vision["clip_id"] == clip)&(vision["pig_id_within_clip"] == str(track["id"]))].set_index("annotation_frame")
        for event in events:
            row = rows.loc[event["frameNumber"]]
            b = event["bbox"]
            assert np.allclose([row.x1,row.y1,row.x2,row.y2],[b['x'],b['y'],b['x']+b['width'],b['y']+b['height']])
            assert row.behavior == aliases.get(event["behaviour"],event["behaviour"])
            assert row.timestamp_source == times[3*event["frameNumber"]]
            exact_count += 1
        for first,second in zip(events,events[1:]):
            f1,f2 = first["frameNumber"],second["frameNumber"]
            if f2-f1 > 1:
                middle=(f1+f2)//2
                fraction=(middle-f1)/(f2-f1)
                expected_x=first['bbox']['x']*(1-fraction)+second['bbox']['x']*fraction
                assert abs(rows.loc[middle].x1-expected_x) < 1e-8
                assert rows.loc[middle].behavior == aliases.get(first['behaviour'],first['behaviour'])
                interpolated_count += 1
                break
checks["source_keyframe_boxes_labels_and_timestamps"] = exact_count
checks["independent_midpoint_interpolation_checks"] = interpolated_count
folds = json.loads((out / "vision" / "leave_one_day_out.json").read_text())
for fold in folds:
    assert not set(fold["train_clips"]) & set(fold["test_clips"])
    assert all(not c.startswith(fold['held_out_day']) for c in fold['train_clips'])
checks["disjoint_day_holdouts"] = len(folds)
rgb = list((raw / "edinburgh").rglob("color.mp4"))
assert len(rgb)==1
capture=cv2.VideoCapture(str(rgb[0]));assert capture.isOpened()
assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT))==1800
capture.release()
labels=list((out / "vision" / "sample_yolo_labels").glob("*.txt"))
crops=list((out / "vision" / "sample_crops").glob("*.jpg"))
assert len(labels)==6 and len(crops)==48
for p in labels:
    rows=p.read_text().splitlines();assert len(rows)==8
    for row in rows:
        fields=row.split();assert fields[0]=='0' and len(fields)==5
        assert all(0<=float(x)<=1 for x in fields[1:])
for p in crops:
    image=cv2.imread(str(p));assert image is not None and image.shape==(128,128,3)
checks['rgb_frames_and_pig_crops']={'frames':len(labels),'crops':len(crops)}
baseline=json.loads((ROOT/'reports'/'acoustic_baseline.json').read_text())
for name in ['random_call_optimistic','team_date_group_proxy','leave_recording_team_out']:
    predictions=pd.read_csv(out/('baseline_'+name+'_predictions.csv'))
    assert len(predictions)==6150 and predictions['audio_filename'].is_unique
    assert abs(balanced_accuracy_score(predictions.actual_negative,predictions.predicted_negative)-baseline[name]['oof_balanced_accuracy'])<1e-12
    assert abs(f1_score(predictions.actual_negative,predictions.predicted_negative,average='macro')-baseline[name]['oof_macro_f1'])<1e-12
checks['baseline_scores_recomputed_from_heldout_predictions']=3
result={'status':'passed','checks':checks,'not_verified':['Model inference on edge hardware','Continuous farm-audio call detection','Real welfare anomaly labels','Synchronized audio/video fusion accuracy','Individual-pig-independent generalization']}
(ROOT/'reports'/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
