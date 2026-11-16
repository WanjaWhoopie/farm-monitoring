"""Reproducible, local preprocessing of the bounded public dataset probe.

Stages: metadata -> fetch selected audio -> audio; vision independently.
Raw files are never modified. These are exploration outputs, not a trained system.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
SEED = 42
ACOUSTIC_FEATURES = ["dur", "amvar", "amrate", "amextent", "q25", "q50", "q75", "fpeak", "harmonicity", "wienentrmean"]


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")


def write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def metadata():
    from sklearn.model_selection import GroupKFold

    frame = pd.read_excel(RAW / "soundwel" / "SoundwelDatasetKey.xlsx")
    frame.columns = [re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") for c in frame.columns]
    for c in frame.select_dtypes(include=["object", "string"]).columns:
        frame[c] = frame[c].map(lambda x: x.strip() if isinstance(x, str) else x)
    if frame["audio_filename"].duplicated().any():
        raise ValueError("Ambiguous audio metadata: duplicate filename")
    index = json.loads((RAW / "soundwel" / "archive_index.json").read_text())
    wav_names = [PurePosixPath(e["name"]).name for e in index if e["name"].lower().endswith(".wav")]
    if len(set(wav_names)) != len(wav_names):
        raise ValueError("Duplicate WAV basenames in archive")
    frame["archive_present"] = frame["audio_filename"].isin(wav_names)
    frame["target_negative"] = frame["valence"].map({"Neg": 1, "Pos": 0})
    if frame["target_negative"].isna().any():
        raise ValueError("Unexpected or missing valence")
    dates = pd.to_datetime(frame["recording_date"], errors="raise")
    frame["recording_date"] = dates.dt.strftime("%Y-%m-%d")
    # Conservative proxy only. Missing dates stay in one bucket per team.
    # Jan 1 dates may encode only a year. Do not invent pig/session identities.
    frame["split_group_proxy"] = frame["recording_team"] + "|" + frame["recording_date"].fillna("DATE_UNKNOWN")
    frame["candidate_group_fold"] = -1
    splitter = GroupKFold(n_splits=5)
    for fold, (_, test) in enumerate(splitter.split(frame, groups=frame["split_group_proxy"])):
        frame.loc[test, "candidate_group_fold"] = fold
    selected = []
    for context, subset in frame[frame["archive_present"]].groupby("context_category", sort=True):
        subset = subset.sort_values("audio_filename")
        selected.extend(subset.sample(n=min(4, len(subset)), random_state=SEED)["audio_filename"].tolist())
    save_json(OUT / "selected_audio_names.json", sorted(selected))
    frame["selected_for_probe"] = frame["audio_filename"].isin(selected)
    write_csv(frame, OUT / "soundwel_metadata.csv")
    context_counts = frame.groupby(["context_category", "valence"]).size().rename("calls").reset_index()
    write_csv(context_counts, OUT / "soundwel_context_counts.csv")
    summary = {
        "metadata_rows": len(frame), "archive_wavs": len(wav_names),
        "archive_wavs_without_key": sorted(set(wav_names) - set(frame["audio_filename"])),
        "key_without_archive": frame.loc[~frame["archive_present"], "audio_filename"].tolist(),
        "valence_counts": frame["valence"].value_counts().to_dict(),
        "context_counts": frame["context_category"].value_counts().to_dict(),
        "recording_team_counts": frame["recording_team"].value_counts().to_dict(),
        "missing_per_source_column": frame.iloc[:, :31].isna().sum().to_dict(),
        "date_groups": int(frame["split_group_proxy"].nunique()),
        "candidate_folds": pd.crosstab(frame["candidate_group_fold"], frame["valence"]).to_dict(),
        "sample_selected": len(selected), "seed": SEED,
        "limitations": ["Team/date grouping is a proxy, not a verified individual-pig split.", "Context, team, filename and valence-derived fields must not be predictive inputs.", "Metadata-derived acoustic features are not measured edge inference latency.", "Contextual negative valence is not a diagnosis or a confirmed distress event."],
    }
    save_json(REPORTS / "soundwel_metadata_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def audio():
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    frame = pd.read_csv(OUT / "soundwel_metadata.csv")
    selected = frame[frame["selected_for_probe"]]
    rows = []
    features = []
    target_rate, target_samples = 16000, 32000
    # HTK mel scale, triangular unnormalized filters; computed locally to avoid
    # a just-in-time compilation dependency for this small preprocessing probe.
    fft_hz = np.fft.rfftfreq(512, d=1/target_rate)
    mel_edges = np.linspace(2595*np.log10(1+50/700), 2595*np.log10(1+8000/700), 66)
    hz_edges = 700*(10**(mel_edges/2595)-1)
    filters = np.maximum(0, np.minimum((fft_hz[None,:]-hz_edges[:-2,None])/(hz_edges[1:-1,None]-hz_edges[:-2,None]), (hz_edges[2:,None]-fft_hz[None,:])/(hz_edges[2:,None]-hz_edges[1:-1,None])))
    periodic_hann = np.hanning(513)[:-1]
    audio_out = OUT / "audio_16khz"
    audio_out.mkdir(parents=True, exist_ok=True)
    for record in selected.to_dict("records"):
        name = record["audio_filename"]
        path = RAW / "soundwel" / "audio" / name
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        if not samples.size or not np.isfinite(samples).all():
            raise ValueError(f"Invalid decoded audio: {name}")
        duration = len(samples) / rate
        mono = samples.mean(axis=1)
        original_peak = float(np.max(np.abs(mono)))
        clipped = int(np.count_nonzero(np.abs(samples) >= 0.999))
        spectrum = np.abs(np.fft.rfft(mono - mono.mean()))**2
        frequencies = np.fft.rfftfreq(len(mono), d=1/rate)
        above_8khz = float(spectrum[frequencies > 8000].sum() / max(float(spectrum.sum()),1e-20))
        divisor = gcd(rate, target_rate)
        waveform = resample_poly(mono, target_rate // divisor, rate // divisor).astype(np.float32)
        # Preserve amplitude; no peak normalization or denoising is applied.
        sf.write(audio_out / name, waveform, target_rate, subtype="FLOAT")
        windows = []
        for start in range(0, len(waveform), target_samples):
            chunk = waveform[start:start + target_samples]
            valid = len(chunk)
            padded = np.pad(chunk, (0, target_samples-valid))
            windows_fft = np.lib.stride_tricks.sliding_window_view(padded,512)[::160]
            power = np.abs(np.fft.rfft(windows_fft*periodic_hann,axis=1))**2
            logmel = (10*np.log10(np.maximum(filters @ power.T,1e-10))).astype(np.float32)
            if not np.isfinite(logmel).all():
                raise ValueError("Nonfinite log-mel features")
            windows.append(logmel)
            features.append({"file": name, "window": len(windows)-1, "start_sample": start, "valid_samples": valid, "label": record["valence"]})
        np.save(OUT / "audio_16khz" / (Path(name).stem + ".logmel.npy"), np.stack(windows))
        rows.append({"audio_filename": name, "context_category": record["context_category"], "valence": record["valence"], "source_rate_hz": rate, "channels": samples.shape[1], "duration_s": duration, "source_peak": original_peak, "clipped_sample_values": clipped, "energy_fraction_above_8khz": above_8khz, "resampled_samples": len(waveform), "windows": len(windows), "sha256_raw": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = pd.DataFrame(rows)
    write_csv(manifest, OUT / "audio_probe_manifest.csv")
    write_csv(pd.DataFrame(features), OUT / "audio_window_manifest.csv")
    summary = {
        "decoded_files": len(rows), "contexts": int(manifest["context_category"].nunique()),
        "source_rate_counts": manifest["source_rate_hz"].value_counts().to_dict(),
        "channels_counts": manifest["channels"].value_counts().to_dict(),
        "duration_s": {"min": float(manifest["duration_s"].min()), "median": float(manifest["duration_s"].median()), "max": float(manifest["duration_s"].max()), "sum": float(manifest["duration_s"].sum())},
        "silent_files": int((manifest["source_peak"] == 0).sum()),
        "files_with_peak_clipping": int((manifest["clipped_sample_values"] > 0).sum()),
        "energy_above_8khz_fraction": {"median": float(manifest["energy_fraction_above_8khz"].median()), "max": float(manifest["energy_fraction_above_8khz"].max()), "files_above_5pct": int((manifest["energy_fraction_above_8khz"]>0.05).sum())},
        "target_rate_hz": target_rate, "window_s": 2, "feature_windows": len(features),
        "mel_bins": 64, "n_fft": 512, "hop_length": 160, "amplitude_normalization": False,
        "mel_scale": "HTK", "window": "periodic Hann", "center": False, "log_power_floor": 1e-10,
        "warning": "Sample has four calls per context, not the natural class distribution. All windows from a call must inherit its group/fold. Resampling to 16kHz is a deployment candidate; compare original bandwidth before finalizing because high-frequency calls may lose information.",
    }
    save_json(REPORTS / "audio_probe_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def baseline():
    from sklearn.base import clone
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    source = pd.read_csv(OUT / "soundwel_metadata.csv")
    usable = source[ACOUSTIC_FEATURES].notna().any(axis=1)
    frame = source.loc[usable].reset_index(drop=True)
    x, y = frame[ACOUSTIC_FEATURES], frame["target_negative"]
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))
    results = {}
    designs = {
        "random_call_optimistic": (StratifiedKFold(5, shuffle=True, random_state=SEED), None),
        "team_date_group_proxy": (GroupKFold(5), frame["split_group_proxy"]),
        "leave_recording_team_out": (LeaveOneGroupOut(), frame["recording_team"]),
    }
    for name, (splitter, groups) in designs.items():
        pred = np.full(len(frame), -1)
        folds = []
        for fold, (train, test) in enumerate(splitter.split(x, y, groups)):
            fitted = clone(model).fit(x.iloc[train], y.iloc[train])
            pred[test] = fitted.predict(x.iloc[test])
            if groups is not None:
                assert not set(groups.iloc[train]) & set(groups.iloc[test])
            folds.append({"fold": fold, "train": len(train), "test": len(test), "test_groups": sorted(set(groups.iloc[test])) if groups is not None else [], "test_positive_valence": int((y.iloc[test] == 0).sum()), "test_negative_valence": int((y.iloc[test] == 1).sum())})
        assert np.all(pred >= 0)
        write_csv(pd.DataFrame({"audio_filename": frame["audio_filename"], "actual_negative": y, "predicted_negative": pred}), OUT / ("baseline_" + name + "_predictions.csv"))
        results[name] = {"oof_balanced_accuracy": balanced_accuracy_score(y, pred), "oof_macro_f1": f1_score(y, pred, average="macro"), "confusion_matrix_rows_true_posvalence_negvalence": confusion_matrix(y, pred, labels=[0,1]).tolist(), "folds": folds}
    results["scope"] = {"rows": len(frame), "excluded_all_features_missing": int((~usable).sum()), "excluded_context_counts": source.loc[~usable,"context_category"].value_counts().to_dict(), "features": ACOUSTIC_FEATURES, "seed": SEED, "note": "Exploratory baseline on provided acoustic measurements, no hyperparameter tuning. This does not validate continuous-call detection, distress detection, CNN inference or independent pigs."}
    save_json(REPORTS / "acoustic_baseline.json", results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["metadata", "audio", "baseline"])
    args = parser.parse_args()
    {"metadata": metadata, "audio": audio, "baseline": baseline}[args.stage]()
