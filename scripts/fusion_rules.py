"""Fusion rules on the event record, with injected anomalies and an alert-threshold sweep.

No public dataset has video and audio of the same pigs, so fusion cannot be learned and cannot be
tested on real paired recordings. This script does the honest alternative:

1. Real windows. The six-day detector's tracks and the geometry model's predicted pen budgets
   give 11 real five-minute event records (what the device would compute). They are daytime only.
2. A daily rhythm. Simulated days of 288 five-minute windows, built by sampling the real windows
   as templates and bending them to a smooth day/night rest curve, with window-to-window noise.
   Audio fields come from a call-rate curve and a valence detector with the error rate of Paul's
   final model, read from audio_results/results_final_seeds.csv: "calibrated" uses the balanced
   accuracy under grouped cross-validation, "new farm" the leave-one-team-out value.
3. A baseline band per field and hour of day (median and robust spread) from seven normal days,
   the "normal rhythm" a deployed pen would learn in its first week.
4. Six rules on the z-scored record, each with persistence (fires when the condition holds in
   2 of the last 3 windows) and a quality gate (nothing but the count rule fires when fewer
   than half the pigs are seen). Three severity levels sit on top.
5. Anomalies injected into test days, sudden and gradual, and a sweep of the threshold z:
   detection rate, lag and false-alarm episodes per normal day for every rule.

Everything in step 2 is an assumption and is written to run_info.json. The point of the exercise
is the shape of the trade-off between catching anomalies and false alarms, and which rules need
which modality, not the absolute numbers.

Usage: python scripts/fusion_rules.py [--out DIR] [--quick]
Outputs: real_windows.csv, rules_on_real_windows.csv, sweep_false_alarms.csv, sweep_detection.csv,
         operating_points.csv, severity_per_day.csv, example_day.csv, fusion_sweep_figure.png,
         run_info.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from activity_budget import CLASSES7, PALETTE, ROOT, SEED  # noqa: E402

TRACKS = ROOT / "reports" / "detector_2026-11-23"
BUDGETS_JSON = TRACKS / "detector_eval_budgets.json"
AUDIO_SEEDS = ROOT / "audio_results" / "results_final_seeds.csv"
FULL_RATE_HZ = 6.04
WIN_PER_DAY = 288
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#5a5955", "#e1e0d9"

REST = ["sleep", "lying"]
ACTIVE = ["investigating", "eat", "walk", "drink"]
RULES = ["movement", "rest", "call", "combined", "count", "feeding"]
Z_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
PERSIST_K, PERSIST_N = 2, 3
MIN_CALLS = 5          # the call rule needs at least this many calls in the window
GATE_PIGS = 4.0        # below this many pigs seen, only the count rule may fire
SUSTAINED = 6          # windows (30 min) of one vision rule before it becomes an alert
FEED_HOURS = [7, 16]   # assumed feeding periods, one hour each
ANOMALIES = ["lethargy_sudden", "lethargy_gradual", "distress_sudden", "distress_gradual",
             "sick_pen_sudden", "sick_pen_gradual", "missing_pig", "camera_fault", "feeder_failure"]
TARGETS = {"movement": ["lethargy_sudden", "lethargy_gradual", "sick_pen_sudden", "sick_pen_gradual"],
           "rest": ["lethargy_sudden", "lethargy_gradual", "sick_pen_sudden", "sick_pen_gradual"],
           "call": ["distress_sudden", "distress_gradual", "sick_pen_sudden", "sick_pen_gradual"],
           "combined": ["sick_pen_sudden", "sick_pen_gradual"],
           "count": ["missing_pig", "camera_fault"],
           "feeding": ["feeder_failure"]}
ASSUMPTIONS = {
    "rest_share_by_hour": "0.65 + 0.20 cos(2 pi (h - 2) / 24): 0.85 at 02:00, 0.45 at 14:00",
    "calls_per_window_by_hour": "12 + 8 cos(2 pi (h - 14) / 24): 20 at 14:00, 4 at 02:00, Poisson",
    "true_negative_call_share": 0.15,
    "feeding": "eat share rises to about 0.30 (lognormal noise 0.2) during the hours in FEED_HOURS, other classes shrink in proportion",
    "window_noise": "rest share x lognormal(0, 0.15), pigs seen + N(0, 0.2)",
    "lethargy": "active share x 0.5 (moved to rest), motion x 0.5; gradual = linear ramp over 3 h",
    "distress": "true negative share 0.15 -> 0.60, call rate x 1.5; gradual = ramp over 3 h",
    "sick_pen": "lethargy and distress together",
    "missing_pig": "pigs seen - 1", "camera_fault": "pigs seen 0, shares replaced by noise",
    "feeder_failure": "no feeding peak after onset (eat share stays at its non-feeding level)",
    "onset": "uniform between 06:00 and 18:00, lasts to the end of the day",
    "baseline": "median and 1.4826 x MAD per field and hour of day over 7 normal days, spread floors 0.02 (shares), 0.03 (negative share), 0.25 (pigs seen), 0.01 (eat)",
    "persistence": f"{PERSIST_K} of the last {PERSIST_N} windows", "min_calls_for_call_rule": MIN_CALLS,
    "quality_gate_pigs_seen": GATE_PIGS, "sustained_windows_for_alert": SUSTAINED,
}


# ----------------------------------------------------------------------------- real windows
def real_windows() -> pd.DataFrame:
    budgets = json.loads(BUDGETS_JSON.read_text())
    rows = []
    for clip, b in budgets.items():
        tf = TRACKS / f"track_{clip.replace('/', '_')}.csv"
        if not tf.exists():      # the mislabelled clip's track sits in excluded_mislabelled_clip/
            continue
        tr = pd.read_csv(tf).sort_values(["track_id", "frame"])
        cx, cy = tr["x"] + tr["width"] / 2, tr["y"] + tr["height"] / 2
        same = tr["track_id"].eq(tr["track_id"].shift()) & tr["frame"].eq(tr["frame"].shift() + 1)
        motion = float((np.hypot(cx.diff(), cy.diff())[same] * FULL_RATE_HZ).mean())
        r = {"clip_id": clip, "hour": 12, "pigs_seen": round(len(tr) / tr["frame"].nunique(), 2), "motion_px_s": round(motion, 1)}
        r.update({c: b["predicted_track"][c] for c in CLASSES7})
        rows.append(r)
    return pd.DataFrame(rows)


def audio_quality() -> dict[str, float]:
    """Balanced accuracy of the final audio model: grouped CV (calibrated) and leave-team-out (new farm)."""
    s = pd.read_csv(AUDIO_SEEDS)
    s = s[s["model"].str.startswith("EfficientNet")]
    return {"calibrated": round(float(s[s["scheme"] == "grouped CV"]["bal_acc"].mean()), 3),
            "new farm": round(float(s[s["scheme"] == "leave-one-team-out"]["bal_acc"].mean()), 3)}


# ----------------------------------------------------------------------------- simulation
def rest_curve(h: np.ndarray) -> np.ndarray:
    return 0.65 + 0.20 * np.cos(2 * np.pi * (h - 2) / 24)


def call_curve(h: np.ndarray) -> np.ndarray:
    return 12 + 8 * np.cos(2 * np.pi * (h - 14) / 24)


def simulate_day(rng: np.random.Generator, templates: pd.DataFrame, bal_acc: float, anomaly: str | None = None) -> pd.DataFrame:
    """One day of 288 windows. Returns the event records plus hidden truth columns (anomaly, onset)."""
    n = WIN_PER_DAY
    hour = np.arange(n) * 5 / 60
    t = templates.iloc[rng.integers(0, len(templates), n)].reset_index(drop=True)
    rest_t = t[REST].sum(1).to_numpy(); act_t = t[ACTIVE].sum(1).to_numpy(); stand_t = t["standing"].to_numpy()
    rest = np.clip(rest_curve(hour) * rng.lognormal(0, 0.15, n), 0.05, 0.97)
    other = 1 - rest
    stand = other * stand_t / np.maximum(stand_t + act_t, 1e-9)
    active = other - stand
    motion = t["motion_px_s"].to_numpy() * active / np.maximum(act_t, 1e-3)
    pigs = np.clip(t["pigs_seen"].to_numpy() + rng.normal(0, 0.2, n), 0, 8)
    calls_rate = call_curve(hour)
    p_neg = np.full(n, ASSUMPTIONS["true_negative_call_share"])
    onset = -1
    f_act = np.ones(n)
    feed_scale = np.ones(n)
    if anomaly:
        onset = int(rng.integers(6 * 12, 18 * 12))
        ramp = np.clip((np.arange(n) - onset) / 36, 0, 1)        # 3 h linear ramp
        step = (np.arange(n) >= onset).astype(float)
        prof = ramp if anomaly.endswith("gradual") else step
        if anomaly.startswith(("lethargy", "sick_pen")):
            f_act = 1 - 0.5 * prof
        if anomaly.startswith(("distress", "sick_pen")):
            p_neg = 0.15 + (0.60 - 0.15) * prof
            calls_rate = calls_rate * (1 + 0.5 * prof)
        if anomaly == "missing_pig":
            pigs = np.clip(pigs - step, 0, 8)
        if anomaly == "camera_fault":
            pigs = pigs * (1 - step)
        if anomaly == "feeder_failure":
            feed_scale = 1 - step
    active = active * f_act
    rest = 1 - active - stand
    motion = motion * f_act
    # split rest and active into the seven classes in the template's proportions
    df = pd.DataFrame({"hour": hour, "pigs_seen": pigs, "motion_px_s": motion})
    for grp, total, tot_t in ((REST, rest, rest_t), (ACTIVE, active, act_t)):
        for c in grp:
            share_t = np.where(tot_t > 1e-9, t[c].to_numpy() / np.maximum(tot_t, 1e-9), 1 / len(grp))
            df[c] = total * share_t
    df["standing"] = stand
    # feeding peaks: in feeding hours the eat share rises to about 0.30 and the other six classes
    # shrink in proportion. A feeder failure is that peak not appearing (feed_scale 0 after onset).
    in_feed = np.isin(np.floor(hour).astype(int), FEED_HOURS)
    peak = np.clip(0.30 * rng.lognormal(0, 0.2, n), 0.1, 0.6) * feed_scale
    new_eat = np.where(in_feed, np.maximum(df["eat"].to_numpy(), peak), df["eat"].to_numpy())
    scale = (1 - new_eat) / np.maximum(1 - df["eat"].to_numpy(), 1e-9)
    for c in CLASSES7:
        if c != "eat":
            df[c] = df[c] * scale
    df["eat"] = new_eat
    if anomaly == "camera_fault":
        noise = rng.dirichlet(np.ones(7), n)
        for i, c in enumerate(CLASSES7):
            df[c] = np.where(pigs == 0, noise[:, i], df[c])
        df["motion_px_s"] = np.where(pigs == 0, 0.0, df["motion_px_s"])
    # audio: calls and the valence detector with the model's balanced accuracy as sensitivity and specificity
    calls = rng.poisson(calls_rate)
    true_neg = rng.binomial(calls, p_neg)
    seen_neg = rng.binomial(true_neg, bal_acc) + rng.binomial(calls - true_neg, 1 - bal_acc)
    df["calls"] = calls
    df["neg_valence"] = np.where(calls > 0, seen_neg / np.maximum(calls, 1), np.nan)
    df["anomaly"] = anomaly or "normal"; df["onset"] = onset
    return df


FIELDS = ["active", "rest", "neg_valence", "pigs_seen", "eat"]


def add_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["active"] = df[ACTIVE].sum(1); df["rest"] = df[REST].sum(1)
    df["hod"] = np.floor(df["hour"]).astype(int) % 24
    return df


def fit_baseline(days: list[pd.DataFrame]) -> pd.DataFrame:
    allw = pd.concat([add_fields(d) for d in days], ignore_index=True)
    floors = {"active": 0.02, "rest": 0.02, "neg_valence": 0.03, "pigs_seen": 0.25, "eat": 0.01}
    rows = []
    for hod, g in allw.groupby("hod"):
        r = {"hod": hod}
        for f in FIELDS:
            x = g[f].dropna().to_numpy()
            med = float(np.median(x)); mad = float(np.median(np.abs(x - med))) * 1.4826
            r[f + "_med"] = med; r[f + "_spread"] = max(mad, floors[f])
        rows.append(r)
    return pd.DataFrame(rows).set_index("hod")


def zscores(df: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    df = add_fields(df)
    b = base.loc[df["hod"].to_numpy()]
    for f in FIELDS:
        df["z_" + f] = (df[f].to_numpy() - b[f + "_med"].to_numpy()) / b[f + "_spread"].to_numpy()
    return df


def persist(raw: np.ndarray, k: int = PERSIST_K, n: int = PERSIST_N) -> np.ndarray:
    s = pd.Series(raw.astype(int)).rolling(n, min_periods=1).sum().to_numpy()
    return s >= k


def apply_rules(df: pd.DataFrame, z: float) -> pd.DataFrame:
    """Raw conditions, the quality gate, persistence, and severity. Returns the frame with one bool column per rule."""
    gated = (df["pigs_seen"] < GATE_PIGS).to_numpy()
    in_feed = np.isin(df["hod"].to_numpy(), FEED_HOURS)
    raw = {"movement": (df["z_active"] < -z).to_numpy() & ~gated,
           "rest": (df["z_rest"] > z).to_numpy() & ~gated,
           "call": ((df["z_neg_valence"] > z) & (df["calls"] >= MIN_CALLS)).to_numpy() & ~gated,
           "count": (df["z_pigs_seen"] < -z).to_numpy(),
           "feeding": ((df["z_eat"] < -z) & in_feed).to_numpy() & ~gated}
    raw["combined"] = raw["movement"] & raw["call"]
    out = df.copy()
    for r in RULES:
        out[r] = persist(raw[r])
    vision_any = out["movement"] | out["rest"] | out["feeding"]
    run = pd.Series(vision_any.astype(int)).groupby((~vision_any).cumsum()).cumsum().to_numpy()
    sev = np.zeros(len(out), int)
    sev[(out["movement"] | out["rest"] | out["feeding"] | out["call"]).to_numpy()] = 1
    sev[(out["combined"] | (run >= SUSTAINED)).to_numpy()] = 2
    sev[out["count"].to_numpy()] = 3
    out["severity"] = sev
    return out


def episodes(fired: np.ndarray) -> int:
    f = fired.astype(int)
    return int(((f[1:] == 1) & (f[:-1] == 0)).sum() + (f[0] == 1))


# ----------------------------------------------------------------------------- figure
def draw_figure(det: pd.DataFrame, fa: pd.DataFrame, ops: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colour = dict(zip(RULES, PALETTE[:6]))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.25, 1]})
    for ax in axes:
        ax.set_facecolor(SURFACE); ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors=INK2, labelsize=9)
    a = axes[0]
    for rule in RULES:
        for audio, ls in (("calibrated", "-"), ("new farm", "--")):
            if audio == "new farm" and rule not in ("call", "combined"):
                continue
            d = det[(det["rule"] == rule) & (det["audio"] == audio) & det["anomaly"].isin(TARGETS[rule])]
            d = d.groupby("z", as_index=False)["detection_rate"].mean()
            f = fa[(fa["rule"] == rule) & (fa["audio"] == audio)].set_index("z")["false_alarms_per_day"]
            x = f.loc[d["z"]].to_numpy()
            a.plot(x, d["detection_rate"], ls, color=colour[rule], linewidth=2, marker="o", markersize=4,
                   label=rule + (" (new farm audio)" if audio == "new farm" else ""))
    a.set_xscale("symlog", linthresh=0.1); a.set_xlim(-0.01, 60); a.set_ylim(0, 1.05)
    a.axvline(1, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    a.text(1.15, 0.50, "1 false alarm\nper day", fontsize=8, color=INK2, va="center")
    a.set_xlabel("false-alarm episodes per normal day (threshold z from 5 down to 1)", color=INK2, fontsize=9)
    a.set_ylabel("detection rate on the anomalies the rule targets", color=INK2, fontsize=9)
    a.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=2)
    a.set_title("Catching injected anomalies against false alarms", loc="left", fontsize=9.5, color=INK)
    b = axes[1]
    o = ops[ops["audio"] == "calibrated"].set_index("rule").loc[RULES]
    o2 = ops[ops["audio"] == "new farm"].set_index("rule").loc[RULES]
    xs = np.arange(len(RULES))
    b.bar(xs - 0.18, o["median_lag_min"], width=0.36, color=[colour[r] for r in RULES], label="calibrated audio")
    b.bar(xs + 0.18, o2["median_lag_min"], width=0.36, color=[colour[r] for r in RULES], alpha=0.45, label="new-farm audio")
    for x, (r, row) in zip(xs, o.iterrows()):
        b.text(x - 0.18, row["median_lag_min"] + 2, f"{row['detection_rate']:.2f}", ha="center", fontsize=7.5, color=INK)
    for x, (r, row) in zip(xs, o2.iterrows()):
        b.text(x + 0.18, row["median_lag_min"] + 2, f"{row['detection_rate']:.2f}", ha="center", fontsize=7.5, color=INK2)
    b.set_xticks(xs); b.set_xticklabels(RULES, fontsize=8.5)
    b.set_ylabel("median lag to first alert, minutes", color=INK2, fontsize=9)
    b.legend(frameon=False, fontsize=8, loc="upper left")
    b.set_title("At the threshold giving at most 1 false alarm per day\n(number above each bar = detection rate)", loc="left", fontsize=9.5, color=INK)
    fig.suptitle("Fusion rules on the event record: injected anomalies, threshold sweep", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--train-days", type=int, default=7)
    ap.add_argument("--test-days", type=int, default=60, help="normal days for the false-alarm rate")
    ap.add_argument("--anomaly-days", type=int, default=40, help="days per anomaly type")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.test_days, args.anomaly_days = 10, 8
    out = args.out or ROOT / "reports" / f"fusion_rules_{datetime.date.today().isoformat()}"
    out.mkdir(parents=True, exist_ok=True)

    templates = real_windows()
    templates.to_csv(out / "real_windows.csv", index=False)
    quality = audio_quality()
    print("audio balanced accuracy used:", quality)
    fa_rows, det_rows, sev_rows = [], [], []
    examples = {}
    base_by_audio = {}
    for audio, bal_acc in quality.items():
        rng = np.random.default_rng(SEED)
        base = fit_baseline([simulate_day(rng, templates, bal_acc) for _ in range(args.train_days)])
        base_by_audio[audio] = base
        normal = [zscores(simulate_day(rng, templates, bal_acc), base) for _ in range(args.test_days)]
        anom = {a: [zscores(simulate_day(rng, templates, bal_acc, a), base) for _ in range(args.anomaly_days)] for a in ANOMALIES}
        for z in Z_GRID:
            fired = {r: [] for r in RULES}; sev = []
            for d in normal:
                o = apply_rules(d, z)
                for r in RULES:
                    fired[r].append(episodes(o[r].to_numpy()))
                sev.append(np.bincount(o["severity"], minlength=4))
            for r in RULES:
                fa_rows.append({"audio": audio, "rule": r, "z": z, "false_alarms_per_day": round(float(np.mean(fired[r])), 3)})
            sev_rows.append({"audio": audio, "z": z, "day": "normal", **{f"windows_level_{i}": round(float(v), 1) for i, v in enumerate(np.mean(sev, 0))}})
            for a, days in anom.items():
                hits = {r: [] for r in RULES}; lags = {r: [] for r in RULES}; sev_a = []
                for d in days:
                    o = apply_rules(d, z)
                    onset = int(d["onset"].iloc[0])
                    sev_a.append(np.bincount(o["severity"].to_numpy()[onset:], minlength=4))
                    for r in RULES:
                        f = o[r].to_numpy()[onset:]
                        hit = bool(f.any()); hits[r].append(hit)
                        if hit:
                            lags[r].append(5 * int(np.argmax(f)))
                    if audio == "calibrated" and z == 3.0 and a == "sick_pen_gradual" and a not in examples:
                        examples[a] = o
                for r in RULES:
                    det_rows.append({"audio": audio, "rule": r, "z": z, "anomaly": a, "detection_rate": round(float(np.mean(hits[r])), 3),
                                     "median_lag_min": float(np.median(lags[r])) if lags[r] else np.nan, "n_days": len(days)})
                sev_rows.append({"audio": audio, "z": z, "day": a, **{f"windows_level_{i}": round(float(v), 1) for i, v in enumerate(np.mean(sev_a, 0))}})
            print(f"{audio}: z {z} done", flush=True)
    fa = pd.DataFrame(fa_rows); det = pd.DataFrame(det_rows); sev_tab = pd.DataFrame(sev_rows)
    fa.to_csv(out / "sweep_false_alarms.csv", index=False)
    det.to_csv(out / "sweep_detection.csv", index=False)
    sev_tab.to_csv(out / "severity_per_day.csv", index=False)

    # operating point per rule: the smallest z with at most one false alarm per day
    ops = []
    for audio in quality:
        for r in RULES:
            f = fa[(fa["audio"] == audio) & (fa["rule"] == r)].sort_values("z")
            ok = f[f["false_alarms_per_day"] <= 1.0]
            z = float(ok["z"].iloc[0]) if len(ok) else float(f["z"].iloc[-1])
            d = det[(det["audio"] == audio) & (det["rule"] == r) & (det["z"] == z) & det["anomaly"].isin(TARGETS[r])]
            ops.append({"audio": audio, "rule": r, "z": z, "false_alarms_per_day": float(f[f["z"] == z]["false_alarms_per_day"].iloc[0]),
                        "detection_rate": round(float(d["detection_rate"].mean()), 3),
                        "median_lag_min": float(np.nanmedian(d["median_lag_min"])) if d["median_lag_min"].notna().any() else np.nan,
                        "targets": ", ".join(TARGETS[r])})
    ops = pd.DataFrame(ops)
    ops.to_csv(out / "operating_points.csv", index=False)
    print("\nOperating points (at most 1 false alarm per day):\n" + ops.drop(columns="targets").to_string(index=False))

    # the rules on the real windows, against the calibrated baseline at the operating z of each rule
    real = templates.copy(); real["calls"] = np.nan; real["neg_valence"] = np.nan
    rz = zscores(real, base_by_audio["calibrated"])
    rr = {"clip_id": real["clip_id"]}
    for r in RULES:
        z = float(ops[(ops["audio"] == "calibrated") & (ops["rule"] == r)]["z"].iloc[0])
        rr[r] = apply_rules(rz, z)[r].to_numpy()
    rr = pd.DataFrame(rr); rr.to_csv(out / "rules_on_real_windows.csv", index=False)
    print(f"\nrules firing on the {len(rr)} real windows: {int(rr[RULES].to_numpy().sum())} of {len(rr) * len(RULES)} rule-windows")

    if examples:
        ex = next(iter(examples.values()))
        keep = ["hour", "pigs_seen", "motion_px_s", *CLASSES7, "calls", "neg_valence", "onset", "z_active", "z_neg_valence", *RULES, "severity"]
        ex[keep].round(3).to_csv(out / "example_day.csv", index=False)
    draw_figure(det, fa, ops, out / "fusion_sweep_figure.png")
    info = {"saved_at": datetime.datetime.now().isoformat(timespec="seconds"), "seed": SEED, "z_grid": Z_GRID,
            "train_days": args.train_days, "test_days": args.test_days, "anomaly_days_per_type": args.anomaly_days,
            "audio_balanced_accuracy": quality, "assumptions": ASSUMPTIONS, "rules": RULES, "targets": TARGETS,
            "inputs": {"tracks": str(TRACKS.relative_to(ROOT)), "budgets": str(BUDGETS_JSON.relative_to(ROOT)), "audio_seeds": str(AUDIO_SEEDS.relative_to(ROOT))},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("\nSaved to", out)


if __name__ == "__main__":
    main()
