#!/usr/bin/env python3
"""
n2_far_matched_evaluator.py  (publish-lock v1)

Purpose (locked):
- Compare PSI4 headline (circR) against standard baselines at matched FAR on a reference set.
- Output a small, auditable evaluation: threshold, achieved FAR, first alarm index, lead time.

Works with OLD PSI4 window outputs:
  <psi4_root>/w_00000/n2_primary.json
and (optionally) raw CSV (std) to compute baselines with the same (win, step).

Baselines implemented (domain-generic, no ML):
- RMS on sig1/sig2 (high_bad)
- Excess kurtosis on sig1/sig2 (high_bad)
- |corr(sig1,sig2)| (high_bad)

PSI4 metric:
- circR (low_bad)

Upgrades (locked):
A) Post-failure alarm rates (fraction of windows alarming pre/post failure index)
B) Explicit direction mapping in contract.json
C) Explicit evaluation_mode: separation vs early_warning (auto inference + override)
D) Stability across seeds (bootstrap REF windows): *_stability_runs.csv + *_stability_summary.csv
E) Paired framing (stats-lite): median(psi4_first - baseline_first), psi4_earlier_fraction

Publish-safety additions:
F) baselines_enabled + baselines_disabled_reason (NO silent downgrade)
G) report.html (human readable, reviewer-friendly)
H) Explicit psi4-only mode is allowed ONLY when user asks (--psi4-only).
   If rawcsv is provided but baselines fail -> INVALID_EVAL (hard fail).

Design rule:
- This script is the ONLY place where PSI4 is compared to baselines.
- Toolkits output evidence packs; evaluation outputs comparative evaluation.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import json
import math
import os
import shutil
import glob
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Planned outputs (publish contract)
# -----------------------------
PLANNED_OUTPUTS = [
    "{tag}_per_window.csv",
    "{tag}_evaluation.csv",
    "{tag}_contract.json",
    "{tag}_summary.json",
    "{tag}_provenance.json",
    "{tag}_report.html",
    "index.html",
]

def planned_outputs_for_mode(output_mode: str, tag: str) -> list[str]:
    # Minimal-risk lock: evaluator pack is already compact.
    # Keep all modes identical for now to avoid unnecessary behavior drift.
    return [x.replace("{tag}", tag) for x in PLANNED_OUTPUTS]





def _selftest() -> int:
    """
    Tier-2 selftest:
      - create temp outdir
      - generate tiny synthetic psi4 windows (+ tiny rawcsv baselines)
      - run full pipeline via subprocess
      - assert required artifacts exist
      - scan HTML for nan/inf/infinity
      - delete temp dir (leak-guard)
    """
    print("[SELFTEST] n2_far_matched_evaluator")

    import sys, json, shutil, tempfile, subprocess, math, random
    from pathlib import Path

    tmp_root = Path(tempfile.mkdtemp(prefix="n2_evaluation_selftest_"))
    try:
        ref_root = tmp_root / "ref_psi4"
        test_root = tmp_root / "test_psi4"
        outdir = tmp_root / "out"
        ref_root.mkdir(parents=True, exist_ok=True)
        test_root.mkdir(parents=True, exist_ok=True)
        outdir.mkdir(parents=True, exist_ok=True)

        # Create synthetic psi4 windows with headline.circR
        def make_windows(root: Path, n: int, seed: int, trend: float):
            rng = random.Random(seed)
            for i in range(n):
                wdir = root / f"w_{i:05d}"
                wdir.mkdir(parents=True, exist_ok=True)
                circR = 1.0 + trend * (i / max(1, n-1)) + 0.02*(rng.random()-0.5)
                obj = {"headline": {"circR": float(circR)}}
                (wdir / "n2_primary.json").write_text(json.dumps(obj), encoding="utf-8")

        # Reference: stable-ish, Test: slightly worse drift (so alarms can happen depending on FAR)
        make_windows(ref_root, n=30, seed=0, trend=0.00)
        make_windows(test_root, n=30, seed=1, trend=0.08)

        # Synthetic rawcsv so baselines are enabled (avoid psi4-only path)
        # Provide enough rows to produce >=10 windows with win=32 step=16
        rng = random.Random(2)
        n_rows = 600
        sig1 = [math.sin(2*math.pi*i/50.0) + 0.05*(rng.random()-0.5) for i in range(n_rows)]
        sig2 = [math.cos(2*math.pi*i/60.0) + 0.05*(rng.random()-0.5) for i in range(n_rows)]

        def write_raw(path: Path):
            lines = ["sig1,sig2"]
            for a,b in zip(sig1,sig2):
                lines.append(f"{a:.8g},{b:.8g}")
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        ref_raw = tmp_root / "ref.csv"
        test_raw = tmp_root / "test.csv"
        write_raw(ref_raw)
        write_raw(test_raw)

        tag = "selftest"
        script = Path(__file__).resolve()
        cmd = [
            sys.executable, "-X", "utf8", str(script),
            "--ref-psi4", str(ref_root),
            "--test-psi4", str(test_root),
            "--ref-rawcsv", str(ref_raw),
            "--test-rawcsv", str(test_raw),
            "--sig1-col", "sig1",
            "--sig2-col", "sig2",
            "--win", "32",
            "--step", "16",
            "--far", "0.2",
            "--outdir", str(outdir),
            "--tag", tag,
            "--no-plot",
            "--max-windows", "12",
            "--n-repeats", "1",
            "--quiet",
        ]

        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print("[SELFTEST_FAIL] subprocess failed")
            print("---- STDOUT ----")
            print(r.stdout)
            print("---- STDERR ----")
            print(r.stderr)
            return 2

        must = [
            f"{tag}_per_window.csv",
            f"{tag}_evaluation.csv",
            f"{tag}_contract.json",
            f"{tag}_summary.json",
            f"{tag}_provenance.json",
            f"{tag}_report.html",
            "index.html",
        ]
        missing = [fn for fn in must if not (outdir / fn).exists()]
        if missing:
            print("[SELFTEST_FAIL] missing artifacts:", ", ".join(missing))
            return 2

        # HTML gate (uses the script's own publish gate)
        _assert_html_no_nan_inf(str(outdir / f"{tag}_report.html"))

        print("[SELFTEST_OK]")
        return 0

    finally:
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass

def _assert_html_no_nan_inf(html_path: str) -> None:
    """
    Publish gate: report.html must not contain standalone nan/NaN/inf/Infinity tokens.
    Display-only safety; no math/CSV changes.
    """
    s = Path(html_path).read_text(encoding="utf-8", errors="strict")
    # Guard against numeric NaN/Inf tokens WITHOUT false positives like "infile".
    pat = re.compile(r"(?i)(?<![A-Za-z0-9_])(?:-?inf|infinity|nan)(?![A-Za-z0-9_])")
    if pat.search(s):
        raise RuntimeError(f"INVALID_REPORT_contains_nonfinite_tokens: path={html_path}")
# -----------------------------
# Utilities
# -----------------------------

def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default

def excess_kurtosis(x: np.ndarray) -> float:
    """
    Excess kurtosis (kurtosis - 3). Simple moment estimator.
    Returns NaN if too short or variance ~0.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return float("nan")
    m = float(np.mean(x))
    v = float(np.mean((x - m) ** 2))
    if v <= 0 or not math.isfinite(v):
        return float("nan")
    m4 = float(np.mean((x - m) ** 4))
    k = m4 / (v * v) - 3.0
    return float(k)

def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(x * x)))

def abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 8:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 0 or sy <= 0:
        return float("nan")
    c = float(np.corrcoef(x, y)[0, 1])
    if not math.isfinite(c):
        return float("nan")
    return float(abs(c))

def list_psi4_windows(psi4_root: str, pattern: str = "w_*") -> List[str]:
    psi4_root = os.path.abspath(psi4_root)
    wins = sorted(glob.glob(os.path.join(psi4_root, pattern)))
    return [w for w in wins if os.path.isdir(w)]

def load_n2_circR_from_primary_json(primary_path: str) -> float:
    """
    OLD outputs can store circR in different places. Try common keys.
    Modern: data['headline']['circR']
    """
    with open(primary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "headline" in data and isinstance(data["headline"], dict) and "circR" in data["headline"]:
            return safe_float(data["headline"]["circR"])

        for k in ["H2_circR", "circR", "circR_real", "H2_circR_real"]:
            if k in data:
                return safe_float(data[k])

        if "headline" in data and isinstance(data["headline"], dict):
            for k in ["H2_circR", "circR_real"]:
                if k in data["headline"]:
                    return safe_float(data["headline"][k])

    return float("nan")

def load_psi4_series(psi4_root: str, verbose: bool) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      window, window_index, psi4_circR
    """
    wins = list_psi4_windows(psi4_root)
    if len(wins) == 0:
        raise RuntimeError(f"No PSI4 windows found under: {psi4_root} (expected w_*/ subfolders)")

    rows = []
    for wdir in wins:
        wname = os.path.basename(wdir)  # w_00000
        try:
            idx = int(wname.split("_")[1]) if "_" in wname else len(rows)
        except Exception:
            idx = len(rows)

        primary = os.path.join(wdir, "n2_primary.json")
        if not os.path.exists(primary):
            continue

        circ = load_n2_circR_from_primary_json(primary)
        rows.append({"window": wname, "window_index": idx, "psi4_circR": circ})

    df = pd.DataFrame(rows).sort_values("window_index").reset_index(drop=True)
    log(f"[INFO] psi4 windows found = {len(df)} under {psi4_root}", verbose)
    return df

def slice_windows_from_raw(df: pd.DataFrame, sig1: str, sig2: str, win: int, step: int) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    x = pd.to_numeric(df[sig1], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[sig2], errors="coerce").to_numpy(dtype=float)

    n = len(df)
    out = []
    i = 0
    start = 0
    while start + win <= n:
        xs = x[start:start + win]
        ys = y[start:start + win]
        out.append((i, xs, ys))
        i += 1
        start += step
    return out

def compute_baselines_from_rawcsv(rawcsv_path: str, sig1: str, sig2: str, win: int, step: int, verbose: bool) -> pd.DataFrame:
    rawcsv_path = os.path.abspath(rawcsv_path)
    if not os.path.exists(rawcsv_path):
        raise RuntimeError(f"Raw CSV not found: {rawcsv_path}")

    df = pd.read_csv(rawcsv_path)
    if sig1 not in df.columns or sig2 not in df.columns:
        raise RuntimeError(f"Columns not found in {rawcsv_path}: need {sig1}, {sig2}")

    wins = slice_windows_from_raw(df, sig1, sig2, win, step)
    if len(wins) == 0:
        raise RuntimeError("No windows produced from raw CSV. Check --win/--step vs file length.")

    rows = []
    for idx, xs, ys in wins:
        rows.append({
            "window_index": idx,
            "rms_sig1": rms(xs),
            "rms_sig2": rms(ys),
            "kurt_sig1": excess_kurtosis(xs),
            "kurt_sig2": excess_kurtosis(ys),
            "abs_corr": abs_corr(xs, ys),
        })

    out = pd.DataFrame(rows).sort_values("window_index").reset_index(drop=True)
    log(f"[INFO] baselines windows computed = {len(out)} from {os.path.basename(rawcsv_path)}", verbose)
    return out


# -----------------------------
# Thresholding + alarms
# -----------------------------

@dataclass
class Threshold:
    metric: str
    direction: str  # "low_bad" or "high_bad"
    threshold: float
    ref_far_achieved: float

def calibrate_threshold(series: pd.Series, far: float, direction: str) -> Tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if s.size < 10:
        return float("nan"), float("nan")

    if direction == "low_bad":
        thr = float(np.quantile(s, far))
        far_ach = float(np.mean(s <= thr))
    elif direction == "high_bad":
        thr = float(np.quantile(s, 1.0 - far))
        far_ach = float(np.mean(s >= thr))
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return thr, far_ach

def alarm_mask(series: pd.Series, thr: float, direction: str) -> np.ndarray:
    s = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not math.isfinite(thr):
        return np.zeros(len(s), dtype=bool)
    if direction == "low_bad":
        return s <= thr
    return s >= thr

def first_alarm_index(series: pd.Series, thr: float, direction: str) -> Optional[int]:
    mask = alarm_mask(series, thr, direction)
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return None
    return int(idxs[0])

def alarm_rate_pre_failure(series: pd.Series, thr: float, direction: str, failure_index: Optional[int]) -> float:
    if failure_index is None or not math.isfinite(float(failure_index)):
        return float("nan")
    s = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    upto = int(max(0, min(len(s), int(failure_index))))
    if upto <= 0:
        return 0.0
    m = alarm_mask(pd.Series(s[:upto]), thr, direction)
    return float(np.mean(m)) if m.size else 0.0

def alarm_rate_post_failure(series: pd.Series, thr: float, direction: str, failure_index: Optional[int]) -> float:
    if failure_index is None or not math.isfinite(float(failure_index)):
        return float("nan")
    s = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    start = int(max(0, min(len(s), int(failure_index))))
    if start >= len(s):
        return 0.0
    m = alarm_mask(pd.Series(s[start:]), thr, direction)
    return float(np.mean(m)) if m.size else 0.0

def plot_metric(tag: str, metric: str, test_series: pd.Series, thr: float, direction: str, outpath: str) -> None:
    y = pd.to_numeric(test_series, errors="coerce").to_numpy(dtype=float)
    x = np.arange(len(y))

    plt.figure()
    plt.plot(x, y, marker="o", linewidth=1, label=f"{metric} (test)")

    if math.isfinite(thr):
        plt.axhline(thr, linestyle="--", linewidth=1, label="threshold (FAR-matched)")
        alarm = (y <= thr) if direction == "low_bad" else (y >= thr)
        alarm_idx = np.where(alarm)[0]
        if alarm_idx.size > 0:
            plt.scatter(alarm_idx, y[alarm_idx], s=18, label="alarms")

    plt.xlabel("window index")
    plt.ylabel(metric)
    plt.title(f"{tag}: {metric} @ FAR matched on reference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def infer_evaluation_mode(evaluation: pd.DataFrame, failure_index: Optional[int], explicit: str) -> str:
    """
    C) Explicit evaluation mode:
    - separation: test is already faulty (alarms at start), lead-time is not meaningful
    - early_warning: there is a pre-failure period and a later onset (lead-time meaningful)

    Heuristic for auto:
    - If failure_index is None: separation
    - Else if most metrics alarm at index 0: separation
    - Else: early_warning
    """
    if explicit in ("separation", "early_warning"):
        return explicit

    if failure_index is None:
        return "separation"

    firsts = evaluation["test_first_alarm_index"].to_numpy(dtype=float)
    firsts = firsts[firsts >= 0]  # drop -1 (no alarm)
    if firsts.size == 0:
        return "early_warning"
    frac_zero = float(np.mean(firsts == 0))
    return "separation" if frac_zero >= 0.5 else "early_warning"


# -----------------------------
# Evaluation core (single run)
# -----------------------------

def compute_evaluation_once(
    ref: pd.DataFrame,
    test: pd.DataFrame,
    metric_specs: List[Tuple[str, str]],
    far: float,
    failure_index: Optional[int],
) -> Tuple[pd.DataFrame, List[Threshold]]:
    thresholds: List[Threshold] = []
    for m, direction in metric_specs:
        thr, far_ach = calibrate_threshold(ref[m], far, direction)
        thresholds.append(Threshold(m, direction, thr, far_ach))

    rows = []
    for t in thresholds:
        first = first_alarm_index(test[t.metric], t.threshold, t.direction)

        lead = float("nan")
        if failure_index is not None and first is not None:
            lead = float(failure_index - first)

        rows.append({
            "metric": t.metric,
            "direction": t.direction,
            "threshold": t.threshold,
            "ref_far_achieved": t.ref_far_achieved,
            "test_first_alarm_index": -1 if first is None else first,
            "lead_time_windows": lead,
            "alarm_rate_pre_failure_frac": alarm_rate_pre_failure(test[t.metric], t.threshold, t.direction, failure_index),
            "alarm_rate_post_failure_frac": alarm_rate_post_failure(test[t.metric], t.threshold, t.direction, failure_index),
        })

    evaluation = pd.DataFrame(rows)
    return evaluation, thresholds


# -----------------------------
# Stability across seeds (bootstrap REF)
# -----------------------------

def parse_seed_list(seed_list: str, n_repeats: int) -> List[int]:
    if seed_list is None or str(seed_list).strip() == "":
        return list(range(n_repeats))
    toks = [t.strip() for t in str(seed_list).split(",") if t.strip() != ""]
    out = []
    for t in toks:
        out.append(int(t))
    return out

def bootstrap_ref(ref: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    n = len(ref)
    idx = rng.integers(0, n, size=n)  # sample with replacement
    return ref.iloc[idx].reset_index(drop=True)

def stability_summary(stats: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(stats, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0:
        return {"median": float("nan"), "min": float("nan"), "max": float("nan"), "q25": float("nan"), "q75": float("nan")}
    return {
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
    }


# -----------------------------
# Minimal report.html
# -----------------------------

def write_report_html(outpath: str, ctx: Dict[str, Any]) -> None:

    def esc(s: Any) -> str:
        # display-only: ensure report never shows 'nan'/'inf'
        if s is None:
            s = ""
        else:
            try:
                v = float(s)
                if not math.isfinite(v):
                    s = "N/A"
                else:
                    s = str(s)
            except Exception:
                s = str(s)
        return (s.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;")
                 .replace('"', "&quot;"))

    # basic, readable, stable
    title = f"Evaluation Report - {ctx.get('tag', '')}"
    lines = []
    lines.append("<!doctype html>")
    lines.append("<html><head><meta charset='utf-8'>")
    lines.append(f"<title>{esc(title)}</title>")
    lines.append("<style>")
    lines.append("body{font-family:Arial,Helvetica,sans-serif;max-width:1100px;margin:24px;}")
    lines.append("code,pre{background:#f6f6f6;padding:2px 4px;border-radius:4px;}")
    lines.append("table{border-collapse:collapse;width:100%;margin-top:12px;}")
    lines.append("th,td{border:1px solid #ddd;padding:8px;font-size:14px;}")
    lines.append("th{background:#fafafa;text-align:left;}")
    lines.append(".muted{color:#666;}")
    lines.append("</style></head><body>")
    lines.append(f"<h1>{esc(title)}</h1>")

    lines.append("<h2>Run summary</h2>")
    lines.append("<ul>")
    for k in [
        "tag", "evaluation_mode", "far", "failure_index", "win", "step",
        "baselines_enabled", "baselines_disabled_reason",
        "paired_baseline", "psi4_earlier_fraction", "median_delta_alarm_index",
        "n_repeats",
    ]:
        if k in ctx:
            v = ctx.get(k, None)
            if k in ("psi4_earlier_fraction", "median_delta_alarm_index"):
                try:
                    fv = float(v)
                    if not math.isfinite(fv):
                        v = "N/A"
                except Exception:
                    if v is None or str(v).strip() == "":
                        v = "N/A"
            lines.append(f"<li><b>{esc(k)}</b>: {esc(v)}</li>")
    lines.append("</ul>")

    lines.append("<h2>Files</h2>")
    files = ctx.get("files", {})
    lines.append("<ul>")
    for label, rel in files.items():
        lines.append(f"<li><b>{esc(label)}</b>: <a href='{esc(rel)}'>{esc(rel)}</a></li>")
    lines.append("</ul>")

    lines.append("<h2>evaluation</h2>")
    evaluation_html = ctx.get("evaluation_html", "<p class='muted'>(missing)</p>")
    lines.append(evaluation_html)

    lines.append("<p class='muted'>Note: thresholds calibrated on reference to match FAR; test evaluated with fixed thresholds (no tuning).</p>")
    lines.append("</body></html>")

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    # selftest handled in __main__ early bypass

    ap = argparse.ArgumentParser()

    ap.add_argument("--ref-psi4", required=True, help="Reference psi4 windows root (contains w_*/n2_primary.json)")
    ap.add_argument("--test-psi4", required=True, help="Test psi4 windows root (contains w_*/n2_primary.json)")

    # Baselines are OPTIONAL only if user explicitly chooses psi4-only.
    ap.add_argument("--ref-rawcsv", default="", help="Reference raw CSV (std) for baselines")
    ap.add_argument("--test-rawcsv", default="", help="Test raw CSV (std) for baselines")
    ap.add_argument("--psi4-only", action="store_true", help="Explicitly run psi4-only evaluation (no baselines).")
    ap.add_argument("--sig1-col", default="", required=False)
    ap.add_argument("--sig2-col", default="", required=False)
    ap.add_argument("--win", type=int, default=None, required=False, help="Window length in samples (required for baselines)")
    ap.add_argument("--step", type=int, default=None, required=False, help="Step size in samples (required for baselines)")

    ap.add_argument("--far", type=float, default=0.05, help="Target FAR on reference (default 0.05)")
    ap.add_argument("--failure-index", type=int, default=None, help="Failure window index (for lead time + pre/post rates)")

    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--output-mode", choices=["publish-min", "standard", "full-audit"], default="standard",
                    help="Artifact volume control only. Evaluator pack is already compact, so all modes are currently identical.")

    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Print planned outputs and exit.")
    ap.add_argument("--selftest", action="store_true", help="Run internal self-test and exit.")
    ap.add_argument("--verbose", action="store_true", help="Force verbose printing (compat flag).")
    ap.add_argument("--max-windows", type=int, default=None, help="Optional cap for faster runs (debug)")

    ap.add_argument(
        "--evaluation-mode",
        default="auto",
        choices=["auto", "separation", "early_warning"],
        help="Explicit evaluation mode (default: auto)",
    )

    # stability across seeds (bootstrap ref)
    ap.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        help="Number of stability repeats (bootstrap REF). Use 3 for minimal reviewer proof.",
    )
    ap.add_argument(
        "--seed-list",
        default="",
        help="Optional comma-separated seeds (overrides 0..n_repeats-1). Example: 0,1,2",
    )

    # paired comparison framing
    ap.add_argument(
        "--paired-baseline",
        default="rms_sig1",
        choices=["rms_sig1", "rms_sig2", "kurt_sig1", "kurt_sig2", "abs_corr"],
        help="Baseline metric to compare psi4 against (paired framing). Default: rms_sig1",
    )

    args = ap.parse_args()

    # Conditional arg enforcement (LOCK):
    # - If baselines are requested (NOT --psi4-only), rawcsv+sig cols+win/step must be provided.
    # - If --psi4-only, these are optional unless rawcsv is provided (then baselines compute must be well-defined).
    _rawcsv_provided = (str(args.ref_rawcsv).strip() != "" or str(args.test_rawcsv).strip() != "")
    _need_baseline_params = ((not args.psi4_only) or _rawcsv_provided)

    if _need_baseline_params:
        missing = []
        if str(args.sig1_col).strip() == "":
            missing.append("--sig1-col")
        if str(args.sig2_col).strip() == "":
            missing.append("--sig2-col")
        if args.win is None or int(args.win) <= 0:
            missing.append("--win")
        if args.step is None or int(args.step) <= 0:
            missing.append("--step")
        if missing:
            print("[ERROR] INVALID_EVAL_missing_args_for_baselines: " + ", ".join(missing))
            return 2

    # Atomic output discipline: write into tmp_outdir, then commit to final outdir (no partial folders).
    final_outdir = args.outdir
    tmp_outdir = final_outdir + f".__tmp__{os.getpid()}"
    # Route all subsequent writes through tmp_outdir
    args.outdir = tmp_outdir

    def _fail_invalid(msg: str) -> int:
        # No writes to final_outdir on invalid; cleanup tmp if it exists.
        try:
            if 'tmp_outdir' in locals() and os.path.exists(tmp_outdir):
                shutil.rmtree(tmp_outdir)
        except Exception:
            pass
        print(msg)
        return 2
    if getattr(args, "dry_run", False):
        print(f"[DRY-RUN] OUTPUT_MODE={args.output_mode}")
        print("[DRY-RUN] PLANNED_OUTPUTS:")
        for f in planned_outputs_for_mode(args.output_mode, args.tag):
            print("  - " + f)
        return 0

    # ---------- FAIL-FAST publish gates (no outdir, no artifacts) ----------
    def _has_psi4_windows(root: str) -> bool:
        try:
            root = os.path.abspath(root)
            if not os.path.isdir(root):
                return False
            wins = list_psi4_windows(root)
            if len(wins) == 0:
                return False
            # Require at least one expected primary file (current format: n2_primary.json)
            for w in wins:
                if os.path.exists(os.path.join(w, "n2_primary.json")):
                    return True
            return False
        except Exception:
            return False

    if not _has_psi4_windows(args.ref_psi4):
        print(f"[ERROR] INVALID_EVAL_no_psi4_windows: ref_psi4={args.ref_psi4}")
        return 2
    if not _has_psi4_windows(args.test_psi4):
        print(f"[ERROR] INVALID_EVAL_no_psi4_windows: test_psi4={args.test_psi4}")
        return 2

    # Baselines gate: if user did NOT request --psi4-only, rawcsv must be provided
    if (not args.psi4_only):
        if str(args.ref_rawcsv).strip() == "" or str(args.test_rawcsv).strip() == "":
            print("[ERROR] INVALID_EVAL_missing_rawcsv: baselines require --ref-rawcsv and --test-rawcsv (or pass --psi4-only).")
            return 2

    verbose = (not args.quiet) or args.verbose

    # ---------- helpers (local, stable) ----------
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z')
    def write_json(path: str, obj) -> None:
        # Publish hygiene: strict JSON (no NaN/Inf). This is display/serialization only.
        def _sanitize(x):
            if isinstance(x, float):
                return x if math.isfinite(x) else None
            if isinstance(x, dict):
                return {k: _sanitize(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [_sanitize(v) for v in x]
            return x

        clean = _sanitize(obj)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, allow_nan=False)

    def write_provenance(outdir: str) -> str:
        prov = {
            "tool": "n2_far_matched_evaluator",
            "timestamp_utc": _utc_now_iso(),
            "argv": sys.argv,
            "python": sys.version,
            "platform": sys.platform,
        }
        # optional deps
        try:
            prov["numpy"] = np.__version__
        except Exception:
            prov["numpy"] = "unknown"
        try:
            prov["pandas"] = pd.__version__
        except Exception:
            prov["pandas"] = "unknown"
        try:
            prov["matplotlib"] = getattr(plt, "__version__", "unknown")
        except Exception:
            prov["matplotlib"] = "unknown"

        outp = os.path.join(outdir, f"{args.tag}_provenance.json")
        write_json(outp, prov)
        return outp

    def write_index(outdir: str, files: dict) -> str:
        # stable, minimal, no dependencies
        lines = []
        lines.append("<!doctype html><html><head><meta charset='utf-8'>")
        lines.append(f"<title>Index - {args.tag}</title>")
        lines.append("</head><body>")
        lines.append(f"<h1>Index - {args.tag}</h1>")
        lines.append("<ul>")
        for label, rel in files.items():
            rel = "" if rel is None else str(rel)
            if rel.startswith("(") or rel == "":
                lines.append(f"<li><b>{label}</b>: {rel}</li>")
            else:
                bn = os.path.basename(rel)
                lines.append(f"<li><b>{label}</b>: <a href='{bn}'>{bn}</a></li>")
        lines.append("</ul></body></html>")
        outp = os.path.join(outdir, "index.html")
        with open(outp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return outp

    # ---------- publish gate planning (no compute yet) ----------
    rawcsv_provided = (str(args.ref_rawcsv).strip() != "" or str(args.test_rawcsv).strip() != "")
    baselines_requested = (not args.psi4_only)

    baselines_enabled = False
    baselines_disabled_reason = ""

    invalid_eval_reason = ""  # if set and baselines_requested, we fail-fast with exit 2

    if args.psi4_only:
        baselines_enabled = False
        baselines_disabled_reason = "psi4_only_explicit" if (not rawcsv_provided) else "psi4_only_explicit_even_with_rawcsv"
    else:
        # baselines are required unless user explicitly opts out
        if not rawcsv_provided:
            invalid_eval_reason = "no_rawcsv_provided_and_psi4_only_not_set"
            baselines_enabled = False
            baselines_disabled_reason = invalid_eval_reason
        else:
            baselines_enabled = True
            baselines_disabled_reason = ""

    # We create outdir early ONLY so we can write an INVALID_EVAL evidence pack
    ensure_dir(tmp_outdir)

    # Always load psi4 series (needed for both normal + invalid pack detail)
    ref_psi4 = load_psi4_series(args.ref_psi4, verbose)
    test_psi4 = load_psi4_series(args.test_psi4, verbose)

    # Strengthen gate: require enough finite psi4 values (not just folders)
    n_ref_finite = int(np.isfinite(pd.to_numeric(ref_psi4["psi4_circR"], errors="coerce").to_numpy(dtype=float)).sum())
    n_test_finite = int(np.isfinite(pd.to_numeric(test_psi4["psi4_circR"], errors="coerce").to_numpy(dtype=float)).sum())
    if n_ref_finite < 10 or n_test_finite < 5:
        # This is invalid even for psi4-only because headline is not computable/reliable.
        invalid_eval_reason = f"insufficient_finite_psi4_values: ref_finite={n_ref_finite} test_finite={n_test_finite}"

    # Optional baselines compute (but NEVER silently downgrade)
    ref_base = None
    test_base = None

    if baselines_enabled and invalid_eval_reason == "":
        try:
            ref_base = compute_baselines_from_rawcsv(
                args.ref_rawcsv,
                args.sig1_col,
                args.sig2_col,
                args.win,
                args.step,
                verbose,
            )
        except Exception as e:
            return _fail_invalid(f"[ERROR] INVALID_EVAL_baselines_failed: ref_base={e}")

        try:
            test_base = compute_baselines_from_rawcsv(
                args.test_rawcsv,
                args.sig1_col,
                args.sig2_col,
                args.win,
                args.step,
                verbose,
            )
        except Exception as e:
            return _fail_invalid(f"[ERROR] INVALID_EVAL_baselines_failed: test_base={e}")

    # Alignment gate (only when baselines enabled)
    ref = ref_psi4.copy()
    test = test_psi4.copy()
    if baselines_enabled and ref_base is not None and test_base is not None and invalid_eval_reason == "":
        n_ref = min(len(ref_psi4), len(ref_base))
        n_test = min(len(test_psi4), len(test_base))
        if args.max_windows is not None:
            n_ref = min(n_ref, args.max_windows)
            n_test = min(n_test, args.max_windows)
        if n_ref < 10 or n_test < 5:
            invalid_eval_reason = f"INVALID_EVAL_insufficient_aligned_windows: ref={n_ref} test={n_test}"
            baselines_enabled = False
            baselines_disabled_reason = invalid_eval_reason
        else:
            log(f"[INFO] aligned windows: ref={n_ref} test={n_test}", verbose)
            ref = pd.concat([ref_psi4.iloc[:n_ref].reset_index(drop=True), ref_base.iloc[:n_ref].reset_index(drop=True)], axis=1)
            test = pd.concat([test_psi4.iloc[:n_test].reset_index(drop=True), test_base.iloc[:n_test].reset_index(drop=True)], axis=1)

    # Metric specs + directions
    psi4_specs = [("psi4_circR", "low_bad")]
    baseline_specs = [
        ("rms_sig1", "high_bad"),
        ("rms_sig2", "high_bad"),
        ("kurt_sig1", "high_bad"),
        ("kurt_sig2", "high_bad"),
        ("abs_corr", "high_bad"),
    ]
    metric_specs = psi4_specs + (baseline_specs if baselines_enabled else [])
    direction_map: Dict[str, str] = {m: d for m, d in metric_specs}

    # Contract is ALWAYS written (even invalid), because it is the run's definition
    contract: Dict[str, Any] = {
        "tag": args.tag,
        "far": args.far,
        "failure_index": args.failure_index,
        "win": args.win,
        "step": args.step,
        "signals": {"sig1": args.sig1_col, "sig2": args.sig2_col},
        "direction": direction_map,
        "metrics": [m for m, _ in metric_specs],
        "psi4_metric": {"name": "psi4_circR"},
        "baselines_enabled": bool(baselines_enabled),
        "baselines_disabled_reason": baselines_disabled_reason,
        "baselines": [m for m, _ in baseline_specs],
        "note": (
            "All thresholds are calibrated on the reference set to match FAR; "
            "test is evaluated with fixed thresholds (no tuning). "
            "No silent downgrade: baselines require explicit rawcsv + successful baseline computation. "
            "INVALID_EVAL produces a minimal evidence pack and exits(2)."
        ),
    }
    out_contract = os.path.join(tmp_outdir, f"{args.tag}_contract.json")
    write_json(out_contract, contract)

    # Provenance is ALWAYS written (even invalid)
    out_prov = write_provenance(args.outdir)

    # FAIL-FAST: if invalid and user did NOT explicitly request psi4-only
    # Also fail-fast if psi4 values are insufficient (even in psi4-only)
    if invalid_eval_reason != "" and (baselines_requested or "insufficient_finite_psi4_values" in invalid_eval_reason):
        summary = {
            "tag": args.tag,
            "verdict": "INVALID_EVAL",
            "exit_code": 2,
            "reason": invalid_eval_reason,
            "baselines_requested": bool(baselines_requested),
            "baselines_enabled": bool(baselines_enabled),
            "baselines_disabled_reason": baselines_disabled_reason,
        }
        out_sum = os.path.join(tmp_outdir, f"{args.tag}_summary.json")
        write_json(out_sum, summary)

        # Minimal report + index
        report_path = os.path.join(tmp_outdir, f"{args.tag}_report.html")
        write_report_html(report_path, {
            "tag": args.tag,
            "evaluation_mode": "INVALID_EVAL",
            "far": args.far,
            "failure_index": args.failure_index,
            "win": args.win,
            "step": args.step,
            "baselines_enabled": baselines_enabled,
            "baselines_disabled_reason": baselines_disabled_reason,
            "paired_baseline": "(disabled)",
            "psi4_earlier_fraction": float("nan"),
            "median_delta_alarm_index": float("nan"),
            "n_repeats": 0,
            "evaluation_html": "<p class='muted'>INVALID_EVAL: no evaluation computed.</p>",
            "files": {
                "contract.json": os.path.basename(out_contract),
                "summary.json": os.path.basename(out_sum),
                "provenance.json": os.path.basename(out_prov),
                "report.html": os.path.basename(report_path),
            },
        })
        _assert_html_no_nan_inf(report_path)


        write_index(tmp_outdir, {
            "contract.json": out_contract,
            "summary.json": out_sum,
            "provenance.json": out_prov,
            "report.html": report_path,
        })

        log(f"[INVALID_EVAL] {invalid_eval_reason}", verbose)
        return 2

    # ---------- normal compute path ----------
    evaluation, thresholds = compute_evaluation_once(
        ref=ref,
        test=test,
        metric_specs=metric_specs,
        far=args.far,
        failure_index=args.failure_index,
    )

    mode = infer_evaluation_mode(evaluation, args.failure_index, args.evaluation_mode)

    out_per = os.path.join(tmp_outdir, f"{args.tag}_per_window.csv")
    out_evaluation = os.path.join(tmp_outdir, f"{args.tag}_evaluation.csv")
    test.copy().to_csv(out_per, index=False)
    evaluation.to_csv(out_evaluation, index=False)

    paired_base = str(args.paired_baseline)
    out_stab_runs = os.path.join(tmp_outdir, f"{args.tag}_stability_runs.csv")
    out_stab_sum = os.path.join(tmp_outdir, f"{args.tag}_stability_summary.csv")

    frac_earlier = float("nan")
    med_delta = float("nan")
    n_repeats_actual = 0
    seeds_used: List[int] = []

    if baselines_enabled:
        n_repeats = int(max(1, args.n_repeats))
        seeds = parse_seed_list(args.seed_list, n_repeats)
        if len(seeds) < n_repeats:
            n_repeats = len(seeds)

        stability_rows = []
        for r in range(n_repeats):
            seed = int(seeds[r])
            ref_b = bootstrap_ref(ref, seed)

            evaluation_r, _ = compute_evaluation_once(
                ref=ref_b,
                test=test,
                metric_specs=metric_specs,
                far=args.far,
                failure_index=args.failure_index,
            )

            def get_first(evaluation: pd.DataFrame, metric: str) -> float:
                v = evaluation.loc[evaluation["metric"] == metric, "test_first_alarm_index"]
                if len(v) == 0:
                    return float("nan")
                return float(v.iloc[0])

            psi4_first = get_first(evaluation_r, "psi4_circR")
            base_first = get_first(evaluation_r, paired_base)

            delta = float("nan")
            if math.isfinite(psi4_first) and math.isfinite(base_first) and psi4_first >= 0 and base_first >= 0:
                delta = float(psi4_first - base_first)  # negative => psi4 earlier

            stability_rows.append({
                "repeat": r,
                "seed": seed,
                "paired_baseline": paired_base,
                "psi4_first_alarm_index": psi4_first,
                "baseline_first_alarm_index": base_first,
                "delta_alarm_index_psi4_minus_baseline": delta,
                "psi4_earlier_than_baseline": (1 if math.isfinite(delta) and delta < 0 else 0),
                "psi4_ties_baseline": (1 if math.isfinite(delta) and delta == 0 else 0),
            })

        stability_runs = pd.DataFrame(stability_rows)
        stability_runs.to_csv(out_stab_runs, index=False)

        frac_earlier = float(np.mean(stability_runs["psi4_earlier_than_baseline"].to_numpy(dtype=float))) if len(stability_runs) else float("nan")
        med_delta = stability_summary(stability_runs["delta_alarm_index_psi4_minus_baseline"])["median"]

        pd.DataFrame([{
            "tag": args.tag,
            "paired_baseline": paired_base,
            "n_repeats": int(len(stability_runs)),
            "psi4_earlier_fraction": frac_earlier,
            "median_delta_alarm_index": med_delta,
            "delta_q25": stability_summary(stability_runs["delta_alarm_index_psi4_minus_baseline"])["q25"],
            "delta_q75": stability_summary(stability_runs["delta_alarm_index_psi4_minus_baseline"])["q75"],
            "psi4_first_median": stability_summary(stability_runs["psi4_first_alarm_index"])["median"],
            "baseline_first_median": stability_summary(stability_runs["baseline_first_alarm_index"])["median"],
        }]).to_csv(out_stab_sum, index=False)

        n_repeats_actual = int(len(stability_runs))
        seeds_used = [int(s) for s in stability_runs["seed"].tolist()]

    # Summary (always for normal runs)
    verdict = "OK"
    if args.psi4_only:
        verdict = "OK_PSI4_ONLY"

    out_sum = os.path.join(tmp_outdir, f"{args.tag}_summary.json")
    write_json(out_sum, {
        "tag": args.tag,
        "verdict": verdict,
        "exit_code": 0,
        "evaluation_mode": mode,
        "baselines_enabled": bool(baselines_enabled),
        "baselines_disabled_reason": baselines_disabled_reason,
        "paired_baseline": paired_base if baselines_enabled else "(disabled)",
        "psi4_earlier_fraction": frac_earlier,
        "median_delta_alarm_index": med_delta,
        "n_repeats": n_repeats_actual,
    })

    # Plots
    if not args.no_plot:
        for t in thresholds:
            out_png = os.path.join(tmp_outdir, f"{args.tag}_{t.metric}.png")
            plot_metric(args.tag, t.metric, test[t.metric], t.threshold, t.direction, out_png)

    # report.html + index.html
    report_path = os.path.join(tmp_outdir, f"{args.tag}_report.html")
    keep_cols = [
        "metric", "direction", "threshold", "ref_far_achieved",
        "test_first_alarm_index", "lead_time_windows",
        "alarm_rate_pre_failure_frac", "alarm_rate_post_failure_frac",
    ]
    keep_cols = [c for c in keep_cols if c in evaluation.columns]
    evaluation_disp = evaluation[keep_cols].replace([np.inf, -np.inf], np.nan)
    evaluation_table = evaluation_disp.fillna(np.nan).to_html(index=False, border=0, na_rep="N/A")

    files = {
        "per_window.csv": out_per,
        "evaluation.csv": out_evaluation,
        "contract.json": out_contract,
        "summary.json": out_sum,
        "provenance.json": out_prov,
        "report.html": report_path,
        "stability_runs.csv": out_stab_runs if baselines_enabled else "(disabled)",
        "stability_summary.csv": out_stab_sum if baselines_enabled else "(disabled)",
    }

    write_report_html(report_path, {
        "tag": args.tag,
        "evaluation_mode": mode,
        "far": args.far,
        "failure_index": args.failure_index,
        "win": args.win,
        "step": args.step,
        "baselines_enabled": baselines_enabled,
        "baselines_disabled_reason": baselines_disabled_reason,
        "paired_baseline": paired_base if baselines_enabled else "(disabled)",
        "psi4_earlier_fraction": frac_earlier,
        "median_delta_alarm_index": med_delta,
        "n_repeats": n_repeats_actual,
        "evaluation_html": evaluation_table,
        "files": {k: (os.path.basename(v) if isinstance(v, str) and os.path.exists(v) else str(v)) for k, v in files.items()},
    })
    _assert_html_no_nan_inf(report_path)


    write_index(tmp_outdir, files)

    log(f"[OK] wrote {out_per}", verbose)
    log(f"[OK] wrote {out_evaluation}", verbose)
    log(f"[OK] wrote {out_contract}", verbose)
    log(f"[OK] wrote {out_sum}", verbose)
    log(f"[OK] wrote {out_prov}", verbose)
    log(f"[OK] wrote {report_path}", verbose)
    if baselines_enabled:
        log(f"[OK] wrote {out_stab_runs}", verbose)
        log(f"[OK] wrote {out_stab_sum}", verbose)
    if not args.no_plot:
        log(f"[OK] wrote plots: {args.outdir}\\{args.tag}_*.png", verbose)

    if verbose:
        print("\n[evaluation]")
        cols = [
            "metric", "threshold", "ref_far_achieved",
            "test_first_alarm_index", "lead_time_windows",
            "alarm_rate_pre_failure_frac", "alarm_rate_post_failure_frac",
        ]
        cols = [c for c in cols if c in evaluation.columns]
        evaluation_print = evaluation[cols].copy()
        _na_cols = ["alarm_rate_pre_failure_frac", "alarm_rate_post_failure_frac", "lead_time_windows"]
        for _c in _na_cols:
            if _c in evaluation_print.columns:
                _x = pd.to_numeric(evaluation_print[_c], errors="coerce").to_numpy(dtype=float)
                _m = ~np.isfinite(_x)
                if _m.any():
                    evaluation_print[_c] = evaluation_print[_c].astype(object)
                    evaluation_print.loc[_m, _c] = "N/A"
        print(evaluation_print)
        print(f"\n[MODE] evaluation_mode={mode}")
        print(f"[BASELINES] enabled={baselines_enabled} reason='{baselines_disabled_reason}'")

        if baselines_enabled and n_repeats_actual > 0:
            print("\n[STABILITY + PAIRED]")
            print(f"paired_baseline = {paired_base}")
            print(f"psi4_earlier_fraction = {frac_earlier:.3f}  (over {n_repeats_actual} repeats)")
            print(f"median(delta_alarm_index = psi4 - baseline) = {'N/A' if not math.isfinite(float(med_delta)) else med_delta}")
    # Must-have artifacts (tmp_outdir) before commit
    must = [
        f"{args.tag}_per_window.csv",
        f"{args.tag}_evaluation.csv",
        f"{args.tag}_contract.json",
        f"{args.tag}_summary.json",
        f"{args.tag}_provenance.json",
        f"{args.tag}_report.html",
        "index.html",
    ]
    for fn in must:
        if not os.path.exists(os.path.join(tmp_outdir, fn)):
            # refuse to commit partial packs
            return _fail_invalid(f"[ERROR] INVALID_EVAL_missing_artifact_before_commit: {fn}")

    # Commit: replace final_outdir with tmp_outdir at directory level
    try:
        if os.path.exists(final_outdir):
            shutil.rmtree(final_outdir)
        shutil.move(tmp_outdir, final_outdir)
    except Exception as e:
        return _fail_invalid(f"[ERROR] INVALID_EVAL_commit_failed: {type(e).__name__}: {e}")

    return 0

EXIT_OK = 0
EXIT_INVALID = 2

if __name__ == "__main__":
    try:
        # EARLY_BYPASS_SELFTEST_DRYRUN: avoid argparse required-args for --selftest/--dry-run
        if "--selftest" in sys.argv:
            raise SystemExit(_selftest())
        if "--dry-run" in sys.argv:
            print("[DRY-RUN] PLANNED_OUTPUTS:")
            for f in PLANNED_OUTPUTS:
                print("  - " + f)
            raise SystemExit(0)
        rc = main()
        if rc is None:
            rc = EXIT_OK
        if rc not in (EXIT_OK, EXIT_INVALID):
            rc = EXIT_INVALID
        raise SystemExit(rc)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else EXIT_INVALID
        if code not in (EXIT_OK, EXIT_INVALID):
            code = EXIT_INVALID
        raise SystemExit(code)
    except Exception as e:
        # Any unexpected exception => invalid eval (no traceback, exit 2)
        print(f"[ERROR] INVALID_EVAL_exception: {type(e).__name__}: {e}")
        raise SystemExit(EXIT_INVALID)

