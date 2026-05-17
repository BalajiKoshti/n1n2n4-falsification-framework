# [PATCH] do not duplicate time_reversal_* inside family summaries
# [PATCH] contract wording aligned to median(Kpsi)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
n1_time-order_toolkit.py
========================

External (publish / industry) name (LOCK):
  Time-Order Asymmetry Detection Toolkit (n₁)

One-line purpose (LOCK):
  Detects whether a single time-series contains irreversible temporal structure beyond shuffled / phase-scrambled controls.

What this toolkit DOES (LOCK):
  - Computes a frozen, windowed time-asymmetry summary (Kψ family) on the real signal
  - Audits that summary against falsification-first null families (MANDATORY in v1):
      A) Phase randomization (FFT phase scramble; spectrum-preserving)
      B) Block shuffle (time ordering broken, local block structure preserved)
      C) Time reversal (x reversed; always computable)
  - Reports full null distributions + percentile separation + standardized empirical p-values (no cherry-pick)

What this toolkit MUST NEVER CLAIM (LOCK):
  - “Failure”, “health”, “risk”, “prediction”
  - It is a FILTER / DETECTOR, not a diagnosis.
  - It is NOT “photon physics proof” in v1.

Report section labels (OUTWARD UI only; internals stay A0/A6/A7):
  - A0 → Structure Check
  - A6 → Temporal Drift Scan
  - A7 → Null Robustness Audit

Classification wording (OUTWARD UI only):
  - Noise-Dominated
  - Weak Time-Asymmetry
  - Clear Time-Asymmetry

Executive number (reporting only, NOT a new metric):
  - Asymmetry Confidence (0–1), derived from null percentile separation strength.

Outputs (frozen evidence pack):
  - contract.json
  - summary.json
  - canonical_windows.csv
  - n1_primary.json              (full details; includes diagnostics)
  - n1_null_table.csv            (all null families, p-values, percentiles)
  - n1_plot.png
  - report.html
  - provenance.json

LOCK RULE:
  Any diagnostic (A0 PSD, Arrow-A extras, etc.) is diagnostic_only: true and MUST NOT change verdict.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import importlib.util
# [PATCH] force UTF-8 stdout/stderr for Windows consoles
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import time
import inspect
import difflib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# -----------------------------
# Constants / identity
# -----------------------------
TOOLKIT_NAME = "Time-Order Asymmetry Detection Toolkit (n₁)"
TOOLKIT_INTERNAL_NAME = "ψ₄ n₁ Time-Ordering Toolkit"
TOOLKIT_VERSION = "1.0.0-lock"
TAU = 2.0 * math.pi
EPS = 1e-12


# -----------------------------
# Helpers
# -----------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def clamp01(x: float) -> float:
    if not np.isfinite(x):
        return float("nan")
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def zscore(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.asarray([], dtype=float)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd < EPS:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


def pct_null_below_real(null_vals: List[float], real_val: float) -> float:
    """
    P(null < real) in [0,1]. If null empty -> NaN.
    """
    if not null_vals:
        return float("nan")
    arr = np.asarray(null_vals, dtype=float)
    return float(np.mean(arr < real_val))


def strength_from_percentile(p: float) -> float:
    """
    Percentile separation "strength" in [0.5, 1.0]:
      strength = max(p, 1-p)
    """
    if not np.isfinite(p):
        return float("nan")
    return float(max(p, 1.0 - p))


def confidence_0_1_from_strength(s: float) -> float:
    """
    Map strength in [0.5, 1.0] -> confidence in [0, 1]:
      conf = (s - 0.5) * 2
    """
    if not np.isfinite(s):
        return float("nan")
    return clamp01((s - 0.5) * 2.0)


def empirical_pvals(null_vals: List[float], real_val: float) -> Dict[str, float]:
    """
    Standardized empirical p-values (LOCK):
      p = (k + 1)/(M + 1)
    Provide:
      - p_right: P(null >= real)
      - p_left : P(null <= real)
      - p_two  : 2*min(p_left, p_right) capped at 1
    """
    if not null_vals:
        return {
            "M": 0,
            "k_right": 0,
            "k_left": 0,
            "p_right": float("nan"),
            "p_left": float("nan"),
            "p_two": float("nan"),
        }
    arr = np.asarray(null_vals, dtype=float)
    M = int(arr.size)
    k_right = int(np.sum(arr >= real_val))
    k_left = int(np.sum(arr <= real_val))
    p_right = float((k_right + 1) / (M + 1))
    p_left = float((k_left + 1) / (M + 1))
    p_two = float(min(1.0, 2.0 * min(p_left, p_right)))
    return {"M": M, "k_right": k_right, "k_left": k_left, "p_right": p_right, "p_left": p_left, "p_two": p_two}


def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    a = np.asarray(xs, dtype=float)
    return float(np.mean(a)), float(np.std(a))


def logic_fingerprint() -> str:
    """
    Fingerprint frozen decision logic to detect drift.
    decide_verdict is defined later.
    """
    try:
        src = inspect.getsource(decide_verdict)  # type: ignore[name-defined]
    except Exception:
        src = "<unavailable>"
    return sha256_text(src)


PLANNED_OUTPUTS: list[str] = [
    "contract.json",
    "summary.json",
    "canonical_windows.csv",
    "n1_primary.json",
    "n1_null_table.csv",
    "n1_plot.png",
    "report.html",
    "provenance.json",
    "legacy_headline_pack.json",
    "index.json",
    "index.html",
    "methods_blurb.txt",
    "caption_text.txt",
    "limitations_blurb.txt",
]

PUBLISH_MIN_OPTIONAL: list[str] = [
    "legacy_headline_pack.json",
    "index.json",
    "methods_blurb.txt",
    "caption_text.txt",
    "limitations_blurb.txt",
]

def planned_outputs_for_mode(output_mode: str = "standard") -> list[str]:
    mode = str(output_mode or "standard").strip().lower()
    outs = list(PLANNED_OUTPUTS)
    if mode == "publish-min":
        outs = [x for x in outs if x not in set(PUBLISH_MIN_OPTIONAL)]
    return outs

def planned_outputs() -> list[str]:
    # Backward-compatible default
    return planned_outputs_for_mode("standard")

def prune_optional_artifacts_n1(outdir: Path, output_mode: str) -> None:
    mode = str(output_mode or "standard").strip().lower()
    if mode != "publish-min":
        return
    for name in PUBLISH_MIN_OPTIONAL:
        fp = outdir / name
        try:
            if fp.exists():
                fp.unlink()
        except Exception:
            # publish-min is convenience only; never break a valid run on prune failure
            pass

def preflight_csv(infile: Path, time_col: str | None, sig_col: str) -> Dict[str, object]:
    """
    Preflight: file existence, header columns, basic finite/constant checks.
    Returns ok/error plus warnings (warnings do not affect verdict).

    LOCK intent for n1:
      - time_col is OPTIONAL. If missing or None, we allow and rely on load_csv_signal()
        to synthesize sample-index time deterministically.
      - sig_col is REQUIRED.
    """
    if not infile.exists():
        return {"ok": False, "error": f"Input not found: {infile}"}

    # Normalize None -> default name "t" for consistent messaging
    if time_col is None:
        time_col = "t"

    warns: list[str] = []

    with infile.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {"ok": False, "error": "CSV has no header."}
        cols = list(reader.fieldnames)

        if time_col not in cols:
            warns.append(f"Missing time column '{time_col}'; using synthetic index time.")
        if sig_col not in cols:
            return {"ok": False, "error": f"Missing signal column: {sig_col}"}

    # Light-weight read of just the signal for warnings
    t, x = load_csv_signal(infile, time_col=time_col, sig_col=sig_col)
    n = int(x.size)
    finite_frac = float(np.mean(np.isfinite(x))) if n else 0.0
    sd = float(np.std(x)) if n else float("nan")

    if finite_frac < 0.999:
        warns.append(
            f"Non-finite samples present (finite_frac={finite_frac:.6f}); non-finite are dropped deterministically."
        )
    if np.isfinite(sd) and sd < 1e-12:
        warns.append("Signal is near-constant (std ~ 0); Kψ curve will be flat by construction.")

    return {"ok": True, "rows": int(n), "warnings": warns}


def preflight_windows(N: int, window: int, step: int) -> list[str]:
    warns: list[str] = []
    if window > N:
        warns.append(f"Window ({window}) > N ({N}); reduce --window or provide longer signal.")
    if step <= 0:
        warns.append("Step must be > 0.")
    if N >= window and step > 0:
        nwin = 1 + (N - window) // step
        if nwin < 5:
            warns.append(f"Only {nwin} windows; results may be unstable. Consider smaller --step.")
    return warns


# -----------------------------
# Frozen Contract (written to contract.json)
# -----------------------------
def frozen_contract() -> Dict[str, object]:
    return {
        "external_name": TOOLKIT_NAME,
        "internal_name": TOOLKIT_INTERNAL_NAME,
        "version": TOOLKIT_VERSION,
        "purpose_frozen": "Detect time-order structure vs noise under falsification-first nulls. Not photon physics proof.",
        "claim": (
            "Detects whether a single time-series contains irreversible temporal structure "
            "beyond phase-scrambled / block-shuffled / time-reversed controls."
        ),
        "headline_metric": {
            "name": "Kpsi_median_separation_vs_nulls",
            "definition": "Compare REAL median(Kpsi) to null families; report percentile separation and empirical p-values; no tuned thresholds.",
            "direction_note": "Two-sided evaluation via percentile strength and empirical p_two.",
        },
        "required_controls_v1": [
            "phase_scramble_fft",
            "block_shuffle",
            "time_reversal",
        ],
        "required_surrogate_families_for_verdict": ["phase_scramble_fft", "block_shuffle"],
        "required_deterministic_controls_to_report": ["time_reversal"],
        "required_controls_v1_note": "required_controls_v1 are computed/reported in v1. Verdict p-values (p_two) are evaluated only against the surrogate families listed in required_surrogate_families_for_verdict; deterministic controls are listed separately.",
        "time_reversal_policy_note": "time_reversal is a deterministic single transform (x reversed). It is required-to-report in v1, but it is not a randomized surrogate ensemble; surrogate p-values come from phase_scramble_fft and block_shuffle.",
        "diagnostics": {
            "A0_psd": {"diagnostic_only": True, "note": "Welch-like PSD proxy; never changes verdict."},
            "ArrowA": {"diagnostic_only": True, "note": "Optional/legacy arrow summaries; never changes verdict in v1."},
        },
        "sections_ui": {
            "A0": "Structure Check",
            "A6": "Temporal Drift Scan",
            "A7": "Null Robustness Audit",
        },
        "non_claims": [
            "PASS/FAIL does not mean the system is healthy or unhealthy.",
            "This is not a risk score or failure predictor.",
            "This does not identify root cause; it only audits time-order asymmetry vs falsifiers.",
        ],
        "verdict_policy_v1": {
            "verdict_thresholds_v1_2": {"alpha_weak": 0.05, "alpha_strong": 0.01},
            "verdict_pvalue_note": "Verdict uses empirical two-sided p_two from required surrogate families (phase_scramble_fft, block_shuffle).",
            "rule": "Verdict is based ONLY on presence of required controls + separation strength vs nulls. Diagnostics never affect verdict.",
            "missing_controls_behavior": "NO_CLEAR_MISSING_CONTROLS",
        },
        "units_note": "lowfreq_cut is normalized (cycles/sample) unless a real sampling rate is explicitly provided; default fs=1.0.",
        "defaults": {
            "fs_hz": 1.0,
            "lowfreq_cut_hz": 0.1,
            "window": 8192,
            "step": 2048,
            "block_len": 1024,
            "n_phase_null": 200,
            "n_block_null": 200,
        },
    }


# -----------------------------
# Data IO
# -----------------------------
def load_csv_signal(infile: Path, time_col: str = "t", sig_col: str = "x") -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads CSV columns time_col and sig_col.
    Drops rows where x is non-finite.
    If time is missing or non-finite, synthesizes deterministic sample-index time.
    """
    t_list: List[float] = []
    x_list: List[float] = []
    with infile.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError("CSV has no header.")
        for row in r:
            t = safe_float(row.get(time_col, "nan"))
            x = safe_float(row.get(sig_col, "nan"))
            if np.isfinite(x):
                t_list.append(t)
                x_list.append(x)
    t = np.asarray(t_list, dtype=float)
    x = np.asarray(x_list, dtype=float)
    if t.size == 0:
        return t, x
    if not np.all(np.isfinite(t)):
        t = np.arange(x.size, dtype=float)
    return t, x


# -----------------------------
# Core metric construction (frozen math)
# -----------------------------
def lowfreq_fraction_window(xw: np.ndarray, fs: float, lowfreq_cut: float) -> float:
    """
    Compute fraction of power below lowfreq_cut for the window xw.
    (Frozen math for n1 v1.)
    """
    xw = np.asarray(xw, dtype=float)
    if xw.size < 8:
        return float("nan")
    xw = xw - float(np.mean(xw))
    X = np.fft.rfft(xw)
    P = (np.abs(X) ** 2)
    freqs = np.fft.rfftfreq(xw.size, d=1.0 / max(EPS, fs))
    total = float(np.sum(P)) + EPS
    low = float(np.sum(P[freqs <= lowfreq_cut]))
    return low / total


def compute_windows(
    x: np.ndarray,
    fs: float,
    window: int,
    step: int,
    lowfreq_cut: float,
) -> Dict[str, object]:
    """
    Slide across x computing lowfreq fraction + Kψ = zscore(lowfreq_frac).
    NOTE: this function is x-only by design; time is represented by sample-index centers.
    """
    N = int(x.size)
    if N < window or window <= 0 or step <= 0:
        return {
            "starts": [],
            "centers": [],
            "lowfreq_frac": [],
            "kpsi": [],
            "window": int(window),
            "step": int(step),
            "N": int(N),
        }

    starts = list(range(0, N - window + 1, step))
    centers = [s + window / 2 for s in starts]
    lf = []
    for s in starts:
        lf.append(lowfreq_fraction_window(x[s : s + window], fs=fs, lowfreq_cut=lowfreq_cut))
    lf_arr = np.asarray(lf, dtype=float)
    kpsi = zscore(lf_arr)
    return {
        "starts": starts,
        "centers": centers,
        "lowfreq_frac": lf_arr.tolist(),
        "kpsi": kpsi.tolist(),
        "window": int(window),
        "step": int(step),
        "N": int(N),
    }
# -----------------------------
# Null generators (frozen families)
# -----------------------------
def fft_phase_scramble(x: np.ndarray, seed: int) -> np.ndarray:
    """
    Spectrum-preserving phase randomization (real FFT domain).
    """
    rng = np.random.default_rng(int(seed))
    x = np.asarray(x, dtype=float)
    N = x.size
    if N < 8:
        return x.copy()
    X = np.fft.rfft(x - float(np.mean(x)))
    mag = np.abs(X)
    phase = np.angle(X)

    # Randomize phases except DC and Nyquist (if exists)
    rand_phase = rng.uniform(-math.pi, math.pi, size=phase.shape)
    rand_phase[0] = phase[0]
    if phase.shape[0] > 1 and (N % 2 == 0):
        rand_phase[-1] = phase[-1]

    X_new = mag * np.exp(1j * rand_phase)
    x_new = np.fft.irfft(X_new, n=N)
    return x_new.astype(float)


def block_shuffle(x: np.ndarray, block_len: int, seed: int) -> np.ndarray:
    """
    Breaks global time-order but preserves within-block structure.
    """
    rng = np.random.default_rng(int(seed))
    x = np.asarray(x, dtype=float)
    N = x.size
    L = int(block_len)
    if L <= 1 or N < L:
        return x.copy()

    blocks = []
    for s in range(0, N, L):
        blocks.append(x[s : min(N, s + L)])
    idx = np.arange(len(blocks))
    rng.shuffle(idx)
    y = np.concatenate([blocks[i] for i in idx], axis=0)
    return y.astype(float)


# -----------------------------
# A0 — Structure Check (diagnostic-only)
# -----------------------------
def A0_psd_check(x: np.ndarray, seed: int, lowfreq_norm_cut: float = 0.05) -> Dict[str, object]:
    """
    Diagnostic-only PSD proxy:
      - compare low-frequency fraction vs phase-scrambled null ensemble
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    if N < 64:
        return {
            "diagnostic_only": True,
            "a0_psd_lowfrac_real": float("nan"),
            "a0_psd_lowfrac_null_mean": float("nan"),
            "a0_psd_lowfrac_null_std": float("nan"),
            "a0_psd_ratio": float("nan"),
            "a0_psd_label": "insufficient_data",
        }

    # Use normalized frequency index rather than fs for diagnostic simplicity
    fs = 1.0
    lowcut = max(EPS, float(lowfreq_norm_cut))

    real = lowfreq_fraction_window(x, fs=fs, lowfreq_cut=lowcut)

    # Small null ensemble (diagnostic-only) — keep light
    reps = 32
    nulls = []
    for i in range(reps):
        y = fft_phase_scramble(x, seed=int(seed) + 10_000 + i)
        nulls.append(lowfreq_fraction_window(y, fs=fs, lowfreq_cut=lowcut))
    mu, sd = mean_std(nulls)
    ratio = float(real / (mu + EPS)) if np.isfinite(mu) else float("nan")

    if not np.isfinite(ratio):
        lab = "unknown"
    elif ratio > 1.2:
        lab = "lowfreq_heavy"
    elif ratio < 0.8:
        lab = "lowfreq_light"
    else:
        lab = "typical"

    return {
        "diagnostic_only": True,
        "a0_psd_lowfrac_real": float(real),
        "a0_psd_lowfrac_null_mean": float(mu),
        "a0_psd_lowfrac_null_std": float(sd),
        "a0_psd_ratio": float(ratio),
        "a0_psd_label": str(lab),
    }


# -----------------------------
# Arrow-A diagnostics (never affects verdict)
# -----------------------------
def arrow_A_legacy(kpsi_curve: np.ndarray) -> Dict[str, object]:
    """
    Legacy arrow diagnostic: simple skew between first and last thirds.
    """
    k = np.asarray(kpsi_curve, dtype=float)
    if k.size < 9:
        return {"diagnostic_only": True, "arrow_mode": "legacy", "arrow_A": float("nan")}
    n = k.size
    a = float(np.mean(k[: n // 3]))
    b = float(np.mean(k[2 * n // 3 :]))
    return {"diagnostic_only": True, "arrow_mode": "legacy", "arrow_A": float(b - a)}


def arrow_A_event_local(kpsi_curve: np.ndarray, center_idx: int, pad: int = 20) -> Dict[str, object]:
    """
    Event-centered arrow diagnostic: compare pre vs post around a chosen window index.
    """
    k = np.asarray(kpsi_curve, dtype=float)
    n = k.size
    c = int(center_idx)
    p = int(pad)
    if n == 0 or c < 0 or c >= n:
        return {"diagnostic_only": True, "arrow_mode": "event", "arrow_event_A_mean": float("nan")}
    lo = max(0, c - p)
    hi = min(n, c + p + 1)
    seg = k[lo:hi]
    pre = seg[: max(0, c - lo)]
    post = seg[max(0, c - lo) + 1 :]

    pre_mean = float(np.mean(pre)) if pre.size else float("nan")
    post_mean = float(np.mean(post)) if post.size else float("nan")

    return {
        "diagnostic_only": True,
        "arrow_mode": "event",
        "arrow_event_center_idx": int(c),
        "arrow_event_pad_windows": int(p),
        "arrow_event_pre_mean": pre_mean,
        "arrow_event_post_mean": post_mean,
        "arrow_event_A_mean": float(post_mean - pre_mean),
    }


# -----------------------------
# A6 — Temporal Drift Scan (windowed curve)
# -----------------------------
def A6_temporal_drift(scan: Dict[str, object]) -> Dict[str, object]:
    centers = np.asarray(scan["centers"], dtype=float)
    kpsi = np.asarray(scan["kpsi"], dtype=float)

    if kpsi.size == 0:
        return {
            "n_windows": 0,
            "kpsi_max": float("nan"),
            "kpsi_min": float("nan"),
            "kpsi_max_center": float("nan"),
            "kpsi_min_center": float("nan"),
        }

    i_max = int(np.argmax(kpsi))
    i_min = int(np.argmin(kpsi))

    return {
        "n_windows": int(kpsi.size),
        "kpsi_max": float(kpsi[i_max]),
        "kpsi_min": float(kpsi[i_min]),
        "kpsi_max_center": float(centers[i_max]) if centers.size else float("nan"),
        "kpsi_min_center": float(centers[i_min]) if centers.size else float("nan"),
    }


# -----------------------------
# A7 — Null Robustness Audit (REQUIRED controls)
# -----------------------------
def compute_A7_null_audit(primary: Dict[str, object], x: np.ndarray, args: argparse.Namespace) -> Dict[str, object]:
    """
    Required controls (v1):
      - phase_scramble_fft
      - block_shuffle
      - time_reversal
    A7 must produce:
      - controls_present dict
      - real_kpsi_mean/median
      - null_family_summaries list entries containing p_null_lt_real etc.
      - raw null distributions (for table + plot)
    """
    scan = primary["scan"]
    real_k = np.asarray(scan["kpsi"], dtype=float)
    real_mean = float(np.mean(real_k)) if real_k.size else float("nan")
    real_median = float(np.median(real_k)) if real_k.size else float("nan")

    fs = float(primary.get("fs_hz", 1.0))
    window = int(args.timeline_win)
    step = int(args.timeline_step)
    lowfreq_cut = float(args.lowfreq_cut)
    block_len = int(args.block_len)

    n_phase = int(args.n_phase_null)
    n_block = int(args.n_block_null)

    null_rows: List[Dict[str, object]] = []

    # --- phase scramble nulls ---
    phase_means: List[float] = []
    phase_medians: List[float] = []
    for i in range(n_phase):
        y = fft_phase_scramble(x, seed=int(args.seed) + 1000 + i)
        sc = compute_windows(y, fs=fs, window=window, step=step, lowfreq_cut=lowfreq_cut)
        kk = np.asarray(sc["kpsi"], dtype=float)
        phase_means.append(float(np.mean(kk)) if kk.size else float("nan"))
        phase_medians.append(float(np.median(kk)) if kk.size else float("nan"))

    # --- block shuffle nulls ---
    block_means: List[float] = []
    block_medians: List[float] = []
    for i in range(n_block):
        y = block_shuffle(x, block_len=block_len, seed=int(args.seed) + 2000 + i)
        sc = compute_windows(y, fs=fs, window=window, step=step, lowfreq_cut=lowfreq_cut)
        kk = np.asarray(sc["kpsi"], dtype=float)
        block_means.append(float(np.mean(kk)) if kk.size else float("nan"))
        block_medians.append(float(np.median(kk)) if kk.size else float("nan"))

    # --- time reversal control (single) ---
    y_rev = x[::-1].copy()
    sc_rev = compute_windows(y_rev, fs=fs, window=window, step=step, lowfreq_cut=lowfreq_cut)
    kk_rev = np.asarray(sc_rev["kpsi"], dtype=float)
    rev_mean = float(np.mean(kk_rev)) if kk_rev.size else float("nan")
    rev_median = float(np.median(kk_rev)) if kk_rev.size else float("nan")

    # Family summaries (use median as frozen headline in decide_verdict)
    fams: List[Dict[str, object]] = []

    def add_family(family: str, null_medians: List[float]) -> None:
        p = pct_null_below_real(null_medians, real_median)
        strength = strength_from_percentile(p)
        conf = confidence_0_1_from_strength(strength)
        pv = empirical_pvals(null_medians, real_median)
        mu, sd = mean_std(null_medians)
        fams.append(
            {
                "family": family,
                "n": int(len(null_medians)),
                "real_kpsi_median": float(real_median),
                "mean_null": float(mu),
                "std_null": float(sd),
                "p_null_lt_real": float(p),
                "strength": float(strength),
                "confidence_0_1": float(conf),
                "pvals": pv,
            }
        )
        for v in null_medians:
            null_rows.append(
                {
                    "family": family,
                    "null_value": float(v),
                    "real_value": float(real_median),
                    "p_null_lt_real": float(p),
                    "strength": float(strength),
                    "p_two": float(pv.get("p_two", float("nan"))),
                }
            )

    # Filter non-finite null draws (publish robustness)
    phase_medians_finite = [float(v) for v in phase_medians if np.isfinite(float(v))]
    block_medians_finite = [float(v) for v in block_medians if np.isfinite(float(v))]
    add_family("phase_scramble_fft", phase_medians_finite)
    add_family("block_shuffle", block_medians_finite)
    # time reversal control is deterministic (single); reported separately (not a null family)
    controls_present = {
        "phase_scramble_fft": bool(len(phase_medians_finite) > 0),
        "block_shuffle": bool(len(block_medians_finite) > 0),
        "time_reversal": bool(np.isfinite(rev_median)),
    }

    # Headline across required families: conservative worst-case strength
    strengths = [safe_float(f.get("strength", float("nan"))) for f in fams if f.get("family") in controls_present]
    strengths = [v for v in strengths if np.isfinite(v)]
    confs = [safe_float(f.get("confidence_0_1", float("nan"))) for f in fams if f.get("family") in controls_present]
    confs = [v for v in confs if np.isfinite(v)]
    headline_strength = float(min(strengths)) if strengths else float("nan")
    headline_conf = float(min(confs)) if confs else float("nan")

    return {
        "controls_present": controls_present,
        "real_kpsi_mean": float(real_mean),
        "real_kpsi_median": float(real_median),
        "time_reversal_mean": float(rev_mean),
        "time_reversal_median": float(rev_median),
        "time_reversal_median_lt_real": bool(np.isfinite(rev_median) and np.isfinite(real_median) and (rev_median < real_median)),
        "time_reversal_pass": bool(np.isfinite(rev_median) and np.isfinite(real_median) and (rev_median < real_median)),  # legacy alias (kept for backward compatibility)
        "null_family_summaries": fams,
        "null_table_rows": null_rows,
        "headline_strength": float(headline_strength),
        "headline_confidence_0_1": float(headline_conf),
    }


# -----------------------------
# Verdict logic (frozen) + reason text
# -----------------------------
REASON_TEXT_MAP = {
    "missing_required_controls": "Controls missing → contract failure (NO_CLEAR).",
    "insufficient_data": "Insufficient data or NaN headline; cannot evaluate.",
    "null_table_empty": "Null table missing/empty; cannot evaluate required controls.",
    "no_clear_two_sided": "Two-sided / reversed separation under at least one required control.",
    "pass_strong": "Real exceeds all required null families strongly (high percentile separation).",
    "pass_weak": "Real exceeds required null families but separation is weaker/mixed.",
    "nulls_not_separated": "Required null families did not separate clearly from real under this test.",
    "nan_pvals": "Empirical p-values missing/NaN under at least one required control (NO_CLEAR).",
    "p_two_both_le_strong": "Both required surrogate families show rare two-sided separation (p_two <= 0.01).",
    "p_two_both_le_weak": "Both required surrogate families show uncommon two-sided separation (p_two <= 0.05).",
    "no_clear_p_two": "Two-sided separation not rare under at least one required surrogate family (NO_CLEAR).",
}


def decide_verdict(contract: dict, A7: dict) -> dict:
    """
    Frozen verdict policy (v1.2 lock):
      - Verdict uses ONLY empirical two-sided p-values (p_two) from REQUIRED surrogate families:
          * phase_scramble_fft
          * block_shuffle
      - time_reversal is required-to-report (deterministic) but not a surrogate p-value family.
      - Diagnostics never affect verdict.

    Labels:
      PASS_STRONG: both required families have p_two <= 0.01
      PASS_WEAK  : both required families have p_two <= 0.05
      NO_CLEAR   : otherwise
      NO_CLEAR_MISSING_CONTROLS: required families missing (n=0 etc.)
    """
    controls = A7.get("controls_present", {}) or {}
    required = ["phase_scramble_fft", "block_shuffle"]

    missing = [k for k in required if not controls.get(k, False)]
    if missing:
        return {
            "verdict": "NO_CLEAR_MISSING_CONTROLS",
            "verdict_reason_code": "missing_required_controls",
            "verdict_reason_text": "Controls missing -> contract failure (NO_CLEAR).",
            "missing_controls": missing,
        }

    fams = A7.get("null_family_summaries", []) or []

    def get_p_two(family: str) -> float:
        for f in fams:
            if f.get("family") == family:
                pv = (f.get("pvals") or {})
                try:
                    return float(pv.get("p_two"))
                except Exception:
                    return float("nan")
        return float("nan")

    p_phase = get_p_two("phase_scramble_fft")
    p_block = get_p_two("block_shuffle")

    # If p-values are NaN for any reason, fail closed (NO_CLEAR)
    if not (p_phase == p_phase and p_block == p_block):
        return {
            "verdict": "NO_CLEAR",
            "verdict_reason_code": "nan_pvals",
            "verdict_reason_text": "Empirical p-values missing/NaN under at least one required control (NO_CLEAR).",
            "missing_controls": [],
        }

    # Strong/weak thresholds (LOCK)
    alpha_strong = 0.01
    alpha_weak = 0.05

    if p_phase <= alpha_strong and p_block <= alpha_strong:
        return {
            "verdict": "PASS_STRONG",
            "verdict_reason_code": "p_two_both_le_strong",
            "verdict_reason_text": "Both required surrogate families show rare two-sided separation (p_two <= 0.01).",
            "missing_controls": [],
        }

    if p_phase <= alpha_weak and p_block <= alpha_weak:
        return {
            "verdict": "PASS_WEAK",
            "verdict_reason_code": "p_two_both_le_weak",
            "verdict_reason_text": "Both required surrogate families show uncommon two-sided separation (p_two <= 0.05).",
            "missing_controls": [],
        }

    return {
        "verdict": "NO_CLEAR",
        "verdict_reason_code": "no_clear_p_two",
        "verdict_reason_text": "Two-sided separation not rare under at least one required surrogate family (NO_CLEAR).",
        "missing_controls": [],
    }
def compute_n1_primary(infile: Path, args: argparse.Namespace) -> Tuple[Dict[str, object], np.ndarray]:
    """
    The coherent v1 primary path:
      CSV -> arrays -> scan -> diagnostics
    Returns:
      primary dict + the raw x (for null audit)
    """
    tc = args.time_col or "t"
    sc = args.sig_col
    t, x = load_csv_signal(infile, time_col=tc, sig_col=sc)

    fs = 1.0  # v1 lock default
    scan = compute_windows(x, fs=fs, window=int(args.timeline_win), step=int(args.timeline_step), lowfreq_cut=float(args.lowfreq_cut))

    a0 = A0_psd_check(x, seed=int(args.seed))
    arrow_legacy = arrow_A_legacy(np.asarray(scan["kpsi"], dtype=float))
    arrow_event = arrow_A_event_local(np.asarray(scan["kpsi"], dtype=float), center_idx=int(len(scan["kpsi"]) // 2), pad=20)
    a6 = A6_temporal_drift(scan)

    primary = {
        "toolkit": TOOLKIT_INTERNAL_NAME,
        "version": TOOLKIT_VERSION,
        "infile": str(infile),
        "time_col": tc,
        "sig_col": sc,
        "fs_hz": float(fs),
        "params": {
            "timeline_win": int(args.timeline_win),
            "timeline_step": int(args.timeline_step),
            "lowfreq_cut": float(args.lowfreq_cut),
            "block_len": int(args.block_len),
            "n_phase_null": int(args.n_phase_null),
            "n_block_null": int(args.n_block_null),
            "seed": int(args.seed),
        },
        "scan": scan,
        "A0": a0,
        "A6": a6,
        "ArrowA": {
            "diagnostic_only": True,
            "legacy": arrow_legacy,
            "event": arrow_event,
        },
    }

    return primary, x


def build_summary_n1(
    contract: Dict[str, object],
    primary: Dict[str, object],
    A7: Dict[str, object],
    verdict_obj: Dict[str, object],
) -> Dict[str, object]:
    """
    Summary (human-facing) — MUST remain conservative:
      - verdict from decide_verdict only
      - headline uses A7 median separation & confidence
    """
    headline = {
        "real_kpsi_median": float(A7.get("real_kpsi_median", float("nan"))),
        "real_kpsi_mean": float(A7.get("real_kpsi_mean", float("nan"))),
        "strength": float(A7.get("headline_strength", float("nan"))),
        "confidence_0_1": float(A7.get("headline_confidence_0_1", float("nan"))),
    }

    return {
        "toolkit": TOOLKIT_NAME,
        "internal_name": TOOLKIT_INTERNAL_NAME,
        "version": TOOLKIT_VERSION,
        "logic_fingerprint": logic_fingerprint(),
        "verdict": verdict_obj.get("verdict"),
        "verdict_reason_code": verdict_obj.get("verdict_reason_code"),
        "verdict_reason_text": verdict_obj.get("verdict_reason_text"),
        "missing_controls": verdict_obj.get("missing_controls", []),
        "headline": headline,
        "controls_present": A7.get("controls_present", {}),
        "null_family_summaries": A7.get("null_family_summaries", []),
        "diagnostics": {
            "A0": primary.get("A0", {}),
            "A6": primary.get("A6", {}),
            "ArrowA": primary.get("ArrowA", {}),
        },
        "notes": contract.get("non_claims", []),
    }


def build_provenance(args: argparse.Namespace, infile: Path) -> Dict[str, object]:
    return {
        "run_time_unix": int(time.time()),
        "infile": str(infile),
        "infile_sha256": sha256_file(infile),
        "args": vars(args),
        "python": sys.version,
        "platform": sys.platform,

        "versions": {
            "pandas": (__import__("pandas").__version__ if importlib.util.find_spec("pandas") else "unknown"),
            "scipy": (__import__("scipy").__version__ if importlib.util.find_spec("scipy") else "unknown"),
            "numpy": getattr(np, "__version__", "unknown"),
            "matplotlib": getattr(matplotlib, "__version__", "unknown"),
        },
    }


# -----------------------------
# Writers
# -----------------------------
def write_canonical_windows_csv(outdir: Path, primary: Dict[str, object]) -> None:
    scan = primary["scan"]
    starts = scan["starts"]
    centers = scan["centers"]
    lf = scan["lowfreq_frac"]
    kpsi = scan["kpsi"]

    path = outdir / "canonical_windows.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_start", "window_center", "lowfreq_frac", "kpsi"])
        for s, c, a, k in zip(starts, centers, lf, kpsi):
            w.writerow([s, c, a, k])


def write_n1_null_table_csv(outdir: Path, A7: Dict[str, object]) -> None:
    """
    Reviewer-auditable null table (reporting-only; never affects verdict).

    We emit TWO kinds of rows:
      1) distribution rows from A7["null_table_rows"] (one row per null draw)
      2) summary rows from A7["null_family_summaries"] (one row per family; includes pvals, mean/std, etc.)

    This replaces the previous minimal schema.
    """
    # [PATCH] expanded null table schema v1 (reviewer-audIT)
    path = outdir / "n1_null_table.csv"

    header = [
        "row_type",              # "null_draw" or "family_summary"
        "family",
        "seed",
        "M",
        "null_value",
        "real_kpsi_median",
        "real_kpsi_mean",
        "time_reversal_median",
        "time_reversal_pass",
        "mean_null",
        "std_null",
        "p_null_lt_real",
        "strength",
        "confidence_0_1",
        "k_right",
        "k_left",
        "p_right",
        "p_left",
        "p_two",
    ]

    rows_out: list[dict[str, object]] = []

    # 1) Null draw rows (existing)
    null_rows = A7.get("null_table_rows", [])
    if isinstance(null_rows, list):
        for r in null_rows:
            if not isinstance(r, dict):
                continue
            rows_out.append({
                "row_type": "null_draw",
                "family": r.get("family", ""),
                "seed": "",
                "M": "",
                "null_value": r.get("null_value", ""),
                "real_kpsi_median": r.get("real_value", r.get("real_kpsi_median", "")),
                "real_kpsi_mean": "",
                # [PATCH] fill reversal columns
                "time_reversal_median": A7.get("time_reversal_median", ""),
                "time_reversal_pass": A7.get("time_reversal_pass", ""),
                "mean_null": "",
                "std_null": "",
                "p_null_lt_real": r.get("p_null_lt_real", ""),
                "strength": r.get("strength", ""),
                "confidence_0_1": "",
                "k_right": "",
                "k_left": "",
                "p_right": "",
                "p_left": "",
                "p_two": r.get("p_two", ""),
            })

    # 2) Family summary rows (preferred audit row)
    fams = A7.get("null_family_summaries", [])
    if isinstance(fams, list):
        for frow in fams:
            if not isinstance(frow, dict):
                continue
            pv = frow.get("pvals", {})
            if not isinstance(pv, dict):
                pv = {}
            rows_out.append({
                "row_type": "family_summary",
                "family": frow.get("family", ""),
                "seed": A7.get("seed", ""),
                "M": pv.get("M", frow.get("n", "")),
                "null_value": "",
                "real_kpsi_median": frow.get("real_kpsi_median", A7.get("real_kpsi_median", "")),
                "real_kpsi_mean": frow.get("real_kpsi_mean", A7.get("real_kpsi_mean", "")),
                # [PATCH] fill reversal columns
                "time_reversal_median": A7.get("time_reversal_median", ""),
                "time_reversal_pass": A7.get("time_reversal_pass", ""),
                "mean_null": frow.get("mean_null", ""),
                "std_null": frow.get("std_null", ""),
                "p_null_lt_real": frow.get("p_null_lt_real", ""),
                "strength": frow.get("strength", ""),
                "confidence_0_1": frow.get("confidence_0_1", ""),
                "k_right": pv.get("k_right", ""),
                "k_left": pv.get("k_left", ""),
                "p_right": pv.get("p_right", ""),
                "p_left": pv.get("p_left", ""),
                "p_two": pv.get("p_two", frow.get("p_two", "")),
            })

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow(r)


def write_error_json(outdir: Path, code: str, message: str, extra: dict | None = None) -> None:
    """
    Minimal publish-safe error artifact. Written only on early-stop conditions.
    """
    payload = {"ok": False, "error_code": str(code), "error_message": str(message)}
    if isinstance(extra, dict) and extra:
        payload["extra"] = extra
    (outdir / "error.json").write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")

def make_n1_plot(outdir: Path, primary: Dict[str, object], A7: Dict[str, object]) -> None:
    scan = primary["scan"]
    centers = np.asarray(scan["centers"], dtype=float)
    kpsi = np.asarray(scan["kpsi"], dtype=float)

    plt.figure(figsize=(10, 4))
    plt.plot(centers, kpsi)
    plt.xlabel("Window center (sample index)")
    plt.ylabel("Kψ (z-score of lowfreq fraction)")
    plt.title("n1: Kψ(t) scan (real signal)")
    plt.tight_layout()
    plt.savefig(outdir / "n1_plot.png", dpi=150)
    plt.close()


def write_report_html(outdir: Path, contract: Dict[str, object], summary: Dict[str, object], primary: Dict[str, object], A7: Dict[str, object]) -> None:
    verdict = str(summary.get("verdict", ""))
    reason = str(summary.get("verdict_reason_text", ""))
    conf = summary.get("headline", {}).get("confidence_0_1", float("nan"))
    try:
        import numpy as _np
        conf = conf if _np.isfinite(float(conf)) else "(not computed)"
    except Exception:
        conf = "(not computed)"


    fam_rows = ""
    fams = summary.get("null_family_summaries", [])
    if isinstance(fams, list):
        for r in fams:
            fam_rows += "<tr>" + "".join(
                [
                    f"<td>{r.get('family','')}</td>",
                    f"<td>{r.get('n','')}</td>",
                    f"<td>{r.get('p_null_lt_real','')}</td>",
                    f"<td>{r.get('strength','')}</td>",
                    f"<td>{(r.get('pvals',{}) or {}).get('p_two','')}</td>",
                ]
            ) + "</tr>"

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>n1 report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; }}
    td, th {{ border: 1px solid #ccc; padding: 6px 8px; }}
    .banner {{ padding: 10px; border-radius: 6px; background: #f3f3f3; }}
  </style>
</head>
<body>
  <h1>{TOOLKIT_NAME}</h1>
  <p><b>Version:</b> {TOOLKIT_VERSION}</p>

  <div class="banner">
    <h2>Verdict: {verdict}</h2>
    <p>{reason}</p>
    <p><b>Asymmetry Confidence (0–1):</b> {conf}</p>
    <p style="font-size: 12px;">
      PASS/NO_CLEAR are results of this audit. This is not a health score, risk score, or root-cause diagnosis.
    </p>
  </div>

  <h2>Plot</h2>
  <p><img src="n1_plot.png" style="max-width: 900px;"></p>

  <h2>Required controls (A7)</h2>
  <table>
    <tr><th>Family</th><th>N</th><th>P(null &lt; real)</th><th>Strength</th><th>p_two</th></tr>
    {fam_rows}
  </table>

  <h2>Artifacts</h2>
  <ul>
    <li><a href="summary.json">summary.json</a></li>
    <li><a href="contract.json">contract.json</a></li>
    <li><a href="n1_primary.json">n1_primary.json</a></li>
    <li><a href="n1_null_table.csv">n1_null_table.csv</a></li>
    <li><a href="canonical_windows.csv">canonical_windows.csv</a></li>
    <li><a href="provenance.json">provenance.json</a></li>
  </ul>

  <h2>What this is NOT</h2>
  <ul>
    <li>Not a classifier, predictor, RUL model, root-cause engine, or safety certification.</li>
    <li>PASS/NO_CLEAR are results; they do not imply health or failure.</li>
  </ul>
</body>
</html>
"""
    (outdir / "report.html").write_text(html, encoding="utf-8")


# -----------------------------
# Index + publish blurbs (already defined in Part 1)
# -----------------------------
# -----------------------------
# Publish helpers (methods/caption/limitations blurbs)
# -----------------------------
def write_publish_blurbs(outdir: Path) -> None:
    """
    Write publish-safe explanatory blurbs.
    These are NOT used in verdict logic; they are reporting-only artifacts.
    """
    methods = (
        "Methods (v1): We compute a windowed low-frequency power fraction and transform it into Kpsi(t) via z-scoring across windows. "
        "We compare REAL median(Kpsi) against falsification-first null families: FFT phase scramble (spectrum-preserving) and block shuffle (time-order broken); "
        "plus time reversal as an additional deterministic control. "
        "All p-values are standardized empirical p = (k+1)/(M+1)."
    )
    caption = (
        "Figure: n1 time-order asymmetry audit. The Kpsi(t) curve is derived from the real signal and compared against null families. "
        "Separation is summarized by percentile strength and empirical two-sided p-values."
    )
    limitations = (
        "Limitations (v1): This is a detector/audit, not a diagnosis or predictor. PASS/NO_CLEAR does not imply health or failure. "
        "Results depend on data length, sampling, and whether null families are appropriate falsifiers for the acquisition pipeline."
    )
    (outdir / "methods_blurb.txt").write_text(methods, encoding="utf-8")
    (outdir / "caption_text.txt").write_text(caption, encoding="utf-8")
    (outdir / "limitations_blurb.txt").write_text(limitations, encoding="utf-8")


def write_index_files(outdir: Path) -> None:
    """
    Minimal index.json + index.html for browsing the evidence pack.
    Reporting-only.
    """
    files = []
    for name in planned_outputs():
        p = outdir / name
        if p.exists():
            files.append(name)

    index = {
        "toolkit": TOOLKIT_NAME,
        "version": TOOLKIT_VERSION,
        "outdir": str(outdir),
        "files": files,
    }
    (outdir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    lis = "\n".join([f'<li><a href="{fn}">{fn}</a></li>' for fn in files])
    html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"/><title>n1 index</title></head>
<body>
<h1>{TOOLKIT_NAME}</h1>
<p><b>Version:</b> {TOOLKIT_VERSION}</p>
<ul>
{lis}
</ul>
</body>
</html>
"""
    (outdir / "index.html").write_text(html, encoding="utf-8")




def write_legacy_headline_pack(outdir: Path, primary: dict, summary: dict) -> None:
    """
    Reporting-only artifact. NEVER affects verdict logic.
    Writes: legacy_headline_pack.json
    """
    try:
        pack = {
            "toolkit": TOOLKIT_NAME,
            "version": TOOLKIT_VERSION,
            "verdict": summary.get("verdict"),
            "logic_fingerprint": summary.get("logic_fingerprint"),
            "ArrowA": primary.get("ArrowA", {}),
            "diagnostic_only": True,
        }
        (outdir / "legacy_headline_pack.json").write_text(
            json.dumps(pack, indent=2), encoding="utf-8"
        )
    except Exception as e:
        # Never fail the run because a diagnostic artifact couldn't be written.
        # Keep silence in publish runs; provenance already captures core outputs.
        return
def write_text_artifacts(outdir: Path) -> None:
    write_publish_blurbs(outdir)


def write_index_json_html(outdir: Path, title: str) -> None:
    write_index_files(outdir)


def print_executive_summary(
    outdir: Path,
    contract: Dict[str, object],
    summary: Dict[str, object],
    artifact_list: List[str],
) -> None:
    fp = str(summary.get("logic_fingerprint", ""))
    verdict = str(summary.get("verdict", ""))
    reason_code = str(summary.get("verdict_reason_code", ""))
    reason_text = str(summary.get("verdict_reason_text", ""))

    print("\n=== EXECUTIVE SUMMARY ===")
    print(f"Toolkit: {contract.get('internal_name', TOOLKIT_INTERNAL_NAME)}")
    print(f"Version: {contract.get('version', TOOLKIT_VERSION)}")
    print(f"Outdir:  {outdir}")
    print(f"Logic fingerprint (decide_verdict): {fp}\n")

    headline = summary.get("headline", {})
    if isinstance(headline, dict):
        for k in ("real_kpsi_median", "real_kpsi_mean", "confidence_0_1"):
            if k in headline:
                if k == "confidence_0_1" and verdict.startswith(("NO_CLEAR", "INVALID")):
                    print("confidence_0_1: (not computed - publish gate)")
                else:
                    print(f"{k}: {headline[k]}")

    print()
    banner = ""
    if verdict not in ("PASS_STRONG", "PASS_WEAK"):
        banner = "  (FAIL/NO_CLEAR ARE RESULTS — see reason below)"
    print(f"Verdict: {verdict}{banner}")
    print(f"Reason:  {reason_code} — {reason_text}\n")

    controls = summary.get("controls_present", {})
    if isinstance(controls, dict):
        print("Controls:")
        for k, v in controls.items():
            print(f"  ✓ {k} : {bool(v)}")
        print()

    print("Artifacts:")
    print(f"  {outdir}")
    for f in artifact_list:
        print(f"  - {f}")
    print("=========================")



def explain_text() -> str:
    """
    Human-facing explanation (reporting-only). NEVER affects verdict logic.
    """
    return (
        f"{TOOLKIT_NAME}\n"
        f"Version: {TOOLKIT_VERSION}\n\n"
        "Purpose (LOCK): Detect whether a single time-series contains irreversible time-order structure\n"
        "beyond falsification-first surrogate controls.\n\n"
        "Required surrogate families for verdict (LOCK v1):\n"
        "  - phase_scramble_fft (spectrum-preserving)\n"
        "  - block_shuffle (breaks global time order, preserves within-block structure)\n\n"
        "Deterministic control (required-to-report):\n"
        "  - time_reversal (x reversed; deterministic; NOT a randomized surrogate family)\n\n"
        "Verdict policy (LOCK): uses empirical two-sided p_two from the required surrogate families.\n"
        "  PASS_STRONG if both p_two <= 0.01\n"
        "  PASS_WEAK   if both p_two <= 0.05\n"
        "  else NO_CLEAR\n\n"
        "Units note (LOCK): lowfreq_cut is normalized cycles/sample when fs=1.0 (default).\n\n"
        "What this toolkit must never claim (LOCK):\n"
        "  - failure/health/risk/prediction/root cause\n"
        "  - diagnosis or certification\n\n"
        "Outputs: contract.json, summary.json, canonical_windows.csv, n1_primary.json, n1_null_table.csv,\n"
        "         n1_plot.png, report.html, provenance.json, index.json, index.html, and publish blurbs.\n"
    )

# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    class HintingParser(argparse.ArgumentParser):
        def error(self, message):
            if "unrecognized arguments:" in message:
                bad = message.split("unrecognized arguments:")[-1].strip().split()[0]
                candidates = []
                for a in self._actions:
                    for opt in getattr(a, "option_strings", []):
                        candidates.append(opt)
                suggestion = difflib.get_close_matches(bad, candidates, n=1)
                if suggestion:
                    message += f"\nDid you mean: {suggestion[0]} ?"
            super().error(message)

    p = HintingParser(
        prog="n1_time-order_toolkit.py",
        description="ψ4 n1 time-order asymmetry audit (publish-safe v1).",
        add_help=True,
    )

    p.add_argument("--infile", required=False, help="Input CSV file path.")
    p.add_argument("--sig-col", default="x", help="Signal column name (default: x).")
    p.add_argument("--time-col", default=None, help="Optional time column.")
    p.add_argument("--idx-col", default=None, help="Alias for --time-col (kept for compatibility).")

    p.add_argument("--out", required=False, help="Output directory.")
    p.add_argument("--outdir", default=None, help="Alias for --out (kept for compatibility).")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block-len", type=int, default=1024)
    p.add_argument("--n-phase-null", type=int, default=50)
    p.add_argument("--n-block-null", type=int, default=50)

    p.add_argument("--timeline-win", type=int, default=8192)
    p.add_argument("--timeline-step", type=int, default=2048)
    p.add_argument("--lowfreq-cut", type=float, default=0.1)

    p.add_argument("--check", action="store_true", help="Preflight only; no computation, no files.")
    p.add_argument("--dry-run", action="store_true", help="Print planned outputs; no computation, no files.")
    p.add_argument("--explain", action="store_true", help="Print what/when/means/NOT and exit.")
    p.add_argument("--output-mode", choices=["publish-min", "standard", "full-audit"], default="standard",
                   help="Artifact volume control only. Does not change math, verdicts, or required core outputs.")

    p.add_argument("--dataset-tag", default="", help="Optional dataset tag (for reporting only).")
    p.add_argument("--selftest", action="store_true", help="Run internal regression self-test and exit.")

    return p


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.outdir and args.out:
        raise SystemExit("[ERROR] Provide only one of --out or --outdir (alias).")
    if args.outdir and not args.out:
        print("[WARN] --outdir is an alias for --out (kept for compatibility).")
        args.out = args.outdir

    if args.idx_col and args.time_col:
        raise SystemExit("[ERROR] Provide only one of --time-col or --idx-col (alias).")
    if args.idx_col and not args.time_col:
        print("[WARN] --idx-col is an alias for --time-col (kept for compatibility).")
        args.time_col = args.idx_col

    return args


def run_check_or_dryrun(args: argparse.Namespace) -> int:
    if args.explain:
        print(explain_text())
        return 0

    if not args.infile:
        raise SystemExit("[ERROR] --infile is required (unless --explain).")
    if not args.out and not args.check and not args.dry_run:
        raise SystemExit("[ERROR] --out is required for compute runs.")

    infile = Path(args.infile)
    if not infile.exists():
        raise SystemExit(f"[ERROR] Input not found: {infile}")

    # Preflight (array-based; time optional)
    pf = preflight_csv(infile, time_col=(args.time_col or "t"), sig_col=args.sig_col)
    if not bool(pf.get("ok", False)):
        raise SystemExit(f"[ERROR] {pf.get('error', 'Preflight failed')}")

    if args.check:
        print("=== CHECK ONLY (no computation, no files) ===")
        print(f"Input: {infile}")
        print("============================================")
        return 0

    if args.dry_run:
        outdir = Path(args.out) if args.out else Path(".")
        print("=== DRY RUN (no computation, no files) ===")
        print(f"Outdir: {outdir}")
        print(f"Output mode: {args.output_mode}")
        print("Planned outputs:")
        for f in planned_outputs_for_mode(args.output_mode):
            print(f"  - {f}")
        print("==========================================")
        return 0

    return -1



def run_selftest() -> None:
    """
    Internal regression self-test (publish-safe).

    Purpose:
      - Ensures evidence pack is produced
      - Ensures locked schema invariants hold
      - Ensures required controls / reversal fields are present
    """
    import os
    import sys
    import json
    import shutil
    import subprocess
    from pathlib import Path

    print("=== SELFTEST (n1) ===")

    # Paths
    script_path = Path(__file__).resolve()
    scratch_dir = script_path.parent
    repo_root = scratch_dir.parent if scratch_dir.name.lower() == "scratch" else scratch_dir
    infile = scratch_dir / "_selftest_n1.csv"
    outdir = repo_root / "out_selftest_n1"

    # Build deterministic synthetic dataset (enough length for windowing)
    # Keep it simple: fixed waveform + tiny noise (seeded).
    import numpy as np
    rng = np.random.default_rng(0)
    n = 2048
    t = np.arange(n, dtype=float)
    x = np.sin(2 * np.pi * t / 64.0) + 0.25 * np.sin(2 * np.pi * t / 19.0)
    x = x + 0.01 * rng.standard_normal(n)

    infile.write_text("t,x\n" + "\n".join(f"{i},{float(x[i])}" for i in range(n)) + "\n", encoding="utf-8")
    print("Wrote:", str(infile))

    # Fresh outdir
    if outdir.exists():
        shutil.rmtree(outdir, ignore_errors=True)

    # Execute the toolkit via subprocess (tests real CLI path)
    cmd = [
        sys.executable,
        str(script_path),
        "--infile", str(infile),
        "--out", str(outdir),
        "--seed", "0",
        "--n-phase-null", "8",
        "--n-block-null", "8",
        "--timeline-win", "128",
        "--timeline-step", "32",
    ]
    print("Running:", " ".join(cmd))
    
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

    # Publish-safe (Option B): allow 0 (PASS_*/NO_CLEAR) or 2 (publish-gate/invalid)
    if r.returncode not in (0, 2):
        print("SELFTEST FAIL: unexpected returncode =", r.returncode)
        print("---- STDOUT ----")
        print(r.stdout)
        print("---- STDERR ----")
        print(r.stderr)
        raise SystemExit(2)

    # If we produced an evidence pack, verdict must exist and match exit policy.
    # (Summary is written by the toolkit run itself.)
    try:
        sj = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
        v = str(sj.get("verdict", ""))
    except Exception as e:
        print("SELFTEST FAIL: could not read/parse summary.json:", e)
        raise SystemExit(2)

    if v.startswith("PASS"):
        if r.returncode != 0:
            print("SELFTEST FAIL: verdict is PASS but returncode != 0")
            print("verdict:", v, "returncode:", r.returncode)
            raise SystemExit(2)
    elif v == "NO_CLEAR_MISSING_CONTROLS":
        if r.returncode != 2:
            print("SELFTEST FAIL: verdict is NO_CLEAR_MISSING_CONTROLS but returncode != 2")
            print("verdict:", v, "returncode:", r.returncode)
            raise SystemExit(2)
    elif v.startswith("NO_CLEAR"):
        if r.returncode != 0:
            print("SELFTEST FAIL: verdict is NO_CLEAR but returncode != 0")
            print("verdict:", v, "returncode:", r.returncode)
            raise SystemExit(2)
    else:
        print("SELFTEST FAIL: unknown verdict label:", v)
        raise SystemExit(2)

    # Required evidence pack files
    required = [
        "contract.json",
        "summary.json",
        "n1_primary.json",
        "n1_null_table.csv",
        "canonical_windows.csv",
        "n1_plot.png",
        "report.html",
        "provenance.json",
        "index.html",
    ]
    missing_files = [name for name in required if not (outdir / name).exists()]
    if missing_files:
        raise SystemExit(f"SELFTEST FAIL: missing files: {missing_files}")

    # Parse outputs
    primary = json.loads((outdir / "n1_primary.json").read_text(encoding="utf-8"))
    summary = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))

    # Schema invariants
    if "A7" not in primary or not isinstance(primary["A7"], dict):
        raise SystemExit("SELFTEST FAIL: primary missing dict A7")
    A7 = primary["A7"]

    # Top-level reversal keys must exist
    for k in ("time_reversal_mean", "time_reversal_median", "time_reversal_pass"):
        if k not in A7:
            raise SystemExit(f"SELFTEST FAIL: A7 missing required key: {k}")

    # Controls invariants
    cp = summary.get("controls_present", {})
    if not isinstance(cp, dict) or ("time_reversal" not in cp):
        raise SystemExit("SELFTEST FAIL: summary.controls_present missing 'time_reversal' key")

    # time_reversal presence should reflect whether A7 time_reversal_median is finite
    import numpy as np
    rev_med = A7.get("time_reversal_median", float("nan"))
    expected = bool(np.isfinite(float(rev_med))) if rev_med is not None else False
    if bool(cp.get("time_reversal", False)) != expected:
        raise SystemExit(f"SELFTEST FAIL: time_reversal present mismatch: controls_present={cp.get('time_reversal')} expected={expected}")

    mc = summary.get("missing_controls", None)
    if mc != []:
        raise SystemExit(f"SELFTEST FAIL: summary.missing_controls expected [], got {mc}")

    # Reason code should be one of the known keys
    rc = summary.get("verdict_reason_code", "")
    if not isinstance(rc, str) or rc not in REASON_TEXT_MAP:
        raise SystemExit(f"SELFTEST FAIL: unknown verdict_reason_code: {rc}")

    
    # 2) GATE/INVALID case: required null families missing (must fail-fast, no pack)
    outdir_invalid = repo_root / "out_selftest_n1_invalid"
    if outdir_invalid.exists():
        shutil.rmtree(outdir_invalid, ignore_errors=True)

    cmd_invalid = [
        sys.executable,
        str(script_path),
        "--infile", str(infile),
        "--out", str(outdir_invalid),
        "--seed", "0",
        "--n-phase-null", "0",
        "--n-block-null", "0",
        "--timeline-win", "128",
        "--timeline-step", "32",
    ]
    print("[SELFTEST] running GATE/INVALID case")
    r_inv = subprocess.run(cmd_invalid, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r_inv.returncode != 2:
        print("SELFTEST FAIL: gate/invalid expected exitcode=2, got", r_inv.returncode)
        print("---- STDOUT ----"); print(r_inv.stdout)
        print("---- STDERR ----"); print(r_inv.stderr)
        raise SystemExit(2)

    msg = (r_inv.stdout or "") + "\n" + (r_inv.stderr or "")
    if "Publish gate failed" not in msg:
        print("SELFTEST FAIL: gate/invalid did not print publish-gate error line")
        print("---- STDOUT ----"); print(r_inv.stdout)
        print("---- STDERR ----"); print(r_inv.stderr)
        raise SystemExit(2)

    # Publish hygiene: invalid run must not print valid-looking headline numbers (nor NaN)
    lowered = msg.lower()
    if ("real_kpsi_median" in lowered) or ("confidence_0_1" in lowered) or ("nan" in lowered):
        print("SELFTEST FAIL: gate/invalid printed headline fields or NaN (publish hygiene violated)")
        print("---- STDOUT ----"); print(r_inv.stdout)
        print("---- STDERR ----"); print(r_inv.stderr)
        raise SystemExit(2)

    # Evidence pack must NOT be written on publish-gate stop
    if (outdir_invalid / "summary.json").exists():
        raise SystemExit("SELFTEST FAIL: gate/invalid wrote summary.json (should have exited before writing).")

    print("SELFTEST PASS")
    print("================")

def main() -> int:
    t0 = time.time()
    parser = build_argparser()
    args = resolve_args(parser.parse_args())

    # Internal selftest
    if getattr(args, "selftest", False):
        run_selftest()
        return 0

    # Explain / check / dry-run + preflight (treat failures as publish-invalid)
    try:
        rc = run_check_or_dryrun(args)
    except SystemExit as e:
        msg = str(e)
        if msg:
            print(msg)
        return 2
    if rc >= 0:
        return int(rc)

    # Publish gate: required null families must be viable BEFORE headline
    missing = []
    if int(getattr(args, "n_phase_null", 0)) < 1:
        missing.append("phase_scramble_fft")
    if int(getattr(args, "n_block_null", 0)) < 1:
        missing.append("block_shuffle")
    if missing:
        log("[ERROR] Publish gate failed: required null families missing/zero reps: " + ", ".join(missing))
        return 2

    # Compute run
    infile = Path(args.infile)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    contract = frozen_contract()

    primary, x = compute_n1_primary(infile, args)
    A7 = compute_A7_null_audit(primary, x, args)
    primary["A7"] = A7  # persist evidence

    verdict_obj = decide_verdict(contract, A7)
    summary = build_summary_n1(contract, primary, A7, verdict_obj)
    prov = build_provenance(args, infile)


    # Non-finite headline publish gate:
    # If headline would be NaN/inf (e.g., too few windows / degenerate data), do NOT write a full evidence pack.
    _hm = summary.get("headline", {}) if isinstance(summary, dict) else {}
    _bad = []
    try:
        import numpy as _np
        for _k in ("real_kpsi_median", "confidence_0_1"):
            _v = _hm.get(_k, float("nan")) if isinstance(_hm, dict) else float("nan")
            if not _np.isfinite(float(_v)):
                _bad.append(_k)
    except Exception:
        _bad = ["headline"]

    if _bad:
        # Write only minimal artifacts for debugging + audit
        (outdir / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
        (outdir / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
        write_error_json(outdir, "PREFLIGHT_NONFINITE_HEADLINE",
                         "Non-finite headline fields (likely insufficient windows/degenerate data).",
                         {"bad_fields": _bad})
        log("[ERROR] Publish gate failed: non-finite headline fields: " + ", ".join(_bad))
        return 2
    # Write artifacts
    # Reporting-only legacy evidence pack (diagnostic-only; NEVER affects verdict).
    write_legacy_headline_pack(outdir, primary, summary)

    (outdir / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    (outdir / "n1_primary.json").write_text(json.dumps(primary, indent=2), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")

    write_canonical_windows_csv(outdir, primary)
    write_n1_null_table_csv(outdir, A7)

    make_n1_plot(outdir, primary, A7)
    write_report_html(outdir, contract, summary, primary, A7)

    write_text_artifacts(outdir)
    write_index_json_html(outdir, title=f"{TOOLKIT_NAME} v{TOOLKIT_VERSION}")
    prune_optional_artifacts_n1(outdir, args.output_mode)

    # One-screen trust output
    print_executive_summary(outdir, contract, summary, planned_outputs_for_mode(args.output_mode))

    # Exit policy (LOCK v1.0.0 semantics):
    #   0 = PASS_* / FAIL_* (completed run)
    #   0 = PASS_* / FAIL_* / NO_CLEAR (completed run; NO_CLEAR is a valid outcome)
    #   2 = publish-gate / invalid / contract failure (missing required controls, bad config, etc.)
    v = summary.get("verdict", "") if isinstance(summary, dict) else ""

    if isinstance(v, str):
        if v.startswith("PASS") or v.startswith("FAIL"):
            return 0
        if v == "NO_CLEAR_MISSING_CONTROLS":
            return 2
        if v.startswith("NO_CLEAR"):
            return 0

    # Defensive fallback: unknown verdict label => treat as invalid
    return 2

if __name__ == "__main__":
    try:
        rc = main()
        if rc is None:
            rc = 0
        # Enforce only {0,2}
        if rc not in (0, 2):
            rc = 2
        sys.exit(rc)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 2
        if code not in (0, 2):
            code = 2
        sys.exit(code)
    except Exception as e:
        print(f"[ERROR] INVALID_RUN_exception: {type(e).__name__}: {e}")
        sys.exit(2)
