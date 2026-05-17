#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ψ4 n₂ Interaction (Handshake) Toolkit — v1.0.0-lock (publish-safe)
==================================================================

FROZEN OBJECT (P0)
- Residual: r(t) = wrap(φ2(t) − 2 φ1(t))  in (-pi, pi]
- Headline: circR = | < exp(i r) > |     (ONLY claim-driving number)
- Diagnostics (never claim-driving): H2_mean, H2_MAD

FROZEN NULL FAMILIES (P1)
- phase_scramble   : preferred FFT-phase randomization if raw signals exist; else phase-increment scramble
- block_shuffle    : block reorder (destroys global time order, preserves local structure)
- mismatch_pairing : MANDATORY (decisive falsifier). Implemented as random circular shifts of φ2 vs φ1.

NON-NEGOTIABLE V1 RULE
- mismatch_pairing MUST run every time with n_mismatch >= 50.
  If mismatch cannot run -> verdict = INVALID_RUN (hard fail). No silent downgrade.

P2 (Multi-scale stability)
- Default block sweep: 64,128,256,512,1024 (log2 scale)
- Reports null means/stds vs block_len

OUTPUT EVIDENCE PACK (always written in normal runs; minimal pack for --check/--dry-run/--explain)
- contract.json
- summary.json
- provenance.json
- canonical_phases.csv
- n2_null_table.csv
- n2_ablation_table.csv
- n2_multiscale_table.csv
- circR_vs_time.csv
- n2_plot.png
- n2_multiscale_plot.png
- circR_vs_time.png
- report.html
- index.json
- index.html
- methods_blurb.txt
- caption_text.txt
- limitations_blurb.txt
- (optional) reversal_metrics.json  (diagnostic only)
- (optional) n2_primary.json  (machine-readable convenience)

Verdict labels (frozen, no tuning knobs)
- PASS_STRONG : mismatch breaks directionally AND (phase_scramble OR block_shuffle) also breaks
- PASS_WEAK   : mismatch breaks directionally, but phase_scramble and block_shuffle do not both/any break
- FAIL        : mismatch does NOT break directionally
- NO_CLEAR    : circR_real is NaN / insufficient data
- INVALID_RUN : mismatch missing / n_mismatch < 50 / cannot execute required controls

Direction-only break rule (no tuned thresholds):
- breaks_directional = (mean_null < circR_real) OR (P(null < real) > 0.5)

This toolkit is a coordination/interaction audit. It is NOT fault prediction, RUL, or root-cause attribution.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import hilbert
import difflib


# -----------------------------
# Identity / Contract (P0)
# -----------------------------

TOOLKIT_NAME = "ψ4 n₂ Interaction (Handshake) Toolkit"
TOOLKIT_VERSION = "1.0.0-lock"

FREEZE_STATEMENT = "All claims and verdicts are driven by circR. H2_mean and H2_MAD are diagnostic only."

FROZEN_CLAIM = (
    "The frozen n₂ observable detects non-random phase-coupling structure between two synchronized streams, "
    "and this structure collapses under nulls that destroy coupling (phase scrambling, time-order shuffles, mismatched pairing (within-record circular shift))."
)

CLAIM_CONTRACT = {
    "toolkit": "n2_interaction",
    "version": TOOLKIT_VERSION,
    "headline_metric": "circR",
    "headline_label": "Interaction Strength (circR)",
    "frozen_residual": "r(t)=wrap(phi2(t)-2*phi1(t))",
    "diagnostics": ["H2_mean", "H2_MAD"],
    "diagnostic_only": ["H2_mean", "H2_MAD"],
    "freeze_statement": FREEZE_STATEMENT,
    "directional_claim": FROZEN_CLAIM,
    "required_null_families_v1": ["phase_scramble", "block_shuffle", "mismatch_pairing"],
    "required_mismatch_min_reps_v1": 50,
    "verdict_labels_v1": ["PASS_STRONG", "PASS_WEAK", "FAIL", "NO_CLEAR", "INVALID_RUN"],
    "break_rule_v1": "breaks if (mean_null < real) OR (P(null < real) > 0.5); mismatch is decisive and mandatory.",
    "never_claims": [
        "fault detection",
        "failure mode classification",
        "root cause attribution",
        "prediction",
        "remaining useful life (RUL)",
        "event forecasting",
    ],
}

REASON_TEXT_MAP: Dict[str, str] = {
    # decide_verdict reasons (frozen codes)
    "null_table_empty": "Null table missing/empty; cannot evaluate required controls.",
    "missing_required_nulls": "Required null families missing or had zero reps.",
    "mismatch_did_not_break": "Mismatch pairing did not reduce circR; coupling not alignment-dependent under this test. (Mismatch = within-record circular shift pairing.)",
    "mismatch_and_other_break": "Mismatch breaks and at least one other null family also breaks. (Mismatch = within-record circular shift pairing.)",
    "mismatch_only_break": "Mismatch breaks but other nulls do not; weaker directional evidence. (Mismatch = within-record circular shift pairing.)",
    # non-verdict helper reasons used in early exits
    "insufficient_data_or_nan": "Insufficient samples or circR is NaN; cannot evaluate.",
    "n_mismatch_null_below_v1_min": "Mismatch null reps below v1 minimum; run invalid by contract.",
}

METHODS_BLURB = (
    "ψ4 n₂ Interaction Toolkit (v1.0-lock)\n"
    "- Residual: r(t)=wrap(φ2(t)−2φ1(t))\n"
    "- Headline: circR = |⟨exp(i r)⟩|\n"
    "- Verdict uses circR ONLY. Diagnostics are non-claim.\n"
    "- Required falsifier: mismatch_pairing (random circular shifts of φ2 vs φ1).\n"
)

CAPTION_TEXT = (
    "n₂ interaction audit. Higher circR indicates stronger phase-lock consistency of φ2 relative to 2·φ1. "
    "PASS indicates alignment-dependent coupling that collapses under mismatch nulls (within-record circular shift); "
    "FAIL indicates coupling did not collapse under mismatch (not alignment-dependent under this test)."
)

LIMITATIONS_BLURB = (
    "Limitations / Safety:\n"
    "- This toolkit does NOT detect faults or predict failures.\n"
    "- PASS/FAIL refers only to alignment-dependent coordination under the frozen residual.\n"
    "- Results depend on phase extraction quality (Hilbert) and on the chosen streams.\n"
    "- Use alongside domain baselines; do not treat as operational safety certification.\n"
)


N2_PLANNED_OUTPUTS = [
    "contract.json",
    "summary.json",
    "provenance.json",
    "canonical_phases.csv",
    "n2_null_table.csv",
    "n2_ablation_table.csv",
    "n2_multiscale_table.csv",
    "circR_vs_time.csv",
    "n2_plot.png",
    "n2_multiscale_plot.png",
    "circR_vs_time.png",
    "report.html",
    "index.json",
    "index.html",
    "methods_blurb.txt",
    "caption_text.txt",
    "limitations_blurb.txt",
    "n2_primary.json",
]

N2_PUBLISH_MIN_OPTIONAL = [
    "n2_ablation_table.csv",
    "n2_multiscale_table.csv",
    "circR_vs_time.csv",
    "n2_multiscale_plot.png",
    "circR_vs_time.png",
    "index.json",
    "methods_blurb.txt",
    "caption_text.txt",
    "limitations_blurb.txt",
]

def planned_outputs_for_mode(output_mode: str = "standard", reversal_check: bool = False) -> list[str]:
    mode = str(output_mode or "standard").strip().lower()
    outs = list(N2_PLANNED_OUTPUTS)
    if reversal_check:
        outs.append("reversal_metrics.json")
    if mode == "publish-min":
        outs = [x for x in outs if x not in set(N2_PUBLISH_MIN_OPTIONAL)]
    return outs

def prune_optional_artifacts_n2(outdir: Path, output_mode: str, reversal_check: bool = False) -> None:
    mode = str(output_mode or "standard").strip().lower()
    if mode != "publish-min":
        return
    for name in N2_PUBLISH_MIN_OPTIONAL:
        fp = outdir / name
        try:
            if fp.exists():
                fp.unlink()
        except Exception:
            # Never break a valid scientific run over optional artifact pruning
            pass

# -----------------------------
# Utils
# -----------------------------

def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _nan_to_none(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def wrap_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    x = np.asarray(x, dtype=float)
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def mad(x: np.ndarray) -> float:
    """Median absolute deviation around median."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    m = np.median(x)
    return float(np.median(np.abs(x - m)))


def circR_from_residual(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    return float(np.abs(np.mean(np.exp(1j * r))))


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def logic_fingerprint_decide_verdict() -> str:
    """
    Trust feature (patch-only): hash of the *actual* frozen decide_verdict source at runtime.
    This is NOT used for any scientific decision.
    """
    try:
        src = inspect.getsource(decide_verdict)
    except Exception:
        return "unavailable"
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def instantaneous_phase_from_signal(sig: np.ndarray) -> np.ndarray:
    """Hilbert analytic phase (wrapped)."""
    sig = np.asarray(sig, dtype=float)
    sig = sig - np.nanmean(sig)
    if np.any(~np.isfinite(sig)):
        m = np.nanmean(sig)
        sig = np.where(np.isfinite(sig), sig, m)
    a = hilbert(sig)
    return wrap_pi(np.angle(a))


def fft_phase_scramble(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """FFT phase randomization: preserves amplitude spectrum, randomizes phases."""
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    if np.any(~np.isfinite(x)):
        m = np.nanmean(x)
        x = np.where(np.isfinite(x), x, m)

    X = np.fft.rfft(x)
    mag = np.abs(X)
    rnd_phase = rng.uniform(0.0, 2.0 * np.pi, size=X.shape)
    Xn = mag * np.exp(1j * rnd_phase)
    y = np.fft.irfft(Xn, n=x.size)
    return y.astype(float)


def phase_scramble_increments(phi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-increment scramble: preserves increment distribution but destroys temporal order."""
    phi = np.asarray(phi, dtype=float)
    d = wrap_pi(np.diff(phi, prepend=phi[0]))
    d_perm = d[rng.permutation(d.size)]
    out = wrap_pi(np.cumsum(d_perm))
    return out


def block_shuffle(x: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Shuffle contiguous blocks; preserves within-block structure."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n <= 1:
        return x.copy()
    if block_len <= 1 or block_len >= n:
        return x[rng.permutation(n)]

    n_blocks = n // block_len
    cut = n_blocks * block_len
    blocks = x[:cut].reshape(n_blocks, block_len)
    perm = rng.permutation(n_blocks)
    out = blocks[perm].reshape(-1)
    if cut < n:
        out = np.concatenate([out, x[cut:]], axis=0)
    return out


def percentile_null_below_real(real: float, null_samples: np.ndarray) -> float:
    """Return P(null < real)."""
    null_samples = np.asarray(null_samples, dtype=float)
    null_samples = null_samples[np.isfinite(null_samples)]
    if null_samples.size == 0 or not np.isfinite(real):
        return float("nan")
    return float(np.mean(null_samples < real))


def empirical_p_value_ge_real(real: float, null_samples: np.ndarray) -> float:
    """Conservative empirical p-value: (k+1)/(M+1), k = #null >= real."""
    null_samples = np.asarray(null_samples, dtype=float)
    null_samples = null_samples[np.isfinite(null_samples)]
    if null_samples.size == 0 or not np.isfinite(real):
        return float("nan")
    k = int(np.sum(null_samples >= real))
    M = int(null_samples.size)
    return float((k + 1) / (M + 1))


def directional_breaks(real: float, null_mean: float, pct_null_below_real: float) -> bool:
    """
    Frozen, no-threshold rule (directional only):
      breaks if mean_null < real OR P(null < real) > 0.5
    """
    if np.isfinite(null_mean) and np.isfinite(real) and (null_mean < real):
        return True
    if np.isfinite(pct_null_below_real) and pct_null_below_real > 0.5:
        return True
    return False


def log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg, flush=True)


# -----------------------------
# Core computations
# -----------------------------

@dataclass
class RealMetrics:
    n: int
    circR: float
    H2_mean: float
    H2_MAD: float


def compute_real(phi1: np.ndarray, phi2: np.ndarray) -> Tuple[np.ndarray, RealMetrics]:
    n = int(min(phi1.size, phi2.size))
    phi1 = np.asarray(phi1[:n], dtype=float)
    phi2 = np.asarray(phi2[:n], dtype=float)
    r = wrap_pi(phi2 - 2.0 * phi1)
    m = RealMetrics(
        n=n,
        circR=_safe_float(circR_from_residual(r)),
        H2_mean=_safe_float(np.nanmean(r)),
        H2_MAD=_safe_float(mad(r)),
    )
    return r, m


def compute_timeline(r: np.ndarray, idx: np.ndarray, win: int, step: int) -> pd.DataFrame:
    r = np.asarray(r, dtype=float)
    idx = np.asarray(idx, dtype=float)
    n = r.size

    win = int(max(16, win))
    step = int(max(1, step))
    win = min(win, n)

    rows = []
    k = 0
    for s in range(0, n - win + 1, step):
        rr = r[s:s + win]
        rows.append(
            {
                "k": k,
                "start": int(s),
                "end": int(s + win - 1),
                "idx_start": _safe_float(idx[s]),
                "idx_end": _safe_float(idx[s + win - 1]),
                "circR": _safe_float(circR_from_residual(rr)),
            }
        )
        k += 1
    return pd.DataFrame(rows)


def sample_null_residual(
    family: str,
    phi1: np.ndarray,
    phi2: np.ndarray,
    block_len: int,
    rng: np.random.Generator,
    *,
    sig1: Optional[np.ndarray] = None,
    sig2: Optional[np.ndarray] = None,
    mismatch_min_shift: Optional[int] = None,
) -> np.ndarray:
    """
    Return one null residual sample r_null.
    """
    n = int(min(phi1.size, phi2.size))
    phi1 = phi1[:n]
    phi2 = phi2[:n]

    if family == "phase_scramble":
        if sig1 is not None and sig2 is not None:
            s1n = fft_phase_scramble(sig1[:n], rng)
            s2n = fft_phase_scramble(sig2[:n], rng)
            p1n = instantaneous_phase_from_signal(s1n)
            p2n = instantaneous_phase_from_signal(s2n)
            return wrap_pi(p2n - 2.0 * p1n)
        p1n = phase_scramble_increments(phi1, rng)
        p2n = phase_scramble_increments(phi2, rng)
        return wrap_pi(p2n - 2.0 * p1n)

    if family == "block_shuffle":
        p1n = block_shuffle(phi1, block_len, rng)
        p2n = block_shuffle(phi2, block_len, rng)
        return wrap_pi(p2n - 2.0 * p1n)

    if family == "mismatch_pairing":
        # MANDATORY decisive falsifier: circular shift of φ2 relative to φ1
        N = n
        if N < 10:
            return np.full((N,), np.nan, dtype=float)

        ms = mismatch_min_shift
        if ms is None:
            ms = int(max(1, block_len))
        ms = int(ms)
        if ms >= N:
            ms = max(1, N // 2)

        s = int(rng.integers(ms, N))
        phi2m = np.roll(phi2, s)
        return wrap_pi(phi2m - 2.0 * phi1)

    raise ValueError(f"Unknown null family: {family}")


def run_null_family(
    family: str,
    phi1: np.ndarray,
    phi2: np.ndarray,
    block_len: int,
    n_reps: int,
    rng: np.random.Generator,
    *,
    sig1: Optional[np.ndarray] = None,
    sig2: Optional[np.ndarray] = None,
    mismatch_min_shift: Optional[int] = None,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Run null reps. Returns stats dict and raw circR samples.
    """
    n_reps = int(n_reps)
    circ_samples = []
    mean_samples = []
    mad_samples = []

    for _ in range(n_reps):
        r_null = sample_null_residual(
            family,
            phi1,
            phi2,
            block_len,
            rng,
            sig1=sig1,
            sig2=sig2,
            mismatch_min_shift=mismatch_min_shift,
        )
        circ_samples.append(circR_from_residual(r_null))
        mean_samples.append(_safe_float(np.nanmean(r_null)))
        mad_samples.append(_safe_float(mad(r_null)))

    c = np.asarray(circ_samples, dtype=float)
    m = np.asarray(mean_samples, dtype=float)
    d = np.asarray(mad_samples, dtype=float)

    stats = {
        "n_reps": int(n_reps),
        "circR_null_mean": _safe_float(np.nanmean(c)),
        "circR_null_std": _safe_float(np.nanstd(c, ddof=1) if c.size > 1 else 0.0),
        "H2_mean_null_mean": _safe_float(np.nanmean(m)),
        "H2_mean_null_std": _safe_float(np.nanstd(m, ddof=1) if m.size > 1 else 0.0),
        "H2_MAD_null_mean": _safe_float(np.nanmean(d)),
        "H2_MAD_null_std": _safe_float(np.nanstd(d, ddof=1) if d.size > 1 else 0.0),
    }
    return stats, c


# -----------------------------
# Plotting
# -----------------------------

def plot_residual_distributions(
    outdir: Path,
    r_real: np.ndarray,
    r_phase: Optional[np.ndarray],
    r_block: Optional[np.ndarray],
    r_mismatch: Optional[np.ndarray],
    bins: int = 72,
) -> None:
    plt.figure(figsize=(8, 4))
    rr = r_real[np.isfinite(r_real)]
    plt.hist(rr, bins=bins, density=True, alpha=0.50, label="REAL residual")

    if r_phase is not None:
        rp = r_phase[np.isfinite(r_phase)]
        plt.hist(rp, bins=bins, density=True, alpha=0.30, label="NULL phase-scramble")

    if r_block is not None:
        rb = r_block[np.isfinite(r_block)]
        plt.hist(rb, bins=bins, density=True, alpha=0.30, label="NULL block-shuffle")

    if r_mismatch is not None:
        rm = r_mismatch[np.isfinite(r_mismatch)]
        plt.hist(rm, bins=bins, density=True, alpha=0.30, label="NULL mismatch pairing")

    plt.title("n₂ residual distribution: REAL vs nulls")
    plt.xlabel("r(t) = wrap(φ₂(t) − 2 φ₁(t))  (radians)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "n2_plot.png", dpi=160)
    plt.close()


def plot_timeline(outdir: Path, tl: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 3.8))
    if len(tl) > 0:
        plt.plot(
            tl["start"].to_numpy(),
            tl["circR"].to_numpy(),
            marker="o",
            linestyle="-",
            label="circR (windowed)",
        )
        plt.legend()
    plt.xlabel("sample_start")
    plt.ylabel("Interaction Strength (circR)")
    plt.title("n₂ sliding-window interaction strength timeline")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "circR_vs_time.png", dpi=180)
    plt.close()


def plot_multiscale(outdir: Path, df: pd.DataFrame) -> None:
    plt.figure(figsize=(8.5, 4.2))
    if df.empty:
        plt.text(0.5, 0.5, "No multiscale results", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(outdir / "n2_multiscale_plot.png", dpi=170)
        plt.close()
        return

    x = df["block_len"].to_numpy(dtype=float)
    real = df["circR_real"].to_numpy(dtype=float)
    plt.plot(x, real, marker="o", linestyle="-", label="REAL")

    for fam, lab in [
        ("phase_scramble", "NULL phase-scramble"),
        ("block_shuffle", "NULL block-shuffle"),
        ("mismatch_pairing", "NULL mismatch"),
    ]:
        mu = df[f"{fam}_mean"].to_numpy(dtype=float)
        sd = df[f"{fam}_std"].to_numpy(dtype=float)
        plt.errorbar(x, mu, yerr=sd, marker="o", linestyle="--", capsize=3, label=lab)

    plt.xscale("log", base=2)
    plt.xticks(x, [str(int(v)) for v in x])
    plt.xlabel("block_len")
    plt.ylabel("circR")
    plt.title("n₂ multi-scale robustness: REAL vs null families")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "n2_multiscale_plot.png", dpi=170)
    plt.close()


# -----------------------------
# Verdict + Reporting (core verdict logic unchanged)
# -----------------------------

def decide_verdict(null_table: pd.DataFrame, required_families: List[str]) -> Tuple[str, Dict[str, object]]:
    """
    mismatch_pairing is decisive and mandatory.

    Returns:
      verdict_label, details dict
    """
    details: Dict[str, object] = {"required_families": list(required_families), "missing": [], "family_breaks": {}}

    if null_table is None or null_table.empty:
        return "INVALID_RUN", {**details, "reason": "null_table_empty"}

    # ensure required families present with reps>0
    fam_rows = {str(r["family"]): r for _, r in null_table.iterrows()}
    for fam in required_families:
        if fam not in fam_rows:
            details["missing"].append(fam)
            continue
        if int(fam_rows[fam].get("n_reps", 0)) <= 0:
            details["missing"].append(fam)

    if details["missing"]:
        return "INVALID_RUN", {**details, "reason": "missing_required_nulls"}

    # pull break flags
    for fam in required_families:
        details["family_breaks"][fam] = bool(fam_rows[fam].get("breaks_directional", False))

    mismatch_breaks = bool(details["family_breaks"].get("mismatch_pairing", False))
    if mismatch_breaks is False:
        return "FAIL", {**details, "reason": "mismatch_did_not_break"}

    # mismatch breaks: strong if at least one other family also breaks
    phase_breaks = bool(details["family_breaks"].get("phase_scramble", False))
    block_breaks = bool(details["family_breaks"].get("block_shuffle", False))
    if phase_breaks or block_breaks:
        return "PASS_STRONG", {**details, "reason": "mismatch_and_other_break"}
    return "PASS_WEAK", {**details, "reason": "mismatch_only_break"}


def make_report_html(
    outdir: Path,
    meta: dict,
    contract: dict,
    summary: dict,
    null_table: pd.DataFrame,
    ablation_df: pd.DataFrame,
    multiscale_df: pd.DataFrame,
) -> None:
    def _img_tag(name: str) -> str:
        p = outdir / name
        return (
            f"<div style='margin:10px 0'><img src='{p.name}' style='max-width: 100%; border: 1px solid #ddd;'/></div>"
            if p.exists()
            else ""
        )

    scope_box = (
        "<b>Scope / Never claim</b><br>"
        "n₂ does NOT detect faults; it detects loss/distortion of coordination between interacting subsystems.<br>"
        "Never claims root cause, failure mode, prediction, or RUL estimation.<br>"
        f"<b>Frozen rule:</b> {contract.get('freeze_statement','')}<br>"
        f"<b>Frozen directional claim:</b> {contract.get('directional_claim','')}"
    )

    html = []
    html.append("<html><head><meta charset='utf-8'><title>ψ4 n2 report</title></head><body>")
    html.append(f"<h2>{TOOLKIT_NAME} v{TOOLKIT_VERSION}</h2>")
    html.append(f"<div style='padding:10px;border:1px solid #ccc;background:#fafafa'>{scope_box}</div>")

    # Linky nav (pairs with index.html)
    html.append("<p><b>Quick links:</b> "
                "<a href='index.html'>index.html</a> | "
                "<a href='summary.json'>summary.json</a> | "
                "<a href='contract.json'>contract.json</a> | "
                "<a href='provenance.json'>provenance.json</a></p>")

    html.append("<h3>Summary</h3>")
    html.append("<pre style='white-space:pre-wrap'>")
    html.append(json.dumps(summary, indent=2))
    html.append("</pre>")

    html.append("<h3>P1 Null Safety (frozen table)</h3>")
    html.append(null_table.to_html(index=False))

    html.append("<h3>Timeline (windowed circR)</h3>")
    html.append(_img_tag("circR_vs_time.png"))

    html.append("<h3>Residual distribution (REAL vs one sample from each null)</h3>")
    html.append(_img_tag("n2_plot.png"))

    html.append("<h3>P2 Multi-scale robustness (block_len sweep)</h3>")
    html.append(_img_tag("n2_multiscale_plot.png"))
    if multiscale_df is not None and len(multiscale_df) > 0:
        html.append(multiscale_df.to_html(index=False))

    html.append("<h3>Ablations (diagnostic robustness)</h3>")
    html.append(ablation_df.to_html(index=False))

    html.append("<h3>Provenance (meta)</h3>")
    html.append("<pre style='white-space:pre-wrap'>")
    html.append(json.dumps(meta, indent=2))
    html.append("</pre>")

    html.append("</body></html>")
    (outdir / "report.html").write_text("\n".join(html), encoding="utf-8")


# -----------------------------
# CLI helpers (alias hygiene, explain/check/dry-run, preflight, exec summary, index)
# -----------------------------

def apply_aliases(argv: List[str]) -> Tuple[List[str], List[str]]:
    """Rewrite alias CLI aliases to canonical flags.

    Supports both '--flag value' and '--flag=value' forms.

    Aliases (alias):
      --outdir   -> --out
      --win      -> --timeline-win
      --step     -> --timeline-step
    """
    out: List[str] = []
    warnings: List[str] = []
    i = 0

    def _emit(alias_flag: str, canonical_flag: str, value: str, msg: str) -> None:
        out.extend([canonical_flag, value])
        warnings.append(msg)

    while i < len(argv):
        a = argv[i]

        # Handle --flag=value form
        if a.startswith('--outdir='):
            _emit('--outdir', '--out', a.split('=', 1)[1],
                  '[WARN] --outdir is an alias for --out and is an alias kept for compatibility.')
            i += 1
            continue
        if a.startswith('--win='):
            _emit('--win', '--timeline-win', a.split('=', 1)[1],
                  '[WARN] --win is an alias for --timeline-win and is an alias kept for compatibility.')
            i += 1
            continue
        if a.startswith('--step='):
            _emit('--step', '--timeline-step', a.split('=', 1)[1],
                  '[WARN] --step is an alias for --timeline-step and is an alias kept for compatibility.')
            i += 1
            continue

        # Handle space-separated form: --flag value
        if a in ('--outdir', '--win', '--step'):
            if i + 1 >= len(argv):
                # Let argparse raise a proper error; don't swallow.
                out.append(a)
                i += 1
                continue
            v = argv[i + 1]
            if a == '--outdir':
                _emit(a, '--out', v, '[WARN] --outdir is an alias for --out and is an alias kept for compatibility.')
            elif a == '--win':
                _emit(a, '--timeline-win', v, '[WARN] --win is an alias for --timeline-win and is an alias kept for compatibility.')
            elif a == '--step':
                _emit(a, '--timeline-step', v, '[WARN] --step is an alias for --timeline-step and is an alias kept for compatibility.')
            i += 2
            continue

        out.append(a)
        i += 1

    return out, warnings


def check_alias_conflicts(argv: List[str]) -> None:
    """Fail fast if both canonical flag and alias alias are supplied."""
    conflicts = [
        ("--out", "--outdir"),
        ("--timeline-win", "--win"),
        ("--timeline-step", "--step"),
    ]

    present = set(a.split("=", 1)[0] for a in argv if a.startswith("--"))
    for canon, alias in conflicts:
        if canon in present and alias in present:
            raise SystemExit(
                f"ERROR: Conflicting flags: {canon} and {alias} (alias). "
                f"Use only {canon}. ({alias} is alias)"
            )

def print_explain() -> None:
    # Contract quick facts (publish-safe)
    v1min = int(CLAIM_CONTRACT.get('required_mismatch_min_reps_v1', 50))
    req = CLAIM_CONTRACT.get('required_null_families_v1', [])
    print(f"Contract (v1): required_null_families={req}; required_mismatch_min_reps_v1={v1min}")
    print("")
    txt = (
        "ψ4 n₂ Interaction (Handshake) Toolkit — v1.0-lock\n"
        "\n"
        "What it measures:\n"
        "  Residual r(t)=wrap(φ2(t)−2φ1(t)), headline circR=|⟨exp(i r)⟩|.\n"
        "\n"
        "When it should PASS:\n"
        "  True alignment-dependent coordination (phase-lock structure) where mismatch pairing collapses circR.\n"
        "\n"
        "When it should FAIL:\n"
        "  Amplitude-only similarities or non-alignment-dependent effects where mismatch does NOT collapse circR.\n"
        "\n"
        "What FAIL means:\n"
        "  The decisive falsifier (mismatch_pairing) did not break the coupling under the frozen rule.\n"
        "  FAIL is a RESULT: the run executed correctly; controls did not support the claim.\n"
        "\n"
        "Examples (fixed):\n"
        "  PASS domain: two synchronized oscillatory streams with φ2≈2·φ1.\n"
        "  FAIL domain: independent channels with similar spectra but no phase alignment.\n"
    )
    print(txt, flush=True)


def preflight_checks(
    df: pd.DataFrame,
    *,
    sig1_col: Optional[str],
    sig2_col: Optional[str],
    phi1_col: Optional[str],
    phi2_col: Optional[str],
    idx_col: str,
    timeline_win: int,
    timeline_step: int,
    quiet: bool,
) -> Tuple[bool, List[str]]:
    """
    Display-only guardrails. Never mutates df or args.
    Returns: (ok, notes). ok=False only when required columns are missing.
    """
    notes: List[str] = []
    ok = True

    n = int(len(df))
    if n <= 0:
        return False, ["[WARN] Empty CSV (0 rows)."]

    # idx col optional
    if idx_col not in df.columns:
        notes.append(f"[WARN] idx-col '{idx_col}' not found; using 0..N-1.")

    use_phase = (phi1_col is not None) and (phi2_col is not None)
    use_sig = (sig1_col is not None) and (sig2_col is not None)

    if not (use_phase or use_sig):
        ok = False
        notes.append("[WARN] Provide either (phi1-col & phi2-col) OR (sig1-col & sig2-col).")

    # required cols exist
    if use_phase:
        if (phi1_col not in df.columns) or (phi2_col not in df.columns):
            ok = False
            notes.append(f"[WARN] Missing phase cols: need '{phi1_col}' and '{phi2_col}'.")
    if use_sig:
        if (sig1_col not in df.columns) or (sig2_col not in df.columns):
            ok = False
            notes.append(f"[WARN] Missing signal cols: need '{sig1_col}' and '{sig2_col}'.")

    # nan% / const checks (only if cols exist)
    def _col_warn(c: str) -> None:
        if c is None or c not in df.columns:
            return
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        frac = float(np.mean(~np.isfinite(x)))
        if frac > 0:
            notes.append(f"[WARN] NaN/inf share in {c}: {100.0*frac:.2f}%")
        xs = x[np.isfinite(x)]
        if xs.size > 0:
            s = float(np.std(xs))
            if s < 1e-12:
                notes.append(f"[WARN] near-constant signal in {c} (std≈0).")

    if use_phase:
        _col_warn(phi1_col)
        _col_warn(phi2_col)
    if use_sig:
        _col_warn(sig1_col)
        _col_warn(sig2_col)

    # timeline sanity
    if int(timeline_win) > n:
        notes.append(f"[WARN] timeline_win ({int(timeline_win)}) > N ({n}). Suggest <= N/4≈{max(16, n//4)}.")
    if int(timeline_step) > int(timeline_win):
        notes.append(f"[WARN] timeline_step ({int(timeline_step)}) > timeline_win ({int(timeline_win)}).")

    return ok, notes


def add_cli_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--infile", required=True, help="Input CSV file")

    # Input mode A: phase columns
    p.add_argument("--phi1-col", default=None, help="Phase column φ1(t) (if already computed)")
    p.add_argument("--phi2-col", default=None, help="Phase column φ2(t) (if already computed)")

    # Input mode B: raw signals -> Hilbert phase
    p.add_argument("--sig1-col", default=None, help="Raw signal column for stream 1 (Hilbert→φ1)")
    p.add_argument("--sig2-col", default=None, help="Raw signal column for stream 2 (Hilbert→φ2)")

    p.add_argument("--idx-col", default="idx", help="Index/time column (default: idx). If missing, 0..N-1 is used.")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--seed", type=int, default=123, help="RNG seed (default: 123)")
    p.add_argument("--quiet", action="store_true", help="Suppress console output")
    p.add_argument("--output-mode", choices=["publish-min", "standard", "full-audit"], default="standard",
                   help="Artifact volume control only. Does not change math, nulls, verdicts, or required core outputs.")

    # UX-only modes (no science impact)
    p.add_argument("--check", action="store_true", help="Preflight-only: load CSV, print warnings, exit (no files).")
    p.add_argument("--dry-run", action="store_true", help="Print resolved run plan and outputs, exit (no files).")
    p.add_argument("--explain", action="store_true", help="Print deterministic explanation and exit.")

    p.add_argument("--selftest", action="store_true", help="Run internal regression self-test and exit.")

    # Null knobs (keep minimal, publish-safe)
    p.add_argument("--block-len", type=int, default=1024, help="Block length for block-based nulls (default: 1024)")
    p.add_argument("--n-phase-null", type=int, default=200, help="Reps for phase-scramble null (default: 200)")
    p.add_argument("--n-block-null", type=int, default=200, help="Reps for block-shuffle null (default: 200)")

    # Mismatch null (MANDATORY; decisive falsifier)
    p.add_argument(
        "--n-mismatch-null",
        type=int,
        default=50,
        help="Reps for mismatch pairing null (MANDATORY; default: 50; must be >= 50).",
    )
    p.add_argument(
        "--mismatch-min-shift",
        type=int,
        default=None,
        help="Minimum circular shift for mismatch null. Recommended: > block_len. Default: block_len.",
    )

    # Timeline defaults (always on)
    p.add_argument("--timeline-win", type=int, default=4096, help="Window size for circR timeline (default: 4096)")
    p.add_argument("--timeline-step", type=int, default=1024, help="Step size for circR timeline (default: 1024)")

    # P2 multiscale sweep
    p.add_argument(
        "--block-sweep",
        type=str,
        default="64,128,256,512,1024",
        help="Comma-separated block lengths for P2 sweep (default: 64,128,256,512,1024)",
    )
    p.add_argument(
        "--sweep-reps",
        type=int,
        default=50,
        help="Reps per null family per block_len in P2 sweep (default: 50)",
    )

    # Diagnostics
    p.add_argument("--reversal-check", action="store_true", help="Write reversal_metrics.json (diagnostic)")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ψ4 n₂ Interaction (Handshake) Toolkit — v1.0.0-lock")

    add_cli_arguments(p)

    return p


def _parse_int_list(csv: str) -> List[int]:
    s = (csv or "").strip()
    if not s:
        return []
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


# --- Part 2 continues with:
#     - index writers + exec summary printer
#     - finalize_run(...) (unifies exit paths)
#     - main() with alias hygiene + preflight printing every run

def write_index_artifacts(outdir: Path, artifacts: List[Dict[str, str]]) -> None:
    """
    Phase 7: Evidence Pack UX (self-navigating)
    Writes:
      - index.json : list of artifacts + roles
      - index.html : clickable links
    """
    index = {"toolkit": TOOLKIT_NAME, "version": TOOLKIT_VERSION, "artifacts": artifacts}
    (outdir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    rows = []
    for a in artifacts:
        fn = a.get("file", "")
        role = a.get("role", "")
        desc = a.get("desc", "")
        rows.append(
            f"<tr><td><a href='{fn}'>{fn}</a></td><td>{role}</td><td>{desc}</td></tr>"
        )
    html = []
    html.append("<html><head><meta charset='utf-8'><title>index</title></head><body>")
    html.append(f"<h2>{TOOLKIT_NAME} v{TOOLKIT_VERSION}</h2>")
    html.append("<p>Run folder index (artifacts + roles)</p>")
    html.append("<table border='1' cellpadding='6' cellspacing='0'>")
    html.append("<tr><th>File</th><th>Role</th><th>Description</th></tr>")
    html.extend(rows)
    html.append("</table>")
    html.append("</body></html>")
    (outdir / "index.html").write_text("\n".join(html), encoding="utf-8")



def _safe_print(s: str) -> None:
    """Print without crashing on Windows consoles that can't encode Greek letters."""
    try:
        print(s, flush=True)
    except UnicodeEncodeError:
        # Replace unencodable characters with '''
        safe = s.encode('ascii', 'replace').decode('ascii')
        print(safe, flush=True)

def exec_summary_block(
    *,
    args: argparse.Namespace,
    logic_fp: str,
    headline_name: str,
    headline_value: float,
    verdict_label: str,
    verdict_reason_code: str,
    verdict_reason_text: str,
    required_controls: List[str],
    missing_controls: List[str],
    outdir: Path,
    key_files: List[str],
    quiet: bool,
) -> None:
    """
    Phase 1: Executive Summary Block (printed at END always, all exit paths)
    """
    if quiet:
        return

    banner = verdict_label
    if verdict_label in ("FAIL", "NO_CLEAR", "INVALID_RUN"):
        banner = f"{verdict_label}  (FAIL IS A RESULT — see reason below)"

    lines = []
    lines.append("")
    lines.append("=== EXECUTIVE SUMMARY ===")
    lines.append(f"Toolkit: {TOOLKIT_NAME}")
    lines.append(f"Version: {TOOLKIT_VERSION}")
    lines.append(f"Input:   {args.infile}")
    lines.append(f"Outdir:  {str(outdir)}")
    lines.append(f"Logic fingerprint (decide_verdict): {logic_fp}")
    lines.append("")
    if verdict_label in ("INVALID_RUN", "NO_CLEAR"):
        # Publish hygiene: never print NaN headline on invalid/no-clear runs
        lines.append("Headline: (not computed - publish gate)")
    elif np.isfinite(headline_value):
        lines.append(f"Headline: {headline_name} = {headline_value:.6g}")
    else:
        # Extremely defensive: non-finite headline should not occur on a normal run
        lines.append("Headline: (non-finite ? input invalid)")
    lines.append("")
    lines.append(f"Verdict: {banner}")
    lines.append(f"Reason:  {verdict_reason_code} — {verdict_reason_text}")
    lines.append("")
    lines.append("Controls:")
    for c in required_controls:
        if c in missing_controls:
            lines.append(f"  ✗ {c} (missing / invalid)")
        else:
            lines.append(f"  ✓ {c}")
    lines.append("")
    lines.append("Artifacts:")
    lines.append(f"  {str(outdir)}")
    for f in key_files:
        lines.append(f"  - {f}")
    lines.append("=========================")
    _safe_print("\n".join(lines))


class _HintingArgumentParser(argparse.ArgumentParser):
    """
    Phase 2.2: “Did you mean…'” hints on argparse failure (UX only).
    """
    def error(self, message: str) -> None:
        # collect known flags
        known = []
        for a in self._actions:
            for opt in a.option_strings:
                known.append(opt)
        # try to suggest for unknown option
        toks = message.split()
        bad = None
        for t in toks:
            if t.startswith("-"):
                bad = t
                break
        hint = ""
        if bad:
            matches = difflib.get_close_matches(bad, known, n=3, cutoff=0.4)
            if matches:
                hint = f"\nDid you mean: {', '.join(matches)}\nUse --help to see all options."
            else:
                hint = "\nUse --help to see all options."
        self.print_usage(sys.stderr)
        raise SystemExit(f"{self.prog}: error: {message}{hint}")


def build_argparser_hinting() -> argparse.ArgumentParser:
    """
    Same flags as build_argparser(), but with hinting parser class.
    (No science changes.)
    """
    p = _HintingArgumentParser(description="ψ4 n₂ Interaction (Handshake) Toolkit — v1.0.0-lock")
    # Flags shared with build_argparser()
    add_cli_arguments(p)

    return p


def write_text_artifacts(outdir: Path) -> None:
    (outdir / "methods_blurb.txt").write_text(METHODS_BLURB, encoding="utf-8")
    (outdir / "caption_text.txt").write_text(CAPTION_TEXT, encoding="utf-8")
    (outdir / "limitations_blurb.txt").write_text(LIMITATIONS_BLURB, encoding="utf-8")


def write_provenance(outdir: Path, args: argparse.Namespace, infile_hash: str) -> Dict[str, object]:
    prov = {
        "toolkit": TOOLKIT_NAME,
        "version": TOOLKIT_VERSION,
        "timestamp_unix": int(time.time()),
        "cmdline": " ".join(sys.argv),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "argv": sys.argv[1:],
        "infile": args.infile,
        "infile_sha256": infile_hash,
        "versions": {
            "numpy": getattr(np, "__version__", "unknown"),
            "pandas": getattr(pd, "__version__", "unknown"),
            "matplotlib": getattr(matplotlib, "__version__", "unknown"),
            "scipy": (lambda: __import__("scipy").__version__ if __import__("importlib").util.find_spec("scipy") else "unknown")(),
        },
    }
    (outdir / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    return prov


def write_contract(outdir: Path, args: argparse.Namespace, input_mode: str) -> Dict[str, object]:
    contract = dict(CLAIM_CONTRACT)
    contract.update(
        {
            "toolkit_name": TOOLKIT_NAME,
            "toolkit_key": "n2_interaction",
            "version": TOOLKIT_VERSION,
            "freeze_statement": FREEZE_STATEMENT,
            "directional_claim": FROZEN_CLAIM,
            "input_mode": input_mode,
            "args": {
                "infile": args.infile,
                "out": args.out,
                "seed": int(args.seed),
                "phi1_col": args.phi1_col,
                "phi2_col": args.phi2_col,
                "sig1_col": args.sig1_col,
                "sig2_col": args.sig2_col,
                "idx_col": args.idx_col,
                "block_len": int(args.block_len),
                "n_phase_null": int(args.n_phase_null),
                "n_block_null": int(args.n_block_null),
                "n_mismatch_null": int(args.n_mismatch_null),
                "mismatch_min_shift": args.mismatch_min_shift,
                "timeline_win": int(args.timeline_win),
                "timeline_step": int(args.timeline_step),
                "block_sweep": args.block_sweep,
                "sweep_reps": int(args.sweep_reps),
                "reversal_check": bool(args.reversal_check),
            },
        }
    )
    (outdir / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return contract


def compute_ablations(phi1: np.ndarray, phi2: np.ndarray) -> pd.DataFrame:
    """
    Diagnostic-only ablations: circR stability under simple truncations.
    Does not affect verdict.
    """
    n = int(min(phi1.size, phi2.size))
    if n <= 0:
        return pd.DataFrame([{"ablation": "full", "n": 0, "circR": float("nan")}])

    fracs = [1.0, 0.75, 0.50, 0.25]
    rows = []
    for f in fracs:
        m = int(max(10, round(n * f)))
        r, met = compute_real(phi1[:m], phi2[:m])
        rows.append({"ablation": f"head_{int(f*100)}pct", "n": m, "circR": _safe_float(met.circR)})
    return pd.DataFrame(rows)


def compute_reversal_metrics(phi1: np.ndarray, phi2: np.ndarray) -> Dict[str, object]:
    """
    Diagnostic-only reversal check: compute headline on reversed series.
    """
    _, m_fwd = compute_real(phi1, phi2)
    _, m_rev = compute_real(phi1[::-1], phi2[::-1])
    return {
        "circR_forward": _safe_float(m_fwd.circR),
        "circR_reversed": _safe_float(m_rev.circR),
        "note": "Diagnostic only. Not used for verdict.",
    }


def build_null_table(
    *,
    families: List[str],
    phi1: np.ndarray,
    phi2: np.ndarray,
    sig1: Optional[np.ndarray],
    sig2: Optional[np.ndarray],
    block_len: int,
    reps_map: Dict[str, int],
    rng: np.random.Generator,
    mismatch_min_shift: Optional[int],
) -> pd.DataFrame:
    rows = []
    # compute real once for consistent circR_real field
    _, real_m = compute_real(phi1, phi2)
    circR_real = _safe_float(real_m.circR)

    for fam in families:
        n_reps = int(reps_map.get(fam, 0))
        stats, c = run_null_family(
            fam,
            phi1,
            phi2,
            block_len,
            n_reps,
            rng,
            sig1=sig1,
            sig2=sig2,
            mismatch_min_shift=mismatch_min_shift,
        )
        pct = percentile_null_below_real(circR_real, c)
        p_ge = empirical_p_value_ge_real(circR_real, c)
        breaks = directional_breaks(circR_real, float(stats["circR_null_mean"]), pct)

        rows.append(
            {
                "family": fam,
                "n_reps": int(stats["n_reps"]),
                "circR_real": circR_real,
                "circR_null_mean": float(stats["circR_null_mean"]),
                "circR_null_std": float(stats["circR_null_std"]),
                "pct_null_below_real": _safe_float(pct),
                "p_ge_real": _safe_float(p_ge),
                "delta_real_minus_null_mean": _safe_float(circR_real - float(stats["circR_null_mean"])),
                "z_real_vs_null_mean": _safe_float((circR_real - float(stats["circR_null_mean"])) / float(stats["circR_null_std"]) if float(stats["circR_null_std"]) > 0 else float("nan")),

                "breaks_directional": bool(breaks),
            }
        )
    return pd.DataFrame(rows)


def build_multiscale_table(
    *,
    sweep_blocks: List[int],
    phi1: np.ndarray,
    phi2: np.ndarray,
    sig1: Optional[np.ndarray],
    sig2: Optional[np.ndarray],
    sweep_reps: int,
    rng: np.random.Generator,
    mismatch_min_shift: Optional[int],
) -> pd.DataFrame:
    """
    P2 diagnostic-only multi-scale sweep.
    """
    _, real_m = compute_real(phi1, phi2)
    circR_real = _safe_float(real_m.circR)

    rows = []
    for b in sweep_blocks:
        row: Dict[str, object] = {"block_len": int(b), "circR_real": circR_real}
        for fam in ["phase_scramble", "block_shuffle", "mismatch_pairing"]:
            stats, _c = run_null_family(
                fam,
                phi1,
                phi2,
                int(b),
                int(sweep_reps),
                rng,
                sig1=sig1,
                sig2=sig2,
                mismatch_min_shift=mismatch_min_shift,
            )
            row[f"{fam}_mean"] = _safe_float(stats["circR_null_mean"])
            row[f"{fam}_std"] = _safe_float(stats["circR_null_std"])
        rows.append(row)

    return pd.DataFrame(rows)


def finalize_and_write_all(
    *,
    outdir: Path,
    args: argparse.Namespace,
    input_mode: str,
    infile_hash: str,
    real_metrics: RealMetrics,
    verdict_label: str,
    verdict_reason_code: str,
    verdict_reason_text: str,
    verdict_details: Dict[str, object],
    null_table: pd.DataFrame,
    ablation_df: pd.DataFrame,
    multiscale_df: pd.DataFrame,
    tl_df: pd.DataFrame,
    meta_extra: Dict[str, object],
    logic_fp: str,
    quiet: bool,
) -> None:
    """
    Single writer path for publish-grade evidence pack.
    """
    ensure_dir(outdir)

    # required text artifacts
    write_text_artifacts(outdir)

    # provenance + contract
    prov = write_provenance(outdir, args, infile_hash)
    contract = write_contract(outdir, args, input_mode=input_mode)

    # core CSVs
    null_table.to_csv(outdir / "n2_null_table.csv", index=False)
    ablation_df.to_csv(outdir / "n2_ablation_table.csv", index=False)
    multiscale_df.to_csv(outdir / "n2_multiscale_table.csv", index=False)
    tl_df.to_csv(outdir / "circR_vs_time.csv", index=False)

    # summary.json
    summary = {
        "toolkit": TOOLKIT_NAME,
        "version": TOOLKIT_VERSION,
        "logic_fingerprint_decide_verdict": logic_fp,
        "headline_metric": "circR",
        "headline_value": _safe_float(real_metrics.circR),
        "diagnostics": {"H2_mean": _safe_float(real_metrics.H2_mean), "H2_MAD": _safe_float(real_metrics.H2_MAD)},
        "verdict_label": verdict_label,
        "verdict": verdict_label,  # alias for compatibility
        
        "verdict_reason_code": verdict_reason_code,
        "verdict_reason_text": verdict_reason_text,
        "verdict_details": verdict_details,
        "required_controls": CLAIM_CONTRACT["required_null_families_v1"],
        "missing_controls": verdict_details.get("missing", []),
        "meta": meta_extra,
    }
    (outdir / "summary.json").write_text(json.dumps(_nan_to_none(summary), indent=2), encoding="utf-8")

    # report
    make_report_html(outdir, {**prov, **meta_extra}, contract, summary, null_table, ablation_df, multiscale_df)

    # index artifacts
    artifacts = [
        {"file": "summary.json", "role": "summary", "desc": "Headline + verdict + reason codes"},
        {"file": "contract.json", "role": "contract", "desc": "Frozen definitions + parameters"},
        {"file": "provenance.json", "role": "provenance", "desc": "Command + environment + hashes"},
        {"file": "n2_null_table.csv", "role": "controls", "desc": "Required null families + directional break flags"},
        {"file": "n2_ablation_table.csv", "role": "diagnostic", "desc": "Ablations (diagnostic-only)"},
        {"file": "n2_multiscale_table.csv", "role": "diagnostic", "desc": "Multi-scale sweep (diagnostic-only)"},
        {"file": "circR_vs_time.csv", "role": "diagnostic", "desc": "Windowed circR timeline"},
        {"file": "n2_plot.png", "role": "plot", "desc": "Residual distribution: REAL vs null samples"},
        {"file": "n2_multiscale_plot.png", "role": "plot", "desc": "Multi-scale: REAL vs null means"},
        {"file": "circR_vs_time.png", "role": "plot", "desc": "Timeline plot"},
        {"file": "report.html", "role": "report", "desc": "Human-readable report"},
        {"file": "methods_blurb.txt", "role": "text", "desc": "Methods blurb"},
        {"file": "caption_text.txt", "role": "text", "desc": "Figure caption"},
        {"file": "limitations_blurb.txt", "role": "text", "desc": "Limitations / safety notes"},
    ]
    # optional convenience primary json
    primary = {"circR": _safe_float(real_metrics.circR), "verdict": verdict_label, "reason": verdict_reason_code}
    (outdir / "n2_primary.json").write_text(json.dumps(_nan_to_none(primary), indent=2), encoding="utf-8")
    artifacts.append({"file": "n2_primary.json", "role": "convenience", "desc": "Machine-readable headline + verdict"})

    # reversal
    if args.reversal_check:
        rev_path = outdir / "reversal_metrics.json"
        # caller already computed meta_extra; keep reversal metrics there too, but also write standalone
        rev_payload = meta_extra.get("reversal_metrics", {})
        rev_path.write_text(json.dumps(_nan_to_none(rev_payload), indent=2), encoding="utf-8")
        artifacts.append({"file": "reversal_metrics.json", "role": "diagnostic", "desc": "Reversal check (diagnostic-only)"})

    write_index_artifacts(outdir, artifacts)

    # executive summary (console)
    key_files = ["summary.json", "contract.json", "provenance.json", "n2_null_table.csv", "report.html", "index.html"]
    exec_summary_block(
        args=args,
        logic_fp=logic_fp,
        headline_name="circR",
        headline_value=_safe_float(real_metrics.circR),
        verdict_label=verdict_label,
        verdict_reason_code=verdict_reason_code,
        verdict_reason_text=verdict_reason_text,
        required_controls=list(CLAIM_CONTRACT["required_null_families_v1"]),
        missing_controls=list(verdict_details.get("missing", [])) if isinstance(verdict_details.get("missing", []), list) else [],
        outdir=outdir,
        key_files=key_files,
        quiet=quiet,
    )



# -----------------------------
# Selftest (publish-safety regression)
# -----------------------------
def run_selftest_n2() -> None:
    """
    Runs 3 deterministic checks:
      1) PASS domain should PASS_STRONG on synth_n2_pass.csv
      2) Real-ish domain should NOT PASS on normal_std.csv
      3) Missing required nulls should produce INVALID_RUN (exit code 2)
    """
    import sys
    import json
    import subprocess
    from pathlib import Path

    py = sys.executable
    script = str(Path(__file__).resolve())

    root = Path("results")
    out_pass = root / "_selftest_n2_pass"
    out_fail = root / "_selftest_n2_fail"
    out_invalid = root / "_selftest_n2_invalid"

    # clean
    for d in (out_pass, out_fail, out_invalid):
        if d.exists():
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    # 1) PASS case
    cmd1 = [
        py, script,
        "--infile", str(Path("scratch/data/synth_n2_pass.csv")),
        "--sig1-col", "DE",
        "--sig2-col", "FE",
        "--idx-col", "idx",
        "--out", str(out_pass),
        "--seed", "0",
        "--block-len", "1024",
        "--n-phase-null", "50",
        "--n-block-null", "50",
        "--n-mismatch-null", "50",
        "--timeline-win", "8192",
        "--timeline-step", "2048",
    ]
    r1 = subprocess.run(cmd1, capture_output=True, text=True)
    if r1.returncode != 0:
        print("SELFTEST FAIL: PASS run returned", r1.returncode)
        print("---- STDOUT ----"); print(r1.stdout)
        print("---- STDERR ----"); print(r1.stderr)
        raise SystemExit(2)

    s1 = json.loads((out_pass / "summary.json").read_text(encoding="utf-8"))
    if s1.get("verdict_label") != "PASS_STRONG":
        raise SystemExit(f"SELFTEST FAIL: expected PASS_STRONG, got {s1.get('verdict_label')}")

    required = [
        "contract.json","summary.json","provenance.json","canonical_phases.csv",
        "n2_null_table.csv","n2_ablation_table.csv","n2_multiscale_table.csv","circR_vs_time.csv",
        "n2_plot.png","n2_multiscale_plot.png","circR_vs_time.png",
        "report.html","index.json","index.html","methods_blurb.txt","caption_text.txt","limitations_blurb.txt","n2_primary.json"
    ]
    missing = [f for f in required if not (out_pass / f).exists()]
    if missing:
        raise SystemExit(f"SELFTEST FAIL: PASS run missing files: {missing}")

    # 2) FAIL case (must not PASS)
    cmd2 = [
        py, script,
        "--infile", str(Path("scratch/data/normal_std.csv")),
        "--sig1-col", "DE",
        "--sig2-col", "FE",
        "--out", str(out_fail),
        "--seed", "0",
        "--block-len", "1024",
        "--n-phase-null", "50",
        "--n-block-null", "50",
        "--n-mismatch-null", "50",
        "--timeline-win", "8192",
        "--timeline-step", "2048",
    ]
    print('[SELFTEST] running FAIL case')
    r2 = subprocess.run(cmd2, capture_output=True, text=True)
    if r2.returncode not in (0, 2):
        print("SELFTEST FAIL: FAIL run returned", r2.returncode)
        print("---- STDOUT ----"); print(r2.stdout)
        print("---- STDERR ----"); print(r2.stderr)
        raise SystemExit(2)

    s2 = json.loads((out_fail / "summary.json").read_text(encoding="utf-8"))
    if s2.get("verdict_label") in ("PASS_STRONG", "PASS_WEAK"):
        raise SystemExit(f"SELFTEST FAIL: expected not-PASS, got {s2.get('verdict_label')}")

    # 3) INVALID: missing required null families (force phase+block reps to 0)
    cmd3 = [
        py, script,
        "--infile", str(Path("scratch/data/synth_n2_pass.csv")),
        "--sig1-col", "DE",
        "--sig2-col", "FE",
        "--idx-col", "idx",
        "--out", str(out_invalid),
        "--seed", "0",
        "--block-len", "1024",
        "--n-phase-null", "0",
        "--n-block-null", "0",
        "--n-mismatch-null", "50",
    ]
    print('[SELFTEST] running INVALID case')
    r3 = subprocess.run(cmd3, capture_output=True, text=True)
    if r3.returncode != 2:
        print("SELFTEST FAIL: INVALID expected exitcode=2, got", r3.returncode)
        print("---- STDOUT ----"); print(r3.stdout)
        print("---- STDERR ----"); print(r3.stderr)
        raise SystemExit(2)

    # Publish hygiene: invalid run must NOT print a NaN headline
    if ("Headline:" in (r3.stdout or "")) and ("NaN" in (r3.stdout or "")):
        print("SELFTEST FAIL: invalid run printed NaN headline (publish gate violated)")
        print("---- STDOUT ----"); print(r3.stdout)
        print("---- STDERR ----"); print(r3.stderr)
        raise SystemExit(2)

    summary_path = (out_invalid / "summary.json")
    if summary_path.exists():
        s3 = json.loads(summary_path.read_text(encoding="utf-8"))
        if s3.get("verdict_label") != "INVALID_RUN":
            raise SystemExit(f"SELFTEST FAIL: expected INVALID_RUN, got {s3.get('verdict_label')}")
    else:
        # Publish-gate mode: tool exits before writing evidence pack; assert gate message was emitted.
        msg = (r3.stdout or "") + "\n" + (r3.stderr or "")
        if "Publish gate failed" not in msg:
            raise SystemExit("SELFTEST FAIL: invalid run produced no summary.json and no publish-gate message")

    print("SELFTEST PASS")
    print("================")





def enforce_required_nulls_from_table(null_rows) -> None:
    """
    Publish gate (v1 lock):
      If required null families are missing or have 0 reps, STOP BEFORE headline.
      This prevents NaN headline + INVALID_RUN on normal user runs.
    """
    # null_rows can be list[dict] or a pandas DataFrame
    rows = None
    try:
        import pandas as pd  # optional
        if isinstance(null_rows, pd.DataFrame):
            rows = null_rows.to_dict(orient="records")
    except Exception:
        pass
    if rows is None:
        rows = list(null_rows) if null_rows is not None else []

    need = {"phase_scramble", "block_shuffle", "mismatch_pairing"}
    got = {}
    for r in rows:
        fam = str(r.get("family", "")).strip()
        reps = r.get("n_reps", r.get("reps", None))
        try:
            reps = int(reps)
        except Exception:
            reps = 0
        if fam:
            got[fam] = reps

    missing = []
    for fam in sorted(need):
        if got.get(fam, 0) < 1:
            missing.append(f"{fam} (reps={got.get(fam,0)})")

    if missing:
        print("[ERROR] Publish gate failed: required null families missing/zero reps: " + ", ".join(missing))
        raise SystemExit(2)


def main() -> None:
    # EARLY_EXPLAIN_EXIT (v1 lock): allow --explain without --infile/--out
    import sys

    # EARLY_SELFTEST_EXIT (v1 lock): allow --selftest without --infile/--out
    if "--selftest" in sys.argv:
        run_selftest_n2()
        return

    raw_argv = sys.argv[1:]
    check_alias_conflicts(raw_argv)
    argv2, alias_warnings = apply_aliases(raw_argv)
    if ('--explain' in argv2) or ('-E' in argv2):
        print_explain()
        return

    parser = build_argparser_hinting()
    args = parser.parse_args(argv2)

    # EARLY_MISSING_NULLS_AFTER_PARSE (v1 lock): required null families cannot have zero reps.
    # Publish gate: STOP BEFORE headline to avoid NaN/INVALID looking like a valid run.
    missing = []
    if int(args.n_phase_null) <= 0:
        missing.append("phase_scramble")
    if int(args.n_block_null) <= 0:
        missing.append("block_shuffle")

    v1min = int(CLAIM_CONTRACT.get("required_mismatch_min_reps_v1", 1))
    if int(args.n_mismatch_null) < v1min:
        missing.append(f"mismatch_pairing (min {v1min})")

    if missing:
        print("[ERROR] Publish gate failed: required null families missing/zero reps: " + ", ".join(missing))
        raise SystemExit(2)


    quiet = bool(args.quiet)

    for w in alias_warnings:
        msg = w if str(w).lstrip().startswith("[WARN]") else f"[WARN] {w}"
        log(msg, quiet)

    # Phase 5 --explain
    if args.explain:
        print_explain()
        return

    infile = Path(args.infile)
    outdir = Path(args.out)

    # Load input early (even for check/dry-run)
    if not infile.exists():
        raise SystemExit(f"ERROR: infile not found: {str(infile)}")

    df = pd.read_csv(infile)
    infile_hash = sha256_file(infile)

    # Preflight (Phase 3): auto-run on every mode (warnings-only)
    ok, notes = preflight_checks(
        df,
        sig1_col=args.sig1_col,
        sig2_col=args.sig2_col,
        phi1_col=args.phi1_col,
        phi2_col=args.phi2_col,
        idx_col=args.idx_col,
        timeline_win=int(args.timeline_win),
        timeline_step=int(args.timeline_step),
        quiet=quiet,
    )

    if not quiet:
        print("=== PREFLIGHT ===")
        print(f"Rows: {len(df)}")
        for n in notes:
            print(n)
        print("PREFLIGHT:", "OK" if ok else "NOT_OK")
        print("=================")

    # Phase 2.3 --dry-run (no files)
    if args.dry_run:
        planned = planned_outputs_for_mode(args.output_mode, bool(args.reversal_check))
        if not quiet:
            print("=== DRY RUN (no computation, no files) ===")
            print("Outdir:", str(outdir))
            print("Output mode:", str(args.output_mode))
            print("Planned outputs:")
            for p in planned:
                print("  -", p)
            print("=========================================")
        # Exit status: if NOT_OK preflight, return nonzero
        if not ok:
            raise SystemExit(2)
        return

    # Phase 3 --check (no files)
    if args.check:
        if not quiet:
            print("=== CHECK ONLY (no computation, no files) ===")
            print("Input:", str(infile))
            print("============================================")
        if not ok:
            raise SystemExit(2)
        return

    # From here: normal run (writes evidence pack)
    # Enforce mandatory mismatch reps v1 rule BEFORE compute
    if int(args.n_mismatch_null) < int(CLAIM_CONTRACT["required_mismatch_min_reps_v1"]):
        # INVALID_RUN but still write minimal pack to outdir (publish-safe)
        ensure_dir(outdir)
        logic_fp = logic_fingerprint_decide_verdict()
        verdict_label = "INVALID_RUN"
        reason_code = "n_mismatch_null_below_v1_min"
        reason_text = REASON_TEXT_MAP.get(reason_code, reason_code)

        # minimal placeholders
        real_metrics = RealMetrics(n=0, circR=float("nan"), H2_mean=float("nan"), H2_MAD=float("nan"))
        null_table = pd.DataFrame([])
        ablation_df = pd.DataFrame([])
        multiscale_df = pd.DataFrame([])
        tl_df = pd.DataFrame([])
        details = {"missing": list(CLAIM_CONTRACT["required_null_families_v1"]), "family_breaks": {}}

        finalize_and_write_all(
            outdir=outdir,
            args=args,
            input_mode="unknown",
            infile_hash=infile_hash,
            real_metrics=real_metrics,
            verdict_label=verdict_label,
            verdict_reason_code=reason_code,
            verdict_reason_text=reason_text,
            verdict_details=details,
            null_table=null_table,
            ablation_df=ablation_df,
            multiscale_df=multiscale_df,
            tl_df=tl_df,
            meta_extra={"preflight_notes": notes},
            logic_fp=logic_fp,
            quiet=quiet,
        )
        raise SystemExit(2)

    # Determine input mode + extract phases
    use_phase = (args.phi1_col is not None) and (args.phi2_col is not None)
    use_sig = (args.sig1_col is not None) and (args.sig2_col is not None)

    if not (use_phase or use_sig):
        raise SystemExit("ERROR: Provide either --phi1-col/--phi2-col OR --sig1-col/--sig2-col")

    if args.idx_col in df.columns:
        idx = pd.to_numeric(df[args.idx_col], errors="coerce").to_numpy(dtype=float)
    else:
        idx = np.arange(len(df), dtype=float)

    sig1 = None
    sig2 = None
    if use_phase:
        phi1 = pd.to_numeric(df[args.phi1_col], errors="coerce").to_numpy(dtype=float)
        phi2 = pd.to_numeric(df[args.phi2_col], errors="coerce").to_numpy(dtype=float)
        input_mode = "phases"
    else:
        sig1 = pd.to_numeric(df[args.sig1_col], errors="coerce").to_numpy(dtype=float)
        sig2 = pd.to_numeric(df[args.sig2_col], errors="coerce").to_numpy(dtype=float)
        phi1 = instantaneous_phase_from_signal(sig1)
        phi2 = instantaneous_phase_from_signal(sig2)
        input_mode = "signals_hilbert"

    # truncate to common finite length
    n = int(min(len(phi1), len(phi2), len(idx)))
    phi1 = phi1[:n]
    phi2 = phi2[:n]
    idx = idx[:n]
    if sig1 is not None:
        sig1 = sig1[:n]
    if sig2 is not None:
        sig2 = sig2[:n]

    # real metrics + residual
    r_real, real_m = compute_real(phi1, phi2)
    if (not np.isfinite(real_m.circR)) or (real_m.n < 10):
        # NO_CLEAR but still write pack
        ensure_dir(outdir)
        logic_fp = logic_fingerprint_decide_verdict()
        verdict_label = "NO_CLEAR"
        reason_code = "insufficient_data_or_nan"
        reason_text = REASON_TEXT_MAP.get(reason_code, reason_code)
        details = {"missing": list(CLAIM_CONTRACT["required_null_families_v1"]), "family_breaks": {}}

        finalize_and_write_all(
            outdir=outdir,
            args=args,
            input_mode=input_mode,
            infile_hash=infile_hash,
            real_metrics=real_m,
            verdict_label=verdict_label,
            verdict_reason_code=reason_code,
            verdict_reason_text=reason_text,
            verdict_details=details,
            null_table=pd.DataFrame([]),
            ablation_df=pd.DataFrame([]),
            multiscale_df=pd.DataFrame([]),
            tl_df=pd.DataFrame([]),
            meta_extra={"preflight_notes": notes},
            logic_fp=logic_fp,
            quiet=quiet,
        )
        raise SystemExit(2)

    # rng
    rng = np.random.default_rng(int(args.seed))

    # canonical phases output
    ensure_dir(outdir)
    pd.DataFrame(
        {
            "idx": idx,
            "phi1": phi1,
            "phi2": phi2,
            "residual_r": r_real,
        }
    ).to_csv(outdir / "canonical_phases.csv", index=False)

    # timeline
    tl_df = compute_timeline(r_real, idx, int(args.timeline_win), int(args.timeline_step))
    plot_timeline(outdir, tl_df)

    # null table (P1)
    reps_map = {
        "phase_scramble": int(args.n_phase_null),
        "block_shuffle": int(args.n_block_null),
        "mismatch_pairing": int(args.n_mismatch_null),
    }
    mismatch_min_shift = args.mismatch_min_shift
    null_table = build_null_table(
        families=list(CLAIM_CONTRACT["required_null_families_v1"]),
        phi1=phi1,
        phi2=phi2,
        sig1=sig1,
        sig2=sig2,
        block_len=int(args.block_len),
        reps_map=reps_map,
        rng=rng,
        mismatch_min_shift=mismatch_min_shift,
    )

    # ensure mismatch executed with valid reps (hard fail closed)
    mis_row = null_table[null_table["family"] == "mismatch_pairing"]
    if mis_row.empty or int(mis_row.iloc[0]["n_reps"]) < int(CLAIM_CONTRACT["required_mismatch_min_reps_v1"]):
        logic_fp = logic_fingerprint_decide_verdict()
        verdict_label = "INVALID_RUN"
        reason_code = "n_mismatch_null_below_v1_min"
        reason_text = REASON_TEXT_MAP.get(reason_code, reason_code)
        details = {"missing": ["mismatch_pairing"], "family_breaks": {}}

        finalize_and_write_all(
            outdir=outdir,
            args=args,
            input_mode=input_mode,
            infile_hash=infile_hash,
            real_metrics=real_m,
            verdict_label=verdict_label,
            verdict_reason_code=reason_code,
            verdict_reason_text=reason_text,
            verdict_details=details,
            null_table=null_table,
            ablation_df=pd.DataFrame([]),
            multiscale_df=pd.DataFrame([]),
            tl_df=tl_df,
            meta_extra={"preflight_notes": notes},
            logic_fp=logic_fp,
            quiet=quiet,
        )
        raise SystemExit(2)

    # verdict (FROZEN)
    verdict_label, verdict_details = decide_verdict(null_table, list(CLAIM_CONTRACT["required_null_families_v1"]))
    verdict_reason_code = str(verdict_details.get("reason", "unknown_reason"))
    verdict_reason_text = REASON_TEXT_MAP.get(verdict_reason_code, verdict_reason_code)

    # diagnostics: ablations + multiscale sweep
    ablation_df = compute_ablations(phi1, phi2)

    sweep_blocks = _parse_int_list(str(args.block_sweep))
    if not sweep_blocks:
        sweep_blocks = [64, 128, 256, 512, 1024]
    multiscale_df = build_multiscale_table(
        sweep_blocks=sweep_blocks,
        phi1=phi1,
        phi2=phi2,
        sig1=sig1,
        sig2=sig2,
        sweep_reps=int(args.sweep_reps),
        rng=rng,
        mismatch_min_shift=mismatch_min_shift,
    )
    plot_multiscale(outdir, multiscale_df)

    # one-sample residuals for distribution plot (UX / intuition only)
    r_phase = sample_null_residual("phase_scramble", phi1, phi2, int(args.block_len), rng, sig1=sig1, sig2=sig2, mismatch_min_shift=mismatch_min_shift)
    r_block = sample_null_residual("block_shuffle", phi1, phi2, int(args.block_len), rng, sig1=sig1, sig2=sig2, mismatch_min_shift=mismatch_min_shift)
    r_mismatch = sample_null_residual("mismatch_pairing", phi1, phi2, int(args.block_len), rng, sig1=sig1, sig2=sig2, mismatch_min_shift=mismatch_min_shift)
    plot_residual_distributions(outdir, r_real, r_phase, r_block, r_mismatch)

    # meta extras
    meta_extra: Dict[str, object] = {
        "preflight_notes": notes,
        "freeze_statement": FREEZE_STATEMENT,
        "directional_claim": FROZEN_CLAIM,
    }
    if args.reversal_check:
        meta_extra["reversal_metrics"] = compute_reversal_metrics(phi1, phi2)

    # fingerprint (Phase 0.1)
    logic_fp = logic_fingerprint_decide_verdict()

    # write full evidence pack
    finalize_and_write_all(
        outdir=outdir,
        args=args,
        input_mode=input_mode,
        infile_hash=infile_hash,
        real_metrics=real_m,
        verdict_label=verdict_label,
        verdict_reason_code=verdict_reason_code,
        verdict_reason_text=verdict_reason_text,
        verdict_details=verdict_details,
        null_table=null_table,
        ablation_df=ablation_df,
        multiscale_df=multiscale_df,
        tl_df=tl_df,
        meta_extra=meta_extra,
        logic_fp=logic_fp,
        quiet=quiet,
    )
    prune_optional_artifacts_n2(outdir, args.output_mode, bool(args.reversal_check))

    # exit code convention (publish-safe): FAIL/NO_CLEAR/INVALID -> nonzero
    if verdict_label in ("NO_CLEAR", "INVALID_RUN"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
