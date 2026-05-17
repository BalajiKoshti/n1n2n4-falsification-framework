#!/usr/bin/env python3
# ================================================================
#   psi4 n4 Closure Toolkit (Publish-safe, single-file) — v1.1.0-lock
#   ------------------------------------------------------------
#   Base: n4_closure_toolkit.py v1.0.0 (frozen math + contract)
#   UX restored: psi4_n4_toolkit.py v0.92 “one-screen trust” console
#
#   Frozen purpose:
#     Closure audit of a FROZEN upstream estimator (UE/UM), with
#     mandatory kill-switch controls. NOT an atomic prediction.
#
#   Frozen headline family (ONE family, fixed):
#     ratio_i = (UE_i + eps) / (UM_i + eps)
#     C4_i    = log(ratio_i)
#
#   Always report (headline summaries):
#     ratio_median, C4_median, abs_C4_median, C4_MAD
#
#   Diagnostics (explicitly not headline):
#     C4_violation_frac_pi = frac(|C4| > pi)
#     C4_median_first25 / mid50 / last25
#
#   Mandatory kill-switch controls (v1 Option A):
#     - gauss_zero
#     - em_swap
#   If either is missing => verdict = NO_CLEAR_MISSING_CONTROLS
#
#   Frozen verdict rule (direction-only, no tuned thresholds):
#     PASS_directional_controls iff BOTH controls satisfy:
#       abs_C4_median(control) > abs_C4_median(real)
#       AND (C4_MAD(control) > C4_MAD(real) OR
#            C4_violation_frac_pi(control) > C4_violation_frac_pi(real))
#
#   Operator guardrails (UX-only, no science change):
#     --check   : preflight only, writes nothing
#     --dry-run : prints planned outputs, writes nothing
#     --explain : short deterministic “what/when/means/NOT” text
#     --outdir  : alias for --out (warn)
#     “Did you mean …pi” for unknown flags
#
#   Evidence pack outputs (always for compute runs):
#     - contract.json
#     - summary.json
#     - provenance.json
#     - n4_primary.json
#     - n4_controls_table.csv
#     - n4_plot.png
#     - report.html
#     - index.json
#     - index.html
#     - methods_blurb.txt
#     - caption_text.txt
#     - limitations_blurb.txt
#     - (optional) n4_robustness_table.csv
#
#   Safety sentences (must appear in report):
#     “n4 is intended to run after interaction is established (e.g. via n2).”
#     “PASS indicates estimator-level closure sensitivity, not system safety/correctness.”
# ================================================================

from __future__ import annotations

import argparse
import difflib
import hashlib
import inspect
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless-safe plotting (publish guardrail; no science change)
import matplotlib.pyplot as plt


TOOLKIT_NAME = "psi4 n4 Closure Toolkit"
TOOLKIT_VERSION = "1.1.0-lock"

DEFAULT_EPS = 1e-30
MIN_FRAMES = 3




# ----------------------------- planned outputs (single source of truth) -----------------------------
# Keep this list frozen and reuse it everywhere to avoid output drift.
PLANNED_OUTPUTS = {
    "contract.json": "Frozen contract (definitions + rules)",
    "summary.json": "Run summary + verdict + reason codes",
    "provenance.json": "Runtime provenance (argv, versions, hashes)",
    "n4_primary.json": "Primary (REAL) metrics (headline + diagnostics)",
    "n4_controls_table.csv": "REAL vs controls numeric table",
    "n4_plot.png": "Plot (REAL vs controls)",
    "report.html": "Human-readable report (start here)",
    "index.json": "Index manifest",
    "index.html": "Self-navigating run folder index",
    "methods_blurb.txt": "Short methods blurb (deterministic)",
    "caption_text.txt": "Suggested figure caption",
    "limitations_blurb.txt": "Limitations / claim boundaries",
}

OPTIONAL_OUTPUTS = {
    "n4_robustness_table.csv": "Optional robustness table across multiple REAL runs",
}

PUBLISH_MIN_OPTIONAL = {
    "index.json": "Compact paper-facing pack does not require machine index",
    "methods_blurb.txt": "Optional helper text",
    "caption_text.txt": "Optional helper text",
    "limitations_blurb.txt": "Optional helper text",
    "n4_robustness_table.csv": "Optional robustness table across multiple REAL runs",
}

def planned_outputs_for_mode(output_mode: str = "standard") -> Dict[str, str]:
    mode = str(output_mode or "standard").strip().lower()
    outs = dict(PLANNED_OUTPUTS)
    outs.update(OPTIONAL_OUTPUTS)
    if mode == "publish-min":
        for k in list(PUBLISH_MIN_OPTIONAL.keys()):
            outs.pop(k, None)
    return outs

def prune_optional_artifacts_n4(outdir: Path, output_mode: str) -> None:
    mode = str(output_mode or "standard").strip().lower()
    if mode != "publish-min":
        return
    for name in PUBLISH_MIN_OPTIONAL.keys():
        fp = outdir / name
        try:
            if fp.exists():
                fp.unlink()
        except Exception:
            pass
# ----------------------------- small utils -----------------------------

def log(msg: str) -> None:
    print(msg, flush=True)

def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)

def err(msg: str) -> None:
    print(f"[ERROR] {msg}", flush=True)

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def sha256_self() -> str:
    try:
        return sha256_file(__file__)
    except Exception:
        return "unavailable"

def mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    m = np.median(x)
    return float(np.median(np.abs(x - m)))

def _nan_to_none(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _pct(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{100.0*x:.2f}%"


# ----------------------------- CLI helpers -----------------------------

class DidYouMeanArgumentParser(argparse.ArgumentParser):
    """
    Adds 'Did you mean ...pi' suggestions for unknown flags.
    """
    def error(self, message: str) -> None:
        # Default argparse message often looks like: "unrecognized arguments: --foo"
        tokens = message.split()
        unknown = None
        if "unrecognized arguments:" in message:
            i = tokens.index("arguments:") + 1
            if i < len(tokens):
                unknown = tokens[i]
        if unknown and unknown.startswith("-"):
            opts = []
            for a in self._actions:
                opts.extend(a.option_strings)
            suggestion = difflib.get_close_matches(unknown, opts, n=1, cutoff=0.55)
            if suggestion:
                message = f"{message}\nDid you mean: {suggestion[0]} ?"
        super().error(message)

def _apply_argv_aliases(argv: List[str]) -> List[str]:
    argv = list(argv)

    # --outdir alias for --out
    if "--outdir" in argv:
        idx = argv.index("--outdir")
        if idx + 1 >= len(argv):
            return argv
        outdir_val = argv[idx + 1]
        # only map if --out not already present
        if "--out" not in argv:
            warn("Legacy alias in use: --outdir -> --out")
            argv[idx] = "--out"
            argv[idx + 1] = outdir_val
        else:
            warn("Both --outdir and --out provided; using --out and ignoring --outdir.")
            # remove --outdir pair
            del argv[idx:idx + 2]
    return argv


# ----------------------------- IO -----------------------------

def load_frames_table(path: str | Path) -> pd.DataFrame:
    """
    Accept .csv, .json, .jsonl.
    Must contain UE and UM (case-insensitive).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    suf = path.suffix.lower()
    if suf == ".csv":
        df = pd.read_csv(path)
    elif suf in (".json", ".jsonl"):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Empty file: {path}")
        if suf == ".jsonl":
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            df = pd.DataFrame(rows)
        else:
            obj = json.loads(text)
            if isinstance(obj, dict) and "frames" in obj and isinstance(obj["frames"], list):
                df = pd.DataFrame(obj["frames"])
            elif isinstance(obj, list):
                df = pd.DataFrame(obj)
            elif isinstance(obj, dict):
                df = pd.DataFrame([obj])
            else:
                raise ValueError("Unsupported JSON structure.")
    else:
        raise ValueError(f"Unsupported file type: {suf}. Use .csv/.json/.jsonl")

    cols_lower = {c.lower(): c for c in df.columns}
    if "ue" not in cols_lower or "um" not in cols_lower:
        raise ValueError(f"Input must contain UE and UM columns. Found: {list(df.columns)}")

    df = df.copy()
    df.rename(columns={cols_lower["ue"]: "UE", cols_lower["um"]: "UM"}, inplace=True)
    return df


# ----------------------------- metrics (FROZEN) -----------------------------

def safe_log_ratio(UE: np.ndarray, UM: np.ndarray, eps: float = DEFAULT_EPS) -> Tuple[np.ndarray, np.ndarray]:
    UE = np.asarray(UE, dtype=float)
    UM = np.asarray(UM, dtype=float)
    ratio = (UE + eps) / (UM + eps)
    C4 = np.log(ratio)
    return ratio, C4

def compute_case_metrics(df: pd.DataFrame, eps: float = DEFAULT_EPS) -> Dict[str, float]:
    """
    FROZEN compute block (do NOT change formulas):
      ratio_i = (UE_i+eps)/(UM_i+eps)
      C4_i    = log(ratio_i)

    Headline summaries (frozen):
      ratio_median, C4_median, abs_C4_median, C4_MAD

    Diagnostics-only:
      C4_violation_frac_pi, time-chunk medians
    """
    UE = df["UE"].to_numpy(dtype=float)
    UM = df["UM"].to_numpy(dtype=float)
    ratio, C4 = safe_log_ratio(UE, UM, eps=eps)

    m = np.isfinite(ratio) & np.isfinite(C4)
    ratio = ratio[m]
    C4 = C4[m]

    if C4.size > 0:
        vio = float(np.mean(np.abs(C4) > np.pi))
        n = int(C4.size)
        i25 = max(1, int(0.25 * n))
        i75 = max(i25 + 1, int(0.75 * n))
        c_first = float(np.median(C4[:i25]))
        c_mid = float(np.median(C4[i25:i75]))
        c_last = float(np.median(C4[i75:]))
    else:
        vio = float("nan")
        c_first = float("nan")
        c_mid = float("nan")
        c_last = float("nan")

    return {
        "n_frames": float(C4.size),

        # Headline (frozen)
        "ratio_median": float(np.median(ratio)) if ratio.size else float("nan"),
        "C4_median": float(np.median(C4)) if C4.size else float("nan"),
        "abs_C4_median": float(np.median(np.abs(C4))) if C4.size else float("nan"),
        "C4_MAD": mad(C4),

        # Secondary reporting (explicitly not verdict thresholds)
        "ratio_mean": float(np.mean(ratio)) if ratio.size else float("nan"),
        "C4_mean": float(np.mean(C4)) if C4.size else float("nan"),
        "C4_std": float(np.std(C4)) if C4.size else float("nan"),
        "ratio_MAD": mad(ratio),

        # Diagnostics only (explicit)
        "C4_violation_frac_pi": vio,
        "C4_median_first25": c_first,
        "C4_median_mid50": c_mid,
        "C4_median_last25": c_last,
    }

def directional_controls_verdict(real: Dict[str, float],
                                 gz: Dict[str, float],
                                 em: Dict[str, float]) -> Dict[str, Any]:
    """
    Frozen v1 rule (Option A): BOTH controls must worsen by 2-axis directional rule.
    """
    r_abs = float(real.get("abs_C4_median", float("nan")))
    r_mad = float(real.get("C4_MAD", float("nan")))
    r_vio = float(real.get("C4_violation_frac_pi", float("nan")))

    def check_one(ctrl: Dict[str, float]) -> Dict[str, Any]:
        c_abs = float(ctrl.get("abs_C4_median", float("nan")))
        c_mad = float(ctrl.get("C4_MAD", float("nan")))
        c_vio = float(ctrl.get("C4_violation_frac_pi", float("nan")))

        worsens_abs = (np.isfinite(c_abs) and np.isfinite(r_abs) and (c_abs > r_abs))
        worsens_mad = (np.isfinite(c_mad) and np.isfinite(r_mad) and (c_mad > r_mad))
        worsens_vio = (np.isfinite(c_vio) and np.isfinite(r_vio) and (c_vio > r_vio))
        pass_two_axis = bool(worsens_abs and (worsens_mad or worsens_vio))

        return {
            "abs_C4_median": c_abs,
            "C4_MAD": c_mad,
            "C4_violation_frac_pi": c_vio,
            "worsens_abs_C4_median": worsens_abs,
            "worsens_C4_MAD": worsens_mad,
            "worsens_violation_frac_pi": worsens_vio,
            "pass_two_axis": pass_two_axis,
        }

    gz_chk = check_one(gz)
    em_chk = check_one(em)
    ok_all = bool(gz_chk["pass_two_axis"] and em_chk["pass_two_axis"])
    verdict = "PASS_directional_controls" if ok_all else "FAIL_or_inconclusive_directional_controls"

    return {
        "verdict": verdict,
        "rule": "Each control must worsen abs_C4_median AND (C4_MAD OR violation_frac_pi) vs REAL (no tuned thresholds).",
        "real": {"abs_C4_median": r_abs, "C4_MAD": r_mad, "C4_violation_frac_pi": r_vio},
        "checks": {"gauss_zero": gz_chk, "em_swap": em_chk},
    }


# ----------------------------- preflight -----------------------------

@dataclass
class PreflightResult:
    ok: bool
    warnings: List[str]
    error_code: Optional[str]
    error_text: Optional[str]
    rows: Optional[int]

REASON_ENUM = [
    "missing_required_controls",
    "preflight_missing_columns",
    "nonfinite_or_nonpositive_energy",
    "too_few_frames",
    "controls_did_not_worsen_directionally",
    "no_clear_numerical_tie",
    "invalid_input",
]

def preflight_file(path: Optional[str], label: str) -> Tuple[bool, Optional[str]]:
    if not path:
        return False, f"{label} path missing"
    p = Path(path)
    if not p.exists():
        return False, f"{label} not found: {p}"
    return True, None

def preflight_table(path: str, eps: float) -> PreflightResult:
    warnings: List[str] = []
    try:
        df = load_frames_table(path)
    except Exception as e:
        return PreflightResult(
            ok=False,
            warnings=[],
            error_code="invalid_input",
            error_text=f"{type(e).__name__}: {e}",
            rows=None,
        )

    n_rows = int(len(df))
    if n_rows < MIN_FRAMES:
        return PreflightResult(
            ok=False,
            warnings=[],
            error_code="too_few_frames",
            error_text=f"too few frames: {n_rows} < {MIN_FRAMES}",
            rows=n_rows,
        )

    UE = df["UE"].to_numpy(dtype=float)
    UM = df["UM"].to_numpy(dtype=float)

    finite = np.isfinite(UE) & np.isfinite(UM)
    nonfinite_frac = 1.0 - float(np.mean(finite)) if finite.size else float("nan")
    if np.isfinite(nonfinite_frac) and nonfinite_frac > 0:
        warnings.append(f"non-finite UE/UM rows: {_pct(nonfinite_frac)} (will be dropped)")

    # log undefined if <=0 (beyond numerical noise). We do not tune a threshold;
    # we only warn here and fail-closed if ALL valid rows are lost.
    nonpos = (UE <= 0) | (UM <= 0)
    nonpos_frac = float(np.mean(nonpos)) if nonpos.size else float("nan")
    if np.isfinite(nonpos_frac) and nonpos_frac > 0:
        warnings.append(f"UE<=0 or UM<=0 rows: {_pct(nonpos_frac)} (log ratio may be invalid)")

    # Try computing to detect "all dropped" / numerical collapse
    metrics = compute_case_metrics(df, eps=eps)
    n_eff = int(metrics.get("n_frames", 0.0) or 0.0)
    if n_eff < MIN_FRAMES:
        return PreflightResult(
            ok=False,
            warnings=warnings,
            error_code="nonfinite_or_nonpositive_energy",
            error_text=f"after dropping invalid rows, effective frames {n_eff} < {MIN_FRAMES}",
            rows=n_rows,
        )

    return PreflightResult(ok=True, warnings=warnings, error_code=None, error_text=None, rows=n_rows)


# ----------------------------- verdict wrapper (FROZEN RULE + NO_CLEAR logic) -----------------------------

def decide_verdict(real_m: Dict[str, float],
                   gz_m: Optional[Dict[str, float]],
                   em_m: Optional[Dict[str, float]],
                   controls_present: bool) -> Tuple[str, str, str, Dict[str, Any]]:
    """
    Returns (verdict, reason_code, reason_text, details)

    - Missing required controls => NO_CLEAR_MISSING_CONTROLS (not FAIL)
    - Numerical tie => NO_CLEAR (tie is 'no clear' not fail)
    - Otherwise use frozen directional_controls_verdict
    """
    if not controls_present or gz_m is None or em_m is None:
        return (
            "NO_CLEAR_MISSING_CONTROLS",
            "missing_required_controls",
            "Controls missing → contract failure (NO_CLEAR).",
            {"controls_present": False},
        )

    # Numerical tie guard (direction-only; if headline comparisons are not strictly >, treat as "no clear")
    # We only tag as tie if abs_C4 medians are exactly equal (within float equality) for either control.
    r_abs = float(real_m.get("abs_C4_median", float("nan")))
    gz_abs = float(gz_m.get("abs_C4_median", float("nan")))
    em_abs = float(em_m.get("abs_C4_median", float("nan")))
    if (np.isfinite(r_abs) and np.isfinite(gz_abs) and gz_abs == r_abs) or (np.isfinite(r_abs) and np.isfinite(em_abs) and em_abs == r_abs):
        return (
            "NO_CLEAR",
            "no_clear_numerical_tie",
            "Numerical tie: control abs_C4_median equals REAL; directional rule requires strict worsening.",
            {"controls_present": True, "tie": True},
        )

    dv = directional_controls_verdict(real_m, gz_m, em_m)
    if dv.get("verdict") == "PASS_directional_controls":
        return (
            "PASS_directional_controls",
            "controls_worsen_directionally",
            "Both kill-switch controls worsen closure under frozen directional rule.",
            dv,
        )

    return (
        "FAIL",
        "controls_did_not_worsen_directionally",
        "One or both kill-switch controls did not worsen closure directionally under the frozen rule.",
        dv,
    )


def logic_fingerprint() -> str:
    """
    Fingerprint frozen logic blocks to detect accidental drift.
    """
    parts = []
    for fn in (compute_case_metrics, directional_controls_verdict, decide_verdict):
        try:
            parts.append(inspect.getsource(fn))
        except Exception:
            parts.append(repr(fn))
    return sha256_text("\n\n".join(parts))


# ===================== END OF 1st HALF =====================
# Paste the 2nd half immediately after this line in the same file.
# The 2nd half includes: report/index writers, plotting, bench-pack,
# executive console blocks, and main().
# ===================== CUT HERE (2nd half follows) =====================

# ===================== BEGIN 2nd HALF =====================

# ----------------------------- text blurbs (deterministic) -----------------------------

def methods_blurb_text() -> str:
    return (
        "ψ4 n4 Closure Toolkit — Methods (frozen v1)\n"
        "\n"
        "Purpose (frozen): Closure audit of a frozen upstream estimator that emits UE and UM per frame.\n"
        "Primary observable (frozen): ratio_i = (UE_i+eps)/(UM_i+eps), C4_i = log(ratio_i).\n"
        "Headline summaries (frozen): ratio_median, C4_median, abs_C4_median, C4_MAD.\n"
        "Diagnostics (not headline): violation fraction beyond π and time-chunk medians.\n"
        "\n"
        "Controls (mandatory, v1): gauss_zero and em_swap.\n"
        "Directional rule (threshold-free): each control must worsen abs_C4_median and also worsen\n"
        "either C4_MAD or violation_frac_pi vs REAL. PASS requires BOTH controls satisfy this.\n"
        "\n"
        "Important: PASS indicates estimator-level closure sensitivity under kill-switch controls.\n"
        "It is not a claim about nature’s constants and not a system safety guarantee.\n"
    )

def caption_text() -> str:
    return (
        "Figure (ψ4 n4 Closure): REAL vs kill-switch controls for closure magnitude and tightness.\n"
        "Headline: abs_C4_median and C4_MAD (with diagnostic violation_frac_pi).\n"
        "PASS requires both controls worsen closure directionally under the frozen rule.\n"
    )

def limitations_blurb_text() -> str:
    return (
        "ψ4 n4 Closure Toolkit — Limitations (v1)\n"
        "\n"
        "- This is a closure audit of a frozen estimator (UE/UM). It does not infer root cause,\n"
        "  remaining useful life, or physical constants.\n"
        "- PASS indicates directional sensitivity under the required controls (gauss_zero, em_swap).\n"
        "- FAIL is a valid outcome: it means the estimator’s closure signal did not break\n"
        "  under the frozen kill-switch tests.\n"
        "- NO_CLEAR indicates contract failure (e.g., missing controls) or invalid/insufficient data.\n"
    )

def explain_text() -> str:
    return (
        "ψ4 n4 Closure Toolkit (v1) — What / When / Means / NOT\n"
        "\n"
        "What it measures (frozen):\n"
        "  Per frame: ratio = (UE+eps)/(UM+eps), C4 = log(ratio).\n"
        "  Closure means C4 stays near 0 and stable.\n"
        "\n"
        "When it should PASS:\n"
        "  If REAL is relatively closed, then BOTH kill-switch controls (gauss_zero and em_swap)\n"
        "  should worsen closure directionally under the frozen rule:\n"
        "    abs_C4_median(control) > abs_C4_median(real)\n"
        "    AND (C4_MAD(control) > C4_MAD(real) OR violation_frac_pi(control) > violation_frac_pi(real))\n"
        "\n"
        "What FAIL means:\n"
        "  Controls did not worsen closure directionally. FAIL is a result.\n"
        "\n"
        "What NO_CLEAR means:\n"
        "  Contract failure (e.g., missing required controls) or invalid/insufficient data.\n"
        "\n"
        "ψ4 is NOT:\n"
        "  A classifier, predictor, RUL model, root-cause engine, or a claim about nature’s constants.\n"
    )


# ----------------------------- writers -----------------------------

def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(_nan_to_none(obj), indent=2, default=_json_default), encoding="utf-8")

def write_text(path: Path, s: str) -> None:
    path.write_text(s, encoding="utf-8")

def df_to_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)

def df_to_dark_html(df: pd.DataFrame) -> str:
    html = df.to_html(index=False, border=0, escape=False)
    html = html.replace(
        'class="dataframe"',
        'style="background:#181818;color:#f0f0f0;border-collapse:collapse;"'
    )
    html = html.replace("<th>", '<th style="border:1px solid #333;padding:6px;">')
    html = html.replace("<td>", '<td style="border:1px solid #333;padding:6px;">')
    return html


def _json_default(o):
    # Minimal safe serializer for numpy/pandas scalar types (and any stray objects).
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    return str(o)

def save_report_html(outdir: Path,
                     contract: Dict[str, Any],
                     summary: Dict[str, Any],
                     meta: Dict[str, Any],
                     controls_table: pd.DataFrame,
                     deltas_table: pd.DataFrame) -> None:
    verdict = summary.get("verdict", "UNKNOWN")
    safety_block = """
<ul style="max-width:980px; line-height:1.45; color:#d0d0d0;">
  <li><b>n4 is intended to run after interaction is established (e.g., via n2).</b></li>
  <li><b>PASS indicates estimator-level closure sensitivity, not system safety/correctness.</b></li>
  <li><b>PASS does NOT mean</b> an atomic prediction, a nature-constant claim, or a guarantee about the underlying physical system.</li>
</ul>
""".strip()

    html = f"""
<html>
<head><meta charset="utf-8"/><title>{TOOLKIT_NAME} Report</title></head>
<body style="font-family:Arial; background-color:#101010; color:#f0f0f0;">
  <h1>{TOOLKIT_NAME} v{TOOLKIT_VERSION}</h1>

  <h2>Verdict</h2>
  <p style="font-size:1.10em;"><b>{verdict}</b></p>
  <p style="max-width:980px; color:#d0d0d0;"><b>Frozen rule:</b> {contract["verdict_rule"]}</p>

  <h3>Safety / claim boundary</h3>
  {safety_block}

  <h2>Top table (REAL vs CONTROLS with Δ)</h2>
  {df_to_dark_html(deltas_table)}

  <h2>Controls table (raw summaries)</h2>
  {df_to_dark_html(controls_table)}

  <h2>Plot</h2>
  <img src="n4_plot.png" width="900"/>

  <h2>Contract</h2>
  <pre style="background:#181818; padding:10px; white-space:pre-wrap;">{json.dumps(contract, indent=2, default=_json_default)}</pre>

  <h2>Summary</h2>
  <pre style="background:#181818; padding:10px; white-space:pre-wrap;">{json.dumps(summary, indent=2, default=_json_default)}</pre>

  <h2>Run metadata</h2>
  <pre style="background:#181818; padding:10px; white-space:pre-wrap;">{json.dumps(meta, indent=2, default=_json_default)}</pre>

  <p style="margin-top:20px; font-size:0.9em; color:#aaaaaa;">
    Generated by {TOOLKIT_NAME} v{TOOLKIT_VERSION}.
  </p>
</body>
</html>
""".strip()

    (outdir / "report.html").write_text(html, encoding="utf-8")

def save_index(outdir: Path, artifacts: Dict[str, str]) -> None:
    """
    artifacts: mapping filename -> role/description
    """
    idx = {
        "toolkit": TOOLKIT_NAME,
        "toolkit_version": TOOLKIT_VERSION,
        "outdir": str(outdir),
        "artifacts": [{"file": k, "role": v} for k, v in artifacts.items()],
    }
    write_json(outdir / "index.json", idx)

    links = []
    for k, v in artifacts.items():
        if k.endswith(".html") or k.endswith(".png") or k.endswith(".json") or k.endswith(".csv") or k.endswith(".txt"):
            links.append(f'<li><a href="{k}">{k}</a> — {v}</li>')
        else:
            links.append(f"<li>{k} — {v}</li>")

    html = f"""
<html>
<head><meta charset="utf-8"/><title>{TOOLKIT_NAME} Index</title></head>
<body style="font-family:Arial; background-color:#101010; color:#f0f0f0;">
  <h1>{TOOLKIT_NAME} v{TOOLKIT_VERSION}</h1>
  <p style="max-width:980px; color:#d0d0d0;">
    Self-navigating run folder. Start with <b>report.html</b> or the plots.
  </p>
  <ul style="line-height:1.6;">
    {''.join(links)}
  </ul>
</body>
</html>
""".strip()

    (outdir / "index.html").write_text(html, encoding="utf-8")


# ----------------------------- plotting -----------------------------

def plot_n4(outdir: Path,
            real: Dict[str, float],
            gz: Optional[Dict[str, float]],
            em: Optional[Dict[str, float]]) -> None:
    """
    Simple, stable plot: bars for abs_C4_median and C4_MAD (and vio as text).
    """
    labels = ["REAL", "gauss_zero", "em_swap"]
    abs_vals = [
        float(real.get("abs_C4_median", float("nan"))),
        float(gz.get("abs_C4_median", float("nan"))) if gz else float("nan"),
        float(em.get("abs_C4_median", float("nan"))) if em else float("nan"),
    ]
    mad_vals = [
        float(real.get("C4_MAD", float("nan"))),
        float(gz.get("C4_MAD", float("nan"))) if gz else float("nan"),
        float(em.get("C4_MAD", float("nan"))) if em else float("nan"),
    ]
    vio_vals = [
        float(real.get("C4_violation_frac_pi", float("nan"))),
        float(gz.get("C4_violation_frac_pi", float("nan"))) if gz else float("nan"),
        float(em.get("C4_violation_frac_pi", float("nan"))) if em else float("nan"),
    ]

    x = np.arange(len(labels))

    plt.figure()
    plt.bar(x - 0.18, abs_vals, width=0.36, label="abs_C4_median")
    plt.bar(x + 0.18, mad_vals, width=0.36, label="C4_MAD")
    plt.xticks(x, labels)
    plt.ylabel("magnitude")
    plt.title("ψ4 n4 Closure — REAL vs controls")
    plt.legend()

    # annotate vio as small text on top
    for i, v in enumerate(vio_vals):
        if np.isfinite(v):
            plt.text(i, max(abs_vals[i] if np.isfinite(abs_vals[i]) else 0.0,
                            mad_vals[i] if np.isfinite(mad_vals[i]) else 0.0) * 1.02,
                     f"vioπ={v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(outdir / "n4_plot.png", dpi=140)
    plt.close()


# ----------------------------- bench-pack (engineering-only) -----------------------------



def make_controls_from_real(real_csv: Path, outdir: Path, seed: int = 0) -> tuple[Path, Path]:
    """
    Generate the two required n4 controls from a REAL file.

    IMPORTANT: These are usability/provenance helpers only (CSV-world approximations).
    They are designed to be "kill-switch" controls that destroy pairing/mechanism:

      1) n4_gauss_zero.csv : keep UM real, replace UE with independent Gaussian (same mean/std as real UE)
                             -> breaks coupling while keeping one side fixed (mechanism-removal spirit)
      2) n4_em_swap.csv    : pair-scramble mismatch control (permute UM relative to UE)
                             -> keeps marginals, destroys pairing

    This does NOT change n4 headline logic.
    """
    import numpy as np
    import pandas as pd

    outdir.mkdir(parents=True, exist_ok=True)

    df = load_frames_table(real_csv)
    ue = df["UE"].to_numpy(dtype=float)
    um = df["UM"].to_numpy(dtype=float)

    rng = np.random.default_rng(int(seed))

    # -------- gauss_zero (kill-switch): keep UM fixed (real), randomize UE --------
    ue_fin = ue[np.isfinite(ue)]
    ue_mu = float(np.mean(ue_fin)) if ue_fin.size else 1.0
    ue_sd = float(np.std(ue_fin) + 1e-12) if ue_fin.size else 1e-6
    ue_g  = rng.normal(loc=ue_mu, scale=ue_sd, size=len(ue))


    # gauss_zero numerical safety (control-only): keep finite & strictly positive
    eps = 1e-9
    ue_g = np.asarray(ue_g, dtype=float)
    ue_g = np.where(np.isfinite(ue_g), ue_g, ue_mu)
    ue_g = np.clip(ue_g, eps, None)
    um_safe = np.asarray(um, dtype=float)
    um_safe = np.where(np.isfinite(um_safe), um_safe, np.nanmedian(um_safe))
    um_safe = np.clip(um_safe, eps, None)
    # if your energies are expected positive, keep this safe (prevents NaNs from log with eps)
    # (minimal intervention; only affects rare negative draws)
    ue_g = np.where(np.isfinite(ue_g), ue_g, ue_mu)
    ue_g = np.maximum(ue_g, 0.0)

    df_g = pd.DataFrame({"UE": ue_g, "UM": um_safe})
    out_g = outdir / "n4_gauss_zero.csv"
    df_g.to_csv(out_g, index=False)

    # -------- em_swap (mismatch kill-switch): pair-scramble UM relative to UE --------
    perm = rng.permutation(len(um))
    um_p = um[perm]
    df_s = pd.DataFrame({"UE": ue, "UM": um_p})
    out_s = outdir / "n4_em_swap.csv"
    df_s.to_csv(out_s, index=False)

    return out_g, out_s

def make_bench_pack(bench_dir: Path, seed: int = 12345) -> Dict[str, Any]:
    """
    Writes deterministic synthetic REAL + controls.
    Purpose: detect accidental code drift, not scientific evidence.
    """
    ensure_dir(bench_dir)
    rng = np.random.default_rng(seed)
    n = 256

    # REAL: UE ~ UM (close)
    UM_real = np.abs(rng.normal(1.0, 0.05, size=n)) + 0.01
    UE_real = UM_real * np.exp(rng.normal(0.0, 0.03, size=n))

    # gauss_zero: break closure
    UM_gz = UM_real.copy()
    UE_gz = np.abs(rng.normal(0.05, 0.02, size=n)) + 0.001

    # em_swap: break closure (regression-only synthetic)
    UM_em = UE_real.copy()
    UE_em = UM_real * np.exp(rng.normal(0.0, 0.06, size=n))

    def write_csv(path: Path, UE: np.ndarray, UM: np.ndarray) -> None:
        pd.DataFrame({"UE": UE.astype(float), "UM": UM.astype(float)}).to_csv(path, index=False)

    write_csv(bench_dir / "real.csv", UE_real, UM_real)
    write_csv(bench_dir / "gauss_zero.csv", UE_gz, UM_gz)
    write_csv(bench_dir / "em_swap.csv", UE_em, UM_em)

    real_m = compute_case_metrics(pd.DataFrame({"UE": UE_real, "UM": UM_real}))
    gz_m = compute_case_metrics(pd.DataFrame({"UE": UE_gz, "UM": UM_gz}))
    em_m = compute_case_metrics(pd.DataFrame({"UE": UE_em, "UM": UM_em}))

    expected = {
        "note": "Regression-only bench pack. If these shift materially, you introduced drift.",
        "seed": int(seed),
        "n": int(n),
        "expected_metrics": {
            "REAL": {k: float(real_m[k]) for k in ("abs_C4_median", "C4_MAD", "C4_violation_frac_pi")},
            "gauss_zero": {k: float(gz_m[k]) for k in ("abs_C4_median", "C4_MAD", "C4_violation_frac_pi")},
            "em_swap": {k: float(em_m[k]) for k in ("abs_C4_median", "C4_MAD", "C4_violation_frac_pi")},
        },
    }
    write_json(bench_dir / "bench_expected.json", expected)
    return expected


# ----------------------------- executive console blocks (trust UX) -----------------------------

def print_header_block() -> None:
    log("========================================")
    log(f"{TOOLKIT_NAME}")
    log(f"Version: {TOOLKIT_VERSION}")
    log("========================================")

def print_real_block(m: Dict[str, float]) -> None:
    log("=== REAL HEADLINE (frozen) ===")
    log(f"n_frames        : {int(m.get('n_frames', 0.0) or 0)}")
    log(f"ratio_median    : {m.get('ratio_median')}")
    log(f"C4_median       : {m.get('C4_median')}")
    log(f"abs_C4_median   : {m.get('abs_C4_median')}")
    log(f"C4_MAD          : {m.get('C4_MAD')}")
    log("==============================")

def print_diagnostics_block(m: Dict[str, float]) -> None:
    log("=== DIAGNOSTICS (NOT headline) ===")
    log(f"violation_frac_pi : {m.get('C4_violation_frac_pi')}")
    log(f"C4_median_first25 : {m.get('C4_median_first25')}")
    log(f"C4_median_mid50   : {m.get('C4_median_mid50')}")
    log(f"C4_median_last25  : {m.get('C4_median_last25')}")
    log("=================================")

def print_controls_block(real: Dict[str, float],
                         gz: Optional[Dict[str, float]],
                         em: Optional[Dict[str, float]]) -> None:
    log("=== CONTROLS (expected to worsen closure) ===")
    if gz is None:
        log("gauss_zero : [MISSING]")
    else:
        log(f"gauss_zero abs_C4_median={gz.get('abs_C4_median')}  C4_MAD={gz.get('C4_MAD')}  vioπ={gz.get('C4_violation_frac_pi')}")
    if em is None:
        log("em_swap    : [MISSING]")
    else:
        log(f"em_swap    abs_C4_median={em.get('abs_C4_median')}  C4_MAD={em.get('C4_MAD')}  vioπ={em.get('C4_violation_frac_pi')}")
    log("===========================================")

def print_executive_summary(summary: Dict[str, Any], outdir: Path) -> None:
    log("\n=== EXECUTIVE SUMMARY ===")
    log(f"Toolkit: {TOOLKIT_NAME}")
    log(f"Version: {TOOLKIT_VERSION}")
    log(f"Outdir:  {outdir}")
    log(f"Logic fingerprint: {summary.get('logic_fingerprint')}")
    log("")
    log("Headline (REAL):")
    log(f"  abs_C4_median = {summary.get('real', {}).get('abs_C4_median')}")
    log(f"  C4_MAD        = {summary.get('real', {}).get('C4_MAD')}")
    log(f"  ratio_median  = {summary.get('real', {}).get('ratio_median')}")
    log("")
    verdict = summary.get("verdict")
    if verdict != "PASS_directional_controls":
        log("Verdict: " + str(verdict) + "  (FAIL/NO_CLEAR ARE RESULTS — see reason below)")
    else:
        log("Verdict: " + str(verdict))
    log(f"Reason:  {summary.get('verdict_reason_code')} — {summary.get('verdict_reason_text')}")
    log("")
    c = summary.get("controls_present", {})
    log("Controls:")
    log(f"  ✓ gauss_zero : {bool(c.get('gauss_zero', False))}")
    log(f"  ✓ em_swap    : {bool(c.get('em_swap', False))}")
    log("")
    log("Artifacts:")
    log(f"  {outdir}")

    # Print artifacts from the single source-of-truth list (prevents drift)
    for fn in planned_outputs_for_mode(summary.get("output_mode", "standard")).keys():
        if (outdir / fn).exists():
            log(f"  - {fn}")
    log("=========================")


# ----------------------------- compute run helpers -----------------------------

HEADLINE_KEYS = ["ratio_median", "C4_median", "abs_C4_median", "C4_MAD"]
DIAG_KEYS = ["C4_violation_frac_pi", "C4_median_first25", "C4_median_mid50", "C4_median_last25"]

def build_controls_table(real_m: Dict[str, float],
                         gz_m: Optional[Dict[str, float]],
                         em_m: Optional[Dict[str, float]]) -> pd.DataFrame:
    rows = []
    def add(name: str, m: Optional[Dict[str, float]]) -> None:
        if m is None:
            rows.append({"case": name, **{k: None for k in (["n_frames"] + HEADLINE_KEYS + DIAG_KEYS)}})
            return
        rows.append({
            "case": name,
            "n_frames": int(m.get("n_frames", 0.0) or 0),
            **{k: m.get(k) for k in HEADLINE_KEYS},
            **{k: m.get(k) for k in DIAG_KEYS},
        })
    add("REAL", real_m)
    add("gauss_zero", gz_m)
    add("em_swap", em_m)
    return pd.DataFrame(rows)

def build_deltas_table(controls: pd.DataFrame) -> pd.DataFrame:
    """
    One view: REAL row, then control rows with Δ vs REAL for headline + key diagnostics.
    """
    df = controls.copy()
    if df.empty:
        return df

    real_row = df[df["case"] == "REAL"]
    if real_row.empty:
        return df

    real_vals = real_row.iloc[0].to_dict()

    out_rows = []
    for _, r in df.iterrows():
        row = r.to_dict()
        if row.get("case") != "REAL":
            for k in HEADLINE_KEYS + ["C4_violation_frac_pi"]:
                v = row.get(k)
                rv = real_vals.get(k)
                if v is None or rv is None:
                    row[f"Δ_{k}"] = None
                else:
                    try:
                        row[f"Δ_{k}"] = float(v) - float(rv)
                    except Exception:
                        row[f"Δ_{k}"] = None
        else:
            for k in HEADLINE_KEYS + ["C4_violation_frac_pi"]:
                row[f"Δ_{k}"] = 0.0
        out_rows.append(row)

    cols = ["case", "n_frames"] + HEADLINE_KEYS + ["C4_violation_frac_pi"] + [f"Δ_{k}" for k in (HEADLINE_KEYS + ["C4_violation_frac_pi"])]
    return pd.DataFrame(out_rows)[cols]

def parse_real_runs_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts

def compute_robustness_table(real_paths: List[str], eps: float) -> Optional[pd.DataFrame]:
    if not real_paths:
        return None
    rows = []
    for p in real_paths:
        try:
            df = load_frames_table(p)
            m = compute_case_metrics(df, eps=eps)
            rows.append({
                "path": p,
                "n_frames": int(m.get("n_frames", 0.0) or 0),
                **{k: m.get(k) for k in HEADLINE_KEYS},
                **{k: m.get(k) for k in DIAG_KEYS},
            })
        except Exception as e:
            rows.append({"path": p, "error": f"{type(e).__name__}: {e}"})
    return pd.DataFrame(rows)


# ----------------------------- main -----------------------------


def run_selftest() -> None:
    """
    Internal regression selftest.
    - Generates bench pack
    - Runs one compute
    - Asserts core artifacts exist
    """
    import subprocess, sys
    from pathlib import Path
    from datetime import datetime

    root = Path("scratch/lock_n4")
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("selftest_%Y%m%d_%H%M%S")
    base = root / stamp
    bench = base / "bench"
    out = base / "out"
    bench.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    # 1) make bench pack
    subprocess.check_call([
        sys.executable, __file__,
        "--make-bench-pack",
        "--out", str(bench),
    ])

    real = bench / "bench_pack" / "real.csv"
    gz   = bench / "bench_pack" / "gauss_zero.csv"
    em   = bench / "bench_pack" / "em_swap.csv"


    # SELFTEST gate regression: missing controls must hard-stop (publish gate)
    # - exit code must be 2
    # - must NOT print executive summary
    # - must NOT create outdir
    gate_out = base / "out_gate_missing_controls"
    if gate_out.exists():
        import shutil
        shutil.rmtree(gate_out)

    r_gate = subprocess.run(
        [sys.executable, __file__, "--real", str(real), "--out", str(gate_out)],
        capture_output=True,
        text=True,
    )

    if r_gate.returncode != 2:
        print("SELFTEST FAIL: expected exit code 2 for missing-controls publish gate")
        print("returncode=", r_gate.returncode)
        print("---- STDOUT ----"); print(r_gate.stdout)
        print("---- STDERR ----"); print(r_gate.stderr)
        raise SystemExit(2)

    out_text = (r_gate.stdout or "") + "\n" + (r_gate.stderr or "")
    if "Publish gate failed: missing required controls" not in out_text:
        print("SELFTEST FAIL: missing-controls gate did not print expected error message")
        print("---- OUTPUT ----"); print(out_text)
        raise SystemExit(2)

    if "=== EXECUTIVE SUMMARY ===" in out_text:
        print("SELFTEST FAIL: gate run printed EXECUTIVE SUMMARY (must fail-fast before headline)")
        print("---- OUTPUT ----"); print(out_text)
        raise SystemExit(2)

    if gate_out.exists():
        print("SELFTEST FAIL: gate outdir exists (must not be created on publish-gate stop):", gate_out)
        raise SystemExit(2)
    # 2) run compute
    subprocess.check_call([
        sys.executable, __file__,
        "--real", str(real),
        "--gauss-zero", str(gz),
        "--em-swap", str(em),
        "--out", str(out),
    ])

    # 3) assert core artifacts
    required = list(PLANNED_OUTPUTS.keys())
    missing = [f for f in required if not (out / f).exists()]
    if missing:
        raise RuntimeError("SELFTEST failed, missing artifacts: " + ", ".join(missing))

    # publish-gate: provenance schema must be complete
    prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    for k in ("versions", "matplotlib_backend", "toolkit", "toolkit_version"):
        if k not in prov:
            raise RuntimeError(f"SELFTEST failed: provenance missing {k}")

    print("SELFTEST OK")
def main() -> None:
    argv = _apply_argv_aliases(sys.argv[1:])

    ap = DidYouMeanArgumentParser(description=f"{TOOLKIT_NAME} (publish-safe closure audit)")

    ap.add_argument("--real", required=False, help="REAL per-frame table (.csv/.json/.jsonl) containing UE and UM.")
    ap.add_argument("--gauss-zero", required=False, default=None, help="gauss_zero control table (UE/UM). (MANDATORY for PASS)")
    ap.add_argument("--em-swap", required=False, default=None, help="em_swap control table (UE/UM). (MANDATORY for PASS)")

    ap.add_argument("--real-runs", default="", help="Comma list of additional REAL runs for robustness CSV (optional).")
    ap.add_argument("--eps", type=float, default=DEFAULT_EPS, help="Numerical eps for ratio stabilization.")
    ap.add_argument("--out", required=False, help="Output directory (evidence pack).")

    ap.add_argument("--check", action="store_true", help="Preflight only; writes nothing.")
    ap.add_argument("--dry-run", action="store_true", help="Print planned outputs; writes nothing.")
    ap.add_argument("--explain", action="store_true", help="Explain what this toolkit measures and exit.")
    ap.add_argument("--output-mode", choices=["publish-min", "standard", "full-audit"], default="standard",
                    help="Artifact volume control only. Does not change closure math, controls, verdicts, or required core outputs.")

    ap.add_argument("--make-bench-pack", action="store_true", help="Write bench_pack/ (regression-only synthetic pack) and exit.")
    ap.add_argument("--make-controls", action="store_true", help="Generate n4_gauss_zero.csv + n4_em_swap.csv from REAL and exit.")
    ap.add_argument("--controls-outdir", default="", help="Directory to write controls into (default: same dir as --real).")
    ap.add_argument("--controls-seed", type=int, default=0, help="Seed for gauss-zero control generation (reproducible).")
    ap.add_argument("--selftest", action="store_true", help="Run internal regression selftest (bench-pack + smoke run) and exit.")
    ap.add_argument("--bench-seed", type=int, default=12345, help="Seed for bench-pack generation.")
    ap.add_argument("--no-hash-input", action="store_true", help="Disable input hashing (not recommended).")

    ap.add_argument("--dataset-tag", default="", help="Optional short tag (recorded only).")
    ap.add_argument("--estimator-name", default="UEUM_estimator", help="Frozen upstream estimator name (recorded).")
    ap.add_argument("--estimator-version", default="v1", help="Frozen upstream estimator version string (recorded).")
    ap.add_argument("--estimator-config", default=None, help="Path to JSON config used upstream (recorded & hashed).")

    args = ap.parse_args(argv)

    # [PATCH] EARLY_MAKE_CONTROLS_EXIT (generate controls without requiring --gauss-zero/--em-swap)
    if getattr(args, "make_controls", False):
        # UX: --out is for evidence packs; make-controls writes to --controls-outdir (or REAL dir)
        if str(getattr(args, "out", "")).strip():
            print("[WARN] --out is ignored for --make-controls; use --controls-outdir to choose where controls are written.")
        real_csv = Path(args.real)
        outdir = Path(args.controls_outdir) if str(getattr(args, "controls_outdir", "")).strip() else real_csv.parent
        gpath, spath = make_controls_from_real(real_csv, outdir, seed=int(getattr(args, "controls_seed", 0)))
        print("Wrote controls:")
        print(f"  {gpath}")
        print(f"  {spath}")
        return
    if getattr(args, "selftest", False):
        run_selftest()
        return

    # --explain
    if args.explain:
        print(explain_text())
        return

    # Bench pack (regression only)
    if args.make_bench_pack:
        if not args.out:
            err("--out is required for --make-bench-pack")
            sys.exit(2)
        outdir = Path(args.out)
        ensure_dir(outdir)
        bench_dir = outdir / "bench_pack"
        make_bench_pack(bench_dir, seed=int(args.bench_seed))
        log(f"Wrote bench pack to: {bench_dir}")
        return

    # DRY RUN: should not require --real; prints what would be produced and exits.
    if args.dry_run:
        log("=== DRY RUN ===")
        log("This prints planned outputs only. No files are written.")
        log(f"Output mode: {args.output_mode}")
        planned = list(planned_outputs_for_mode(args.output_mode).keys())
        log("Planned outputs:")
        for fn in planned:
            log(f"  - {fn}")
        sys.exit(0)


    # Validate minimal required args for normal modes
    if not args.real:
        err("--real is required.")

    # preflight for required files
    ok_real, msg_real = preflight_file(args.real, "REAL")
    ok_gz, msg_gz = preflight_file(args.gauss_zero, "gauss_zero") if args.gauss_zero else (False, "gauss_zero path missing")
    ok_em, msg_em = preflight_file(args.em_swap, "em_swap") if args.em_swap else (False, "em_swap path missing")

    # Controls are mandatory for PASS; missing => NO_CLEAR_MISSING_CONTROLS, but we still allow compute of REAL
    controls_present = bool(ok_gz and ok_em)


    # Publish gate (v1 lock): for compute runs, BOTH controls must be present.
    # Exempt modes: --check, --dry-run, --explain, --selftest, --make-controls, --make-bench-pack
    if (not args.check) and (not args.dry_run) and (not args.explain) and (not args.selftest) and (not args.make_controls) and (not args.make_bench_pack):
        if not controls_present:
            err("Publish gate failed: missing required controls for compute run. Provide BOTH --gauss-zero and --em-swap.")
            raise SystemExit(2)
    # Preflight tables
    log("=== PREFLIGHT ===")
    if not ok_real:
        err(msg_real or "REAL preflight failed")

    pf_real = preflight_table(args.real, eps=float(args.eps))
    if pf_real.rows is not None:
        log(f"Rows: {pf_real.rows}")
    if pf_real.warnings:
        for w in pf_real.warnings:
            warn(w)
    if not pf_real.ok:
        err(f"PREFLIGHT: {pf_real.error_code} — {pf_real.error_text}")
        log("=================")
        if args.check or args.dry_run:
            # check/dry-run should still exit cleanly with preflight failure code
            sys.exit(2)
        # For compute mode, treat as NO_CLEAR (invalid input)
        if not args.out:
            err("--out is required for computation")
            sys.exit(2)
        outdir = Path(args.out)
        ensure_dir(outdir)
        # minimal evidence pack for invalid runs
        contract = {
            "toolkit": TOOLKIT_NAME,
            "toolkit_version": TOOLKIT_VERSION,
            "frozen_purpose": "Closure audit of frozen UE/UM estimator with mandatory kill-switch controls (not atomic prediction).",
            "headline_family": "ratio_i=(UE_i+eps)/(UM_i+eps); C4_i=log(ratio_i)",
            "headline_summaries": HEADLINE_KEYS,
            "diagnostics_only": DIAG_KEYS,
            "required_controls_v1": ["gauss_zero", "em_swap"],
            "missing_control_behavior": "NO_CLEAR_MISSING_CONTROLS",
            "verdict_rule": "Each control must worsen abs_C4_median AND (C4_MAD OR violation_frac_pi) vs REAL (no tuned thresholds).",
            "verdict_reason_codes_v1": REASON_ENUM,
            "frozen_functions_fingerprint": ["compute_case_metrics", "directional_controls_verdict", "decide_verdict"],
        }
        meta = {
            "timestamp_utc": _now_utc_iso(),
            "platform": {"python": sys.version, "platform": platform.platform()},
            "argv": sys.argv,
            "self_sha256": sha256_self(),
        }
        summary = {
            "verdict": "NO_CLEAR",
            "verdict_reason_code": pf_real.error_code,
            "verdict_reason_text": pf_real.error_text,
            "logic_fingerprint": logic_fingerprint(),
            "controls_present": {"gauss_zero": bool(ok_gz), "em_swap": bool(ok_em)},
            "output_mode": str(args.output_mode),
        }
        write_json(outdir / "contract.json", contract)
        
        # publish-gate: provenance schema must be complete even for NO_CLEAR
        meta["toolkit"] = TOOLKIT_NAME
        meta["toolkit_version"] = TOOLKIT_VERSION
        def _safe_version(modname: str) -> str:
            try:
                mod = __import__(modname)
                return str(getattr(mod, "__version__", "unknown"))
            except Exception:
                return "not_installed"
        meta["versions"] = {
            "python": platform.python_version(),
            "numpy": _safe_version("numpy"),
            "pandas": _safe_version("pandas"),
            "matplotlib": _safe_version("matplotlib"),
            "scipy": _safe_version("scipy"),
        }
        try:
            import matplotlib
            meta["matplotlib_backend"] = matplotlib.get_backend()
        except Exception:
            meta["matplotlib_backend"] = "unknown"

        write_json(outdir / "provenance.json", meta)
        write_json(outdir / "summary.json", summary)
        # index helpers
        write_text(outdir / "methods_blurb.txt", methods_blurb_text())
        write_text(outdir / "caption_text.txt", caption_text())
        write_text(outdir / "limitations_blurb.txt", limitations_blurb_text())
        save_index(outdir, {
            "summary.json": "Run summary + verdict",
            "contract.json": "Frozen contract (definitions + rules)",
            "provenance.json": "Runtime provenance",
            "methods_blurb.txt": "Short methods blurb",
            "caption_text.txt": "Suggested figure caption",
            "limitations_blurb.txt": "Limitations / claim boundaries",
        })
        print_executive_summary(summary, outdir)

        # PUBLISH_EXIT_POLICY_V3
        # PASS/FAIL => 0 (valid scientific outcomes)
        # NO_CLEAR/INVALID => 2 (publish hard stop / contract failure)
        v = summary.get('verdict', '') if isinstance(summary, dict) else ''
        if isinstance(v, str) and (v.startswith('PASS') or v.startswith('FAIL')):
            return 0
        return 2
    log("PREFLIGHT: OK")
    log("=================")

    # check-only exits here (no computation, no files)
    if args.check:
        log("=== CHECK ONLY (no computation, no files) ===")
        log(f"Input: {args.real}")
        log("============================================")
        return

    # dry-run exits here (no computation, no files)
    if args.dry_run:
        if not args.out:
            err("--out is required for --dry-run (to plan outputs)")
            sys.exit(2)
        log("=== DRY RUN (no computation, no files) ===")
        log(f"Outdir: {args.out}")
        planned = [
            "contract.json",
            "summary.json",
            "provenance.json",
            "n4_primary.json",
            "n4_controls_table.csv",
            "n4_plot.png",
            "report.html",
            "index.json",
            "index.html",
            "methods_blurb.txt",
            "caption_text.txt",
            "limitations_blurb.txt",
        ]
        # optional
        if args.real_runs.strip():
            planned.append("n4_robustness_table.csv")
        log("Planned outputs:")
        for p in planned:
            log(f"  - {p}")
        log("=========================================")
        return

    # compute run requires --out
    if not args.out:
        err("--out is required for computation")

    outdir = Path(args.out)
    ensure_dir(outdir)

    # Contract (write first)
    contract = {
        "toolkit": TOOLKIT_NAME,
        "toolkit_version": TOOLKIT_VERSION,
        "frozen_purpose": "Closure audit of frozen UE/UM estimator with mandatory kill-switch controls (not atomic prediction).",
        "headline_family": "ratio_i=(UE_i+eps)/(UM_i+eps); C4_i=log(ratio_i)",
        "headline_summaries": HEADLINE_KEYS,
        "diagnostics_only": DIAG_KEYS,
        "required_controls_v1": ["gauss_zero", "em_swap"],
        "missing_control_behavior": "NO_CLEAR_MISSING_CONTROLS",
        "verdict_rule": "Each control must worsen abs_C4_median AND (C4_MAD OR violation_frac_pi) vs REAL (no tuned thresholds).",
        "verdict_reason_codes_v1": REASON_ENUM,
        "frozen_functions_fingerprint": ["compute_case_metrics", "directional_controls_verdict", "decide_verdict"],
        "notes": [
            "n4 is intended to run after interaction is established (e.g., via n2).",
            "PASS indicates estimator-level closure sensitivity, not system safety/correctness.",
            "ψ4 is NOT: classifier, predictor, RUL model, root-cause engine, or nature-constant claim.",
        ],
    }
    write_json(outdir / "contract.json", contract)

    # Provenance (include input hashes unless disabled)
    meta: Dict[str, Any] = {
        "timestamp_utc": _now_utc_iso(),
        "platform": {"python": sys.version, "platform": platform.platform()},
        "argv": sys.argv,
        "self_sha256": sha256_self(),
        "dataset_tag": args.dataset_tag,
        "estimator": {
            "name": args.estimator_name,
            "version": args.estimator_version,
            "config_path": args.estimator_config,
        },
        "inputs": {
            "real": str(args.real),
            "gauss_zero": str(args.gauss_zero) if args.gauss_zero else None,
            "em_swap": str(args.em_swap) if args.em_swap else None,
        },
    }

    if not args.no_hash_input:
        meta["input_sha256"] = {
            "real": sha256_file(args.real),
            "gauss_zero": sha256_file(args.gauss_zero) if (args.gauss_zero and Path(args.gauss_zero).exists()) else None,
            "em_swap": sha256_file(args.em_swap) if (args.em_swap and Path(args.em_swap).exists()) else None,
            "estimator_config": sha256_file(args.estimator_config) if (args.estimator_config and Path(args.estimator_config).exists()) else None,
        }
    # --- reproducibility metadata (audit only; no science impact)
    def _safe_version(modname: str) -> str:
        try:
            mod = __import__(modname)
            return str(getattr(mod, "__version__", "unknown"))
        except Exception:
            return "not_installed"

    meta["toolkit"] = TOOLKIT_NAME
    meta["toolkit_version"] = TOOLKIT_VERSION
    meta["versions"] = {
        "python": platform.python_version(),
        "numpy": _safe_version("numpy"),
        "pandas": _safe_version("pandas"),
        "matplotlib": _safe_version("matplotlib"),
        "scipy": _safe_version("scipy"),
    }
    try:
        import matplotlib
        meta["matplotlib_backend"] = matplotlib.get_backend()
    except Exception:
        meta["matplotlib_backend"] = "unknown"
    write_json(outdir / "provenance.json", meta)

    # Write deterministic helper blurbs
    write_text(outdir / "methods_blurb.txt", methods_blurb_text())
    write_text(outdir / "caption_text.txt", caption_text())
    write_text(outdir / "limitations_blurb.txt", limitations_blurb_text())

    # Compute metrics
    print_header_block()

    df_real = load_frames_table(args.real)
    real_m = compute_case_metrics(df_real, eps=float(args.eps))

    gz_m = None
    em_m = None
    if ok_gz:
        df_gz = load_frames_table(args.gauss_zero)
        gz_m = compute_case_metrics(df_gz, eps=float(args.eps))
    if ok_em:
        df_em = load_frames_table(args.em_swap)
        em_m = compute_case_metrics(df_em, eps=float(args.eps))

    print_real_block(real_m)
    print_diagnostics_block(real_m)
    print_controls_block(real_m, gz_m, em_m)

    verdict, reason_code, reason_text, details = decide_verdict(real_m, gz_m, em_m, controls_present=controls_present)

    # Primary JSON (REAL only, frozen headline + diagnostics)
    primary = {
        "n_frames": int(real_m.get("n_frames", 0.0) or 0),
        "headline": {k: real_m.get(k) for k in HEADLINE_KEYS},
        "diagnostics": {k: real_m.get(k) for k in DIAG_KEYS},
        "secondary_reporting": {k: real_m.get(k) for k in ["ratio_mean", "C4_mean", "C4_std", "ratio_MAD"]},
    }
    write_json(outdir / "n4_primary.json", primary)

    # Tables
    controls_table = build_controls_table(real_m, gz_m, em_m)
    deltas_table = build_deltas_table(controls_table)
    df_to_csv(outdir / "n4_controls_table.csv", controls_table)

    # Optional robustness
    rr = parse_real_runs_list(args.real_runs)
    robustness_table = compute_robustness_table(rr, eps=float(args.eps))
    if robustness_table is not None:
        df_to_csv(outdir / "n4_robustness_table.csv", robustness_table)

    # Plot + report
    plot_n4(outdir, real_m, gz_m, em_m)
    save_report_html(outdir, contract, {
        "verdict": verdict,
        "verdict_reason_code": reason_code,
        "verdict_reason_text": reason_text,
        "logic_fingerprint": logic_fingerprint(),
        "controls_present": {"gauss_zero": bool(ok_gz), "em_swap": bool(ok_em)},
        "real": {k: real_m.get(k) for k in HEADLINE_KEYS + DIAG_KEYS + ["n_frames"]},
        "gauss_zero": {k: (gz_m.get(k) if gz_m else None) for k in HEADLINE_KEYS + DIAG_KEYS + ["n_frames"]},
        "em_swap": {k: (em_m.get(k) if em_m else None) for k in HEADLINE_KEYS + DIAG_KEYS + ["n_frames"]},
        "directional_details": details,
    }, meta, controls_table, deltas_table)

    # Summary JSON (machine-stable)
    summary = {
        "toolkit": TOOLKIT_NAME,
        "toolkit_version": TOOLKIT_VERSION,
        "dataset_tag": args.dataset_tag,
        "output_mode": str(args.output_mode),
        "verdict": verdict,
        "verdict_reason_code": reason_code,
        "verdict_reason_text": reason_text,
        "logic_fingerprint": logic_fingerprint(),
        "controls_present": {"gauss_zero": bool(ok_gz), "em_swap": bool(ok_em)},
        "real": {k: real_m.get(k) for k in HEADLINE_KEYS + DIAG_KEYS + ["n_frames"]},
        "gauss_zero": {k: (gz_m.get(k) if gz_m else None) for k in HEADLINE_KEYS + DIAG_KEYS + ["n_frames"]},
        "em_swap": {k: (em_m.get(k) if em_m else None) for k in HEADLINE_KEYS + DIAG_KEYS + ["n_frames"]},
        "directional_details": details,
    }
    write_json(outdir / "summary.json", summary)

    # Index
    artifacts = {
        "report.html": "Human-readable report (start here)",
        "n4_plot.png": "Plot (REAL vs controls)",
        "summary.json": "Run summary + verdict + reason codes",
        "contract.json": "Frozen definitions + rules + reason-code enum",
        "provenance.json": "Runtime provenance (argv, versions, hashes)",
        "n4_primary.json": "Primary (REAL) metrics (headline + diagnostics)",
        "n4_controls_table.csv": "REAL vs controls numeric table",
        "methods_blurb.txt": "Short methods blurb (deterministic)",
        "caption_text.txt": "Suggested figure caption",
        "limitations_blurb.txt": "Limitations / claim boundaries",
        "index.html": "This index",
        "index.json": "Index manifest",
    }
    if robustness_table is not None:
        artifacts["n4_robustness_table.csv"] = "Optional robustness table across multiple REAL runs"

    save_index(outdir, artifacts)
    prune_optional_artifacts_n4(outdir, args.output_mode)

    # Final executive summary to console
    print_executive_summary(summary, outdir)

    # PUBLISH_EXIT_POLICY_V3
    # PASS/FAIL => 0 (valid scientific outcomes)
    # NO_CLEAR/INVALID => 2 (publish hard stop / contract failure)
    v = summary.get('verdict', '') if isinstance(summary, dict) else ''
    if isinstance(v, str) and (v.startswith('PASS') or v.startswith('FAIL')):
        return 0
    return 2
if __name__ == "__main__":
    import sys
    sys.exit(main())
