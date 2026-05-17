#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
n1_falsification_ledger.py

Purpose (frozen):
- A publish-grade "ledger" that repeatedly calls the LOCKED n1_time-order_toolkit.py
  as a BLACK BOX, and summarizes falsification / null-safety behavior across cases.

What it IS:
- Driver / evaluator: run cases × repeats, collect verdicts, produce an auditable ledger.

What it is NOT:
- NOT a detector
- NOT a baseline fitter
- NOT allowed to import toolkit internals (no from n1_time-order_toolkit import ...)
- NO threshold changes, NO math changes, NO touching toolkit logic.

Outputs (stable):
- ledger.csv              (row per case×repeat)
- ledger_summary.csv      (one row per case with counts)
- ledger_summary.json     (same as csv, machine-readable)
- contract.json           (frozen scope + rules)
- provenance.json         (cmdline + environment)
- report.html             (simple human view)
- runs/<case>/run_XXX/    (full toolkit outputs per run, for audit; omitted when --no-runs is used)
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


EXIT_OK = 0
EXIT_INVALID = 2


def _selftest() -> int:
    """
    Strong selftest (scoreboard-style):
    - Creates a temp outdir
    - Runs a tiny real ledger (repeats=2) using the included cases file
    - Asserts required artifacts exist
    - Asserts report.html has no NaN/Inf/Infinity tokens
    - Cleans up temp outdir (leak guard)
    """
    import tempfile
    import subprocess
    import shutil
    from pathlib import Path

    print("[SELFTEST] n1_falsification_ledger")

    # Prefer repo cases file. If not present, fall back to invariants-only.
    candidates = [
        Path("scratch") / "cases_n1.json",
        Path(__file__).resolve().parent / "cases_n1.json",
    ]
    cases_path = next((p for p in candidates if p.exists()), None)
    if cases_path is None:
        assert isinstance(PLANNED_OUTPUTS, list) and len(PLANNED_OUTPUTS) >= 3
        print("[SELFTEST_OK] (no cases_n1.json found; invariants-only)")
        return EXIT_OK

    tmp_root = Path(tempfile.mkdtemp(prefix="n1_selftest_"))
    outdir   = tmp_root / "out"

    try:
        cmd = [
            sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
            "--cases", str(cases_path),
            "--outdir", str(outdir),
            "--repeats", "2",
            "--base-seed", "0",
            "--no-runs",
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)

        if p.returncode != 0:
            raise AssertionError(
                f"selftest run failed rc={p.returncode}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
            )

        must = [
            "ledger.csv", "ledger_summary.csv", "ledger_summary.json",
            "contract.json", "provenance.json", "report.html",
        ]
        for f in must:
            fp = outdir / f
            if not fp.exists():
                raise AssertionError(f"missing artifact in selftest: {fp}")

        html = (outdir / "report.html").read_text(encoding="utf-8", errors="replace")
        # Guard against numeric NaN/Inf tokens WITHOUT false-positives like "infile".
        # Match standalone tokens (optionally with leading "-"), case-insensitive.
        pat = re.compile(r"(?i)(?<![A-Za-z0-9_])(?:-?inf|infinity|nan)(?![A-Za-z0-9_])")
        if pat.search(html):
            raise AssertionError("report.html contains numeric NaN/Inf token(s)")

        print("[SELFTEST_OK]")
        return EXIT_OK
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

DEFAULT_TOOLKIT = Path("scratch") / "n1_time-order_toolkit.py"
PLANNED_OUTPUTS_BASE = [
    "ledger.csv",
    "ledger_summary.csv",
    "ledger_summary.json",
    "contract.json",
    "provenance.json",
    "report.html",
]

PLANNED_OUTPUTS_WITH_RUNS = PLANNED_OUTPUTS_BASE + ["runs/"]

# Default for early-bypass paths (no argparse args yet): assume runs/ is included.
PLANNED_OUTPUTS = PLANNED_OUTPUTS_WITH_RUNS
# Early bypass: allow --selftest without required argparse args
if "--selftest" in sys.argv:
    raise SystemExit(_selftest())

# Early bypass: allow --dry-run without required argparse args
if "--dry-run" in sys.argv and "--cases" not in sys.argv and "--outdir" not in sys.argv:
    print("[DRY-RUN] PLANNED_OUTPUTS:")
    for x in PLANNED_OUTPUTS:
        print(f"  - {x}")
    raise SystemExit(EXIT_OK)

# -------------------------
# Small utilities (no deps)
# -------------------------

def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def read_json_bom_safe(path: Path) -> Any:
    # Accept UTF-8 with or without BOM (Windows-safe).
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sanitize_json(x: Any) -> Any:
    """
    Publish-safe JSON: replace NaN/Inf with None recursively.
    Keeps schema stable and prevents non-standard JSON tokens.
    """
    import math
    if isinstance(x, float):
        if not math.isfinite(x):
            return None
        return x
    if isinstance(x, dict):
        return {k: _sanitize_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_sanitize_json(v) for v in x]
    if isinstance(x, tuple):
        return [_sanitize_json(v) for v in x]
    return x


def write_json(path: Path, obj: Any) -> None:
    safe = _sanitize_json(obj)
    path.write_text(
        json.dumps(safe, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8"
    )


def now_utc_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def verdict_bucket(verdict: Optional[str]) -> str:
    """
    Normalize toolkit verdict strings into a small stable bucket set.
    Toolkit v1 uses at least:
      PASS_STRONG, PASS_WEAK, NO_CLEAR, NO_CLEAR_MISSING_CONTROLS
    """
    if verdict is None:
        return "NO_OUTPUT"
    v = str(verdict).strip().upper()
    if v.startswith("PASS_"):
        return "PASS"
    if v.startswith("NO_CLEAR"):
        return "NO_CLEAR"
    return v


def expected_ok(expected: str, bucket: str) -> bool:
    """
    expected values allowed in cases.json:
      - "PASS"             => require PASS
      - "NO_CLEAR"         => require NO_CLEAR
      - "PASS_OR_NO_CLEAR" => accept either
      - "ANY"              => accept any bucket (still must be exit0)
    """
    exp = expected.strip().upper()
    if exp == "ANY":
        return True
    if exp == "PASS_OR_NO_CLEAR":
        return bucket in ("PASS", "NO_CLEAR")
    return bucket == exp


# -------------------------
# Black-box runner
# -------------------------

def run_one(
    toolkit: Path,
    case: Dict[str, Any],
    repeat_index: int,
    run_dir: Path,
    base_seed: int,
    quiet: bool = False,
) -> Tuple[int, Optional[str], Optional[str], str]:
    """
    Returns:
      (exit_code, verdict_raw, verdict_bucket, stdout_tail)
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    infile = str(case["infile"])
    signal_col = str(case["signal_col"])

    # Defaults (safe + consistent)
    fs = str(case.get("fs", 0))
    win = str(case.get("win", 1024))
    step = str(case.get("step", 256))
    far = str(case.get("far", 0.05))
    n_null = str(case.get("n_null", 50))
    seed = str(int(base_seed) + int(repeat_index))

    # Optional knobs forwarded verbatim if present
    # (keeps ledger generic but still black-box)
    extra_args: List[str] = []
    passthru_keys = [
        "lowfreq_cut_hz",  # if toolkit uses it (harmless if unused when omitted)
        "max_windows",
        "window_family",
        "null_family",
        "no_plot",
        "verbose",
        "quiet",
    ]
    for k in passthru_keys:
        if k not in case:
            continue
        v = case[k]
        # boolean flags
        if isinstance(v, bool):
            if v:
                extra_args.append(f"--{k.replace('_', '-')}")
        else:
            extra_args.extend([f"--{k.replace('_', '-')}", str(v)])

    toolkit_args = case.get("toolkit_args", {}) if isinstance(case, dict) else {}
    # Required base args
    cmd = [
        sys.executable, "-X", "utf8", str(toolkit),
        "--infile", infile,
        "--sig-col", signal_col,
        "--seed", seed,
        "--out", str(run_dir),
    ]

    # Forward locked toolkit knobs (if present)
    # (Only add flags when values exist; keeps this judge black-box and schema-driven)
    def add_k(flag, key):
        if isinstance(toolkit_args, dict) and key in toolkit_args and toolkit_args[key] is not None:
            cmd.extend([flag, str(toolkit_args[key])])

    add_k("--block-len",     "block_len")
    add_k("--n-phase-null",  "n_phase_null")
    add_k("--n-block-null",  "n_block_null")
    add_k("--timeline-win",  "timeline_win")
    add_k("--timeline-step", "timeline_step")
    add_k("--lowfreq-cut",   "lowfreq_cut")
    add_k("--dataset-tag",   "dataset_tag")

    cmd = cmd + extra_args

    if quiet:
        cmd.append("--quiet")

    p = subprocess.run(cmd, capture_output=True, text=True)
    code = int(p.returncode)

    # Capture a short tail for debugging only (ledger still deterministic).
    combined = (p.stdout or "") + "\n" + (p.stderr or "")
    tail = "\n".join([ln for ln in combined.splitlines() if ln.strip()][-30:]).strip()

    verdict_raw: Optional[str] = None
    bucket: Optional[str] = None

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summ = read_json_bom_safe(summary_path)
            verdict_raw = str(summ.get("verdict", "")).strip() or None
            bucket = verdict_bucket(verdict_raw)
        except Exception as ex:
            verdict_raw = None
            bucket = "SUMMARY_PARSE_FAIL"
            tail = (tail + f"\n[LEDGER] summary.json parse error: {type(ex).__name__}: {ex}").strip()

    return code, verdict_raw, bucket, tail


# -------------------------
# Reporting
# -------------------------

def write_report_html(outdir: Path, rows: List[Dict[str, Any]], summary_rows: List[Dict[str, Any]], contract: Dict[str, Any]) -> None:
    def esc(s: Any) -> str:
        s = "" if s is None else str(s)
        return (
            s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
        )

    # Simple, reviewer-friendly HTML (no JS).
    parts = []
    parts.append("<!doctype html><html><head><meta charset='utf-8' />")
    parts.append("<title>n1 falsification ledger</title>")
    parts.append("<style>body{font-family:Arial,Helvetica,sans-serif;max-width:1100px;margin:20px;} table{border-collapse:collapse;} th,td{border:1px solid #ddd;padding:6px 8px;} th{background:#f5f5f5;} code{background:#f7f7f7;padding:1px 4px;border-radius:4px;}</style>")
    parts.append("</head><body>")
    parts.append("<h1>n1 falsification ledger</h1>")

    parts.append("<h2>Contract</h2>")
    parts.append("<pre>" + esc(json.dumps(contract, indent=2, ensure_ascii=False)) + "</pre>")

    parts.append("<h2>Summary by case</h2>")
    parts.append("<table><thead><tr>")
    for k in summary_rows[0].keys() if summary_rows else []:
        parts.append("<th>" + esc(k) + "</th>")
    parts.append("</tr></thead><tbody>")
    for r in summary_rows:
        parts.append("<tr>")
        for k in r.keys():
            parts.append("<td>" + esc(r.get(k)) + "</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")

    parts.append("<h2>Runs (case × repeat)</h2>")
    parts.append("<table><thead><tr>")
    for k in rows[0].keys() if rows else []:
        parts.append("<th>" + esc(k) + "</th>")
    parts.append("</tr></thead><tbody>")
    for r in rows:
        parts.append("<tr>")
        for k in r.keys():
            parts.append("<td>" + esc(r.get(k)) + "</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")

    parts.append("</body></html>")
    (outdir / "report.html").write_text("\n".join(parts), encoding="utf-8")


# -------------------------
# Main
# -------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="n1 falsification ledger (black-box runner for n1_time-order_toolkit.py)",
    )
    ap.add_argument("--cases", required=True, help="Path to cases.json (UTF-8; BOM ok).")
    ap.add_argument("--outdir", required=True, help="Output directory (will be created/overwritten).")
    ap.add_argument("--toolkit", default=str(DEFAULT_TOOLKIT), help="Path to n1 toolkit script.")
    ap.add_argument("--repeats", type=int, default=20, help="Repeats per case (default: 20).")
    ap.add_argument("--base-seed", type=int, default=0, help="Base seed for repeat seeds (default: 0).")
    ap.add_argument("--no-runs", action="store_true", help="Do not keep per-run toolkit outputs (no runs/ folder).")
    ap.add_argument("--output-mode", choices=["publish-min", "standard", "full-audit"], default="standard",
                    help="Artifact volume control only. publish-min aliases to --no-runs; does not change toolkit verdict logic.")
    ap.add_argument("--dry-run", action="store_true", help="Print PLANNED_OUTPUTS and exit 0 (no writes).")
    ap.add_argument("--check", action="store_true", help="Validate inputs and exit 0 (no writes).")
    ap.add_argument("--quiet", action="store_true", help="Reduce child stdout/stderr captured (still recorded).")
    args = ap.parse_args()

    if str(args.output_mode).strip().lower() == "publish-min":
        args.no_runs = True

    if args.dry_run:
        planned = PLANNED_OUTPUTS_BASE if args.no_runs else PLANNED_OUTPUTS_WITH_RUNS
        print(f"[DRY-RUN] OUTPUT_MODE={args.output_mode}")
        print("[DRY-RUN] PLANNED_OUTPUTS:")
        for x in planned:
            print(f"  - {x}")
        return EXIT_OK

    outdir = Path(args.outdir)
    toolkit = Path(args.toolkit)

    # Publish-gate: toolkit must exist
    if not toolkit.exists():
        eprint(f"[ERROR] INVALID_INPUT_missing_toolkit: toolkit={toolkit}")
        return EXIT_INVALID

    cases_path = Path(args.cases)
    if not cases_path.exists():
        eprint(f"[ERROR] INVALID_INPUT_missing_cases: cases={cases_path}")
        return EXIT_INVALID

    # Load cases (BOM-safe)
    try:
        cases = read_json_bom_safe(cases_path)
    except Exception as ex:
        eprint(f"[ERROR] INVALID_INPUT_cases_parse_failed: {type(ex).__name__}: {ex}")
        return EXIT_INVALID

    if not isinstance(cases, list) or not cases:
        eprint("[ERROR] INVALID_INPUT_cases_empty_or_not_list")
        return EXIT_INVALID

    # Validate case schema minimally
    for c in cases:
        if not isinstance(c, dict):
            eprint("[ERROR] INVALID_INPUT_case_not_object")
            return EXIT_INVALID
        for k in ("name", "infile", "signal_col", "expected"):
            if k not in c:
                eprint(f"[ERROR] INVALID_INPUT_case_missing_key: key={k} case={c}")
                return EXIT_INVALID


    # Publish-gate: each case infile must exist BEFORE we create outdir
    # (prevents partial artifact packs on invalid inputs)
    for c in cases:
        inf = Path(str(c["infile"]))
        if not inf.exists():
            eprint(f"[ERROR] INVALID_INPUT_missing_infile: infile={inf}")
            return EXIT_INVALID
        if inf.is_dir():
            eprint(f"[ERROR] INVALID_INPUT_infile_is_dir: infile={inf}")
            return EXIT_INVALID

        # Optional but strong: require signal_col present in CSV header
        # (cheap check; still black-box: we do not interpret data)
        sigcol = str(c["signal_col"])
        try:
            with inf.open("r", encoding="utf-8-sig", newline="") as f:
                header = f.readline().strip()
        except Exception as ex:
            eprint(f"[ERROR] INVALID_INPUT_infile_unreadable: infile={inf} err={type(ex).__name__}:{ex}")
            return EXIT_INVALID

        # Accept comma-separated header; if no header, this will fail cleanly
        # Parse header robustly: support common delimiters and quoted fields
        cols = []
        if header:
            try:
                import csv as _csv
                # Sniff delimiter from the first line only (cheap)
                dialect = _csv.Sniffer().sniff(header, delimiters=[",",";","	","|"])
                cols = next(_csv.reader([header], dialect))
            except Exception:
                # Fallback: split on common delimiters
                cols = re.split(r"[;,	|]", header)
            cols = [c.strip().strip('"').strip("'") for c in cols if c is not None]
        if sigcol not in cols:
            eprint(f"[ERROR] INVALID_INPUT_missing_signal_col: infile={inf} signal_col={sigcol}")
            return EXIT_INVALID

    # If --check: all validations passed; do not create outdir, do not run toolkit.
    if args.check:
        print("[OK] CHECK passed")
        print(f"  toolkit: {toolkit}")
        print(f"  cases: {cases_path}  (n_cases={len(cases)})")
        return EXIT_OK

    # Fresh outdir (ledger is its own artifact pack)
    ensure_clean_dir(outdir)
    runs_dir = outdir / "runs"
    if not args.no_runs:
        runs_dir.mkdir(parents=True, exist_ok=True)

    planned_outputs = PLANNED_OUTPUTS_BASE if args.no_runs else PLANNED_OUTPUTS_WITH_RUNS

    contract = {
        "tool": "n1_falsification_ledger",
        "scope": {
            "black_box_only": True,
            "no_toolkit_internal_imports": True,
            "no_threshold_changes": True,
            "no_math_changes": True,
        },
        "expected_semantics": {
            "exit_codes_allowed": [0, 2],
            "PASS_bucket": "verdict starts with PASS_",
            "NO_CLEAR_bucket": "verdict starts with NO_CLEAR",
            "cases_expected_values": ["PASS", "NO_CLEAR", "PASS_OR_NO_CLEAR", "ANY"],
        },
        "planned_outputs": planned_outputs,
        "created_utc": now_utc_iso(),
    }
    write_json(outdir / "contract.json", contract)

    provenance = {
        "utc": now_utc_iso(),
        "argv": sys.argv,
        "python": sys.version,
        "executable": sys.executable,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "toolkit_path": str(toolkit),
        "cases_path": str(cases_path),
        "repeats": int(args.repeats),
        "base_seed": int(args.base_seed),
    }
    write_json(outdir / "provenance.json", provenance)

    rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    # HARD SAFETY: fail hard if any null case yields PASS (if expected NO_CLEAR)
    null_violation_total = 0
    invalid_exit_total = 0

    for case in cases:
        name = str(case["name"])
        expected = str(case["expected"]).strip().upper()
        case_stats = {
            "case": name,
            "expected": expected,
            "total_runs": 0,
            "exit0": 0,
            "exit1": 0,
            "exit2": 0,
            "exit_other": 0,
            "PASS": 0,
            "NO_CLEAR": 0,
            "NO_OUTPUT": 0,
            "SUMMARY_PARSE_FAIL": 0,
            "other_bucket": 0,
            "expected_mismatch": 0,
            "null_violation": 0,
        }
        if args.no_runs:
            import tempfile
            case_runs_base = Path(tempfile.mkdtemp(prefix=f"n1_runs_{name}_"))
        else:
            case_runs_base = runs_dir / name
            case_runs_base.mkdir(parents=True, exist_ok=True)
        for r in range(int(args.repeats)):
            run_dir = case_runs_base / f"run_{r:03d}"

            code, verdict_raw, bucket, tail = run_one(
                toolkit=toolkit,
                case=case,
                repeat_index=r,
                run_dir=run_dir,
                base_seed=int(args.base_seed),
                quiet=bool(args.quiet),
            )

            # Enforce exit discipline (locked n1 policy: only {0,2})
            if code not in (0, 2):
                invalid_exit_total += 1

            # Bucketing
            b = (bucket if bucket is not None else "NO_OUTPUT")
            bucket_norm = b if b in ("PASS", "NO_CLEAR", "NO_OUTPUT", "SUMMARY_PARSE_FAIL") else "OTHER"

            # Check expected vs observed (only meaningful when exit0 + has bucket)
            mismatch = 0
            if code == 0 and bucket is not None:
                if expected in ("PASS", "NO_CLEAR", "PASS_OR_NO_CLEAR", "ANY"):
                    if not expected_ok(expected, verdict_bucket(verdict_raw)):
                        mismatch = 1

            # Null violation rule: if expected NO_CLEAR but we got PASS on exit0
            null_violation = 0
            if expected == "NO_CLEAR" and code == 0 and verdict_bucket(verdict_raw) == "PASS":
                null_violation = 1
                null_violation_total += 1

            # Record row
            rows.append({
                "case": name,
                "repeat": r,
                "seed": int(args.base_seed) + r,
                "exit_code": code,
                "verdict_raw": verdict_raw if verdict_raw is not None else "",
                "bucket": verdict_bucket(verdict_raw) if verdict_raw is not None else (bucket or "NO_OUTPUT"),
                "expected": expected,
                "expected_mismatch": mismatch,
                "null_violation": null_violation,
                "run_dir": str(run_dir).replace("\\", "/"),
                "stdout_stderr_tail": tail,
            })

            # Update stats
            case_stats["total_runs"] += 1
            if code == 0:
                case_stats["exit0"] += 1
            elif code == 1:
                case_stats["exit1"] += 1
            elif code == 2:
                case_stats["exit2"] += 1
            else:
                case_stats["exit_other"] += 1

            # bucket counts
            vb = verdict_bucket(verdict_raw) if verdict_raw is not None else (bucket or "NO_OUTPUT")
            if vb == "PASS":
                case_stats["PASS"] += 1
            elif vb == "NO_CLEAR":
                case_stats["NO_CLEAR"] += 1
            elif vb == "NO_OUTPUT":
                case_stats["NO_OUTPUT"] += 1
            elif vb == "SUMMARY_PARSE_FAIL":
                case_stats["SUMMARY_PARSE_FAIL"] += 1
            else:
                case_stats["other_bucket"] += 1

            case_stats["expected_mismatch"] += mismatch
            case_stats["null_violation"] += null_violation

            # Optionally delete run artifacts (still keep ledger row)
            if args.no_runs:
                shutil.rmtree(run_dir, ignore_errors=True)

        summary_rows.append(case_stats)

        if args.no_runs:
            shutil.rmtree(case_runs_base, ignore_errors=True)

        # HARD FAIL immediately if null violation occurs
        if case_stats["null_violation"] > 0:
            eprint(f"[ERROR] NULL_SAFETY_VIOLATION: case={name} violations={case_stats['null_violation']}")
            eprint("This ledger is designed to FAIL HARD on null safety violations.")
            # Still write what we have so far for debugging
            break

        # HARD FAIL if any non {0,1,2} exit appeared
        if case_stats["exit_other"] > 0:
            eprint(f"[ERROR] INVALID_EXIT_CODE: case={name} exit_other={case_stats['exit_other']}")
            break

    # Write outputs
    # ledger.csv (row per run)
    ledger_csv = outdir / "ledger.csv"
    with ledger_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    # ledger_summary.csv / json
    summary_csv = outdir / "ledger_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            w.writeheader()
            w.writerows(summary_rows)

    write_json(outdir / "ledger_summary.json", summary_rows)

    # report.html
    if summary_rows:
        write_report_html(outdir, rows, summary_rows, contract)

    print("[OK] Ledger complete")
    print(f"  outdir: {outdir}")
    print(f"  cases: {len(cases)}")
    print(f"  repeats: {int(args.repeats)}")
    if null_violation_total:
        print(f"[WARN] null violations: {null_violation_total}")

    # Final exit policy:
    # - If any null violation -> exit 2
    # - If any invalid exit code -> exit 2
    # - Else exit 0
    if null_violation_total > 0:
        return EXIT_INVALID
    if invalid_exit_total > 0:
        eprint(f"[ERROR] INVALID_EXIT_CODES_SEEN: count={invalid_exit_total}")
        return EXIT_INVALID

    return EXIT_OK


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        eprint("[ERROR] INTERRUPTED")
        sys.exit(EXIT_INVALID)
    except Exception as ex:
        eprint(f"[ERROR] EXCEPTION: {type(ex).__name__}: {ex}")
        sys.exit(EXIT_INVALID)

