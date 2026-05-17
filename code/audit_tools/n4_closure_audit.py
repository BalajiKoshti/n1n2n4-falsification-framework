#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
n4_closure_audit.py  â€”  Ïˆ4 n4 Closure Audit Matrix (Tool B)

Purpose (frozen):
  A consistency audit, not a detector.
  Question: "Does closure survive only in REAL systems and collapse under closure-breaking controls?"

Non-negotiable rules:
  - No statistics, no significance, no thresholds, no detector language, no competition.
  - Direction-only audit rule:
        control_pass = (closure(control) is worse than closure(REAL))
    using a single closure scalar derived from the frozen n4 closure contract.
  - Final verdict:
        PASS iff REAL is best AND all controls worsen vs REAL.

Exit codes:
  - 0: valid audit completed
  - 2: invalid input / missing file / runtime failure

Outputs (locked):
  - audit_matrix.json
  - audit_matrix.txt
  - summary.json
  - contract.json
  - provenance.json
  - index.html
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import csv
import os
import shutil
import sys
import tempfile
import math
from pathlib import Path
from typing import Any, Dict

# ----------------------------- Constants -----------------------------

PLANNED_OUTPUTS = [
    "audit_matrix.json",
    "audit_matrix.txt",
    "summary.json",
    "contract.json",
    "provenance.json",
    "index.html",
]

def planned_outputs_for_mode(output_mode: str) -> list[str]:
    # Minimal-risk lock: audit pack is already compact.
    # Keep all modes identical for now to avoid behavior drift.
    return list(PLANNED_OUTPUTS)




# We treat "closure" as: abs_C4_median from the frozen n4 closure toolkit.
# Convention: LOWER is better closure consistency.
CLOSURE_FIELD = "abs_C4_median"
CLOSURE_DIRECTION = "lower_is_better"


def _selftest() -> int:
    """
    Tier-2 selftest:
      - create temp outdir
      - generate tiny synthetic inputs internally
      - run full pipeline (black-box subprocess call to this script)
      - assert evidence-pack artifacts exist
      - scan HTML/JSON for nan/inf/infinity tokens
      - delete temp dir (leak-guard)
    """
    print("[SELFTEST] n4_closure_audit")

    # EARLY-SAFE: keep everything local (selftest may run before helper defs are used elsewhere).
    import sys, json, shutil, tempfile, subprocess, math, random
    from pathlib import Path
    import re as _re

    tmp_root = Path(tempfile.mkdtemp(prefix="n4_audit_selftest_"))
    try:
        data_dir = tmp_root / "data"
        out_dir  = tmp_root / "out"
        data_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Synthetic CSVs with UE/UM columns (minimum required by toolkit contract) ---
        def write_csv(path: Path, UE, UM):
            path.write_text(
                "UE,UM\n" + "\n".join(f"{ue:.10g},{um:.10g}" for ue, um in zip(UE, UM)) + "\n",
                encoding="utf-8"
            )

        n = 2048
        t = range(n)

        # REAL: relatively tight UE~UM with mild structure
        rng = random.Random(0)
        UE_real = [1.0 + 0.01*math.sin(2*math.pi*i/128.0) + 0.002*(rng.random()-0.5) for i in t]
        UM_real = [1.0 + 0.002*(rng.random()-0.5) for i in t]

        # GAUSS_ZERO: noisier imbalance
        rng = random.Random(1)
        UE_gz = [1.0 + 0.05*(rng.random()-0.5) for i in t]
        UM_gz = [1.0 + 0.05*(rng.random()-0.5) for i in t]

        # EM_SWAP: different systematic mismatch pattern
        rng = random.Random(2)
        UE_em = [1.0 + 0.02*(rng.random()-0.5) for i in t]
        UM_em = [1.0 + 0.03*math.sin(2*math.pi*i/64.0) + 0.02*(rng.random()-0.5) for i in t]

        real_csv  = data_dir / "real.csv"
        gz_csv    = data_dir / "gauss_zero.csv"
        em_csv    = data_dir / "em_swap.csv"
        write_csv(real_csv, UE_real, UM_real)
        write_csv(gz_csv,   UE_gz,   UM_gz)
        write_csv(em_csv,   UE_em,   UM_em)

        # Run this audit script as a black-box subprocess (full pipeline includes toolkit run)
        audit_py = Path(__file__).resolve()
        cmd = [
            sys.executable, "-X", "utf8", str(audit_py),
            "--real", str(real_csv),
            "--gauss-zero", str(gz_csv),
            "--em-swap", str(em_csv),
            "--out", str(out_dir),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print("[SELFTEST_FAIL] audit subprocess failed")
            print("---- STDOUT ----")
            print(r.stdout)
            print("---- STDERR ----")
            print(r.stderr)
            return 2

        required = [
            "audit_matrix.json",
            "audit_matrix.txt",
            "summary.json",
            "contract.json",
            "provenance.json",
            "index.html",
        ]
        missing = [f for f in required if not (out_dir / f).exists()]
        if missing:
            print("[SELFTEST_FAIL] missing artifacts:", ", ".join(missing))
            return 2

        # HTML gate scan
        html = (out_dir / "index.html").read_text(encoding="utf-8", errors="replace")
        if _re.search(r"(?i)(?<![A-Za-z0-9_])(?:-?inf|infinity|nan)(?![A-Za-z0-9_])", html, flags=_re.IGNORECASE):
            print("[SELFTEST_FAIL] HTML contains nan/inf/infinity tokens")
            return 2

        # JSON parse + token scan
        for jf in ["audit_matrix.json", "summary.json", "contract.json", "provenance.json"]:
            txt = (out_dir / jf).read_text(encoding="utf-8", errors="replace")
            if _re.search(r"(?i)(?<![A-Za-z0-9_])(?:-?inf|infinity|nan)(?![A-Za-z0-9_])", txt, flags=_re.IGNORECASE):
                print(f"[SELFTEST_FAIL] JSON contains nan/inf/infinity token: {jf}")
                return 2
            try:
                json.loads(txt)
            except Exception:
                print(f"[SELFTEST_FAIL] JSON parse failed: {jf}")
                return 2

        print("[SELFTEST_OK]")
        return 0

    finally:
        # leak guard: always remove temp root (includes out_dir)
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass
def _utc_now_iso() -> str:
    # timezone-aware UTC, no deprecation warnings
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _print_err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _safe_rm(path: Path) -> None:
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _atomic_commit(tmp_out: Path, final_out: Path) -> None:
    # Remove existing final_out (if any) then move tmp_out into place
    if final_out.exists():
        shutil.rmtree(final_out, ignore_errors=True)
    tmp_out.replace(final_out)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_text(path: Path, s: str) -> None:
    path.write_text(s, encoding="utf-8")


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&#39;")
    )


def _fmt_num(x: Any) -> str:
    # display-only formatting (no math changes)
    try:
        if x is None:
            return "N/A"
        if isinstance(x, (int, float)):
            if x != x:  # NaN
                return "N/A"
            if x == float("inf") or x == float("-inf"):
                return "N/A"
            return f"{x:.6g}"
        return str(x)
    except Exception:
        return "N/A"


# ----------------------------- Black-box toolkit runner -----------------------------

def _toolkit_script_path() -> Path:
    # Expect toolkit beside this audit in scratch/
    return (Path(__file__).resolve().parent / "n4_closure_toolkit.py")

def _run_toolkit_blackbox(real_path: Path, gauss_path: Path, emswap_path: Path, outdir: Path) -> None:
    """
    Black-box run of n4_closure_toolkit.py.
    Writes an evidence pack to outdir (temporary), then audit reads ONLY JSON outputs.
    """
    import subprocess

    tk = _toolkit_script_path()
    if not tk.exists():
        raise RuntimeError(f"Missing toolkit script: {tk}")

    cmd = [
        sys.executable, "-X", "utf8", str(tk),
        "--real", str(real_path),
        "--gauss-zero", str(gauss_path),
        "--em-swap", str(emswap_path),
        "--out", str(outdir),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        # Bubble up toolkit stderr/stdout as part of the audit failure message (publish debugging)
        raise RuntimeError(
            "Toolkit run failed "
            f"(rc={r.returncode}).\\n---- STDOUT ----\\n{r.stdout}\\n---- STDERR ----\\n{r.stderr}"
        )

def _read_json_if_exists(p: Path):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def _extract_absC4(obj, label: str) -> float:
    """
    Robustly extract abs_C4_median from toolkit outputs.
    Supports common shapes:
      - {"abs_C4_median": 0.123}
      - {"REAL": {"abs_C4_median": ...}, "GAUSS_ZERO": {...}, ...}
      - {"cases": {"REAL": {...}, ...}}
      - [{"case":"REAL","abs_C4_median":...}, ...]
    """
    key = CLOSURE_FIELD

    def find_in_dict(d):
        if isinstance(d, dict):
            if key in d:
                return d[key]
            # common nesting
            for k in ("cases", "rows", "metrics"):
                if k in d and isinstance(d[k], (dict, list)):
                    v = find_in_dict(d[k])
                    if v is not None:
                        return v
        if isinstance(d, list):
            for it in d:
                v = find_in_dict(it)
                if v is not None:
                    return v
        return None

    if not isinstance(obj, (dict, list)):
        raise RuntimeError(f"Toolkit output for {label} is not JSON object/list")

    # Try labeled branches first (REAL/GAUSS_ZERO/EM_SWAP)
    if isinstance(obj, dict):
        for branch in (label, label.upper()):
            if branch in obj:
                v = find_in_dict(obj[branch])
                if v is not None:
                    return float(v)
        # common: obj["cases"][label]
        if "cases" in obj and isinstance(obj["cases"], dict) and label in obj["cases"]:
            v = find_in_dict(obj["cases"][label])
            if v is not None:
                return float(v)

    # fallback: search any structure (last resort)
    v = find_in_dict(obj)
    if v is None:
        raise RuntimeError(f"Could not find {key} for {label} in toolkit outputs")
    return float(v)

def _read_closures_from_toolkit_outdir(run_dir: Path) -> dict:
    """
    Extract closure scalars from the BLACK-BOX toolkit output directory.

    Preferred source (stable for REAL+controls): n4_controls_table.csv
      - case labels may be:
          REAL
          CONTROL: gauss_zero
          CONTROL: em_swap

    Fallback: n4_primary.json (REAL only) ? but controls are mandatory for audit,
    so if controls table is missing, we treat as INVALID.
    """
    controls_csv = run_dir / "n4_controls_table.csv"
    if controls_csv.exists():
        with controls_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows or ("case" not in rows[0]):
            raise RuntimeError("n4_controls_table.csv present but missing 'case' column")

        def pick(case_name: str):
            want = str(case_name).strip()
            want2 = f"CONTROL: {want}"
            for r in rows:
                c = str(r.get("case", "")).strip()
                if c == want or c == want2:
                    v = r.get(CLOSURE_FIELD, None)
                    if v is None:
                        return None
                    sv = str(v).strip()
                    if sv == "" or sv.lower() == "none":
                        return None
                    return float(sv)
            return None

        real = pick("REAL")
        gz   = pick("gauss_zero")
        em   = pick("em_swap")

        if real is None:
            raise RuntimeError(f"Could not find {CLOSURE_FIELD} for REAL in n4_controls_table.csv")
        if gz is None:
            raise RuntimeError(f"Could not find {CLOSURE_FIELD} for gauss_zero in n4_controls_table.csv")
        if em is None:
            raise RuntimeError(f"Could not find {CLOSURE_FIELD} for em_swap in n4_controls_table.csv")

        return {"REAL": real, "GAUSS_ZERO": gz, "EM_SWAP": em}

    # Fallback: REAL only from n4_primary.json (but controls are REQUIRED for audit)
    primary = _read_json_if_exists(run_dir / "n4_primary.json")
    if not isinstance(primary, dict):
        raise RuntimeError("Toolkit outputs missing n4_controls_table.csv and unreadable n4_primary.json")

    headline = primary.get("headline", {})
    if not isinstance(headline, dict) or (CLOSURE_FIELD not in headline):
        raise RuntimeError(f"Could not find {CLOSURE_FIELD} under n4_primary.json['headline']")

    # REAL exists, but controls missing => invalid for audit
    _ = float(headline[CLOSURE_FIELD])
    raise RuntimeError(
        "Toolkit outputs missing n4_controls_table.csv; cannot extract GAUSS_ZERO/EM_SWAP closures (required)."
    )

# ----------------------------- Core audit -----------------------------
# ----------------------------- Core audit -----------------------------


def run_audit(real_path: Path, gauss_path: Path, emswap_path: Path) -> Dict[str, Any]:
    # Black-box: run toolkit in a temp dir, read only its frozen JSON outputs.
    tmp_run = Path(tempfile.mkdtemp(prefix="n4_audit_toolkit_", dir=str(real_path.parent)))
    try:
        _run_toolkit_blackbox(real_path, gauss_path, emswap_path, tmp_run)
        clos = _read_closures_from_toolkit_outdir(tmp_run)

        real_closure = float(clos["REAL"])
        gaus_closure = float(clos["GAUSS_ZERO"])
        emsw_closure = float(clos["EM_SWAP"])

        def _finite(x):
            return isinstance(x, (int, float)) and math.isfinite(float(x))

        if not (_finite(real_closure) and _finite(gaus_closure) and _finite(emsw_closure)):
            raise RuntimeError(
                f"Non-finite closure metric: REAL={real_closure} GAUSS_ZERO={gaus_closure} EM_SWAP={emsw_closure}"
            )

        def worsen(control_val: float, real_val: float) -> bool:
            # direction-only; strict
            # lower_is_better => worse means greater
            return control_val > real_val

        row_real = {
            "row": "REAL",
            "infile": str(real_path.as_posix()),
            "closure_field": CLOSURE_FIELD,
            "closure_direction": CLOSURE_DIRECTION,
            "closure_value": real_closure,
            "worsen_vs_real": None,
            "expected_behavior": "Baseline reference (best closure expected).",
            "pass_fail": True,
        }
        row_gaus = {
            "row": "GAUSS_ZERO",
            "infile": str(gauss_path.as_posix()),
            "closure_field": CLOSURE_FIELD,
            "closure_direction": CLOSURE_DIRECTION,
            "closure_value": gaus_closure,
            "worsen_vs_real": worsen(gaus_closure, real_closure),
            "expected_behavior": "Mandatory closure-breaking control; closure must degrade vs REAL.",
            "pass_fail": worsen(gaus_closure, real_closure),
        }
        row_emsw = {
            "row": "EM_SWAP",
            "infile": str(emswap_path.as_posix()),
            "closure_field": CLOSURE_FIELD,
            "closure_direction": CLOSURE_DIRECTION,
            "closure_value": emsw_closure,
            "worsen_vs_real": worsen(emsw_closure, real_closure),
            "expected_behavior": "Mandatory closure-breaking control; closure must degrade vs REAL.",
            "pass_fail": worsen(emsw_closure, real_closure),
        }

        real_is_best = (real_closure < gaus_closure) and (real_closure < emsw_closure)
        controls_pass = bool(row_gaus["pass_fail"]) and bool(row_emsw["pass_fail"])
        audit_pass = bool(real_is_best and controls_pass)

        matrix = {
            "tool": "n4_closure_audit",
            "version": "v1",
            "closure_field": CLOSURE_FIELD,
            "closure_direction": CLOSURE_DIRECTION,
            "rows": [row_real, row_gaus, row_emsw],
            "rules": {
                "direction_only": True,
                "control_pass_rule": "control_pass = (closure(control) worse than closure(REAL))",
                "final_verdict_rule": "PASS iff REAL best AND ALL controls worsen vs REAL",
            },
            "verdict": "PASS" if audit_pass else "FAIL",
            "checks": {
                "real_is_best": real_is_best,
                "controls_worsen": controls_pass,
            },
            "toolkit_outputs_used": {
                "tmp_run_dir": str(tmp_run.as_posix()),
                "preferred_json": "n4_primary.json (fallback: summary.json)",
            },
        }
        return matrix
    finally:
        _safe_rm(tmp_run)


def render_matrix_txt(matrix: Dict[str, Any]) -> str:
    rows = matrix["rows"]
    verdict = matrix["verdict"]
    lines = []
    lines.append("Ïˆ4 n4 Closure Audit Matrix (Tool B)")
    lines.append("=" * 72)
    lines.append(f"closure_field     : {matrix['closure_field']}")
    lines.append(f"closure_direction : {matrix['closure_direction']}")
    lines.append(f"verdict           : {verdict}")
    lines.append("")
    lines.append("Rows (direction-only audit)")
    lines.append("-" * 72)

    # Simple fixed-width table
    hdr = f"{'ROW':<12} {'closure_value':>14} {'worsen_vs_real':>14} {'pass_fail':>10}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for r in rows:
        row = r["row"]
        cv = _fmt_num(r["closure_value"])
        wv = r["worsen_vs_real"]
        wv_s = "N/A" if wv is None else ("TRUE" if wv else "FALSE")
        pf = "TRUE" if r["pass_fail"] else "FALSE"
        lines.append(f"{row:<12} {cv:>14} {wv_s:>14} {pf:>10}")

    lines.append("")
    lines.append("Expected behavior (frozen)")
    lines.append("-" * 72)
    for r in rows:
        lines.append(f"- {r['row']}: {r['expected_behavior']}")
    lines.append("")
    lines.append("Publish-safe sentence (verbatim)")
    lines.append("-" * 72)
    lines.append(
        "We report a closure audit matrix evaluating whether estimator self-consistency "
        "degrades under mandatory closure-breaking controls, without invoking statistical "
        "rarity or detection performance."
    )
    lines.append("")
    return "\n".join(lines)


def write_index_html(outdir: Path, matrix: Dict[str, Any]) -> None:
    # Minimal index linking outputs + embedding small HTML table
    rows = matrix["rows"]
    verdict = matrix["verdict"]
    checks = matrix["checks"]

    table_rows = []
    for r in rows:
        wv = r["worsen_vs_real"]
        wv_s = "N/A" if wv is None else ("TRUE" if wv else "FALSE")
        pf = "TRUE" if r["pass_fail"] else "FALSE"
        table_rows.append(
            "<tr>"
            f"<td>{_html_escape(r['row'])}</td>"
            f"<td>{_html_escape(_fmt_num(r['closure_value']))}</td>"
            f"<td>{_html_escape(wv_s)}</td>"
            f"<td>{_html_escape(pf)}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>n4_closure_audit - {verdict}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    code, pre {{ background: #f6f6f6; padding: 2px 6px; border-radius: 4px; }}
    table {{ border-collapse: collapse; margin-top: 10px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; }}
    th {{ background: #f3f3f3; }}
  </style>
</head>
<body>
  <h1>n4_closure_audit</h1>
  <p><b>Verdict:</b> {verdict}</p>
  <ul>
    <li><b>closure_field</b>: {_html_escape(matrix['closure_field'])}</li>
    <li><b>closure_direction</b>: {_html_escape(matrix['closure_direction'])}</li>
    <li><b>real_is_best</b>: {_html_escape(str(checks.get('real_is_best')))}</li>
    <li><b>controls_worsen</b>: {_html_escape(str(checks.get('controls_worsen')))}</li>
  </ul>

  <h2>Audit matrix</h2>
  <table>
    <thead>
      <tr>
        <th>row</th>
        <th>{_html_escape(matrix['closure_field'])}</th>
        <th>worsen_vs_real</th>
        <th>pass_fail</th>
      </tr>
    </thead>
    <tbody>
      {''.join(table_rows)}
    </tbody>
  </table>

  <h2>Outputs</h2>
  <ul>
    <li><a href="audit_matrix.json">audit_matrix.json</a></li>
    <li><a href="audit_matrix.txt">audit_matrix.txt</a></li>
    <li><a href="summary.json">summary.json</a></li>
    <li><a href="contract.json">contract.json</a></li>
    <li><a href="provenance.json">provenance.json</a></li>
  </ul>

  <h2>Publish-safe sentence</h2>
  <p>
    We report a closure audit matrix evaluating whether estimator self-consistency
    degrades under mandatory closure-breaking controls, without invoking statistical
    rarity or detection performance.
  </p>
</body>
</html>
"""
    _write_text(outdir / "index.html", html)


def contract_blob() -> Dict[str, Any]:
    return {
        "tool": "n4_closure_audit",
        "version": "v1",
        "purpose_frozen": "Consistency audit (not a detector): closure must degrade under closure-breaking controls.",
        "must_not": [
            "statistics",
            "rarity/significance",
            "thresholds",
            "detector language",
            "competition",
        ],
        "direction_only_rule": {
            "closure_field": CLOSURE_FIELD,
            "closure_direction": CLOSURE_DIRECTION,
            "control_pass": "closure(control) worse than closure(REAL)",
            "final_verdict": "PASS iff REAL best AND ALL controls worsen vs REAL",
        },
        "outputs_locked": PLANNED_OUTPUTS,
        "exit_codes": {"0": "valid audit", "2": "invalid input / invalid audit"},
    }


def provenance_blob(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "created_utc": _utc_now_iso(),
        "argv": sys.argv[:],
        "python": sys.version,
        "platform": sys.platform,
        "inputs": {
            "real": str(args.real),
            "gauss_zero": str(args.gauss_zero),
            "em_swap": str(args.em_swap),
        },
    }


# ----------------------------- Main -----------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ïˆ4 n4 Closure Audit Matrix (Tool B)")
    ap.add_argument("--real", required=True, help="REAL input table (.csv/.json/.jsonl) with UE/UM columns.")
    ap.add_argument("--gauss-zero", dest="gauss_zero", required=True, help="GAUSS_ZERO control input table.")
    ap.add_argument("--em-swap", dest="em_swap", required=True, help="EM_SWAP control input table.")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--output-mode", choices=["publish-min", "standard", "full-audit"], default="standard",
                    help="Artifact volume control only. Audit pack is already compact, so all modes are currently identical.")
    ap.add_argument("--dry-run", action="store_true", help="Print planned outputs and exit 0 (no writes).")
    ap.add_argument("--check", action="store_true", help="Validate inputs + print closure scalars (no writes).")
    ap.add_argument("--selftest", action="store_true", help="Run deterministic self-test and exit 0 (no writes).")
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    # dry-run fast path (no filesystem writes)
    if getattr(args, "dry_run", False):
        print(f"[DRY-RUN] OUTPUT_MODE={args.output_mode}")
        print("[DRY-RUN] PLANNED_OUTPUTS:")
        for f in planned_outputs_for_mode(args.output_mode):
            print(f"  - {f}")
        return 0


    # check-only mode (no filesystem writes)
    if getattr(args, "check", False):
        # validate existence (same gate as compute path)
        real_path = Path(args.real)
        gauss_path = Path(args.gauss_zero)
        emswap_path = Path(args.em_swap)
        for key, pp in [("real", real_path), ("gauss_zero", gauss_path), ("em_swap", emswap_path)]:
            if not pp.exists():
                _print_err(f"[ERROR] INVALID_INPUT_missing_file: {key}={pp}")
                return 2

        # run audit (uses publish-grade gates in run_audit)
        try:
            matrix = run_audit(real_path, gauss_path, emswap_path)
        except Exception as e:
            msg = str(e)
            if ("Non-finite closure metric:" in msg) or ("Non-finite metrics in " in msg):
                _print_err("[ERROR] INVALID_INPUT_nonfinite_closure")
                _print_err(msg)
                return 2
            _print_err(f"[ERROR] INVALID_EVAL_exception: {type(e).__name__}: {e}")
            return 2

        rows = {r["row"]: r for r in matrix["rows"]}
        print("=== CHECK ONLY (no files) ===")
        print(f"REAL       {CLOSURE_FIELD} = {rows['REAL']['closure_value']}")
        print(f"GAUSS_ZERO {CLOSURE_FIELD} = {rows['GAUSS_ZERO']['closure_value']}")
        print(f"EM_SWAP    {CLOSURE_FIELD} = {rows['EM_SWAP']['closure_value']}")
        print(f"real_is_best   = {matrix.get('checks',{}).get('real_is_best')}")
        print(f"controls_worsen= {matrix.get('checks',{}).get('controls_worsen')}")
        print(f"expected_verdict = {matrix.get('verdict')}")
        print("================================")
        return 0

    outdir = Path(args.out)
    # publish gate: on invalid -> exit 2 and write nothing
    real_path = Path(args.real)
    gauss_path = Path(args.gauss_zero)
    emswap_path = Path(args.em_swap)

    for key, p in [("real", real_path), ("gauss_zero", gauss_path), ("em_swap", emswap_path)]:
        if not p.exists():
            _print_err(f"[ERROR] INVALID_INPUT_missing_file: {key}={p}")
            return 2

    # tmp out for atomic commit
    tmp_out = Path(tempfile.mkdtemp(prefix="n4_audit_tmp_", dir=str(outdir.parent) if outdir.parent.exists() else None))
    try:
        # run audit
        try:
            matrix = run_audit(real_path, gauss_path, emswap_path)
        except Exception as e:
            msg = str(e)
            if ("Non-finite closure metric:" in msg) or ("Non-finite metrics in " in msg):
                _print_err("[ERROR] INVALID_INPUT_nonfinite_closure")
                _print_err(msg)
                return 2
            _print_err(f"[ERROR] INVALID_EVAL_exception: {type(e).__name__}: {e}")
            return 2

        # write outputs into tmp_out
        _write_json(tmp_out / "contract.json", contract_blob())
        _write_json(tmp_out / "provenance.json", provenance_blob(args))

        _write_json(tmp_out / "audit_matrix.json", matrix)
        _write_text(tmp_out / "audit_matrix.txt", render_matrix_txt(matrix))

        summary = {
            "tool": "n4_closure_audit",
            "version": "v1",
            "verdict": matrix["verdict"],
            "checks": matrix.get("checks", {}),
            "closure_field": matrix["closure_field"],
            "closure_direction": matrix["closure_direction"],
        }
        _write_json(tmp_out / "summary.json", summary)

        write_index_html(tmp_out, matrix)

        # ensure planned outputs exist before commit
        for f in PLANNED_OUTPUTS:
            if not (tmp_out / f).exists():
                _print_err(f"[ERROR] INVALID_RUN_missing_artifact: {f}")
                return 2

        # atomic commit to final outdir
        _atomic_commit(tmp_out, outdir)

        print("[OK] n4 closure audit complete")
        print(f"  outdir : {outdir.as_posix()}")
        print(f"  verdict: {summary['verdict']}")
        return 0

    finally:
        # if tmp_out still exists (e.g., early return), remove it
        _safe_rm(tmp_out)


if __name__ == "__main__":
    try:
        # EARLY_BYPASS_SELFTEST_DRYRUN: avoid argparse required-args for --selftest/--dry-run
        if "--selftest" in sys.argv:
            raise SystemExit(_selftest())
        if "--dry-run" in sys.argv:
            print("[DRY-RUN] PLANNED_OUTPUTS:")
            for f in PLANNED_OUTPUTS:
                print(f"  - {f}")
            raise SystemExit(0)
        rc = main(sys.argv[1:])
        sys.exit(rc)
    except Exception as e:
        _print_err(f"[ERROR] INVALID_EVAL_exception: {type(e).__name__}: {e}")
        sys.exit(2)



