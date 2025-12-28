#!/usr/bin/env python3
"""
Run the full agent benchmark suite (L0-L6) and generate reports.
Supports L5 iterative tuning and presets.
"""

import os
import sys
import shutil
import time
import json
import logging
import argparse
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add repo root to python path so we can import backend
sys.path.insert(0, os.getcwd())

from backend.benchmarks.benchmark_runner import BenchmarkSuite, resolve_levels

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_agent_benchmarks")

REPORTS_DIR = Path("reports")
SNAPSHOTS_DIR = REPORTS_DIR / "snapshots"

# --- Constants ---

PRESET_VERSION = "l5_poly_v1"

GLOBAL_BENCH_POLY_PRESET_V1 = {
    # --- Stage B: Separation (force ON + make it harder to skip) ---
    "stage_b.separation.enabled": True,
    "stage_b.separation.model": "htdemucs",          # optional; keep if you trust it most
    "stage_b.separation.synthetic_model": False,     # for L5 "real-ish" mixes
    "stage_b.separation.overlap": 0.75,
    "stage_b.separation.shifts": 4,
    "stage_b.separation.harmonic_masking.enabled": True,
    "stage_b.separation.harmonic_masking.mask_width": 0.03,

    # If your Stage B separation logic reads "gates", this is HIGH ROI.
    # Safe even if ignored (it's inside a dict field).
    "stage_b.separation.gates.min_mixture_score": 0.10,
    "stage_b.separation.gates.bypass_if_synthetic_like": False,

    # --- Stage B: Poly recall ---
    "stage_b.polyphonic_peeling.max_layers": 6,
    "stage_b.confidence_voicing_threshold": 0.50,
    "stage_b.melody_filtering.voiced_prob_threshold": 0.35,

    # --- Stage C: de-fragmentation / stability ---
    "stage_c.min_note_duration_ms_poly": 70.0,
    "stage_c.gap_filling.max_gap_ms": 70.0,
    "stage_c.confidence_threshold": 0.20,
    "stage_c.confidence_hysteresis.start": 0.60,
    "stage_c.confidence_hysteresis.end": 0.40,

    # --- Stage D: reduce quantization damage (in case notes are scored post-D) ---
    "stage_d.quantization_mode": "light_rubato",
    "stage_d.light_rubato_snap_ms": 20.0,
}

TUNING_PASSES = {
  1: [
    {"stage_b.separation.gates.min_mixture_score": 0.05},
    {"stage_b.separation.gates.min_mixture_score": 0.10},
    {"stage_b.separation.gates.min_mixture_score": 0.20},
  ],
  2: [
    {"stage_b.polyphonic_peeling.max_layers": 4, "stage_b.confidence_voicing_threshold": 0.55},
    {"stage_b.polyphonic_peeling.max_layers": 6, "stage_b.confidence_voicing_threshold": 0.50},
    {"stage_b.polyphonic_peeling.max_layers": 8, "stage_b.confidence_voicing_threshold": 0.45},
  ],
  3: [
    {"stage_c.min_note_duration_ms_poly": 55.0, "stage_c.gap_filling.max_gap_ms": 60.0},
    {"stage_c.min_note_duration_ms_poly": 70.0, "stage_c.gap_filling.max_gap_ms": 70.0},
    {"stage_c.min_note_duration_ms_poly": 85.0, "stage_c.gap_filling.max_gap_ms": 90.0},
  ],
}

# --- Helpers ---

def ensure_determinism(seed: int, torch_deterministic: bool = False) -> None:
    import os, random
    import numpy as np

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    except Exception:
        pass

def unflatten_overrides(flat: dict) -> dict:
    out: dict = {}
    for k, v in flat.items():
        cur = out
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out

def append_sweep_row(root_out: str, run_out: str, level: str, pass_id: int, variant: int, overrides_flat: dict, seed: int, dry_run: bool = False) -> None:
    import json, os

    payload = {
        "level": level,
        "pass": pass_id,
        "variant": variant,
        "seed": seed,
        "preset_version": PRESET_VERSION,
        "dry_run": dry_run,
        "run_dir": os.path.relpath(run_out, root_out),
    }

    # Generate stable hash from flat overrides + seed + level
    hash_input = json.dumps(overrides_flat, sort_keys=True) + str(seed) + level + PRESET_VERSION
    h = hashlib.sha1(hash_input.encode("utf-8")).hexdigest()[:12]
    payload["overrides_sha"] = h

    if dry_run:
        payload.update({
             "note_f1": 0.0,
             "onset_mae_ms": 0.0,
             "fragmentation_score": 0.0,
             "note_count": 0,
             "gt_count": 0,
             "voiced_ratio": 0.0,
        })
    else:
        summary_path = os.path.join(run_out, "summary.json")
        if not os.path.exists(summary_path):
            return
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            if not rows:
                return
            r = rows[-1]
            payload.update({
                "note_f1": r.get("note_f1"),
                "onset_mae_ms": r.get("onset_mae_ms"),
                "fragmentation_score": r.get("fragmentation_score"),
                "note_count": r.get("note_count"),
                "gt_count": r.get("gt_count"),
                "voiced_ratio": r.get("voiced_ratio"),
            })
        except Exception as e:
            logger.error(f"Failed to read summary from {summary_path}: {e}")
            return

    out_path = os.path.join(root_out, "l5_sweep_results.jsonl")
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")

# --- Legacy Report Generation ---

def collect_artifacts(run_dir: Path):
    """Copy critical artifacts to reports/ folder."""
    logger.info("Collecting artifacts...")

    # 1. benchmark_results.json (summary snapshot)
    summary_src = run_dir / "summary.json"
    if summary_src.exists():
        shutil.copy(summary_src, REPORTS_DIR / "benchmark_results.json")
        # Also save a timestamped snapshot
        shutil.copy(summary_src, SNAPSHOTS_DIR / f"benchmark_results_{int(time.time())}.json")

    # 2. stage_metrics.json (summary CSV converted or raw)
    # The runner produces summary.csv, let's copy that too
    summary_csv = run_dir / "summary.csv"
    if summary_csv.exists():
        shutil.copy(summary_csv, REPORTS_DIR / "summary.csv")

    # 3. Aggregated stage metrics from all levels
    stage_metrics = {}
    for f in run_dir.glob("*_metrics.json"):
        try:
            data = json.loads(f.read_text())
            key = f"{data.get('level', 'unknown')}_{data.get('name', 'unknown')}"
            stage_metrics[key] = data
        except Exception:
            pass

    with open(REPORTS_DIR / "stage_metrics.json", "w") as f:
        json.dump(stage_metrics, f, indent=2)

    # 4. Generate Health Report
    generate_health_report(stage_metrics)

    # 5. Generate Regression Flags
    generate_regression_flags(stage_metrics)

def generate_health_report(metrics: dict):
    """Generate Markdown health report."""
    lines = ["# Stage Health Report", "", f"Run Date: {time.ctime()}", ""]

    lines.append("## Level Summary")
    lines.append("| Level | Name | Note F1 | Onset MAE (ms) | Notes |")
    lines.append("|---|---|---|---|---|")

    for key, m in sorted(metrics.items()):
        f1 = m.get("note_f1", 0.0)
        mae = m.get("onset_mae_ms", "")
        if isinstance(mae, (float, int)):
            mae = f"{mae:.1f}"
        else:
            mae = "-"
        count = m.get("predicted_count", 0)
        lines.append(f"| {m.get('level')} | {m.get('name')} | {f1:.3f} | {mae} | {count} |")

    lines.append("")
    lines.append("## Polyphonic Diagnostics")
    # Check L2, L3, L5, L6 specifically
    poly_levels = [k for k in metrics.keys() if any(x in k for x in ["L2", "L3", "L5", "L6"])]
    if not poly_levels:
        lines.append("_No polyphonic levels run._")
    else:
        for key in sorted(poly_levels):
            m = metrics[key]
            lines.append(f"### {key}")
            lines.append(f"- **Voiced Ratio**: {m.get('voiced_ratio', 0.0):.3f}")
            lines.append(f"- **Pitch Jump Rate**: {m.get('pitch_jump_rate_cents_sec', 0.0):.1f} cents/sec")
            lines.append(f"- **Vocal Band Ratio**: {m.get('vocal_band_ratio', 0.0):.3f}")
            lines.append("")

    with open(REPORTS_DIR / "stage_health_report.md", "w") as f:
        f.write("\n".join(lines))

def generate_regression_flags(metrics: dict):
    """Generate Regression Flags report."""
    flags = []

    # Define thresholds
    MIN_F1 = 0.1  # Very low baseline to catch catastrophic failures
    MAX_MAE = 200.0

    for key, m in metrics.items():
        level = m.get("level", "")

        # 1. F1 Drop
        f1 = m.get("note_f1")
        if f1 is not None and f1 < MIN_F1 and level not in ["L4"]: # L4 doesn't have F1
            flags.append(f"🔴 {key}: F1 score {f1:.3f} is below safety threshold {MIN_F1}")

        # 2. Onset Accuracy
        mae = m.get("onset_mae_ms")
        if mae is not None and mae > MAX_MAE:
            flags.append(f"🟠 {key}: Onset MAE {mae:.1f}ms is high (> {MAX_MAE}ms)")

        # 3. Empty Output
        if m.get("predicted_count", 0) == 0:
            flags.append(f"🔴 {key}: Zero notes predicted!")

        # 4. Fragmentation
        frag = m.get("fragmentation_score", 0.0)
        if frag > 0.5:
             flags.append(f"🟡 {key}: High fragmentation score {frag:.2f}")

    content = ["# Regression Flags", "", f"Generated: {time.ctime()}", ""]
    if not flags:
        content.append("✅ No obvious regressions detected.")
    else:
        content.extend(flags)

    with open(REPORTS_DIR / "regression_flags.md", "w") as f:
        f.write("\n".join(content))


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Agent Benchmark Runner")
    parser.add_argument("--level", default="all", help="all, L0, L5.1, etc.")
    parser.add_argument("--output", default=f"results/benchmark_{int(time.time())}", help="Output root directory")
    parser.add_argument("--tuning-pass", type=int, default=0, help="Run tuning pass N (L5 only)")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed")
    parser.add_argument("--preset", choices=["auto", "none", "l5"], default="auto", help="Config preset strategy")
    parser.add_argument("--dry-run", action="store_true", help="Setup dirs/logs but do not run suite")
    parser.add_argument("--torch-deterministic", action="store_true", help="Force torch deterministic algorithms")
    args = parser.parse_args()

    # 1. Ensure determinism
    ensure_determinism(args.seed, args.torch_deterministic)

    # 2. Setup Directories
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # 3. Check L5 deps if needed (Robust)
    levels_to_run = resolve_levels(args.level)
    is_l5_run = any(lvl.startswith("L5") for lvl in levels_to_run)

    if is_l5_run:
        l5_paths = [
            "backend/benchmarks/ladder/L5.1_kal_ho_na_ho.mid",
            "backend/benchmarks/ladder/L5.2_tumhare_hi_rahenge.mid"
        ]

        # Check primary paths
        missing = [p for p in l5_paths if not os.path.exists(p)]

        if missing:
            # Fallback attempts (check relative to backend/benchmarks/ladder/midi/ or similar)
            # But the user asked to "print attempted paths".
            # Let's verify if they are truly missing or just elsewhere.
            # For now, we print error and exit 2 as per plan, but with better messaging.
            logger.error("Missing L5 MIDI assets. Checked:")
            for p in l5_paths:
                logger.error(f"  - {p} {'[MISSING]' if p in missing else '[FOUND]'}")

            # Allow continuing if it's L5 but user might be fixing it, but plan said exit 2.
            sys.exit(2)

    # 4. Determine Base Overrides (Preset)
    base_overrides_flat = {}

    use_preset = False
    if args.preset == "l5":
        use_preset = True
    elif args.preset == "auto":
        # Only auto-apply if explicitly running an L5 level (to avoid surprising behavior on L0-L4)
        if is_l5_run:
            use_preset = True

    if use_preset:
        logger.info(f"Applying L5 Poly Preset ({PRESET_VERSION})")
        base_overrides_flat = dict(GLOBAL_BENCH_POLY_PRESET_V1)

    # 5. Execution Logic
    if args.tuning_pass > 0:
        if not is_l5_run:
            logger.error("--tuning-pass only supported for L5 levels")
            sys.exit(1)

        deltas = TUNING_PASSES.get(args.tuning_pass, [])
        if not deltas:
            logger.error(f"Unknown tuning pass: {args.tuning_pass}")
            sys.exit(1)

        logger.info(f"Running Tuning Pass {args.tuning_pass} with {len(deltas)} variants")

        for i, delta_flat in enumerate(deltas):
            run_dir = Path(args.output) / f"pass{args.tuning_pass}_v{i}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Merge overrides: Base + Variant
            merged_flat = dict(base_overrides_flat)
            merged_flat.update(delta_flat)

            final_overrides = unflatten_overrides(merged_flat)

            logger.info(f"Variant {i}: {delta_flat}")

            if args.dry_run:
                logger.info("Dry run: skipping execution")
                append_sweep_row(args.output, str(run_dir), args.level, args.tuning_pass, i, merged_flat, args.seed, dry_run=True)
                continue

            # Initialize suite pointing to specific variant dir
            suite = BenchmarkSuite(
                str(run_dir),
                pipeline_seed=args.seed,
                deterministic=True,
                deterministic_torch=args.torch_deterministic
            )

            # Execute
            try:
                for lvl in levels_to_run:
                    if lvl == "L5.1":
                        suite.run_L5_1_kal_ho_na_ho(overrides=final_overrides)
                    elif lvl == "L5.2":
                        suite.run_L5_2_tumhare_hi_rahenge(overrides=final_overrides)
                    else:
                        logger.warning(f"Skipping {lvl} in tuning loop (only L5 supported)")

                suite.generate_summary()
                append_sweep_row(args.output, str(run_dir), args.level, args.tuning_pass, i, merged_flat, args.seed, dry_run=False)

            except Exception as e:
                logger.exception(f"Run failed for variant {i}: {e}")

    else:
        # Standard Run (Single Pass)
        logger.info(f"Running Standard Benchmark in {args.output}")

        final_overrides = unflatten_overrides(base_overrides_flat)

        if args.dry_run:
            logger.info("Dry run: skipping execution.")
            logger.info(f"Would run levels: {levels_to_run}")
            if final_overrides:
                logger.info(f"Would apply overrides: {json.dumps(final_overrides, indent=2)}")
            # For standard run dry-run, we don't necessarily write JSONL sweep rows unless asked.
            # But we exit successfully.
            sys.exit(0)

        suite = BenchmarkSuite(
            str(args.output),
            pipeline_seed=args.seed,
            deterministic=True,
            deterministic_torch=args.torch_deterministic
        )

        for lvl in levels_to_run:
            if lvl == "L0":
                suite.run_L0_mono_sanity()
            elif lvl == "L1":
                suite.run_L1_mono_musical()
            elif lvl == "L2":
                suite.run_L2_poly_dominant()
            elif lvl == "L3":
                suite.run_L3_full_poly()
            elif lvl == "L4":
                # L4 supports 'use_preset' bool, but it only loads PIANO_61KEY_CONFIG if true.
                # If we want to use OUR preset (GLOBAL_BENCH_POLY_PRESET_V1), L4 doesn't support overrides.
                # So we stick to its built-in 'use_preset' logic if requested.
                # args.preset="l5" implies we want the preset behavior.
                # args.preset="auto" -> defaults to False for L4 unless logic is changed.
                # Original logic: auto defaults to no preset for L4 (consistent with legacy behavior).
                suite.run_L4_real_songs(use_preset=(args.preset == "l5"))
            elif lvl == "L6":
                suite.run_L6_synthetic_pop_song()
            elif lvl == "L5.1":
                suite.run_L5_1_kal_ho_na_ho(overrides=final_overrides)
            elif lvl == "L5.2":
                suite.run_L5_2_tumhare_hi_rahenge(overrides=final_overrides)

        suite.generate_summary()
        collect_artifacts(Path(args.output))

if __name__ == "__main__":
    main()
