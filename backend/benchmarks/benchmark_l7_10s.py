import os
import sys
import json
import logging
import random
import numpy as np
from typing import Any, Dict

# Ensure root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.pipeline.config import PipelineConfig
from backend.pipeline.utils_config import apply_dotted_overrides
from backend.benchmarks.ladder.runner import NumpyEncoder
from backend.benchmarks.ladder.generators import generate_benchmark_example
from backend.benchmarks.ladder.synth import midi_to_wav_synth
from backend.benchmarks.ladder.metrics.stage_c import calculate_stage_c_metrics
from backend.pipeline.stage_a import load_and_preprocess
from backend.pipeline.stage_b import extract_features
from backend.pipeline.models import AnalysisData
import backend.pipeline.stage_c as stage_c_module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("L7Runner")

def run_l7_benchmark():
    # Define L7
    l7_level = {
        "id": "L7_SYNTH_POLY",
        "description": "L7 Synthetic Polyphony Stress Test (10s)",
        "examples": ["poly_dominant_l7", "poly_full_l7"]
    }

    output_dir = "benchmark_results_l7"
    os.makedirs(output_dir, exist_ok=True)

    config = PipelineConfig()

    # Patch 5 & 8 overrides
    runner_overrides = {
        "stage_b.polyphonic_peeling.force": True,
        "stage_b.polyphonic_peeling.max_layers": 8,
        "stage_b.polyphonic_peeling.residual_voicing_relax": 0.15,
        "bench_profile": "l7_synth", # Triggers Patch 8
    }
    apply_dotted_overrides(config, runner_overrides)

    # Ensure bench_profile is set
    setattr(config, "bench_profile", "l7_synth")

    results = []
    velocities = [50, 60, 75]

    for base_example_id in l7_level["examples"]:
        for vel in velocities:
            example_id = f"{base_example_id}_v{vel}"
            logger.info(f"Running L7 Example: {example_id} (vel={vel})")

            # Deterministic seed
            seed_val = abs(hash(example_id)) % (2**32)
            random.seed(seed_val)
            np.random.seed(seed_val)

            res = {"id": example_id, "velocity": vel, "errors": []}

            try:
                # 1. Gen
                base_name = base_example_id.replace("_l7", "")
                gen_id = f"happy_birthday_{base_name}"

                score = generate_benchmark_example(gen_id, accomp_velocity=vel)
                midi_path = os.path.join(output_dir, f"{example_id}.mid")
                score.write("midi", fp=midi_path)

                # 2. Synth
                wav_path = os.path.join(output_dir, f"{example_id}.wav")
                diag = {}
                midi_to_wav_synth(
                    score,
                    wav_path,
                    diagnostics=diag,
                    articulation_enabled=True,
                    articulation_max_gap_sec=0.1,
                    articulation_frac=0.2
                )
                res["synth_diagnostics"] = diag

                # 3. Stage A
                stage_a_out = load_and_preprocess(wav_path, config=config.stage_a)

                # 4. Stage B
                stage_b_out = extract_features(stage_a_out, config=config)

                # 5. Stage C
                analysis_data = AnalysisData(
                    meta=stage_b_out.meta,
                    stem_timelines=stage_b_out.stem_timelines,
                )
                notes = stage_c_module.apply_theory(analysis_data, config=config)

                # Metrics
                mc = calculate_stage_c_metrics(notes, midi_path)
                res["metrics"] = mc
                logger.info(f"Metrics ({example_id}): {mc}")

            except Exception as e:
                logger.error(f"Failed: {e}", exc_info=True)
                res["errors"].append(str(e))

            results.append(res)

    # Sanity Assert
    expected_count = len(l7_level["examples"]) * len(velocities)
    if len(results) != expected_count:
        logger.error(f"FATAL: Expected {expected_count} results, got {len(results)}")
        sys.exit(1)

    # Summarization
    summary = {
        "results": results,
        "stats_by_velocity": {}
    }

    for vel in velocities:
        vel_res = [r for r in results if r["velocity"] == vel]
        f1_scores = [r.get("metrics", {}).get("Note_F1", 0.0) for r in vel_res]
        mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        summary["stats_by_velocity"][vel] = {
            "mean_note_f1": mean_f1,
            "count": len(vel_res)
        }

    summary_path = os.path.join(output_dir, "l7_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    logger.info(f"L7 Benchmark Complete. Results in {output_dir}")
    logger.info(f"Summary Stats: {json.dumps(summary['stats_by_velocity'], indent=2)}")

    return results

if __name__ == "__main__":
    run_l7_benchmark()
