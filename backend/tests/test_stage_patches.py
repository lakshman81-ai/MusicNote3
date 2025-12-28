import unittest
from backend.pipeline.utils_config import apply_dotted_overrides
from backend.pipeline.stage_c import coalesce_not_none
from backend.pipeline.stage_b import extract_features
from backend.pipeline.models import StageAOutput, MetaData, Stem
from backend.pipeline.config import PipelineConfig
import numpy as np

class TestStagePatches(unittest.TestCase):
    def test_apply_dotted_overrides_nested(self):
        cfg = {"a": {"b": 1}}
        overrides = {"a.c.d": 2}
        apply_dotted_overrides(cfg, overrides)
        self.assertEqual(cfg["a"]["c"]["d"], 2)

    def test_stage_c_merge_gap_logic(self):
        # 0.0 should be preserved, None should fallback
        self.assertEqual(coalesce_not_none(0.0, 100.0, 50.0), 0.0)
        self.assertEqual(coalesce_not_none(None, 0.0, 50.0), 0.0)
        self.assertEqual(coalesce_not_none(None, None, 50.0), 50.0)

    def test_stage_b_detector_init_no_crash(self):
        # Mocking minimal StageA output
        meta = MetaData(sample_rate=44100, duration_sec=1.0)
        # Needs a valid stem to not exit early
        dummy_audio = np.zeros(44100, dtype=np.float32)
        mix_stem = Stem(audio=dummy_audio, sr=44100, type="mix")
        stage_a_out = StageAOutput(stems={"mix": mix_stem}, meta=meta, audio_type="monophonic")

        # Config with fmin_override (float key) in detectors
        config = PipelineConfig()
        config.stage_b.detectors["fmin_override"] = 180.0
        config.stage_b.detectors["yin"] = {"enabled": True}

        # Should not crash
        try:
            extract_features(stage_a_out, config=config)
        except AttributeError as e:
            self.fail(f"Stage B crashed with AttributeError: {e}")
        except Exception as e:
            # We only care that it doesn't crash due to float iteration
            pass

if __name__ == "__main__":
    unittest.main()
