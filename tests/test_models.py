import json
import unittest
from dataclasses import asdict

from llama_doctor.models import (
    PROPS_KEYS,
    MetricsSnapshot,
    ModelsMeta,
    ProbeResult,
    PropsSnapshot,
    ServerSnapshot,
    guess_moe,
    parse_quant,
)
from llama_doctor.promparse import parse_prometheus

from .support import load_fixture


def snapshot_from_fixture(name):
    return MetricsSnapshot.from_samples(parse_prometheus(load_fixture(name)))


class TestMetricsSnapshot(unittest.TestCase):
    def test_moe_fixture_rates(self):
        snap = snapshot_from_fixture("metrics_moe_offload.txt")
        self.assertAlmostEqual(snap.prefill_tps, 40.0, places=6)
        self.assertAlmostEqual(snap.decode_tps, 7.0, places=6)
        self.assertAlmostEqual(snap.cache_hit_ratio, 0.1111111, places=6)
        self.assertAlmostEqual(snap.avg_draft_len, 3.0, places=6)
        self.assertAlmostEqual(snap.draft_acceptance, 0.4, places=6)

    def test_healthy_fixture_rates(self):
        snap = snapshot_from_fixture("metrics_healthy.txt")
        self.assertAlmostEqual(snap.prefill_tps, 2000.0, places=6)
        self.assertAlmostEqual(snap.decode_tps, 50.0, places=6)
        self.assertAlmostEqual(snap.cache_hit_ratio, 0.75, places=6)
        self.assertAlmostEqual(snap.avg_draft_len, 2.0, places=6)
        self.assertAlmostEqual(snap.draft_acceptance, 0.8, places=6)
        self.assertEqual(snap.kv_cache_usage_ratio, 0.42)
        self.assertIsNone(snap.kv_cache_tokens)

    def test_minimal_fixture_degrades(self):
        snap = snapshot_from_fixture("metrics_minimal.txt")
        self.assertAlmostEqual(snap.prefill_tps, 300.0, places=6)
        self.assertAlmostEqual(snap.decode_tps, 30.0, places=6)
        self.assertIsNone(snap.cache_hit_ratio)
        self.assertIsNone(snap.avg_draft_len)
        self.assertIsNone(snap.draft_acceptance)
        self.assertIsNone(snap.n_tokens_max)

    def test_zero_denominator_is_none(self):
        snap = MetricsSnapshot.from_samples(
            {"llamacpp:prompt_tokens_total": 10, "llamacpp:prompt_seconds_total": 0}
        )
        self.assertIsNone(snap.prefill_tps)
        snap = MetricsSnapshot.from_samples(
            {"llamacpp:draft_tokens_total": 10, "llamacpp:drafts_total": 0}
        )
        self.assertIsNone(snap.avg_draft_len)

    def test_negative_counter_is_none(self):
        snap = MetricsSnapshot.from_samples(
            {"llamacpp:prompt_tokens_total": -5, "llamacpp:prompt_seconds_total": 1}
        )
        self.assertIsNone(snap.prompt_tokens_total)
        self.assertIsNone(snap.prefill_tps)
        snap = MetricsSnapshot.from_samples({"llamacpp:tokens_predicted_total": float("nan")})
        self.assertIsNone(snap.tokens_predicted_total)

    def test_alias_lookup(self):
        snap = MetricsSnapshot.from_samples(
            {"n_drafted_total": 50, "llamacpp:n_drafts_total": 25}
        )
        self.assertAlmostEqual(snap.avg_draft_len, 2.0, places=6)


class TestPropsSnapshot(unittest.TestCase):
    def props_from_fixture(self, name):
        return PropsSnapshot.from_payload(json.loads(load_fixture(name)))

    def test_props_full_whitelist(self):
        snap = self.props_from_fixture("props_full.json")
        self.assertEqual(snap.model_name, "Qwen3-30B-A3B-UD-Q4_K_XL")
        self.assertEqual(snap.quant, "UD-Q4_K_XL")
        self.assertEqual(snap.n_ctx, 32768)
        self.assertEqual(snap.total_slots, 1)
        self.assertEqual(snap.build_info, "b6500-abc1234")
        self.assertEqual(snap.n_gpu_layers, 99)
        self.assertEqual(snap.n_threads, 8)
        self.assertEqual(snap.cache_type_k, "q8_0")
        self.assertEqual(snap.cache_type_v, "q8_0")
        self.assertTrue(snap.has_vision)
        self.assertTrue(snap.is_moe)
        self.assertEqual(snap.missing, ())

    def test_props_unsloth_minimal(self):
        snap = self.props_from_fixture("props_unsloth_minimal.json")
        self.assertEqual(snap.model_name, "Qwen3-30B-A3B-GGUF")
        self.assertIsNone(snap.quant)
        self.assertIsNone(snap.n_ctx)
        self.assertEqual(snap.total_slots, 1)
        self.assertEqual(snap.build_info, "unsloth-studio")
        self.assertTrue(snap.is_moe)
        self.assertEqual(
            snap.missing,
            (
                "n_ctx",
                "n_gpu_layers",
                "n_threads",
                "cache_type_k",
                "cache_type_v",
                "modalities",
            ),
        )

    def test_props_never_hold_sentinels(self):
        for name in ("props_full.json", "props_unsloth_minimal.json"):
            snap = self.props_from_fixture(name)
            self.assertNotIn("SENTINEL", repr(snap))
            self.assertNotIn("SENTINEL", str(asdict(snap)))

    def test_props_drops_field_containing_media_marker(self):
        payload = {
            "model_path": "/models/m.gguf",
            "total_slots": 1,
            "media_marker": "<__media_SENTINEL__>",
            "build_info": "<__media_SENTINEL__>",
        }
        snap = PropsSnapshot.from_payload(payload)
        self.assertIsNone(snap.build_info)
        self.assertEqual(snap.total_slots, 1)

    def test_props_non_dict_payload(self):
        expected = PropsSnapshot(missing=PROPS_KEYS)
        self.assertEqual(PropsSnapshot.from_payload(None), expected)
        self.assertEqual(PropsSnapshot.from_payload(["not", "a", "dict"]), expected)

    def test_props_basename_from_long_hf_path(self):
        payload = {
            "model_path": (
                "/home/u/.cache/huggingface/hub/"
                "models--unsloth--Qwen3-30B-A3B-GGUF/snapshots/"
                "d1e2f3a4b5c6098f0e7d6c5b4a3210fedcba9876543210/"
                "Qwen3-30B-A3B-UD-Q4_K_XL.gguf"
            )
        }
        self.assertGreater(len(payload["model_path"]), 128)
        snap = PropsSnapshot.from_payload(payload)
        self.assertEqual(snap.model_name, "Qwen3-30B-A3B-UD-Q4_K_XL")
        self.assertEqual(snap.quant, "UD-Q4_K_XL")

    def test_props_whitespace_basename_is_none(self):
        snap = PropsSnapshot.from_payload({"model_path": "   .gguf"})
        self.assertIsNone(snap.model_name)
        self.assertIsNone(snap.quant)
        self.assertIsNone(snap.is_moe)

    def test_props_marker_after_padding_not_missing(self):
        marker = "<__media_SENTINEL__>"
        payload = {
            "model_path": "Qwen3-30B-A3B-Q4_K_M.gguf",
            "build_info": "x" * 120 + marker,
            "media_marker": marker,
        }
        snap = PropsSnapshot.from_payload(payload)
        self.assertIsNone(snap.build_info)
        self.assertNotIn("build_info", snap.missing)
        self.assertNotIn("SENTINEL", repr(snap))
        self.assertEqual(
            snap.missing,
            ("n_ctx", "total_slots", "n_gpu_layers", "n_threads", "cache_type_k", "cache_type_v", "modalities"),
        )

    def test_props_present_but_invalid_not_missing(self):
        payload = {
            "model_path": "m.gguf",
            "n_ctx": "not-a-number",
            "modalities": {"vision": "yes"},
        }
        snap = PropsSnapshot.from_payload(payload)
        self.assertIsNone(snap.n_ctx)
        self.assertIsNone(snap.has_vision)
        self.assertNotIn("n_ctx", snap.missing)
        self.assertNotIn("modalities", snap.missing)
        self.assertEqual(
            snap.missing,
            ("total_slots", "build_info", "n_gpu_layers", "n_threads", "cache_type_k", "cache_type_v"),
        )


class TestQuantAndMoe(unittest.TestCase):
    def test_parse_quant(self):
        cases = [
            ("Qwen3-30B-A3B-UD-Q4_K_XL", "UD-Q4_K_XL"),
            ("gemma-3-12b-it-Q8_0", "Q8_0"),
            ("model-IQ2_XXS", "IQ2_XXS"),
            ("gpt-oss-20b-MXFP4", "MXFP4"),
            ("Llama-3-8B-Q4_K_M-00001-of-00002", "Q4_K_M"),
            ("llama-7b", None),
        ]
        for name, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(parse_quant(name), expected)

    def test_guess_moe(self):
        for name in ("Qwen3-30B-A3B-Q4_K_M", "Mixtral-8x7B-v0.1", "gpt-oss-20b"):
            with self.subTest(name=name):
                self.assertTrue(guess_moe(name))
        for name in ("Llama-3.1-8B-Instruct", "gemma-3-12b-it"):
            with self.subTest(name=name):
                self.assertIsNone(guess_moe(name))


class TestModelsMeta(unittest.TestCase):
    def test_models_meta_full(self):
        meta = ModelsMeta.from_payload(json.loads(load_fixture("models_full.json")))
        self.assertEqual(meta.model_id, "Qwen3-30B-A3B-UD-Q4_K_XL.gguf")
        self.assertEqual(meta.n_params, 30532122624)
        self.assertEqual(meta.size_bytes, 17716740096)
        self.assertEqual(meta.n_ctx_train, 40960)

    def test_models_meta_empty(self):
        meta = ModelsMeta.from_payload({"data": []})
        self.assertIsNone(meta.model_id)
        self.assertIsNone(meta.n_params)
        self.assertIsNone(meta.size_bytes)
        self.assertIsNone(meta.n_ctx_train)


class TestProbeAndSnapshot(unittest.TestCase):
    def test_probe_result_reliable(self):
        weak = ProbeResult(
            prefill_tps=0.0, decode_tps=0.0, prompt_n=999, predicted_n=0, warmup_s=0.0
        )
        self.assertFalse(weak.reliable)
        strong = ProbeResult(
            prefill_tps=0.0, decode_tps=0.0, prompt_n=1000, predicted_n=0, warmup_s=0.0
        )
        self.assertTrue(strong.reliable)

    def test_server_snapshot_alive(self):
        alive = ServerSnapshot(
            taken_at=1.0,
            metrics=MetricsSnapshot(),
            props=None,
            models=None,
            errors={},
        )
        self.assertTrue(alive.alive)
        dead = ServerSnapshot(
            taken_at=1.0, metrics=None, props=None, models=None, errors={}
        )
        self.assertFalse(dead.alive)
        models_only = ServerSnapshot(
            taken_at=1.0, metrics=None, props=None, models=ModelsMeta(), errors={}
        )
        self.assertTrue(models_only.alive)


if __name__ == "__main__":
    unittest.main()
