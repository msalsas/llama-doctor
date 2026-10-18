import json
import unittest
from dataclasses import asdict, replace

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

    def test_queue_metrics_from_real_exposition(self):
        snap = snapshot_from_fixture("metrics_queue.txt")
        self.assertEqual(snap.requests_processing, 1.0)
        self.assertEqual(snap.requests_deferred, 0.0)

    def test_queue_metrics_sum_labeled_series(self):
        snap = MetricsSnapshot.from_samples(parse_prometheus(
            'llamacpp:requests_processing{slot="0"} 1\n'
            'llamacpp:requests_processing{slot="1"} 2\n'
            'llamacpp:requests_deferred{model="a"} 4\n'
        ))
        self.assertEqual(snap.requests_processing, 3.0)
        self.assertEqual(snap.requests_deferred, 4.0)

    def test_unlabeled_series_wins_over_labeled(self):
        snap = MetricsSnapshot.from_samples(
            {"llamacpp:requests_deferred": 2.0, 'llamacpp:requests_deferred{model="a"}': 9.0}
        )
        self.assertEqual(snap.requests_deferred, 2.0)

    def test_ratio_gauge_with_several_label_sets_is_not_summed(self):
        snap = MetricsSnapshot.from_samples({
            'llamacpp:kv_cache_usage_ratio{a="1"}': 0.5,
            'llamacpp:kv_cache_usage_ratio{a="2"}': 0.6,
        })
        self.assertIsNone(snap.kv_cache_usage_ratio)


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
            ("n_ctx", "modalities"),
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
            ("n_ctx", "total_slots", "modalities"),
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
            ("total_slots", "build_info"),
        )

    def test_props_startup_flags_are_not_required_keys(self):
        for key in ("n_gpu_layers", "n_threads", "cache_type_k", "cache_type_v"):
            self.assertNotIn(key, PROPS_KEYS)

    def test_props_quant_from_model_ftype(self):
        cases = (
            ("Q4_K - Medium", "Q4_K_M"),
            ("Q5_K - Small", "Q5_K_S"),
            ("Q8_0", "Q8_0"),
            ("IQ2_XS - 2.3125 bpw", "IQ2_XS"),
            ("MXFP4 MoE", "MXFP4_MOE"),
            ("BF16", "BF16"),
        )
        for ftype, quant in cases:
            with self.subTest(ftype=ftype):
                snap = PropsSnapshot.from_payload({"model_path": "/m/model.gguf", "model_ftype": ftype})
                self.assertEqual(snap.quant, quant)

    def test_props_filename_quant_wins_over_model_ftype(self):
        snap = PropsSnapshot.from_payload({"model_path": "/m/Qwen3-8B-UD-Q4_K_XL.gguf",
                                           "model_ftype": "Q4_K - Medium"})
        self.assertEqual(snap.quant, "UD-Q4_K_XL")

    def _spec_types(self, params):
        return PropsSnapshot.from_payload({"default_generation_settings": {"params": params}}).spec_types

    def test_props_spec_types(self):
        cases = (
            ({"speculative.types": ["none"]}, ()),
            ({"speculative.types": "none"}, ()),
            ({"speculative.types": []}, ()),
            ({"speculative.types": ["draft"]}, ("draft",)),
            ({"speculative.types": "draft,ngram-cache"}, ("draft", "ngram-cache")),
            ({"speculative.types": ["none", "MTP"]}, ("mtp",)),
            ({"temperature": 0.8}, None),
            ({"speculative.types": 3}, None),
        )
        for params, expected in cases:
            with self.subTest(params=params):
                self.assertEqual(self._spec_types(params), expected)
        self.assertIsNone(PropsSnapshot.from_payload({}).spec_types)


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
        self.assertIsNone(meta.quant)

    def test_models_meta_picks_loaded_entry_not_first(self):
        meta = ModelsMeta.from_payload(UNSLOTH_MODELS)
        self.assertEqual(meta.model_id, "unsloth/Qwen3.8-Flash-Next-GGUF")
        self.assertEqual(meta.quant, "UD-Q3_K_XL")

    def test_models_meta_without_loaded_uses_first(self):
        payload = {
            "data": [
                {"id": "a/First-GGUF", "quant": "Q4_K_M", "loaded": False},
                {"id": "a/Second-GGUF", "quant": "Q8_0"},
            ]
        }
        meta = ModelsMeta.from_payload(payload)
        self.assertEqual(meta.model_id, "a/First-GGUF")
        self.assertEqual(meta.quant, "Q4_K_M")
        self.assertFalse(meta.entry_loaded)

    def test_models_meta_flags_loaded_entry(self):
        self.assertTrue(ModelsMeta.from_payload(UNSLOTH_MODELS).entry_loaded)
        self.assertFalse(ModelsMeta.from_payload({"data": []}).entry_loaded)

    def test_models_meta_loaded_must_be_true_not_truthy(self):
        payload = {
            "data": [
                {"id": "a/First-GGUF", "quant": "Q4_K_M"},
                {"id": "a/Second-GGUF", "quant": "Q8_0", "loaded": "yes"},
            ]
        }
        self.assertEqual(ModelsMeta.from_payload(payload).model_id, "a/First-GGUF")

    def test_models_meta_quant_from_meta_fallback(self):
        payload = {"data": [{"id": "a/M-GGUF", "meta": {"quant": "IQ4_XS"}}]}
        self.assertEqual(ModelsMeta.from_payload(payload).quant, "IQ4_XS")

    def test_models_meta_top_level_quant_wins_over_meta(self):
        payload = {"data": [{"id": "a/M-GGUF", "quant": "Q6_K", "meta": {"quant": "IQ4_XS"}}]}
        self.assertEqual(ModelsMeta.from_payload(payload).quant, "Q6_K")

    def test_models_meta_bad_top_level_quant_falls_back_to_meta(self):
        payload = {"data": [{"id": "a/M-GGUF", "quant": 4, "meta": {"quant": "Q4_0"}}]}
        self.assertEqual(ModelsMeta.from_payload(payload).quant, "Q4_0")

    def test_models_meta_without_any_quant(self):
        payload = {"data": [{"id": "a/M-GGUF", "loaded": True, "meta": "junk"}]}
        meta = ModelsMeta.from_payload(payload)
        self.assertEqual(meta.model_id, "a/M-GGUF")
        self.assertIsNone(meta.quant)

    def test_models_meta_skips_non_dict_entries(self):
        payload = {"data": ["junk", {"id": "a/M-GGUF", "quant": "Q5_K_M", "loaded": True}]}
        self.assertEqual(ModelsMeta.from_payload(payload).quant, "Q5_K_M")


UNSLOTH_MODELS = {
    "data": [
        {"id": "unsloth/Qwen3.8-27B-GGUF", "quant": "UD-IQ3_S", "loaded": False},
        {
            "id": "unsloth/Qwen3.8-Flash-Next-GGUF",
            "quant": "UD-Q3_K_XL",
            "loaded": True,
            "context_length": 80000,
        },
        {"id": "unsloth/gemma-5-12b-it-GGUF", "quant": "Q4_K_M", "loaded": False},
    ]
}


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


class TestServerProps(unittest.TestCase):
    def setUp(self):
        self.unsloth = PropsSnapshot.from_payload(
            json.loads(load_fixture("props_unsloth_minimal.json"))
        )
        self.full = PropsSnapshot.from_payload(json.loads(load_fixture("props_full.json")))
        self.models = ModelsMeta.from_payload(json.loads(load_fixture("models_full.json")))

    def test_quant_backfilled_from_models_id(self):
        snapshot = ServerSnapshot(0.0, None, self.unsloth, self.models, {})
        server = snapshot.server_props
        self.assertEqual(server.quant, "UD-Q4_K_XL")
        self.assertEqual(server.model_name, "Qwen3-30B-A3B-GGUF")
        self.assertEqual(server.missing, self.unsloth.missing)
        self.assertIsNone(self.unsloth.quant)

    def test_model_name_backfilled_when_props_have_none(self):
        props = PropsSnapshot(total_slots=1)
        models = replace(self.models, entry_loaded=True)
        server = ServerSnapshot(0.0, None, props, models, {}).server_props
        self.assertEqual(server.model_name, "Qwen3-30B-A3B-UD-Q4_K_XL")
        self.assertEqual(server.quant, "UD-Q4_K_XL")
        self.assertTrue(server.is_moe)

    def test_props_values_win_over_models(self):
        props = PropsSnapshot(model_name="local-model", quant="Q8_0")
        server = ServerSnapshot(0.0, None, props, self.models, {}).server_props
        self.assertIs(server, props)

    def test_without_models_returns_props(self):
        snapshot = ServerSnapshot(0.0, None, self.unsloth, None, {})
        self.assertIs(snapshot.server_props, self.unsloth)
        no_id = ServerSnapshot(0.0, None, self.unsloth, ModelsMeta(), {})
        self.assertIs(no_id.server_props, self.unsloth)

    def test_quant_from_models_field_when_id_has_no_suffix(self):
        models = ModelsMeta.from_payload(UNSLOTH_MODELS)
        server = ServerSnapshot(0.0, None, self.unsloth, models, {}).server_props
        self.assertEqual(server.quant, "UD-Q3_K_XL")

    def test_id_suffix_wins_over_models_field(self):
        models = ModelsMeta(model_id="Qwen3-30B-A3B-UD-Q4_K_XL.gguf", quant="Q8_0")
        server = ServerSnapshot(0.0, None, self.unsloth, models, {}).server_props
        self.assertEqual(server.quant, "UD-Q4_K_XL")

    def test_props_quant_wins_over_models_field(self):
        props = PropsSnapshot(model_name="local-model", quant="Q8_0")
        models = ModelsMeta.from_payload(UNSLOTH_MODELS)
        self.assertIs(ServerSnapshot(0.0, None, props, models, {}).server_props, props)

    def test_models_quant_used_even_without_id(self):
        models = ModelsMeta(quant="Q5_K_M", entry_loaded=True)
        server = ServerSnapshot(0.0, None, self.unsloth, models, {}).server_props
        self.assertEqual(server.quant, "Q5_K_M")

    def test_no_quant_anywhere_stays_none(self):
        models = ModelsMeta.from_payload({"data": [{"id": "unsloth/Qwen3.8-27B-GGUF"}]})
        server = ServerSnapshot(0.0, None, self.unsloth, models, {}).server_props
        self.assertIsNone(server.quant)

    def test_props_quant_not_overwritten_by_loaded_entry(self):
        props = PropsSnapshot(model_name="Qwen3.8-Flash-Next-GGUF", quant="Q8_0")
        models = ModelsMeta.from_payload(UNSLOTH_MODELS)
        self.assertEqual(ServerSnapshot(0.0, None, props, models, {}).server_props.quant, "Q8_0")

    def test_id_suffix_wins_over_models_field_on_loaded_entry(self):
        models = ModelsMeta.from_payload(
            {"data": [{"id": "Qwen3-30B-A3B-UD-Q4_K_XL.gguf", "quant": "Q8_0", "loaded": True}]}
        )
        server = ServerSnapshot(0.0, None, self.unsloth, models, {}).server_props
        self.assertEqual(server.quant, "UD-Q4_K_XL")

    def test_unloaded_unmatched_entry_not_backfilled(self):
        payload = {"data": [{"id": "unsloth/gemma-5-12b-it-GGUF", "quant": "Q4_K_M"},
                            {"id": "unsloth/Qwen3.8-27B-GGUF", "quant": "UD-IQ3_S"}]}
        models = ModelsMeta.from_payload(payload)
        self.assertIs(ServerSnapshot(0.0, None, self.unsloth, models, {}).server_props, self.unsloth)
        bare = PropsSnapshot(total_slots=1)
        self.assertIs(ServerSnapshot(0.0, None, bare, models, {}).server_props, bare)

    def test_unloaded_entry_matching_model_path_is_backfilled(self):
        props = PropsSnapshot.from_payload({"model_path": "/models/unsloth/Qwen3.8-27B-GGUF"})
        payload = {"data": [{"id": "unsloth/Qwen3.8-27B-GGUF", "quant": "UD-IQ3_S"}]}
        models = ModelsMeta.from_payload(payload)
        self.assertEqual(ServerSnapshot(0.0, None, props, models, {}).server_props.quant, "UD-IQ3_S")

    def test_without_props_returns_none(self):
        self.assertIsNone(ServerSnapshot(0.0, None, None, self.models, {}).server_props)


if __name__ == "__main__":
    unittest.main()
