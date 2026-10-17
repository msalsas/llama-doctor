from collections.abc import Sequence

from . import connectivity, decode, kv, memory, prefill, prefix_cache, props_check, spec
from .base import AnalyzerSpec, DiagnosisContext, Report
from .engine import run_analyzers

BUILTIN_ANALYZERS = (
    AnalyzerSpec("connectivity", "Connectivity", connectivity.analyze),
    AnalyzerSpec("prefill", "Prefill", prefill.analyze),
    AnalyzerSpec("decode", "Decode", decode.analyze),
    AnalyzerSpec("prefix_cache", "Prefix cache", prefix_cache.analyze),
    AnalyzerSpec("kv", "KV cache", kv.analyze),
    AnalyzerSpec("spec", "Speculative decoding", spec.analyze),
    AnalyzerSpec("memory", "Memory", memory.analyze),
    AnalyzerSpec("props", "Properties", props_check.analyze),
)


def diagnose(
    ctx: DiagnosisContext, analyzers: Sequence[AnalyzerSpec] = BUILTIN_ANALYZERS
) -> Report:
    return run_analyzers(ctx, analyzers)
