import argparse
import json
import os
import sys
from collections.abc import Sequence

from . import __version__
from .analysis import diagnose
from .analysis.base import DiagnosisContext, HostHints
from .client import DEFAULT_TIMEOUT, LlamaClient
from .probe import run_probe
from .redact import Secret
from .render.jsondoc import diagnosis_document
from .render.terminal import render_report, use_color

EXIT_TOOL_ERROR = 3


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(EXIT_TOOL_ERROR, f"{self.prog}: error: {message}\n")


def _positive_float(value: str) -> float:
    number = float(value)
    if not number > 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return number


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer > 0") from None
    if number <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Build the llama-doctor command-line parser."""
    parser = _ArgumentParser(prog="llama-doctor")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    diagnose_parser = commands.add_parser("diagnose")
    diagnose_parser.add_argument("url", metavar="URL")
    diagnose_parser.add_argument("--json", action="store_true")
    diagnose_parser.add_argument("--api-key-env", metavar="VAR")
    diagnose_parser.add_argument("--model", metavar="NAME")
    diagnose_parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, metavar="SEC")
    diagnose_parser.add_argument("--probe", action="store_true")
    diagnose_parser.add_argument("--vram-gib", type=float, metavar="N")
    diagnose_parser.add_argument("--ram-gib", type=float, metavar="N")
    diagnose_parser.add_argument("--weights-gib", type=float, metavar="N")
    diagnose_parser.add_argument("--no-color", action="store_true")
    diagnose_parser.add_argument("--no-slots", action="store_true")

    serve_parser = commands.add_parser("serve")
    serve_parser.add_argument("url", metavar="URL")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--interval", type=_positive_float, default=2.0)
    serve_parser.add_argument("--history", type=_positive_int, default=300)
    serve_parser.add_argument("--rate-window", type=_positive_float, default=30.0, metavar="SEC")
    serve_parser.add_argument("--api-key-env", metavar="VAR")
    serve_parser.add_argument("--model", metavar="NAME")
    serve_parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, metavar="SEC")
    serve_parser.add_argument("--vram-gib", type=float, metavar="N")
    serve_parser.add_argument("--ram-gib", type=float, metavar="N")
    serve_parser.add_argument("--weights-gib", type=float, metavar="N")
    serve_parser.add_argument("--no-slots", action="store_true")
    return parser


def _api_key(name: str | None) -> Secret | None:
    if name is None:
        return None
    key = Secret.from_env(name)
    if key is None:
        raise ValueError(f"environment variable {name} is not set or empty")
    return key


def _client(args: argparse.Namespace) -> LlamaClient | None:
    try:
        return LlamaClient(
            args.url, api_key=_api_key(args.api_key_env), timeout=args.timeout, model=args.model
        )
    except ValueError as exc:
        print(f"llama-doctor: {exc}", file=sys.stderr)
        return None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the llama-doctor command-line application."""
    args = build_parser().parse_args(argv)
    if args.command == "diagnose" and args.probe and args.model is not None:
        print("llama-doctor: --probe cannot be combined with --model", file=sys.stderr)
        return EXIT_TOOL_ERROR
    client = _client(args)
    if client is None:
        return EXIT_TOOL_ERROR
    if args.command == "serve":
        from llama_doctor.dashboard.server import run

        return run(
            client,
            host=args.host,
            port=args.port,
            interval=args.interval,
            history=args.history,
            rate_window=args.rate_window,
            hints=HostHints(
                vram_gib=args.vram_gib, ram_gib=args.ram_gib, weights_gib=args.weights_gib
            ),
            slots=not args.no_slots,
            out=sys.stderr,
        )

    snapshot = client.snapshot(include_slots=not args.no_slots)
    probe = None
    if args.probe and snapshot.alive:
        print("probe: warm-up plus one synthetic prompt (prefix cache disabled)", file=sys.stderr)
        probe = run_probe(client)
        if probe is None:
            print("probe: failed, using lifetime counters", file=sys.stderr)
    context = DiagnosisContext(
        snapshot,
        probe,
        HostHints(vram_gib=args.vram_gib, ram_gib=args.ram_gib, weights_gib=args.weights_gib),
    )
    report = diagnose(context)
    if args.json:
        print(
            json.dumps(
                diagnosis_document(report, snapshot, target=client.base_url, probe=probe),
                indent=2,
                allow_nan=False,
            )
        )
    else:
        print(
            render_report(
                report,
                snapshot,
                target=client.base_url,
                probe=probe,
                color=use_color(sys.stdout, os.environ) and not args.no_color,
            )
        )
    return report.exit_code
