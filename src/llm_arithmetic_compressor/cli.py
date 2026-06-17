from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import benchmark_by_lengths, benchmark_traditional, parse_lengths
from .codec import CodecConfig, compress_file, decompress_file, verify_files
from .model import DEFAULT_CONTEXT_WINDOW, DEFAULT_MODEL, DEFAULT_REVISION, ModelConfig, load_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-compress")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--context-window",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW,
        help="Maximum previous tokens to condition on. 0 means use the model maximum.",
    )
    parser.add_argument("--precision-bits", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--quiet", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_compress = sub.add_parser("compress")
    p_compress.add_argument("input", type=Path)
    p_compress.add_argument("output_dir", type=Path)

    p_decompress = sub.add_parser("decompress")
    p_decompress.add_argument("compressed", type=Path)
    p_decompress.add_argument("header", type=Path)
    p_decompress.add_argument("output", type=Path)

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("original", type=Path)
    p_verify.add_argument("recovered", type=Path)

    p_benchmark = sub.add_parser("benchmark")
    p_benchmark.add_argument("input", type=Path)
    p_benchmark.add_argument("output_dir", type=Path)
    p_benchmark.add_argument(
        "--lengths",
        default=None,
        help="Comma-separated character prefix lengths, for example 32,128,512,2048.",
    )
    p_benchmark.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip decompress+byte-compare for each benchmark length.",
    )

    args = parser.parse_args(argv)

    if args.command == "verify":
        ok = verify_files(args.original, args.recovered)
        print("OK: files are byte-identical" if ok else "FAIL: files differ")
        return 0 if ok else 1

    model_config = ModelConfig(
        model_name=args.model,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        context_window=args.context_window,
    )
    model = load_model(model_config)
    codec_config = CodecConfig(
        precision_bits=args.precision_bits,
        min_frequency=1,
        context_window=args.context_window,
        top_k=args.top_k,
    )

    if args.command == "compress":
        metadata = compress_file(args.input, args.output_dir, model, codec_config, not args.quiet)
        _print_json(metadata)
        return 0

    if args.command == "decompress":
        metadata = decompress_file(
            args.compressed,
            args.header,
            args.output,
            model,
            codec_config,
            not args.quiet,
        )
        _print_json({"recovered": str(args.output), "metadata": metadata})
        return 0

    if args.command == "benchmark":
        if args.lengths:
            result = benchmark_by_lengths(
                args.input,
                args.output_dir,
                model,
                codec_config,
                parse_lengths(args.lengths),
                verify_roundtrip=not args.skip_verify,
                show_progress=not args.quiet,
            )
            _print_json(result)
        else:
            metadata = compress_file(args.input, args.output_dir, model, codec_config, not args.quiet)
            traditional = benchmark_traditional(args.input, args.output_dir)
            _print_json({"llm_arithmetic": metadata, "traditional": traditional})
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def _print_json(value: dict) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
