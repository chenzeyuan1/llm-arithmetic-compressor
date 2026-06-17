from __future__ import annotations

import bz2
import gzip
import json
import lzma
import shutil
import subprocess
from pathlib import Path

from .codec import CodecConfig, compress_file, decompress_file, verify_files
from .model import ProbabilityModel


def benchmark_traditional(input_path: Path, output_dir: Path) -> dict:
    data = input_path.read_bytes()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "original": {"size": len(data), "ratio": 1.0 if data else None},
        "gzip": _write_bytes(output_dir / "traditional.gz", gzip.compress(data), len(data)),
        "bz2": _write_bytes(output_dir / "traditional.bz2", bz2.compress(data), len(data)),
        "lzma": _write_bytes(output_dir / "traditional.xz", lzma.compress(data), len(data)),
    }
    zstd = shutil.which("zstd")
    if zstd:
        zstd_path = output_dir / "traditional.zst"
        subprocess.run(
            [zstd, "-q", "-f", str(input_path), "-o", str(zstd_path)],
            check=True,
        )
        results["zstd"] = _size_result(zstd_path.stat().st_size, len(data))
    else:
        results["zstd"] = _benchmark_zstandard_package(output_dir, data)
    return results


def benchmark_by_lengths(
    input_path: Path,
    output_dir: Path,
    model: ProbabilityModel,
    codec_config: CodecConfig,
    lengths: list[int],
    verify_roundtrip: bool = True,
    show_progress: bool = True,
) -> dict:
    text = input_path.read_text(encoding="utf-8")
    selected_lengths = _valid_lengths(lengths, len(text))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for length in selected_lengths:
        sample_text = text[:length]
        sample_dir = output_dir / f"len_{length}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_path = sample_dir / "input.txt"
        sample_path.write_text(sample_text, encoding="utf-8")

        llm_dir = sample_dir / "llm"
        metadata = compress_file(
            sample_path,
            llm_dir,
            model,
            codec_config,
            show_progress=show_progress,
        )

        verified = None
        if verify_roundtrip:
            recovered_path = sample_dir / "recovered.txt"
            decompress_file(
                llm_dir / "compressed.bin",
                llm_dir / "header.bin",
                recovered_path,
                model,
                codec_config,
                show_progress=show_progress,
            )
            verified = verify_files(sample_path, recovered_path)

        traditional = benchmark_traditional(sample_path, sample_dir / "traditional")
        rows.append(
            {
                "length_chars": length,
                "original_size": metadata["original_size"],
                "token_count": metadata["token_count"],
                "llm_arithmetic": {
                    "payload_size": metadata["container"]["payload_size"],
                    "header_size": metadata["container"]["header_size"],
                    "package_size": metadata["container"]["package_size"],
                    "payload_ratio": metadata["container"]["payload_ratio"],
                    "package_ratio": metadata["container"]["package_ratio"],
                    "elapsed_seconds": metadata["timing"]["elapsed_seconds"],
                    "verified": verified,
                },
                "traditional": traditional,
            }
        )

    result = {
        "input": str(input_path),
        "length_unit": "characters",
        "model_name": model.model_name,
        "tokenizer_name": model.tokenizer_name,
        "revision": model.revision,
        "codec": {
            "precision_bits": codec_config.precision_bits,
            "min_frequency": codec_config.min_frequency,
            "context_window": codec_config.context_window,
            "top_k": codec_config.top_k,
            "quantization": "topk_escape_v1",
        },
        "rows": rows,
    }
    (output_dir / "benchmark_lengths.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "benchmark_lengths.md").write_text(
        format_length_benchmark_markdown(result),
        encoding="utf-8",
    )
    return result


def format_length_benchmark_markdown(result: dict) -> str:
    headers = [
        "chars",
        "bytes",
        "tokens",
        "payload size",
        "payload ratio",
        "package size",
        "package ratio",
        "gzip",
        "bz2",
        "lzma",
        "zstd",
        "verified",
    ]
    lines = [
        "# Compression Benchmark by Text Length",
        "",
        f"- model: `{result['model_name']}`",
        f"- tokenizer: `{result['tokenizer_name']}`",
        f"- revision: `{result['revision']}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in result["rows"]:
        trad = row["traditional"]
        llm = row["llm_arithmetic"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["length_chars"]),
                    str(row["original_size"]),
                    str(row["token_count"]),
                    str(llm["payload_size"]),
                    _ratio(llm["payload_ratio"]),
                    str(llm["package_size"]),
                    _ratio(llm["package_ratio"]),
                    _ratio(trad["gzip"]["ratio"]),
                    _ratio(trad["bz2"]["ratio"]),
                    _ratio(trad["lzma"]["ratio"]),
                    _ratio(trad["zstd"]["ratio"]),
                    _verified(llm["verified"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_lengths(value: str) -> list[int]:
    lengths = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        length = int(part)
        if length < 0:
            raise ValueError("lengths must be non-negative")
        lengths.append(length)
    if not lengths:
        raise ValueError("at least one length is required")
    return lengths


def _write_bytes(path: Path, data: bytes, original_size: int) -> dict:
    path.write_bytes(data)
    return _size_result(len(data), original_size)


def _benchmark_zstandard_package(output_dir: Path, data: bytes) -> dict:
    try:
        import zstandard
    except ImportError:
        return {"size": None, "ratio": None, "available": False}

    compressed = zstandard.ZstdCompressor().compress(data)
    result = _write_bytes(output_dir / "traditional.zst", compressed, len(data))
    result["backend"] = "python-zstandard"
    return result


def _size_result(size: int, original_size: int) -> dict:
    return {
        "size": size,
        "ratio": None if original_size == 0 else size / original_size,
    }


def _valid_lengths(lengths: list[int], max_length: int) -> list[int]:
    valid = sorted(set(length for length in lengths if length <= max_length))
    if 0 not in valid:
        valid.insert(0, 0)
    if max_length not in valid:
        valid.append(max_length)
    return valid


def _ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _verified(value: bool | None) -> str:
    if value is None:
        return "skipped"
    return "yes" if value else "no"
