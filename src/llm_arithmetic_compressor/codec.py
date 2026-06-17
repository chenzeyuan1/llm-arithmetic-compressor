from __future__ import annotations

import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .arithmetic import ArithmeticDecoder, ArithmeticEncoder
from .header import Header, read_header, write_header
from .model import ProbabilityModel
from .probability import (
    QUANTIZATION_TOPK_ESCAPE_V1,
    logits_to_topk_escape_distribution,
)


FORMAT_VERSION = 1
LOGITS_MODE_KV_CACHE = "kv_cache"


@dataclass(frozen=True)
class CodecConfig:
    precision_bits: int = 20
    min_frequency: int = 1
    context_window: int = 0
    top_k: int = 4096


def compress_file(
    input_path: Path,
    output_dir: Path,
    model: ProbabilityModel,
    config: CodecConfig,
    show_progress: bool = True,
) -> dict:
    start = time.perf_counter()
    config = _effective_config(config, model)
    data = input_path.read_bytes()
    text = data.decode("utf-8")
    token_ids = model.encode_text(text)

    output_dir.mkdir(parents=True, exist_ok=True)
    compressed_path = output_dir / "compressed.bin"
    header_path = output_dir / "header.bin"

    if not token_ids:
        compressed = b""
        bit_length = 0
    else:
        compressed, bit_length = _compress_tokens_with_kv_cache(
            token_ids,
            model,
            config,
            show_progress,
        )

    compressed_path.write_bytes(compressed)
    elapsed = time.perf_counter() - start
    metadata = _metadata(
        model=model,
        config=config,
        token_count=len(token_ids),
        original_size=len(data),
        compressed_size=len(compressed),
        bit_length=bit_length,
        crc32=zlib.crc32(data) & 0xFFFFFFFF,
        elapsed_seconds=elapsed,
        logits_mode=LOGITS_MODE_KV_CACHE,
    )
    write_header(
        header_path,
        Header(
            token_count=len(token_ids),
            original_size=len(data),
            bit_length=bit_length,
            crc32=zlib.crc32(data) & 0xFFFFFFFF,
        ),
    )
    metadata = _add_container_stats(metadata, header_path.stat().st_size, len(compressed), len(data))
    return metadata


def decompress_file(
    compressed_path: Path,
    metadata_path: Path,
    output_path: Path,
    model: ProbabilityModel,
    config: CodecConfig | None = None,
    show_progress: bool = True,
) -> dict:
    header = read_header(metadata_path)
    config = _effective_config(config or CodecConfig(), model)
    token_count = header.token_count
    compressed = compressed_path.read_bytes()

    if token_count == 0:
        output_path.write_bytes(b"")
        return _metadata_from_header(header, model, config, len(compressed), 0.0)

    token_ids = _decompress_tokens_with_kv_cache(
        compressed,
        header.bit_length,
        token_count,
        model,
        config,
        show_progress,
    )

    text = model.decode_tokens(token_ids)
    data = text.encode("utf-8")
    if len(data) != header.original_size:
        raise RuntimeError("decoded byte length does not match metadata")
    if (zlib.crc32(data) & 0xFFFFFFFF) != header.crc32:
        raise RuntimeError("decoded bytes do not match original crc32")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return _metadata_from_header(header, model, config, len(compressed), 0.0)


def verify_files(original_path: Path, recovered_path: Path) -> bool:
    return original_path.read_bytes() == recovered_path.read_bytes()


def _compress_tokens_with_kv_cache(
    token_ids: list[int],
    model: ProbabilityModel,
    config: CodecConfig,
    show_progress: bool,
) -> tuple[bytes, int]:
    encoder = ArithmeticEncoder()
    session = model.create_logit_session(config.context_window)
    iterator = _progress(token_ids, "compress", show_progress)
    for token_id in iterator:
        _encode_token(encoder, session.current_logits(), token_id, model, config)
        session.advance(token_id)
    return encoder.finish()


def _decompress_tokens_with_kv_cache(
    compressed: bytes,
    bit_length: int,
    token_count: int,
    model: ProbabilityModel,
    config: CodecConfig,
    show_progress: bool,
) -> list[int]:
    decoder = ArithmeticDecoder(compressed, bit_length)
    token_ids: list[int] = []
    session = model.create_logit_session(config.context_window)
    iterator = range(token_count)
    if show_progress:
        iterator = tqdm(iterator, desc="decompress", unit="tok")
    for _ in iterator:
        token_id = _decode_token(decoder, session.current_logits(), model, config)
        token_ids.append(token_id)
        session.advance(token_id)
    return token_ids


def _encode_token(
    encoder: ArithmeticEncoder,
    logits,
    token_id: int,
    model: ProbabilityModel,
    config: CodecConfig,
) -> None:
    dist = logits_to_topk_escape_distribution(
        logits,
        top_k=config.top_k,
        precision_bits=config.precision_bits,
        min_frequency=config.min_frequency,
    )
    positions = np.flatnonzero(dist.token_ids == token_id)
    if len(positions):
        encoder.encode_symbol(dist.table, int(positions[0]))
    else:
        encoder.encode_symbol(dist.table, dist.escape_symbol)
        encoder.encode_symbol(_uniform_token_table(model.vocab_size), token_id)


def _decode_token(
    decoder: ArithmeticDecoder,
    logits,
    model: ProbabilityModel,
    config: CodecConfig,
) -> int:
    dist = logits_to_topk_escape_distribution(
        logits,
        top_k=config.top_k,
        precision_bits=config.precision_bits,
        min_frequency=config.min_frequency,
    )
    symbol = decoder.decode_symbol(dist.table)
    if symbol == dist.escape_symbol:
        return decoder.decode_symbol(_uniform_token_table(model.vocab_size))
    return int(dist.token_ids[symbol])


def _probability_metadata(config: CodecConfig) -> dict:
    return {
        "precision_bits": config.precision_bits,
        "min_frequency": config.min_frequency,
        "quantization": QUANTIZATION_TOPK_ESCAPE_V1,
        "top_k": config.top_k,
    }


_UNIFORM_TABLE_CACHE = {}


def _uniform_token_table(vocab_size: int):
    table = _UNIFORM_TABLE_CACHE.get(vocab_size)
    if table is None:
        from .arithmetic import FrequencyTable

        table = FrequencyTable.from_frequencies(np.ones(vocab_size, dtype=np.int64))
        _UNIFORM_TABLE_CACHE[vocab_size] = table
    return table


def _metadata(
    model: ProbabilityModel,
    config: CodecConfig,
    token_count: int,
    original_size: int,
    compressed_size: int,
    bit_length: int,
    crc32: int,
    elapsed_seconds: float,
    logits_mode: str,
) -> dict:
    ratio = None if original_size == 0 else compressed_size / original_size
    return {
        "format_version": FORMAT_VERSION,
        "backend": "hf_transformers",
        "model_name": model.model_name,
        "tokenizer_name": model.tokenizer_name,
        "revision": model.revision,
        "vocab_size": model.vocab_size,
        "prefix_token_id": model.prefix_token_id,
        "token_count": token_count,
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": ratio,
        "payload_size": compressed_size,
        "payload_ratio": ratio,
        "crc32": f"{crc32:08x}",
        "context_window": config.context_window,
        "modeling": {
            "logits_mode": logits_mode,
            "runtime": model.runtime_info(),
            "max_context_tokens": model.max_context_tokens,
        },
        "probability": _probability_metadata(config),
        "arithmetic": {
            "state_bits": 32,
            "bit_length": bit_length,
        },
        "timing": {
            "elapsed_seconds": elapsed_seconds,
        },
        "config": asdict(config),
    }


def _metadata_from_header(
    header: Header,
    model: ProbabilityModel,
    config: CodecConfig,
    compressed_size: int,
    elapsed_seconds: float,
) -> dict:
    return _metadata(
        model=model,
        config=config,
        token_count=header.token_count,
        original_size=header.original_size,
        compressed_size=compressed_size,
        bit_length=header.bit_length,
        crc32=header.crc32,
        elapsed_seconds=elapsed_seconds,
        logits_mode=LOGITS_MODE_KV_CACHE,
    )


def _add_container_stats(
    metadata: dict,
    header_size: int,
    payload_size: int,
    original_size: int,
) -> dict:
    package_size = payload_size + header_size
    metadata["container"] = {
        "payload_size": payload_size,
        "header_size": header_size,
        "package_size": package_size,
        "payload_ratio": None if original_size == 0 else payload_size / original_size,
        "package_ratio": None if original_size == 0 else package_size / original_size,
    }
    return metadata


def _effective_config(config: CodecConfig, model: ProbabilityModel) -> CodecConfig:
    max_context = max(1, model.max_context_tokens - 1)
    requested_context = config.context_window if config.context_window > 0 else max_context
    return CodecConfig(
        precision_bits=config.precision_bits,
        min_frequency=config.min_frequency,
        context_window=min(requested_context, max_context),
        top_k=min(max(1, config.top_k), model.vocab_size),
    )


def _progress(items, desc: str, show_progress: bool):
    if show_progress:
        return tqdm(items, desc=desc, unit="tok")
    return items
