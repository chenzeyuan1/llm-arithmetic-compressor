from pathlib import Path
import zlib

from llm_arithmetic_compressor.codec import CodecConfig, compress_file, decompress_file, verify_files
from llm_arithmetic_compressor.header import HEADER_STRUCT, read_header
from llm_arithmetic_compressor.model import FakeByteModel


def _roundtrip(tmp_path: Path, text: str):
    model = FakeByteModel()
    config = CodecConfig(precision_bits=12, context_window=16)
    input_path = tmp_path / "input.txt"
    out_dir = tmp_path / "compressed"
    recovered_path = tmp_path / "recovered.txt"
    input_path.write_bytes(text.encode("utf-8"))

    metadata = compress_file(
        input_path,
        out_dir,
        model,
        config,
        show_progress=False,
    )
    decompress_file(
        out_dir / "compressed.bin",
        out_dir / "header.bin",
        recovered_path,
        model,
        config,
        show_progress=False,
    )
    assert metadata["token_count"] == len(text.encode("utf-8"))
    assert metadata["container"]["payload_size"] == metadata["compressed_size"]
    assert metadata["container"]["package_size"] >= metadata["container"]["payload_size"]
    assert verify_files(input_path, recovered_path)


def test_roundtrip_chinese(tmp_path):
    _roundtrip(tmp_path, "早上好，太阳公公。")


def test_roundtrip_english(tmp_path):
    _roundtrip(tmp_path, "The quick brown fox jumps over the lazy dog.")


def test_roundtrip_random(tmp_path):
    _roundtrip(tmp_path, "a8F#91xzPq@@29kLm")


def test_roundtrip_repeated(tmp_path):
    _roundtrip(tmp_path, "哈哈哈哈哈哈哈哈哈哈哈哈")


def test_roundtrip_empty(tmp_path):
    _roundtrip(tmp_path, "")


def test_roundtrip_longer_than_context_window(tmp_path):
    _roundtrip(tmp_path, "早上好。" * 40)


def test_logit_session_matches_window_recompute():
    model = FakeByteModel()
    tokens = model.encode_text("abcdefg")
    context_window = 3
    session = model.create_logit_session(context_window)

    for i, token_id in enumerate(tokens):
        start = max(0, i - context_window)
        context = [model.prefix_token_id] + tokens[start:i]
        assert session.current_logits() == model.next_token_logits(context)
        session.advance(token_id)


def test_compact_header_has_required_fields(tmp_path):
    model = FakeByteModel()
    input_path = tmp_path / "input.txt"
    out_dir = tmp_path / "compressed"
    input_path.write_text("compact header", encoding="utf-8")

    metadata = compress_file(
        input_path,
        out_dir,
        model,
        CodecConfig(precision_bits=12, context_window=16),
        show_progress=False,
    )

    header_path = out_dir / "header.bin"
    header = read_header(header_path)
    assert header_path.stat().st_size == HEADER_STRUCT.size
    assert header.token_count == metadata["token_count"]
    assert header.original_size == metadata["original_size"]
    assert header.bit_length == metadata["arithmetic"]["bit_length"]
    assert header.crc32 == zlib.crc32(input_path.read_bytes()) & 0xFFFFFFFF
    assert metadata["modeling"]["runtime"]["model_type"] == ""


def test_crc_mismatch_fails_after_decode(tmp_path):
    model = FakeByteModel()
    input_path = tmp_path / "input.txt"
    out_dir = tmp_path / "compressed"
    recovered_path = tmp_path / "recovered.txt"
    input_path.write_text("runtime mismatch", encoding="utf-8")

    compress_file(
        input_path,
        out_dir,
        model,
        CodecConfig(precision_bits=12, context_window=16),
        show_progress=False,
    )
    metadata_path = out_dir / "header.bin"
    header_data = bytearray(metadata_path.read_bytes())
    header_data[-1] ^= 0x01
    metadata_path.write_bytes(header_data)

    try:
        decompress_file(
            out_dir / "compressed.bin",
            metadata_path,
            recovered_path,
            model,
            CodecConfig(precision_bits=12, context_window=16),
            show_progress=False,
        )
    except RuntimeError as exc:
        assert "crc32" in str(exc)
    else:
        raise AssertionError("expected crc mismatch to fail")


def test_roundtrip_topk_escape_path(tmp_path):
    model = FakeByteModel()
    input_path = tmp_path / "input.txt"
    out_dir = tmp_path / "compressed"
    recovered_path = tmp_path / "recovered.txt"
    input_path.write_text("a8F#91xzPq@@29kLm", encoding="utf-8")

    metadata = compress_file(
        input_path,
        out_dir,
        model,
        CodecConfig(precision_bits=12, context_window=8, top_k=4),
        show_progress=False,
    )
    decompress_file(
        out_dir / "compressed.bin",
        out_dir / "header.bin",
        recovered_path,
        model,
        CodecConfig(precision_bits=12, context_window=8, top_k=4),
        show_progress=False,
    )
    assert metadata["probability"]["quantization"] == "topk_escape_v1"
    assert verify_files(input_path, recovered_path)
