from llm_arithmetic_compressor.benchmark import benchmark_by_lengths, parse_lengths
from llm_arithmetic_compressor.codec import CodecConfig
from llm_arithmetic_compressor.model import FakeByteModel


def test_parse_lengths():
    assert parse_lengths("8, 16,32") == [8, 16, 32]


def test_benchmark_by_lengths_outputs_json_and_markdown(tmp_path):
    input_path = tmp_path / "article.txt"
    output_dir = tmp_path / "bench"
    input_path.write_text("早上好，太阳公公。The quick brown fox.", encoding="utf-8")

    result = benchmark_by_lengths(
        input_path,
        output_dir,
        FakeByteModel(),
        CodecConfig(precision_bits=12, context_window=8),
        lengths=[4, 12],
        verify_roundtrip=True,
        show_progress=False,
    )

    assert [row["length_chars"] for row in result["rows"]] == [0, 4, 12, len(input_path.read_text(encoding="utf-8"))]
    assert all(row["llm_arithmetic"]["verified"] for row in result["rows"])
    assert (output_dir / "benchmark_lengths.json").exists()
    markdown = (output_dir / "benchmark_lengths.md").read_text(encoding="utf-8")
    assert "Compression Benchmark by Text Length" in markdown
    assert "payload ratio" in markdown
    assert "package ratio" in markdown
