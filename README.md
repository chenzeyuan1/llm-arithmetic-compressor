# LLM Arithmetic Compressor

Strictly lossless UTF-8 text compression with deterministic next-token logits
from a local Hugging Face causal language model and integer arithmetic coding.

The default model is `Qwen/Qwen2.5-0.5B`. Larger Qwen models and Hugging Face
Mamba causal LM checkpoints can also be used when your machine has enough
memory.

## What It Does

For each token, the model predicts the next-token distribution from the previous
context. The compressor quantizes the distribution into an integer top-k +
escape table, then arithmetic-codes the real next token. The decompressor runs
the same model and tokenizer again, reconstructs the same distributions, and
decodes the exact original token sequence.

The stream is lossless only when decompression uses the same model, tokenizer,
revision, quantization parameters, and effective inference behavior.

## Install

Python 3.11 or newer is recommended.

macOS/Linux:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
pip install -e ".[hf,dev,zstd]"
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[hf,dev,zstd]"
```

For unit-test-only development without installing PyTorch and Transformers:

```bash
pip install -e ".[dev,zstd]"
```

The `zstd` extra installs the optional Python `zstandard` package. The benchmark
prefers the external `zstd` command when it is available, then falls back to the
Python package. If neither is installed, zstd is reported as unavailable and the
other benchmark codecs still run.

## Quick Start

macOS/Linux:

```bash
llm-compress compress input.txt out_dir
llm-compress decompress out_dir/compressed.bin out_dir/header.bin recovered.txt
llm-compress verify input.txt recovered.txt
llm-compress benchmark input.txt out_dir
```

Windows PowerShell:

```powershell
llm-compress compress input.txt out_dir
llm-compress decompress out_dir\compressed.bin out_dir\header.bin recovered.txt
llm-compress verify input.txt recovered.txt
llm-compress benchmark input.txt out_dir
```

If you need a Hugging Face mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

## Output Files

Compression writes two files:

- `compressed.bin`: arithmetic-coded payload bits.
- `header.bin`: compact 37-byte binary header.

The compact header stores only:

- magic/version
- token count
- original byte size
- arithmetic bit length
- CRC32

Model and codec settings are software or command-line contract rather than
stored in the file. Decompression must use the same `--model`, `--revision`,
`--dtype`, `--precision-bits`, `--top-k`, and `--context-window` settings used
for compression. For the strongest reproducibility across machines, use
`--device cpu`.

Compression output reports both:

- `payload_ratio`: `compressed.bin` only, excluding `header.bin`.
- `package_ratio`: `compressed.bin + header.bin`.

## Model And Device Options

Apple Silicon GPU:

```bash
llm-compress --device mps compress input.txt out_dir
llm-compress --device mps decompress out_dir/compressed.bin out_dir/header.bin recovered.txt
```

NVIDIA CUDA GPU:

```bash
llm-compress --device cuda compress input.txt out_dir
llm-compress --device cuda decompress out_dir/compressed.bin out_dir/header.bin recovered.txt
```

Windows CUDA uses the same `--device cuda` option. `--device auto` chooses MPS
when available, then CUDA, then CPU.

Use a larger Qwen model:

```bash
llm-compress --model Qwen/Qwen2.5-1.5B --precision-bits 24 compress input.txt out_1_5b
llm-compress --model Qwen/Qwen2.5-1.5B --precision-bits 24 decompress out_1_5b/compressed.bin out_1_5b/header.bin recovered.txt
```

Use a Hugging Face Mamba causal LM:

```bash
llm-compress --model state-spaces/mamba-130m-hf --top-k 256 compress input.txt out_mamba
llm-compress --model state-spaces/mamba-130m-hf --top-k 256 decompress out_mamba/compressed.bin out_mamba/header.bin recovered_mamba.txt
```

For Mamba speed, install the optional fast kernels recommended by
`transformers` for your platform, such as `mamba-ssm`, `kernels`, or
`causal-conv1d`. Without them, Transformers can fall back to a slower
implementation.

## Benchmark Different Text Lengths

```bash
llm-compress benchmark article.txt bench_out --lengths 0,64,256,1024,4096
```

This writes:

- `bench_out/benchmark_lengths.json`
- `bench_out/benchmark_lengths.md`
- per-length artifacts under `bench_out/len_<chars>/`

The benchmark compares LLM arithmetic coding with gzip, bz2, lzma, and zstd
when either the external `zstd` command or the optional Python `zstandard`
package is available. By default every length is also decompressed and
byte-compared; use `--skip-verify` for faster exploratory runs.

## Notes

- Input files are UTF-8 text.
- Empty files are represented by empty payload plus a valid header.
- The compressed stream does not store per-token probabilities or token ranks.
- Transformer models use incremental `past_key_values`; Mamba models use
  `cache_params` when supported by Transformers.
- By default `--context-window 0` uses the model maximum context window. A small
  explicit window reduces memory, but rebuilding a sliding window can be much
  slower for long files.
- Compression ratio may be worse than traditional compressors on some inputs.
  The primary goal is byte-exact recovery with an LLM probability model.

## Tests

```bash
python -m pytest
```
