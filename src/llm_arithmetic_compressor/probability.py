from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .arithmetic import FrequencyTable


QUANTIZATION_TOPK_ESCAPE_V1 = "topk_escape_v1"


@dataclass(frozen=True)
class TopKDistribution:
    table: FrequencyTable
    token_ids: np.ndarray
    escape_symbol: int


def logits_to_topk_escape_distribution(
    logits: Sequence[float],
    top_k: int,
    precision_bits: int = 20,
    min_frequency: int = 1,
) -> TopKDistribution:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if precision_bits <= 0:
        raise ValueError("precision_bits must be positive")

    logits_array = np.asarray(logits, dtype=np.float64)
    if logits_array.ndim != 1:
        logits_array = logits_array.reshape(-1)
    vocab_size = int(logits_array.size)
    if vocab_size == 0:
        raise ValueError("logits cannot be empty")
    top_k = min(top_k, vocab_size)

    finite = np.isfinite(logits_array)
    if not bool(np.all(finite)):
        logits_array = np.where(finite, logits_array, -1.0e30)

    if top_k == vocab_size:
        candidate_ids = np.arange(vocab_size)
    else:
        candidate_ids = np.argpartition(logits_array, -top_k)[-top_k:]
    candidate_logits = logits_array[candidate_ids]
    order = np.lexsort((candidate_ids, -candidate_logits))
    token_ids = candidate_ids[order].astype(np.int64)
    top_logits = logits_array[token_ids]

    max_logit = float(np.max(logits_array))
    all_weights = np.exp(np.maximum(logits_array - max_logit, -745.0))
    denom = float(np.sum(all_weights))
    if not math.isfinite(denom) or denom <= 0:
        top_probs = np.full(top_k, 1.0 / vocab_size, dtype=np.float64)
        escape_prob = max(0.0, 1.0 - float(top_k) / vocab_size)
    else:
        top_weights = np.exp(np.maximum(top_logits - max_logit, -745.0))
        top_probs = top_weights / denom
        escape_prob = max(0.0, 1.0 - float(np.sum(top_probs)))

    probs = np.concatenate([top_probs, np.array([escape_prob], dtype=np.float64)])
    total = 1 << precision_bits
    floor_mass = len(probs) * min_frequency
    if floor_mass > total:
        raise ValueError(
            f"precision_bits={precision_bits} is too small for top_k={top_k}"
        )

    remaining = total - floor_mass
    frequencies = _quantize_raw(probs * remaining, remaining, min_frequency)
    return TopKDistribution(
        table=FrequencyTable.from_frequencies(frequencies),
        token_ids=token_ids,
        escape_symbol=top_k,
    )


def _quantize_raw(raw: np.ndarray, remaining: int, min_frequency: int) -> np.ndarray:
    extras = np.floor(raw).astype(np.int64)
    leftovers = int(remaining - int(np.sum(extras)))
    if leftovers:
        remainders = raw - extras
        order = np.lexsort((np.arange(len(raw)), -remainders))
        extras[order[:leftovers]] += 1
    return extras + min_frequency
