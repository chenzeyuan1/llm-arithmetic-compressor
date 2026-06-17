import numpy as np

from llm_arithmetic_compressor.probability import logits_to_topk_escape_distribution


def test_topk_escape_quantization_is_positive_and_fixed_total():
    dist = logits_to_topk_escape_distribution(
        [0.0, 1.0, -2.0, 3.0],
        top_k=2,
        precision_bits=8,
    )
    assert all(freq > 0 for freq in dist.table.frequencies)
    assert dist.table.total == 256
    assert len(dist.token_ids) == 2
    assert dist.escape_symbol == 2


def test_topk_escape_quantization_is_stable():
    a = logits_to_topk_escape_distribution([0.1, 0.2, 0.3], top_k=2, precision_bits=6)
    b = logits_to_topk_escape_distribution([0.1, 0.2, 0.3], top_k=2, precision_bits=6)
    assert np.array_equal(a.table.frequencies, b.table.frequencies)
    assert np.array_equal(a.token_ids, b.token_ids)
