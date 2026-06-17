from llm_arithmetic_compressor.arithmetic import ArithmeticDecoder, ArithmeticEncoder, FrequencyTable


def test_arithmetic_roundtrip_small_table():
    table = FrequencyTable.from_frequencies([1, 3, 2, 7])
    symbols = [3, 3, 1, 0, 2, 3, 1, 1, 0, 3]

    encoder = ArithmeticEncoder()
    for symbol in symbols:
        encoder.encode_symbol(table, symbol)
    data, bit_length = encoder.finish()

    decoder = ArithmeticDecoder(data, bit_length)
    recovered = [decoder.decode_symbol(table) for _ in symbols]
    assert recovered == symbols
