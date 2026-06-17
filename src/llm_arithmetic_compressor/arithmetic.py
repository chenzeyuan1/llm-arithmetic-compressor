from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


STATE_BITS = 32
FULL_RANGE = 1 << STATE_BITS
HALF_RANGE = FULL_RANGE >> 1
QUARTER_RANGE = HALF_RANGE >> 1
THREE_QUARTER_RANGE = QUARTER_RANGE * 3
STATE_MASK = FULL_RANGE - 1


class BitOutput:
    def __init__(self) -> None:
        self._bytes = bytearray()
        self._current = 0
        self._count = 0
        self.bit_length = 0

    def write(self, bit: int) -> None:
        self._current = (self._current << 1) | (bit & 1)
        self._count += 1
        self.bit_length += 1
        if self._count == 8:
            self._bytes.append(self._current)
            self._current = 0
            self._count = 0

    def finish(self) -> bytes:
        if self._count:
            self._bytes.append(self._current << (8 - self._count))
            self._current = 0
            self._count = 0
        return bytes(self._bytes)


class BitInput:
    def __init__(self, data: bytes, bit_length: int | None = None) -> None:
        self._data = data
        self._bit_length = len(data) * 8 if bit_length is None else bit_length
        self._pos = 0

    def read(self) -> int:
        if self._pos >= self._bit_length:
            return 0
        byte = self._data[self._pos >> 3]
        shift = 7 - (self._pos & 7)
        self._pos += 1
        return (byte >> shift) & 1


@dataclass(frozen=True)
class FrequencyTable:
    frequencies: Sequence[int]
    cumulative: Sequence[int]
    total: int

    @classmethod
    def from_frequencies(cls, frequencies: Sequence[int]) -> "FrequencyTable":
        if len(frequencies) == 0:
            raise ValueError("frequency table is empty")
        freq_array = np.asarray(frequencies, dtype=np.int64)
        if np.any(freq_array <= 0):
            raise ValueError("all frequencies must be positive")
        total = int(freq_array.sum())
        if total >= QUARTER_RANGE:
            raise ValueError("frequency total is too large for 32-bit coder")
        cumulative = np.empty(len(freq_array) + 1, dtype=np.int64)
        cumulative[0] = 0
        cumulative[1:] = np.cumsum(freq_array, dtype=np.int64)
        return cls(frequencies=freq_array, cumulative=cumulative, total=total)

    def symbol_for_count(self, count: int) -> int:
        return int(np.searchsorted(self.cumulative, count, side="right") - 1)


class ArithmeticEncoder:
    def __init__(self) -> None:
        self.low = 0
        self.high = STATE_MASK
        self.pending_bits = 0
        self.output = BitOutput()

    def encode_symbol(self, table: FrequencyTable, symbol: int) -> None:
        if symbol < 0 or symbol >= len(table.frequencies):
            raise ValueError(f"symbol {symbol} outside table")
        self._update(table, table.cumulative[symbol], table.cumulative[symbol + 1])

    def finish(self) -> tuple[bytes, int]:
        self.pending_bits += 1
        if self.low < QUARTER_RANGE:
            self._write_bit_plus_pending(0)
        else:
            self._write_bit_plus_pending(1)
        data = self.output.finish()
        return data, self.output.bit_length

    def _update(self, table: FrequencyTable, sym_low: int, sym_high: int) -> None:
        span = self.high - self.low + 1
        self.high = self.low + (span * sym_high // table.total) - 1
        self.low = self.low + (span * sym_low // table.total)

        while True:
            if self.high < HALF_RANGE:
                self._write_bit_plus_pending(0)
            elif self.low >= HALF_RANGE:
                self._write_bit_plus_pending(1)
                self.low -= HALF_RANGE
                self.high -= HALF_RANGE
            elif self.low >= QUARTER_RANGE and self.high < THREE_QUARTER_RANGE:
                self.pending_bits += 1
                self.low -= QUARTER_RANGE
                self.high -= QUARTER_RANGE
            else:
                break
            self.low = (self.low << 1) & STATE_MASK
            self.high = ((self.high << 1) & STATE_MASK) | 1

    def _write_bit_plus_pending(self, bit: int) -> None:
        self.output.write(bit)
        inverse = bit ^ 1
        while self.pending_bits:
            self.output.write(inverse)
            self.pending_bits -= 1


class ArithmeticDecoder:
    def __init__(self, data: bytes, bit_length: int | None = None) -> None:
        self.low = 0
        self.high = STATE_MASK
        self.input = BitInput(data, bit_length)
        self.code = 0
        for _ in range(STATE_BITS):
            self.code = ((self.code << 1) & STATE_MASK) | self.input.read()

    def decode_symbol(self, table: FrequencyTable) -> int:
        span = self.high - self.low + 1
        offset = self.code - self.low
        count = ((offset + 1) * table.total - 1) // span
        symbol = table.symbol_for_count(count)
        self._update(table, table.cumulative[symbol], table.cumulative[symbol + 1])
        return symbol

    def _update(self, table: FrequencyTable, sym_low: int, sym_high: int) -> None:
        span = self.high - self.low + 1
        self.high = self.low + (span * sym_high // table.total) - 1
        self.low = self.low + (span * sym_low // table.total)

        while True:
            if self.high < HALF_RANGE:
                pass
            elif self.low >= HALF_RANGE:
                self.low -= HALF_RANGE
                self.high -= HALF_RANGE
                self.code -= HALF_RANGE
            elif self.low >= QUARTER_RANGE and self.high < THREE_QUARTER_RANGE:
                self.low -= QUARTER_RANGE
                self.high -= QUARTER_RANGE
                self.code -= QUARTER_RANGE
            else:
                break
            self.low = (self.low << 1) & STATE_MASK
            self.high = ((self.high << 1) & STATE_MASK) | 1
            self.code = ((self.code << 1) & STATE_MASK) | self.input.read()
