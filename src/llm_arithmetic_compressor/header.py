from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path


MAGIC = b"LLMAC2\0\0"
HEADER_VERSION = 2
HEADER_STRUCT = struct.Struct(">8sBQQQI")


@dataclass(frozen=True)
class Header:
    token_count: int
    original_size: int
    bit_length: int
    crc32: int


def write_header(path: Path, header: Header) -> None:
    path.write_bytes(
        HEADER_STRUCT.pack(
            MAGIC,
            HEADER_VERSION,
            header.token_count,
            header.original_size,
            header.bit_length,
            header.crc32 & 0xFFFFFFFF,
        )
    )


def read_header(path: Path) -> Header:
    data = path.read_bytes()
    if len(data) != HEADER_STRUCT.size:
        raise RuntimeError("invalid compact header size")
    magic, version, token_count, original_size, bit_length, crc32 = HEADER_STRUCT.unpack(data)
    if magic != MAGIC:
        raise RuntimeError("not an LLMAC compact header")
    if version != HEADER_VERSION:
        raise RuntimeError("unsupported header version")
    return Header(
        token_count=int(token_count),
        original_size=int(original_size),
        bit_length=int(bit_length),
        crc32=int(crc32),
    )
