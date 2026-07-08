"""
Minimal Protobuf wire-format codec for cTrader Open API.
Supports enough of protobuf v2 to encode/decode the cTrader message set.
"""

import struct
from typing import Any


def _encode_varint(value):
    result = bytearray()
    while value > 0x7F:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value & 0x7F)
    return bytes(result)


def _decode_varint(data: bytes, offset: int):
    result = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        result |= (byte & 0x7F) << shift
        offset += 1
        if not (byte & 0x80):
            return result, offset
        shift += 7
    raise ValueError("Truncated varint")


def _encode_signed(value):
    if value >= 0:
        return _encode_varint(value << 1)
    return _encode_varint(~(value << 1))


def _decode_signed(data, offset):
    raw, offset = _decode_varint(data, offset)
    if raw & 1:
        return ~(raw >> 1), offset
    return raw >> 1, offset


def encode_field(field_number: int, wire_type: int, value: bytes) -> bytes:
    key = (field_number << 3) | wire_type
    return _encode_varint(key) + value


def encode_message(fields: list[tuple[int, int, bytes]]) -> bytes:
    return b"".join(encode_field(f, wt, v) for f, wt, v in fields)


def encode_float(val: float) -> bytes:
    return struct.pack("<f", val)


def encode_double(val: float) -> bytes:
    return struct.pack("<d", val)


def encode_int64(val: int) -> bytes:
    return _encode_varint(val)


def encode_int32(val: int) -> bytes:
    return _encode_varint(val)


def encode_sint64(val: int) -> bytes:
    return _encode_signed(val)


def encode_uint64(val: int) -> bytes:
    return _encode_varint(val)


def encode_bool(val: bool) -> bytes:
    return b"\x01" if val else b"\x00"


def encode_string(val: str) -> bytes:
    encoded = val.encode("utf-8")
    return _encode_varint(len(encoded)) + encoded


def encode_bytes(val: bytes) -> bytes:
    return _encode_varint(len(val)) + val


def encode_embedded(msg: bytes) -> bytes:
    return _encode_varint(len(msg)) + msg


def decode_float(data: bytes) -> float:
    return struct.unpack("<f", data[:4])[0]


def decode_double(data: bytes) -> float:
    return struct.unpack("<d", data[:8])[0]


def decode_int64(data, offset):
    return _decode_varint(data, offset)


def decode_int32(data, offset):
    return _decode_varint(data, offset)


def decode_sint64(data, offset):
    return _decode_signed(data, offset)


def decode_bool(data, offset):
    val, offset = _decode_varint(data, offset)
    return val != 0, offset


def decode_length_delimited(data, offset):
    length, offset = _decode_varint(data, offset)
    if offset + length > len(data):
        raise ValueError("Length-delimited field exceeds buffer")
    sub = data[offset:offset + length]
    return sub, offset + length


class Decoder:
    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def remaining(self):
        return self._pos < len(self._data)

    def read_int32(self, field_num, wire_type=0):
        if self._skip_to(field_num, wire_type):
            val, self._pos = decode_int32(self._data, self._pos)
            return val
        return 0

    def read_int64(self, field_num, wire_type=0):
        if self._skip_to(field_num, wire_type):
            val, self._pos = decode_int64(self._data, self._pos)
            return val
        return 0

    def read_uint64(self, field_num, wire_type=0):
        if self._skip_to(field_num, wire_type):
            val, self._pos = _decode_varint(self._data, self._pos)
            return val
        return 0

    def read_sint64(self, field_num, wire_type=0):
        if self._skip_to(field_num, wire_type):
            val, self._pos = decode_sint64(self._data, self._pos)
            return val
        return 0

    def read_bool(self, field_num, wire_type=0):
        if self._skip_to(field_num, wire_type):
            val, self._pos = decode_bool(self._data, self._pos)
            return val
        return False

    def read_float(self, field_num, wire_type=5):
        if self._skip_to(field_num, wire_type):
            val = decode_float(self._data[self._pos:self._pos + 4])
            self._pos += 4
            return val
        return 0.0

    def read_double(self, field_num, wire_type=1):
        if self._skip_to(field_num, wire_type):
            val = decode_double(self._data[self._pos:self._pos + 8])
            self._pos += 8
            return val
        return 0.0

    def read_string(self, field_num, wire_type=2):
        if self._skip_to(field_num, wire_type):
            sub, self._pos = decode_length_delimited(self._data, self._pos)
            return sub.decode("utf-8")
        return ""

    def read_bytes(self, field_num, wire_type=2):
        if self._skip_to(field_num, wire_type):
            sub, self._pos = decode_length_delimited(self._data, self._pos)
            return sub
        return b""

    def read_embedded(self, field_num):
        return self.read_bytes(field_num, 2)

    def _skip_to(self, field_num, wire_type):
        self._pos = 0
        target_key = (field_num << 3) | wire_type
        while self.remaining():
            key, self._pos = _decode_varint(self._data, self._pos)
            if key == target_key:
                return True
            f_wt = key & 0x07
            if f_wt == 0:
                _, self._pos = _decode_varint(self._data, self._pos)
            elif f_wt == 1:
                self._pos += 8
            elif f_wt == 2:
                _, self._pos = decode_length_delimited(self._data, self._pos)
            elif f_wt == 5:
                self._pos += 4
            else:
                raise ValueError(f"Unknown wire type {f_wt}")
        return False

    def iter_repeated_embedded(self, field_num):
        self._pos = 0
        target_key = (field_num << 3) | 2
        while self.remaining():
            key, self._pos = _decode_varint(self._data, self._pos)
            if key == target_key:
                sub, self._pos = decode_length_delimited(self._data, self._pos)
                yield Decoder(sub)
            else:
                f_wt = key & 0x07
                if f_wt == 0:
                    _, self._pos = _decode_varint(self._data, self._pos)
                elif f_wt == 1:
                    self._pos += 8
                elif f_wt == 2:
                    _, self._pos = decode_length_delimited(self._data, self._pos)
                elif f_wt == 5:
                    self._pos += 4
