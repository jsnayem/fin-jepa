"""
cTrader Open API v2+ message encoding/decoding using minimal protobuf codec.
"""

import time
import zlib

from ctrader.proto_codec import (
    Decoder,
    encode_bool,
    encode_bytes,
    encode_double,
    encode_embedded,
    encode_float,
    encode_int32,
    encode_int64,
    encode_message,
    encode_sint64,
    encode_string,
    encode_uint64,
)

# ---- Protocol constants ----
PROTO_VERSION = 2

CLIENT_MSGID_APPLICATION_AUTH_REQ = 4500
PROTO_MSGID_APPLICATION_AUTH_RES = 4501
CLIENT_MSGID_ACCOUNT_AUTH_REQ = 4502
PROTO_MSGID_ACCOUNT_AUTH_RES = 4503

CLIENT_MSGID_TRENDBAR_REQ = 4508
PROTO_MSGID_TRENDBAR_RES = 4509

CLIENT_MSGID_SUBSCRIBE_SPOTS_REQ = 4512
PROTO_MSGID_SPOT_EVENT = 4513

PROTO_MSGID_RECONCILE_RES = 4505
PROTO_MSGID_HEARTBEAT_EVENT = 4502

# Symbol IDs for common forex pairs on cTrader
FOREX_SYMBOLS = {
    "EURUSD": 1,
    "GBPUSD": 2,
    "USDJPY": 3,
    "USDCHF": 4,
    "AUDUSD": 5,
    "NZDUSD": 6,
    "USDCAD": 7,
}

TRENDBAR_PERIODS = {
    "M1": 1,
    "M2": 2,
    "M3": 3,
    "M4": 4,
    "M5": 5,
    "M10": 10,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H2": 120,
    "H3": 180,
    "H4": 240,
    "H6": 360,
    "H8": 480,
    "D1": 1440,
    "W1": 10080,
    "MN1": 43200,
}


def make_frame(payload_type: int, payload: bytes) -> bytes:
    """Wrap payload in a cTrader websocket frame (length-prefixed with varint)."""
    msg = encode_message([
        (1, 0, encode_int32(payload_type)),
        (2, 2, encode_bytes(payload)),
    ])
    from ctrader.proto_codec import _encode_varint
    return _encode_varint(len(msg)) + msg


def parse_frame(data: bytes) -> tuple[int, bytes]:
    """Parse a cTrader websocket frame, returning (payload_type, payload)."""
    from ctrader.proto_codec import _decode_varint
    _, offset = _decode_varint(data, 0)
    dec = Decoder(data[offset:])
    return dec.read_int32(1), dec.read_bytes(2)


def build_app_auth(client_id: str, client_secret: str) -> bytes:
    payload = encode_message([
        (1, 2, encode_string(client_id)),
        (2, 2, encode_string(client_secret)),
    ])
    return make_frame(CLIENT_MSGID_APPLICATION_AUTH_REQ, payload)


def build_account_auth(access_token: str) -> bytes:
    payload = encode_message([
        (1, 2, encode_string(access_token)),
    ])
    return make_frame(CLIENT_MSGID_ACCOUNT_AUTH_REQ, payload)


def build_trendbar_req(
    account_id: int,
    symbol_id: int,
    period: int,
    from_ts: int,
    to_ts: int,
) -> bytes:
    payload = encode_message([
        (1, 0, encode_int64(account_id)),
        (2, 0, encode_int32(symbol_id)),
        (3, 0, encode_int32(period)),
        (4, 0, encode_int64(from_ts)),
        (5, 0, encode_int64(to_ts)),
    ])
    return make_frame(CLIENT_MSGID_TRENDBAR_REQ, payload)


def decode_trendbar_res(payload: bytes) -> list[dict]:
    """Decode ProtoOATrendbarRes → list of bar dicts."""
    dec = Decoder(payload)
    bars = []
    for bar_dec in dec.iter_repeated_embedded(1):
        bar = {
            "volume": bar_dec.read_int64(1),
            "period": bar_dec.read_int32(2),
            "low": bar_dec.read_sint64(3),
            "delta_open": bar_dec.read_uint64(4),
            "delta_close": bar_dec.read_uint64(5),
            "delta_high": bar_dec.read_uint64(6),
            "delta_low_timestamp": bar_dec.read_uint64(7),
            "datetime_ms": bar_dec.read_int64(8),
            "delta_high_timestamp": bar_dec.read_uint64(9),
        }
        bars.append(bar)
    return bars


def decode_auth_res(payload: bytes) -> dict:
    dec = Decoder(payload)
    return {
        "ctid_trader_account_id": dec.read_int64(1),
        "access_token": dec.read_string(3),
    }
