"""
WebSocket client for cTrader Open API v2.
Handles connection, authentication, and trendbar requests.
"""

import asyncio
import json
import os
import struct
import time
from asyncio import Queue
from typing import Optional

import websockets
from websockets.asyncio.client import ClientConnection

from ctrader.messages import (
    PROTO_MSGID_ACCOUNT_AUTH_RES,
    PROTO_MSGID_APPLICATION_AUTH_RES,
    PROTO_MSGID_TRENDBAR_RES,
    build_account_auth,
    build_app_auth,
    build_trendbar_req,
    decode_auth_res,
    decode_trendbar_res,
    parse_frame,
)


class CTraderClient:
    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        access_token: str = "",
        account_id: int = 0,
        host: str = "demo.ctraderapi.com",
        port: int = 5035,
    ):
        self._client_id = client_id or os.environ.get("CTRADER_CLIENT_ID", "")
        self._client_secret = client_secret or os.environ.get("CTRADER_CLIENT_SECRET", "")
        self._access_token = access_token or os.environ.get("CTRADER_ACCESS_TOKEN", "")
        self._account_id = account_id or int(os.environ.get("CTRADER_ACCOUNT_ID", "0"))
        self._host = host or os.environ.get("CTRADER_HOST", "demo.ctraderapi.com")
        self._port = port or int(os.environ.get("CTRADER_PORT", "5035"))
        self._ws: Optional[ClientConnection] = None
        self._responses: Queue[tuple[int, bytes]] = Queue()
        self._listener_task: Optional[asyncio.Task] = None

    @property
    def account_id(self) -> int:
        return self._account_id

    async def connect(self):
        url = f"wss://{self._host}:{self._port}"
        self._ws = await websockets.connect(url, ping_interval=20, ping_timeout=10)
        self._listener_task = asyncio.create_task(self._listen())

        await self._ws.send(build_app_auth(self._client_id, self._client_secret))
        app_res = await self._wait_for(PROTO_MSGID_APPLICATION_AUTH_RES)

        await self._ws.send(build_account_auth(self._access_token))
        acct_res = await self._wait_for(PROTO_MSGID_ACCOUNT_AUTH_RES)
        acct_data = decode_auth_res(acct_res)
        self._account_id = acct_data["ctid_trader_account_id"]
        return acct_data

    async def _listen(self):
        while self._ws and not self._ws.close_code:
            try:
                data = await self._ws.recv()
            except websockets.ConnectionClosed:
                break
            if isinstance(data, str):
                continue
            try:
                pt, payload = parse_frame(data)
            except Exception:
                continue
            await self._responses.put((pt, payload))

    async def _wait_for(self, payload_type: int, timeout: float = 10.0) -> bytes:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                pt, payload = await asyncio.wait_for(self._responses.get(), timeout=remaining)
                if pt == payload_type:
                    return payload
                self._responses.task_done()
            except asyncio.TimeoutError:
                break
        raise TimeoutError(f"Timed out waiting for payload type {payload_type}")

    async def fetch_trendbars(
        self,
        symbol_id: int,
        period: int,
        from_ts: int,
        to_ts: int,
        timeout: float = 30.0,
    ) -> list[dict]:
        req = build_trendbar_req(self._account_id, symbol_id, period, from_ts, to_ts)
        await self._ws.send(req)
        payload = await self._wait_for(PROTO_MSGID_TRENDBAR_RES, timeout)
        return decode_trendbar_res(payload)

    async def close(self):
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
