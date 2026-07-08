"""
Collector orchestrator: connects to cTrader, checks existing data, fetches only new bars.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ctrader.client import CTraderClient
from ctrader.messages import FOREX_SYMBOLS, TRENDBAR_PERIODS
from ctrader.store import BarStore

logger = logging.getLogger(__name__)


def _ts_ms_from_iso(date_str: str) -> int:
    dt = datetime.fromisoformat(date_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


class Collector:
    def __init__(self, db_path: str | Path):
        self._store = BarStore(db_path)
        self._client: Optional[CTraderClient] = None

    async def _ensure_client(self) -> CTraderClient:
        if self._client is None:
            self._client = CTraderClient()
            await self._client.connect()
            logger.info("Connected and authenticated to cTrader")
        return self._client

    async def collect(
        self,
        symbol: str,
        period: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> int:
        """
        Collect OHLC bars for a symbol/period, only appending new bars not in the DB.

        If from_date is None, starts from the last bar in the DB for this symbol/period.
        If from_date is specified (ISO format), uses it as the start.
        If the DB already has data and no from_date is given, only fetches new bars.
        """
        symbol_id = FOREX_SYMBOLS.get(symbol.upper())
        if symbol_id is None:
            raise ValueError(f"Unknown symbol: {symbol}. Available: {list(FOREX_SYMBOLS)}")

        period_val = TRENDBAR_PERIODS.get(period.upper())
        if period_val is None:
            raise ValueError(f"Unknown period: {period}. Available: {list(TRENDBAR_PERIODS)}")

        end_ts = _ts_ms_from_iso(to_date) if to_date else _now_ms()

        last_ts = self._store.get_last_timestamp(symbol, period)
        if from_date:
            start_ts = _ts_ms_from_iso(from_date)
        elif last_ts is not None:
            start_ts = last_ts + 1
            logger.info(
                "Resuming from last bar at %s (%d bars already stored)",
                datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).isoformat(),
                self._store.count_bars(symbol, period),
            )
        else:
            start_ts = 0
            logger.info("No existing data for %s/%s. Full backfill.", symbol, period)

        if start_ts >= end_ts:
            logger.info("Already up to date for %s/%s", symbol, period)
            return 0

        client = await self._ensure_client()
        logger.info(
            "Fetching %s/%s from %s to %s",
            symbol, period,
            datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc).isoformat(),
            datetime.fromtimestamp(end_ts / 1000, tz=timezone.utc).isoformat(),
        )

        try:
            bars = await client.fetch_trendbars(symbol_id, period_val, start_ts, end_ts)
        except Exception as e:
            self._store.log_collection(
                symbol, period.upper(), start_ts, end_ts, 0, "error", str(e),
            )
            logger.error("Failed to fetch %s/%s: %s", symbol, period, e)
            raise

        if not bars:
            logger.info("No new bars returned for %s/%s", symbol, period)
            self._store.log_collection(
                symbol, period.upper(), start_ts, end_ts, 0, "complete",
            )
            return 0

        inserted = self._store.insert_bars(symbol, period.upper(), bars)
        self._store.log_collection(
            symbol, period.upper(), start_ts, end_ts, inserted, "complete",
        )
        logger.info(
            "Inserted %d new bars for %s/%s (total: %d)",
            inserted, symbol, period,
            self._store.count_bars(symbol, period),
        )
        return inserted

    async def collect_multiple(
        self,
        symbol: str,
        periods: list[str],
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> dict[str, int]:
        results = {}
        for period in periods:
            n = await self.collect(symbol, period, from_date, to_date)
            results[period] = n
        return results

    async def close(self):
        if self._client:
            await self._client.close()
            self._client = None
