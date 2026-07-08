"""
SQLite storage for OHLC candle data with append-only semantics.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


CREATE_BARS_TABLE = """
CREATE TABLE IF NOT EXISTS bars (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    period TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    collected_at TEXT NOT NULL,
    UNIQUE(symbol, period, timestamp_ms)
)
"""

CREATE_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_bars_symbol_period_ts
ON bars(symbol, period, timestamp_ms)
"""

CREATE_COLLECTION_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS collection_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    period TEXT NOT NULL,
    start_ts INTEGER,
    end_ts INTEGER,
    num_bars INTEGER,
    status TEXT NOT NULL,
    error TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT
)
"""


class BarStore:
    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute(CREATE_BARS_TABLE)
            conn.execute(CREATE_INDEX)
            conn.execute(CREATE_COLLECTION_LOG_TABLE)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_last_timestamp(self, symbol: str, period: str) -> Optional[int]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(timestamp_ms) FROM bars WHERE symbol = ? AND period = ?",
                (symbol, period),
            ).fetchone()
            return row[0] if row and row[0] is not None else None

    def insert_bars(self, symbol: str, period: str, bars: list[dict]) -> int:
        """Insert bars with ON CONFLICT IGNORE. Returns number of new bars inserted."""
        now = datetime.now(timezone.utc).isoformat()
        row_dicts = []
        for bar in bars:
            ts_ms = bar["datetime_ms"]
            low = bar["low"]
            delta_open = bar["delta_open"]
            delta_close = bar["delta_close"]
            delta_high = bar["delta_high"]
            open_price = (low + delta_open) / 100000.0
            close_price = (low + delta_close) / 100000.0
            high_price = (low + delta_high) / 100000.0
            low_price = low / 100000.0
            volume = bar["volume"]

            row_dicts.append((
                symbol,
                period,
                ts_ms,
                open_price,
                high_price,
                low_price,
                close_price,
                volume,
                now,
            ))

        with self._conn() as conn:
            before = conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
            conn.executemany(
                """INSERT OR IGNORE INTO bars
                   (symbol, period, timestamp_ms, open, high, low, close, volume, collected_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                row_dicts,
            )
            after = conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
            return after - before

    def log_collection(
        self,
        symbol: str,
        period: str,
        start_ts: int,
        end_ts: int,
        num_bars: int,
        status: str,
        error: Optional[str] = None,
    ):
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO collection_log
                   (symbol, period, start_ts, end_ts, num_bars, status, error, started_at, finished_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (symbol, period, start_ts, end_ts, num_bars, status, error, now, now),
            )

    def count_bars(self, symbol: str, period: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM bars WHERE symbol = ? AND period = ?",
                (symbol, period),
            ).fetchone()
            return row[0]

    def load_bars(
        self, symbol: str, period: str, start_ts: Optional[int] = None, end_ts: Optional[int] = None
    ) -> list[dict]:
        where = ["symbol = ?", "period = ?"]
        params: list = [symbol, period]
        if start_ts is not None:
            where.append("timestamp_ms >= ?")
            params.append(start_ts)
        if end_ts is not None:
            where.append("timestamp_ms <= ?")
            params.append(end_ts)

        query = f"SELECT * FROM bars WHERE {' AND '.join(where)} ORDER BY timestamp_ms"
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
