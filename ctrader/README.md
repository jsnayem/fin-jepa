# cTrader Open API Data Collection System

Collects OHLC candle data from cTrader Open API v2 and stores it in SQLite with append-only semantics.

## Architecture

```
run_collector.py          CLI entry point
    └── collector.py      Orchestrator: checks existing data, fetches only new bars
            ├── client.py     Async WebSocket client (auth + trendbar requests)
            │       └── messages.py    cTrader message encoding/decoding
            │               └── proto_codec.py   Minimal protobuf wire-format codec
            └── store.py      SQLite storage with dedup via UNIQUE constraint
```

## Key Behaviors

- **Append-only**: `INSERT OR IGNORE` on `(symbol, period, timestamp_ms)` — re-running against the same date range will never duplicate or overwrite existing data.
- **Resume-aware**: Automatically detects the last stored timestamp per symbol/period and fetches only newer bars.
- **Full backfill**: When no data exists for a symbol/period, starts from epoch 0 for a complete historical pull.
- **Zero external dependencies**: No protobuf compiler needed — a custom minimal codec handles the cTrader wire format.
- **Collection logging**: Every fetch is recorded in `collection_log` with status, bar count, timestamps, and errors.

## Prerequisites

- cTrader application credentials (get them from [ctrader.com](https://ctrader.com)):
  - `Client ID`
  - `Client Secret`
  - `Access Token`

## Usage

### Set credentials

```bash
export CTRADER_CLIENT_ID="your-client-id"
export CTRADER_CLIENT_SECRET="your-client-secret"
export CTRADER_ACCESS_TOKEN="your-access-token"
```

Optional overrides for non-demo environments:
```bash
export CTRADER_HOST="live.ctraderapi.com"
export CTRADER_PORT="5035"
export CTRADER_ACCOUNT_ID="12345"
```

### Collect data

```bash
# Single period
python -m ctrader.run_collector --symbol EURUSD --period D1

# Multiple periods
python -m ctrader.run_collector --symbol EURUSD --periods D1 H4 H1

# With date range (ISO format)
python -m ctrader.run_collector --symbol EURUSD --period D1 --from 2024-01-01 --to 2025-01-01

# Custom database path
python -m ctrader.run_collector --symbol EURUSD --period D1 --db data/my_bars.db
```

### Check stored data

```bash
python -m ctrader.run_collector --symbol EURUSD --stats
```

### List available symbols and periods

```bash
python -m ctrader.run_collector --list
```

### Cron-based incremental collection

Omit `--from` and `--to` to automatically resume from the last stored bar:

```bash
# Runs every hour, only fetches new bars since last run
0 * * * * cd /path/to/fin-jepa && CTRADER_CLIENT_ID=... python -m ctrader.run_collector --symbol EURUSD --periods D1 H4 H1
```

## Database Schema

### `bars` table

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-increment primary key |
| `symbol` | TEXT | Forex pair (e.g. EURUSD) |
| `period` | TEXT | Timeframe (D1, H4, H1, etc.) |
| `timestamp_ms` | INTEGER | Unix timestamp in milliseconds |
| `open` | REAL | Opening price |
| `high` | REAL | Highest price |
| `low` | REAL | Lowest price |
| `close` | REAL | Closing price |
| `volume` | INTEGER | Tick volume |
| `collected_at` | TEXT | ISO timestamp of when the bar was collected |

**Unique constraint**: `(symbol, period, timestamp_ms)` — prevents duplicate bars on re-run.

### `collection_log` table

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-increment primary key |
| `symbol` | TEXT | Forex pair |
| `period` | TEXT | Timeframe |
| `start_ts` | INTEGER | Requested start timestamp (ms) |
| `end_ts` | INTEGER | Requested end timestamp (ms) |
| `num_bars` | INTEGER | Number of bars inserted |
| `status` | TEXT | `complete` or `error` |
| `error` | TEXT | Error message if status is `error` |
| `started_at` | TEXT | Collection start time (ISO) |
| `finished_at` | TEXT | Collection end time (ISO) |

## Price Reconstruction

cTrader sends OHLC as delta-encoded values relative to the bar's low price:

```
open  = (low + delta_open)  / 100_000
high  = (low + delta_high)  / 100_000
low   =  low                / 100_000
close = (low + delta_close) / 100_000
```

This is handled automatically by `store.insert_bars()`.

## Available Symbols

EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, NZDUSD, USDCAD

## Available Periods

| Code | Duration |
|------|----------|
| M1   | 1 min    |
| M2   | 2 min    |
| M3   | 3 min    |
| M4   | 4 min    |
| M5   | 5 min    |
| M10  | 10 min   |
| M15  | 15 min   |
| M30  | 30 min   |
| H1   | 1 hour   |
| H2   | 2 hours  |
| H3   | 3 hours  |
| H4   | 4 hours  |
| H6   | 6 hours  |
| H8   | 8 hours  |
| D1   | 1 day    |
| W1   | 1 week   |
| MN1  | 1 month  |
