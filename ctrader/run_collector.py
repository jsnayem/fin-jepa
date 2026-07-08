"""
Entry-point script for the cTrader data collection system.

Usage:
    python -m ctrader.run_collector \\
        --symbol EURUSD \\
        --period D1 \\
        [--from 2024-01-01] \\
        [--to 2025-01-01] \\
        [--periods D1 H4 H1] \\
        [--db data/ctrader_bars.db]

Environment variables:
    CTRADER_CLIENT_ID      - cTrader application client ID
    CTRADER_CLIENT_SECRET  - cTrader application client secret
    CTRADER_ACCESS_TOKEN   - cTrader account access token
    CTRADER_HOST           - API host (default: demo.ctraderapi.com)
    CTRADER_PORT           - API port (default: 5035)
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from ctrader.collector import Collector
from ctrader.messages import FOREX_SYMBOLS, TRENDBAR_PERIODS
from ctrader.store import BarStore


def parse_args():
    parser = argparse.ArgumentParser(description="cTrader Open API data collector")
    parser.add_argument("--symbol", default="EURUSD", help="Forex symbol")
    parser.add_argument("--period", default=None, help="Single period (e.g. D1, H4)")
    parser.add_argument("--periods", nargs="+", default=None, help="Multiple periods")
    parser.add_argument("--from", dest="from_date", default=None, help="Start date (ISO format)")
    parser.add_argument("--to", dest="to_date", default=None, help="End date (ISO format)")
    parser.add_argument("--db", default="data/ctrader_bars.db", help="SQLite database path")
    parser.add_argument("--list", action="store_true", help="List available symbols and periods")
    parser.add_argument("--stats", action="store_true", help="Show collection stats for symbol")
    return parser.parse_args()


def show_stats(db_path: str, symbol: str):
    store = BarStore(db_path)
    print(f"\nStats for {symbol}:")
    print(f"{'Period':<8} {'Bars':>10}")
    print("-" * 20)
    for period in TRENDBAR_PERIODS:
        count = store.count_bars(symbol, period)
        if count > 0:
            print(f"{period:<8} {count:>10}")


def show_list():
    print("\nAvailable symbols:")
    for sym in FOREX_SYMBOLS:
        print(f"  {sym}")
    print("\nAvailable periods:")
    for period, val in TRENDBAR_PERIODS.items():
        print(f"  {period:<6} ({val} min)")


async def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.list:
        show_list()
        return

    symbol = args.symbol.upper()

    if args.stats:
        show_stats(args.db, symbol)
        return

    periods = args.periods or ([args.period] if args.period else ["D1"])
    db_path = args.db

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    collector = Collector(db_path)
    try:
        results = await collector.collect_multiple(
            symbol, periods, args.from_date, args.to_date,
        )
        print(f"\nCollection results for {symbol}:")
        for period, count in results.items():
            print(f"  {period}: {count} new bars inserted")
    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(main())
