"""
riskjepa — RiskJEPA data + evaluation pathway for hourly EUR/USD.

Thesis: EUR/USD is symmetric/mean-reverting with ~2-day regime persistence, so
direction is near-random. The edge is *risk-reward*, not sign accuracy: trade
only high-magnitude/high-confidence bars, size by volatility, and FLAT when
uncertain (triple-barrier kill-switch).

Modules:
  features.py  — regime-aware data path: CTX=48/TGT=12, mean-reversion features,
                 vol-normalized forward-return label, triple-barrier sign, embargo.
  metrics.py   — cost-aware risk-reward backtest (profit factor, Sharpe, winrate, %-flat).
  probe.py     — baseline (CPU feature probe) + frozen-encoder probe + backtest sweep.

This package does NOT modify forex_features.py; it imports the original loaders.
"""
from . import features
from . import metrics
from . import probe

__all__ = ["features", "metrics", "probe"]
