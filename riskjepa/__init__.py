"""
riskjepa — RiskJEPA data + model + evaluation pathway for hourly EUR/USD.

Thesis: EUR/USD is symmetric/mean-reverting with ~2-day regime persistence, so
direction is near-random. The edge is *risk-reward*, not sign accuracy: trade
only high-magnitude/high-confidence bars, size by volatility, and FLAT when
uncertain (triple-barrier kill-switch).

Modules:
  features.py  — regime-aware data path: CTX=48/TGT=12, mean-reversion features,
                 vol-normalized forward-return label, triple-barrier sign, embargo.
  model.py     — RiskJEPA: conv-tokenizer encoder + ALiBi predictor + RevIN +
                 ret_head/unc_head/tb_head + SIGReg collapse guard + JEPA SSL aux.
  metrics.py   — cost-aware risk-reward backtest (profit factor, Sharpe, winrate,
                 %-flat) with a triple-barrier OR uncertainty kill-switch.
  walkforward.py — purged walk-forward (expanding-window) K-fold splits + per-fold
                 cost-aware backtest + aggregate mean±std.
  train.py     — train RiskJEPA per fold; SELECT ON OOS PROFIT-FACTOR / SHARPE
                 (not JEPA loss). CPU-smoke + GPU modes.
  probe.py     — CPU feature baseline + frozen RiskJEPA walk-forward backtest.

This package does NOT modify forex_features.py; it imports the original loaders.
"""
from . import features
from . import model
from . import metrics
from . import walkforward
from . import train_riskjepa
from . import probe

__all__ = ["features", "model", "metrics", "walkforward", "train_riskjepa", "probe"]
