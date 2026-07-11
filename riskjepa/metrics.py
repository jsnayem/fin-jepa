"""
riskjepa/metrics.py — Cost-aware risk-reward backtest metrics.

Given a predicted vol-normalized return ŷ_t, an uncertainty (or |ŷ|) estimate
σ̂_t, and the realized vol σ_t, apply the sizing rule and charge realistic FX
costs. Reports the profit-factor / Sharpe / winrate / %-flat that define the
RiskJEPA thesis (trade only high-conviction bars, size by vol, FLAT when unsure).

Sizing rule (from research/new_model_design.md):
    pos_t = 0                                  if |ŷ_t| < c1 * σ̂_t   (flat / kill-switch)
    pos_t = sign(ŷ_t) * tanh(|ŷ_t| / σ̂_t)      otherwise, vol-scaled, capped at 1.0

Here we use |ŷ_t| itself as the confidence proxy (the triple-barrier 0 label
already forces flat in sideways regimes), with c1 controlling how selective we
are. The realized return per bar is de-vol'd by σ_t to keep magnitudes
comparable to the (vol-normalized) prediction.
"""
import numpy as np


def _rankic(pred, y):
    """Spearman rank-IC between prediction and realized (vol-normalized) target."""
    def rank(x):
        order = np.argsort(x)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(1, len(x) + 1)
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        for u in np.where(counts > 1)[0]:
            r[inv == u] = r[inv == u].mean()
        return r
    p = rank(pred); q = rank(y)
    p = p - p.mean(); q = q - q.mean()
    d = np.sqrt((p * p).sum()) * np.sqrt((q * q).sum())
    return float((p * q).sum() / d) if d > 0 else float('nan')


def backtest(y_pred, y_true, tb=None, c1=0.5, spread_bars=0.10,
             vol=None, per_bar=True):
    """Run the cost-aware risk-reward backtest.

    Args:
        y_pred: (N,) model-predicted vol-normalized forward return (z-scored ok,
                sign + magnitude matter).
        y_true: (N,) realized vol-normalized forward return (aligned with y_pred).
        tb:     (N,) optional triple-barrier sign (+1/-1/0); if given, bars with
                tb==0 are forced FLAT regardless of c1 (data-driven kill-switch).
        c1:     conviction threshold as a multiple of std(|y_pred|); below this
                the position is FLAT.
        spread_bars: round-turn cost in 'vol-normalized return' units. Both y_pred
                and y_true are already divided by σ_t (vol60), whose hourly value on
                EUR/USD ≈ 0.18% = 0.0018. One pip = 0.0001 = 0.0001/0.0018 ≈ 0.055
                vol-units; round-turn (enter+exit, ~1-2 pip spread) ≈ 0.06-0.11.
                Default 0.10 (≈1.8-pip round-turn) — realistic for EUR/USD.
        vol:    (N,) optional realized vol σ_t — currently informational; both
                sides already vol-normalized so costs are in the same units.
        per_bar: unused placeholder for API symmetry (P&L is per-bar here).

    Returns dict of metrics:
        n, n_trade, pct_flat, winrate, profit_factor, sharpe, mean_ret, tot_ret,
        rankIC, mean_pos, used_triple_barrier.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if tb is not None:
        tb = np.asarray(tb, dtype=float)

    n = len(y_pred)
    thr = c1 * np.std(np.abs(y_pred)) + 1e-9

    # position sizing (kill-switch applied)
    pos = np.zeros(n, dtype=float)
    active = np.abs(y_pred) >= thr
    if tb is not None:
        active &= (tb != 0.0)               # triple-barrier 0 => always flat
    pos[active] = np.sign(y_pred[active]) * np.tanh(np.abs(y_pred[active]) / thr)

    # realized P&L per bar (vol-normalized): position * forward return
    gross = pos * y_true
    cost = np.abs(pos) * spread_bars        # pay 2x spread implicitly via |pos|
    pnl = gross - cost

    n_trade = int(np.sum(pos != 0.0))
    pct_flat = float(np.mean(pos == 0.0))

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    winrate = float(len(wins) / n_trade) if n_trade > 0 else float('nan')
    gross_win = wins.sum()
    gross_loss = -losses.sum()
    profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else float('inf')

    mean_ret = float(pnl.mean())
    tot_ret = float(pnl.sum())
    # Sharpe (per-bar), annualized-ish by sqrt of bars-per-year (~ 24*365/1).
    sd = pnl.std()
    sharpe = float(mean_ret / sd * np.sqrt(len(pnl))) if sd > 0 else float('nan')

    rank_ic = _rankic(y_pred, y_true)

    used_tb = tb is not None
    return {
        'n': n,
        'n_trade': n_trade,
        'pct_flat': pct_flat,
        'winrate': winrate,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
        'mean_ret': mean_ret,
        'tot_ret': tot_ret,
        'rankIC': rank_ic,
        'mean_pos': float(np.mean(np.abs(pos))),
        'used_triple_barrier': used_tb,
        'c1': c1,
        'spread_bars': spread_bars,
    }


def sweep_c1(y_pred, y_true, tb=None, c1_grid=None, spread_bars=1.0):
    """Sweep the conviction threshold and return a table of metrics.

    Returns a list of dicts (one per c1). Useful to calibrate how selective the
    kill-switch needs to be for EUR/USD's thin edge.
    """
    if c1_grid is None:
        c1_grid = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
    rows = []
    for c1 in c1_grid:
        m = backtest(y_pred, y_true, tb=tb, c1=c1, spread_bars=spread_bars)
        m = {k: m[k] for k in ('c1', 'n_trade', 'pct_flat', 'winrate',
                                'profit_factor', 'sharpe', 'rankIC', 'mean_pos')}
        rows.append(m)
    return rows


if __name__ == "__main__":
    # smoke test with synthetic data
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal(5000)
    # a weak signal: 5% rank-IC, plus a flat regime
    y_pred = 0.05 * y_true + rng.standard_normal(5000) * 0.999
    tb = np.where(np.abs(y_true) > 0.6, np.sign(y_true), 0.0)
    m = backtest(y_pred, y_true, tb=tb)
    print("smoke backtest:", {k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in m.items()})
    print("sweep:")
    for r in sweep_c1(y_pred, y_true, tb=tb):
        print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
