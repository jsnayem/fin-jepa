"""
alphas.py — Operator toolkit + curated formulaic alpha features for a single instrument.

Adapted from Kakushadze (2015), "101 Formulaic Alphas", for a single forex pair (hourly
bars). The cross-sectional operator rank() is replaced by a rolling time-series rank
(ts_rank) over RANK_W bars, since a single instrument has no cross-section. VWAP is a
20-bar volume-weighted average price (uses the volume column). Alphas requiring market
cap or industry classification (IndClass) are excluded.

All operators are vectorized with numpy sliding windows and return pandas Series aligned
to the input index (NaN during warmup).
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=RuntimeWarning)

RANK_W = 120  # rolling window (bars) used to substitute cross-sectional rank/scale


def _safe_argmax(sw):
    out = np.full(sw.shape[0], np.nan)
    valid = ~np.all(np.isnan(sw), axis=1)
    sw2 = np.where(np.isnan(sw), -np.inf, sw)
    out[valid] = np.argmax(sw2[valid], axis=1).astype(float)
    return out


def _safe_argmin(sw):
    out = np.full(sw.shape[0], np.nan)
    valid = ~np.all(np.isnan(sw), axis=1)
    sw2 = np.where(np.isnan(sw), np.inf, sw)
    out[valid] = np.argmin(sw2[valid], axis=1).astype(float)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sliding-window helper
# ─────────────────────────────────────────────────────────────────────────────
def _slid(arr, w):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    w = int(np.floor(w))
    if n < w or w < 1:
        return np.full((n, max(w, 1)), np.nan)
    sw = np.lib.stride_tricks.sliding_window_view(arr, w)  # (n-w+1, w)
    out = np.full((n, w), np.nan)
    out[w - 1:] = sw
    return out


def _series(x):
    return x if isinstance(x, pd.Series) else pd.Series(x)


# ─────────────────────────────────────────────────────────────────────────────
# Time-series operators (vectorized)
# ─────────────────────────────────────────────────────────────────────────────
def ts_min(x, d):
    x = _series(x); sw = _slid(x.to_numpy(float), d)
    return pd.Series(np.nanmin(sw, axis=1), index=x.index)


def ts_max(x, d):
    x = _series(x); sw = _slid(x.to_numpy(float), d)
    return pd.Series(np.nanmax(sw, axis=1), index=x.index)


def ts_argmin(x, d):
    x = _series(x); sw = _slid(x.to_numpy(float), d)
    pos = _safe_argmin(sw)
    pos[~np.all(~np.isnan(sw), axis=1)] = np.nan
    return pd.Series(pos, index=x.index)


def ts_argmax(x, d):
    x = _series(x); sw = _slid(x.to_numpy(float), d)
    pos = _safe_argmax(sw)
    pos[~np.all(~np.isnan(sw), axis=1)] = np.nan
    return pd.Series(pos, index=x.index)


def ts_mean(x, d):
    x = _series(x); sw = _slid(x.to_numpy(float), d)
    return pd.Series(np.nanmean(sw, axis=1), index=x.index)


def ts_sum(x, d):
    x = _series(x); sw = _slid(x.to_numpy(float), d)
    return pd.Series(np.nansum(sw, axis=1), index=x.index)


def ts_std(x, d):
    x = _series(x); sw = _slid(x.to_numpy(float), d)
    return pd.Series(np.nanstd(sw, axis=1), index=x.index)


def ts_product(x, d):
    x = _series(x); sw = _slid(x.to_numpy(float), d)
    return pd.Series(np.nanprod(sw, axis=1), index=x.index)


def ts_rank(x, d):
    """Time-series rank (1..d) of the last value within the window."""
    x = _series(x); arr = x.to_numpy(float)
    w = max(int(np.floor(d)), 2)
    sw = _slid(arr, w)
    last = sw[:, -1]
    r = (sw <= last[:, None]).sum(axis=1).astype(float)
    valid = ~np.all(np.isnan(sw), axis=1)
    r[~valid] = np.nan
    return pd.Series(r, index=x.index)


def ts_corr(a, b, d):
    a = _series(a).to_numpy(float); b = _series(b).to_numpy(float)
    swa = _slid(a, d); swb = _slid(b, d)
    ma = np.nanmean(swa, axis=1); mb = np.nanmean(swb, axis=1)
    da = swa - ma[:, None]; db = swb - mb[:, None]
    cov = np.nanmean(da * db, axis=1)
    sa = np.nanstd(swa, axis=1); sb = np.nanstd(swb, axis=1)
    denom = sa * sb
    corr = np.where(denom > 1e-12, cov / np.where(denom > 1e-12, denom, 1.0), np.nan)
    return pd.Series(corr, index=_series(a).index)


def ts_cov(a, b, d):
    a = _series(a).to_numpy(float); b = _series(b).to_numpy(float)
    swa = _slid(a, d); swb = _slid(b, d)
    ma = np.nanmean(swa, axis=1); mb = np.nanmean(swb, axis=1)
    da = swa - ma[:, None]; db = swb - mb[:, None]
    cov = np.nanmean(da * db, axis=1)
    return pd.Series(cov, index=_series(a).index)


def decay_linear(x, d):
    x = _series(x); w = max(int(np.floor(d)), 1)
    wts = np.arange(1, w + 1)[::-1].astype(float); wts /= wts.sum()
    sw = _slid(x.to_numpy(float), w)
    return pd.Series((sw * wts).sum(axis=1), index=x.index)


def scale(x, a=1.0):
    """Single-series scale: x / rolling-sum(|x|) * a (cross-sectional scale substitute)."""
    x = _series(x)
    denom = x.abs().rolling(RANK_W).sum()
    return x / denom * a


def rank(x):
    """Cross-sectional rank substituted by rolling ts_rank over RANK_W."""
    return ts_rank(x, RANK_W)


def delay(x, d):
    return _series(x).shift(int(d))


def delta(x, d):
    return _series(x) - _series(x).shift(int(d))


def signedpower(x, a):
    x = _series(x).to_numpy(float)
    return pd.Series(np.sign(x) * np.abs(x) ** a, index=_series(x).index)


def log(x):
    return np.log(_series(x).astype(float).clip(lower=1e-12))


def sqrt(x):
    return np.sqrt(_series(x).astype(float).clip(lower=0.0))


def indneutralize(x, g=None):
    raise NotImplementedError("IndClass neutralization not applicable to a single forex pair")


# ─────────────────────────────────────────────────────────────────────────────
# Curated alpha feature set (diverse, no cap / IndClass)
# Families: mean-reversion (MR), momentum (MOM), volume-price (VP), volatility (VOL)
# ─────────────────────────────────────────────────────────────────────────────
def _build_alpha_features(df):
    """df must contain: open, high, low, close, volume, vwap, returns, adv20, adv60
    (column names are matched case-insensitively)."""
    g = {str(c).lower(): df[c] for c in df.columns}
    o, h, l, c, v = g['open'], g['high'], g['low'], g['close'], g['volume']
    vwap, ret, adv20, adv60 = g['vwap'], g['returns'], g['adv20'], g['adv60']
    out = pd.DataFrame(index=df.index)

    # ── Mean-reversion ──
    out['a101'] = (c - o) / ((h - l) + 1e-3)                                   # simple MR intrabar
    out['a42'] = rank(vwap - c) / rank(vwap + c)                              # contrarian close vs vwap
    out['a12'] = -np.sign(delta(v, 1).to_numpy(float)) * delta(c, 1).to_numpy(float)
    out['a53'] = -delta(((c - l) - (h - c)) / (c - l + 1e-6), 9)
    out['a54'] = -((l - c) * (o ** 5)) / ((l - h) * (c ** 5) + 1e-12)
    out['a2'] = -ts_corr(rank(delta(log(v), 2)), rank((c - o) / o), 6)

    # ── Momentum ──
    out['a3'] = log(delay(c, 1) / delay(o, 1) + 1e-9)                         # prior-bar close/open
    out['a38'] = -rank(ts_rank(c, 10)) * rank(c / o)
    out['a34'] = (1 - rank(ts_std(ret, 2) / ts_std(ret, 5))) + (1 - rank(delta(c, 1)))

    # ── Volume–price ──
    out['a15'] = -ts_sum(rank(ts_corr(rank(h), rank(v), 3)), 3)
    out['a26'] = -ts_max(ts_corr(ts_rank(v, 5), ts_rank(h, 5), 5), 3)
    out['a43'] = ts_rank(v / adv20, 20) * ts_rank(-delta(c, 7), 8)
    out['a55'] = -ts_corr(
        rank((c - ts_min(l, 12)) / (ts_max(h, 12) - ts_min(l, 12) + 1e-9)),
        rank(v), 6)
    out['a17'] = -ts_rank(c, 10) * delta(delta(c, 1), 1) * ts_rank(v / adv20, 5)
    out['a35'] = ts_rank(v, 32) * (1 - ts_rank(c + h - l, 16)) * (1 - ts_rank(ret, 32))

    # ── Volatility ──
    cond = (ret < 0).to_numpy()
    base = np.where(cond, ts_std(ret, 20).to_numpy(float), c.to_numpy(float))
    out['a1'] = rank(ts_argmax(signedpower(pd.Series(base, index=df.index), 2), 5)) - 0.5
    out['a40'] = -rank(ts_std(h, 10)) * ts_corr(h, v, 10)

    return out


ALPHA_COLS = ['a101', 'a42', 'a12', 'a53', 'a54', 'a2',
              'a3', 'a38', 'a34',
              'a15', 'a26', 'a43', 'a55', 'a17', 'a35',
              'a1', 'a40']


def compute_alpha_features(df):
    """Return DataFrame of curated alpha channels (NaN during warmup)."""
    return _build_alpha_features(df)[ALPHA_COLS]


def compute_mega_alpha(df, alpha_df):
    """Composite 'mega-alpha' target: standardized avg of MR + momentum + VP + VOL signals."""
    parts = [alpha_df['a101'], alpha_df['a3'], alpha_df['a43'], alpha_df['a40']]
    std = [p.sub(p.mean()).div(p.std() + 1e-9) for p in parts]
    mega = sum(std) / len(std)
    return mega.clip(-3, 3)


if __name__ == "__main__":
    from forex_features import load_eurusd_h1, add_vwap_adv
    df = load_eurusd_h1("data/EURUSD_H1.csv")
    df = add_vwap_adv(df)
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    af = compute_alpha_features(df)
    print("alpha features:", af.shape)
    print("non-nan rows:", int(af.notna().any(axis=1).sum()))
    print(af.describe().T[['mean', 'std', 'min', 'max']].round(3))
