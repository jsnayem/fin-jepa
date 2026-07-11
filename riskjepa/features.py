"""
riskjepa/features.py — Regime-aware (risk-reward) data path for hourly EUR/USD.

Concept: EUR/USD is symmetric (~52/48), mean-reverting, low-vol, with ~2-day
regime persistence (Rivin regime study). Direction is near-random, so the edge
is NOT sign accuracy — it is *risk-reward*: trade only high-magnitude /
high-confidence bars, size by volatility, and FLAT when uncertain.

This module builds the data pipeline that supports that thesis:
  - Short context  CTX=48 (~2 days) / TGT=12 (matches 2-day median persistence).
  - Mean-reversion feature block on top of the base + alpha columns.
  - Label = vol-normalized forward return  r / vol60  (magnitude + direction).
  - Triple-barrier sign: +1 / -1 / 0(sideways => FLAT, the data-driven kill-switch).
  - A TGT-bar EMBARGO between train and val windows (kills train/val leakage).

It does NOT modify forex_features.py — it imports the original loaders and base
feature builders and only changes the LABEL / CONTEXT / FEATURES / SPLIT.
"""
import numpy as np
import pandas as pd

import forex_features as ff   # original module — reused, not edited
from alphas import ALPHA_COLS as ALPHA_COLS_REF


# ── forked (regime-aware) hyperparameters ────────────────────────────────────
CTX = 48       # ~2 trading days (matches Rivin's 2-day median persistence)
TGT = 12       # 12h horizon
GAP_H = 120    # same gap rule as original (weekly closures are not split)
EMBARGO = TGT  # drop val windows within TGT bars of the last train window


# ── mean-reversion feature block (regime-aware) ──────────────────────────────
def add_mr_features(df):
    """Short-lookback mean-reversion features (the paper's prescription for FX).

    These directly encode 'technical mean-reversion, short lookback' on top of
    the existing base/alpha columns:
      - distance-from-VWAP      (close - vwap)/vwap
      - short RSI(14)           (Wilder-style, 0..1)
      - intraday-range reversion (close - mid)/range, mid=(high+low)/2
    """
    df = df.copy()
    vwap = df['vwap'] if 'vwap' in df else ff.add_vwap_adv(df)['vwap']
    c = df['Close']
    df['mr_vwap_dist'] = (c - vwap) / (vwap + 1e-9)

    # Wilder RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['mr_rsi14'] = 1.0 - 1.0 / (1.0 + rs)        # mapped to 0..1 (mean-reversion friendly)

    mid = (df['High'] + df['Low']) / 2.0
    rng = (df['High'] - df['Low']) + 1e-9
    df['mr_range_rev'] = (c - mid) / rng

    return df[['mr_vwap_dist', 'mr_rsi14', 'mr_range_rev']]


MR_COLS = ['mr_vwap_dist', 'mr_rsi14', 'mr_range_rev']


def build_label_and_features(df, tgt=TGT):
    """Return (features_df, vol_norm_return_series, triple_barrier_series).

    label = sum(logret[i:i+tgt]) / vol60   (vol-normalized forward return, z-scored)
    triple-barrier sign: +1 if fwd > +k*vol60, -1 if fwd < -k*vol60, else 0 (sideways/FLAT).
    """
    df = ff.add_vwap_adv(df)
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
    base = ff.build_base_features(df)
    alpha = ff.compute_alpha_features(df)
    mr = add_mr_features(df)

    feats = pd.concat([
        base[ff.BASE_COLS],
        alpha[ALPHA_COLS_REF],
        mr[MR_COLS],
    ], axis=1)

    # forward log-return over the TGT horizon
    fwd = df['returns'].rolling(tgt).sum().shift(-tgt)

    # realized volatility denominator (trailing 60 bars)
    vol60 = df['returns'].rolling(60).std() + 1e-8

    # vol-normalized forward return (the magnitude + direction signal)
    vret = (fwd / vol60).astype(float)

    # triple-barrier: up / down / sideways(0 = FLAT)
    k = 0.5
    barrier = k * vol60
    tb = np.where(fwd > barrier, 1.0,
                  np.where(fwd < -barrier, -1.0, 0.0)).astype(float)

    return feats, vret, pd.Series(tb, index=df.index)


def make_dataset(df, ctx=CTX, tgt=TGT, train_frac=0.9, embargo=EMBARGO):
    """Build the risk-reward dataset: (ctx, F) -> (tgt, F) with labels y, y_tb.

    Splits by time, then drops val windows within `embargo` bars of the train
    cutoff to prevent look-ahead leakage across the train/val boundary.
    """
    feats, vret, tb = build_label_and_features(df, tgt)
    feats = feats.astype(float)
    vret = vret.astype(float)
    tb = tb.astype(float)

    arr = feats.to_numpy()
    n = arr.shape[0]

    # runs (no Friday->Monday stitching)
    gap = df['Time'].diff().dt.total_seconds() > GAP_H * 3600
    run = gap.cumsum().to_numpy()

    # time-based train/val cutoff
    t0, t1 = df['Time'].min(), df['Time'].max()
    cutoff = t0 + (t1 - t0) * train_frac
    is_train_time = (df['Time'] < cutoff).to_numpy()

    # normalizer statistics from training rows (non-NaN) ONLY
    mask_tr = is_train_time & ~np.any(np.isnan(arr), axis=1)
    mean = np.nanmean(arr[mask_tr], axis=0)
    std = np.nanstd(arr[mask_tr], axis=0) + 1e-8
    arr_n = (arr - mean) / std

    # z-score the vol-normalized return label on train
    vmean = np.nanmean(vret[mask_tr])
    vstd = np.nanstd(vret[mask_tr]) + 1e-8
    vret_n = (vret - vmean) / vstd

    # valid window start indices: full window inside one run, fully observed,
    # label present (and triple-barrier defined).
    valid = []
    for i in range(ctx, n - tgt + 1):
        if not np.any(np.isnan(arr_n[i - ctx:i + tgt])):
            if np.all(run[i - ctx:i + tgt] == run[i - ctx]):
                li = i + tgt - 1
                if not np.isnan(vret_n[li]) and tb.iloc[li] == tb.iloc[li]:  # not NaN
                    valid.append(i)
    valid = np.array(valid, dtype=int)

    # split by time
    split = np.array(['train' if is_train_time[i] else 'val' for i in valid])

    # embargo: drop val windows whose start is within `embargo` bars after the
    # last train window (otherwise train context leaks into val).
    last_train = valid[split == 'train'].max() if (split == 'train').any() else -10**9
    is_val = split == 'val'
    inside_embargo = is_val & (valid <= last_train + embargo)
    split = np.where(inside_embargo, '__drop__', split)
    keep = split != '__drop__'
    valid = valid[keep]
    split = split[keep]

    return RiskJEPADataset(arr_n, vret_n, tb.to_numpy(), valid, split, ctx, tgt), {
        'n': n,
        'n_features': arr.shape[1],
        'n_train': int((split == 'train').sum()),
        'n_val': int((split == 'val').sum()),
        'cutoff': str(cutoff),
        'feature_cols': list(feats.columns),
        'mr_cols': MR_COLS,
    }


class RiskJEPADataset:
    """(ctx, F) -> (tgt, F); label y = vol-normalized fwd return at i+tgt-1;
    y_tb = triple-barrier sign (+1/-1/0) at i+tgt-1."""

    def __init__(self, feat, vret, tb, starts, split, ctx, tgt):
        self.feat = feat      # (n, F) normalized
        self.vret = vret      # (n,) vol-normalized forward return (z-scored)
        self.tb = tb          # (n,) triple-barrier sign (+1/-1/0)
        self.starts = starts  # window start indices (ctx begins at start-ctx)
        self.split = split
        self.ctx = ctx
        self.tgt = tgt

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        i = self.starts[idx]
        ctx = self.feat[i - self.ctx:i]
        tgt = self.feat[i:i + self.tgt]
        y = float(self.vret[i + self.tgt - 1])
        y_tb = float(self.tb[i + self.tgt - 1])
        return {'ctx': np.ascontiguousarray(ctx, dtype=np.float32),
                'tgt': np.ascontiguousarray(tgt, dtype=np.float32),
                'y': y, 'y_tb': y_tb, 'start': i}

    # ── lightweight loader used by the probe (mirrors ForexH1Dataset.probe_pairs) ──
    @staticmethod
    def collate(batch):
        out = {k: np.stack([b[k] for b in batch]) for k in ('ctx', 'tgt')}
        for k in ('y', 'y_tb', 'start'):
            out[k] = np.array([b[k] for b in batch], dtype=np.float32)
        return out


if __name__ == "__main__":
    import os
    data = os.environ.get("RJ_DATA", "data/EURUSD_H1.csv")
    df = ff.load_eurusd_h1(data)
    ds, info = make_dataset(df)
    print("dataset info:", info)
    print("sample batch shapes:",
          {k: (v.shape if hasattr(v, 'shape') else v)
           for k, v in ds[0].items()})
