"""
forex_features.py — Load EURUSD_H1, build base + alpha features, handle weekend gaps,
normalize on the training split, and produce sliding windows + a mega-alpha target.

Data: data/EURUSD_H1.csv (tab-separated: Time, Open, High, Low, Close, Volume, [spread]).
The unlabeled 7th column (spread) is dropped.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from alphas import compute_alpha_features, compute_mega_alpha

CTX = 120   # context length (hours ~ 5 days)
TGT = 24    # prediction horizon (hours ~ 1 day)
GAP_H = 120  # max allowed gap (hours) within a run. Weekly FX weekend closures
              # (~48-75h) are NOT split, so the 144-bar (ctx=120+tgt=24) window fits
              # across trading weeks; only genuine outages (>5d) split the series.


# ─────────────────────────────────────────────────────────────────────────────
# Load + preprocess
# ─────────────────────────────────────────────────────────────────────────────
def load_eurusd_h1(path):
    # The file has 7 tab-separated fields (DateTime, OHLC, Volume, Spread) but a
    # 6-name header, so read with explicit names and keep the datetime column.
    cols = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Spread']
    df = pd.read_csv(path, sep='\t', header=None, skiprows=1, names=cols)
    df['Time'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[c] = df[c].astype(float)
    df = df.sort_values('Time').reset_index(drop=True)
    return df


def add_vwap_adv(df):
    """VWAP proxy (20-bar volume-weighted typical price) + average dollar-volume proxies."""
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0
    vol = df['Volume'].clip(lower=0.0)
    adv20 = (df['Close'] * vol).rolling(20).mean()
    adv60 = (df['Close'] * vol).rolling(60).mean()
    vwap_num = (tp * vol).rolling(20).sum()
    vwap_den = vol.rolling(20).sum() + 1e-12
    df = df.copy()
    df['vwap'] = vwap_num / vwap_den
    df['adv20'] = adv20
    df['adv60'] = adv60
    return df


def build_base_features(df):
    o, h, l, c, v = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
    logret = np.log(c / c.shift(1))
    out = pd.DataFrame(index=df.index)
    out['logret'] = logret
    out['close_open'] = np.log(c / o)
    out['range_pct'] = (h - l) / (c + 1e-9)
    out['up_shadow'] = (h - c) / (c + 1e-9)
    out['low_shadow'] = (c - l) / (c + 1e-9)
    out['vol5'] = logret.rolling(5).std()
    out['vol20'] = logret.rolling(20).std()
    out['vol60'] = logret.rolling(60).std()
    out['logvol'] = np.log1p(v.clip(lower=0.0))
    out['volr20'] = v / (v.rolling(20).mean() + 1e-12)
    out['volr60'] = v / (v.rolling(60).mean() + 1e-12)
    out['ma5'] = c / c.rolling(5).mean()
    out['ma20'] = c / c.rolling(20).mean()
    out['ma60'] = c / c.rolling(60).mean()
    out['hl_ratio'] = h / l
    return out


BASE_COLS = ['logret', 'close_open', 'range_pct', 'up_shadow', 'low_shadow',
             'vol5', 'vol20', 'vol60', 'logvol', 'volr20', 'volr60',
             'ma5', 'ma20', 'ma60', 'hl_ratio']


def build_feature_matrix(df):
    """Return (features_df, mega_alpha_series) with NaN during warmup."""
    df = add_vwap_adv(df)
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
    base = build_base_features(df)
    alpha = compute_alpha_features(df)
    mega = compute_mega_alpha(df, alpha)
    feats = pd.concat([base[BASE_COLS], alpha[ALPHA_COLS_REF]], axis=1)
    return feats, mega


# reference to alpha columns (kept here to avoid circular import at module top)
from alphas import ALPHA_COLS as ALPHA_COLS_REF  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Gap handling + normalization + windows
# ─────────────────────────────────────────────────────────────────────────────
def make_dataset(df, ctx=CTX, tgt=TGT, train_frac=0.9):
    feats, mega = build_feature_matrix(df)
    feats = feats.astype(float)
    mega = mega.astype(float)

    arr = feats.to_numpy()
    n = arr.shape[0]

    # runs (no Friday->Monday stitching)
    gap = df['Time'].diff().dt.total_seconds() > GAP_H * 3600
    run = gap.cumsum().to_numpy()

    # time-based train/val cutoff
    t0, t1 = df['Time'].min(), df['Time'].max()
    cutoff = t0 + (t1 - t0) * train_frac
    is_train_time = (df['Time'] < cutoff).to_numpy()

    # normalizer statistics from training rows (non-NaN)
    mask_tr = is_train_time & ~np.any(np.isnan(arr), axis=1)
    mean = np.nanmean(arr[mask_tr], axis=0)
    std = np.nanstd(arr[mask_tr], axis=0) + 1e-8
    arr_n = (arr - mean) / std

    mega_arr = mega.to_numpy()
    mmean = np.nanmean(mega_arr[mask_tr])
    mstd = np.nanstd(mega_arr[mask_tr]) + 1e-8
    mega_n = (mega_arr - mmean) / mstd

    # valid window start indices: full window inside one run, fully observed
    valid = []
    for i in range(ctx, n - tgt + 1):
        if not np.any(np.isnan(arr_n[i - ctx:i + tgt])):
            if np.all(run[i - ctx:i + tgt] == run[i - ctx]):
                valid.append(i)
    valid = np.array(valid, dtype=int)

    split = np.array(['train' if is_train_time[i] else 'val' for i in valid])
    return ForexH1Dataset(arr_n, mega_n, valid, split, ctx, tgt), {
        'n': n, 'n_features': arr.shape[1],
        'n_train': int((split == 'train').sum()),
        'n_val': int((split == 'val').sum()),
        'cutoff': str(cutoff), 'feature_cols': list(feats.columns),
    }


class ForexH1Dataset(Dataset):
    def __init__(self, feat, mega, starts, split, ctx, tgt):
        self.feat = feat          # (n, F) normalized
        self.mega = mega          # (n,)
        self.starts = starts      # window start indices (ctx begins at start-ctx)
        self.split = split
        self.ctx = ctx
        self.tgt = tgt

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        i = self.starts[idx]
        ctx = self.feat[i - self.ctx:i]
        tgt = self.feat[i:i + self.tgt]
        return {'ctx': torch.FloatTensor(ctx), 'tgt': torch.FloatTensor(tgt), 'start': i}


def probe_pairs(self, tau, indices=None, batch=None):
    """Return (ctx_array (B,CTX,F), target_array (B,)) for the mega-alpha at i+tau."""
    if indices is None:
        indices = np.arange(len(self.starts))
    starts = self.starts[indices]
    ctx = np.stack([self.feat[s - self.ctx:s] for s in starts])
    tgt_idx = starts + tau
    ok = (tgt_idx >= 0) & (tgt_idx < len(self.mega))
    y = self.mega[tgt_idx]
    y[~ok] = np.nan
    return ctx, y


# attach as a method
ForexH1Dataset.probe_pairs = probe_pairs


if __name__ == "__main__":
    df = load_eurusd_h1("data/EURUSD_H1.csv")
    ds, info = make_dataset(df)
    print("dataset info:", info)
    print("sample batch shapes:", {k: v.shape for k, v in ds[0].items() if hasattr(v, 'shape')})
