#!/usr/bin/env python3
"""
EUR/USD Next-Trading-Day OHLC Forecast — loads v2 checkpoint, predicts one day ahead.
"""

import json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints" / "eurusd_v2"

OHLC_COLS = ['open', 'high', 'low', 'close']
ALL_COLS  = ['open', 'high', 'low', 'close', 'volume', 'returns',
             'volatility_20', 'volatility_5', 'range_pct',
             'close_ma_5', 'close_ma_20', 'high_low_ratio']


class OHLCTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, n_layers=4, n_heads=4, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4, dropout=0.1,
                activation='gelu', batch_first=True, norm_first=True,
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 4),
        )

    def forward(self, x):
        B, T, F = x.shape
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :T, :]
        x = self.dropout(x)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        for block in self.blocks:
            x = block(x, src_mask=mask, is_causal=False)
        x = self.norm(x)
        return self.head(x)

    def predict_next(self, ctx):
        with torch.no_grad():
            out = self.forward(ctx)
            ohlc = out[:, -1, :]
            o, h, l, c = ohlc[:, 0:1], ohlc[:, 1:2], ohlc[:, 2:3], ohlc[:, 3:4]
            h = torch.maximum(h, torch.maximum(o, c))
            l = torch.minimum(l, torch.minimum(o, c))
            return torch.cat([o, h, l, c], dim=-1)


def main():
    print("=" * 60, flush=True)
    print("EUR/USD Next-Day OHLC Forecast (v2)")
    print("=" * 60, flush=True)

    ckpt_path = CKPT_DIR / "best.pt"
    meta_path = CKPT_DIR / "meta.json"

    if not ckpt_path.exists() or not meta_path.exists():
        print("ERROR: Run train_eurusd_v2.py first.")
        return

    meta = json.load(open(meta_path))
    ohlc_mean = np.array(meta['ohlc_mean'], dtype=np.float32)
    ohlc_std  = np.array(meta['ohlc_std'],  dtype=np.float32)
    all_mean  = np.array(meta['all_mean'],  dtype=np.float32)
    all_std   = np.array(meta['all_std'],   dtype=np.float32)
    n_features = meta['n_features']

    print(f"  Features: {n_features} | Val loss: {meta['best_val_loss']:.6f} "
          f"| Val MAE: {meta.get('best_val_mae', 0):.5f} | Last date: {meta.get('last_date', 'unknown')}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = OHLCTransformer(n_features=n_features, d_model=64, n_layers=4, n_heads=4).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Loaded epoch {ckpt['epoch']}")

    print("\n[1] Building last context from 2025 data...", flush=True)
    import yfinance as yf
    df = yf.download("EURUSD=X", start="2000-01-01", end="2025-12-31", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_values('date').reset_index(drop=True)

    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_5'] = df['returns'].rolling(5).std()
    df['range_pct'] = (df['high'] - df['low']) / df['close']
    df['close_ma_5'] = df['close'] / df['close'].rolling(5).mean()
    df['close_ma_20'] = df['close'] / df['close'].rolling(20).mean()
    df['high_low_ratio'] = df['high'] / df['low']
    df = df.dropna(subset=ALL_COLS).reset_index(drop=True)

    vals = df[ALL_COLS].values.astype(np.float32)
    normalized = (vals - all_mean) / all_std
    last_ctx_norm = normalized[-60:]
    last_close = vals[-1, 3]
    last_date = df['date'].iloc[-1]
    print(f"  Last close: {last_close:.5f} on {last_date.date()}")

    ctx_tensor = torch.FloatTensor(last_ctx_norm).unsqueeze(0).to(DEVICE)

    print(f"\n[2] Predicting next trading day...", flush=True)
    with torch.no_grad():
        next_pred_norm = model.predict_next(ctx_tensor).cpu().numpy()
    next_pred = next_pred_norm[0] * ohlc_std + ohlc_mean

    print("\n" + "=" * 60)
    print(f"Next Trading Day Prediction")
    print(f"  (first trading day after {last_date.date()})")
    print("=" * 60)
    for i, col in enumerate(OHLC_COLS):
        print(f"  {col:>5s}: {next_pred[i]:.5f}")

    change = next_pred[3] - last_close
    pct = (change / last_close) * 100
    print(f"\n  Expected change: {change:+.5f} ({pct:+.3f}%)")

    out_csv = CKPT_DIR / "next_day_prediction_v2.csv"
    pd.DataFrame([next_pred], columns=OHLC_COLS).to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")
    print(f"\n✅ Forecast complete")


if __name__ == "__main__":
    main()
