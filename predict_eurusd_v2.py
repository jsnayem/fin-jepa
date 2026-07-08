#!/usr/bin/env python3
"""
EUR/USD 2026 OHLC Forecast — loads v2 checkpoint, autoregressively predicts 260 days.
"""

import json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints" / "eurusd_v2"

OHLC_COLS = ['open', 'high', 'low', 'close']
ALL_COLS  = ['open', 'high', 'low', 'close', 'volume', 'returns',
             'volatility_20', 'volatility_5', 'range_pct',
             'close_ma_5', 'close_ma_20', 'high_low_ratio']

# ──────────────────────────────────────────────
# Model (same as training)
# ──────────────────────────────────────────────

class OHLCTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, n_layers=3, n_heads=4, max_len=120):
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
        preds = self.head(x)
        return preds

    def autoregressive(self, ctx, n_steps, pbar=None):
        current = ctx  # (1, T_ctx, F)
        preds_all = []
        for _ in range(n_steps):
            if current.shape[1] > 100:
                current = current[:, -100:]
            out = self.forward(current)
            ohlc_next = out[:, -1:, :]  # (1, 1, 4)

            # Enforce OHLC constraints
            ohlc_next = self._ohlc_constraint(ohlc_next)

            preds_all.append(ohlc_next.squeeze(1).detach())

            next_feat = self._build_features(ohlc_next.squeeze(1), current[:, -1, :])
            current = torch.cat([current, next_feat.unsqueeze(1)], dim=1)

            if pbar is not None:
                pbar.update(1)

        return torch.stack(preds_all, dim=0)  # (n_steps, 4)

    @staticmethod
    def _ohlc_constraint(ohlc):
        o, h, l, c = ohlc[..., 0:1], ohlc[..., 1:2], ohlc[..., 2:3], ohlc[..., 3:4]
        h = torch.maximum(h, torch.maximum(o, c))
        l = torch.minimum(l, torch.minimum(o, c))
        return torch.cat([o, h, l, c], dim=-1)

    @staticmethod
    def _build_features(ohlc, prev_feat):
        o, h, l, c = ohlc[..., 0:1], ohlc[..., 1:2], ohlc[..., 2:3], ohlc[..., 3:4]
        prev_c = prev_feat[..., 3:4]
        returns = (c - prev_c) / (prev_c.abs() + 1e-8)
        vol = prev_feat[..., 4:5]
        rp = (h - l) / (c.abs() + 1e-8)
        prev_vol20 = prev_feat[..., 6:7]
        prev_vol5 = prev_feat[..., 7:8]
        prev_cma5 = prev_feat[..., 9:10]
        prev_cma20 = prev_feat[..., 10:11]
        prev_hlr = prev_feat[..., 11:12]

        new_vol20 = 0.95 * prev_vol20 + 0.05 * returns.abs()
        new_vol5  = 0.80 * prev_vol5  + 0.20 * returns.abs()
        new_cma5  = 0.80 * prev_cma5  + 0.20 * (c / (prev_c.abs() + 1e-8))
        new_cma20 = 0.95 * prev_cma20 + 0.05 * (c / (prev_c.abs() + 1e-8))
        new_hlr   = 0.95 * prev_hlr   + 0.05 * (h / (l.abs() + 1e-8))

        return torch.cat([o, h, l, c, vol, returns, new_vol20, new_vol5,
                          rp, new_cma5, new_cma20, new_hlr], dim=-1)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60, flush=True)
    print("EUR/USD 2026 OHLC Forecast (v2)")
    print("=" * 60, flush=True)

    # Load
    ckpt_path  = CKPT_DIR / "best.pt"
    meta_path  = CKPT_DIR / "meta.json"

    if not ckpt_path.exists() or not meta_path.exists():
        print("ERROR: Run train_eurusd_v2.py first.")
        return

    meta = json.load(open(meta_path))
    ohlc_mean = np.array(meta['ohlc_mean'], dtype=np.float32)
    ohlc_std  = np.array(meta['ohlc_std'],  dtype=np.float32)
    all_mean  = np.array(meta['all_mean'],  dtype=np.float32)
    all_std   = np.array(meta['all_std'],   dtype=np.float32)
    n_features = meta['n_features']

    print(f"  Features: {n_features} | Val loss: {meta['best_val_loss']:.6f} | Val MAE: {meta.get('best_val_mae', 0):.5f}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = OHLCTransformer(n_features=n_features, d_model=64, n_layers=4, n_heads=4).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Loaded epoch {ckpt['epoch']}")

    # Re-run data pipeline to get last context with correct features
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

    # Last 60 rows = most recent context
    last_ctx_norm = normalized[-60:]  # (60, F)
    ctx_tensor = torch.FloatTensor(last_ctx_norm).unsqueeze(0).to(DEVICE)  # (1, 60, F)

    # Also get last 60 raw OHLC for reference
    last_raw = df[OHLC_COLS].values[-60:]
    print(f"  Last raw close: {last_raw[-1, 3]:.5f} ({df['date'].iloc[-1].date()})")

    # Predict 2026
    N_DAYS = 260
    print(f"\n[2] Autoregressive rollout: {N_DAYS} trading days...", flush=True)

    with torch.no_grad(), tqdm(total=N_DAYS, desc="  Rolling out", unit="day") as pbar:
        preds_norm = model.autoregressive(ctx_tensor, N_DAYS, pbar=pbar)  # (260, 4)

    preds_np = preds_norm.cpu().numpy()
    preds_ohlc = preds_np * ohlc_std + ohlc_mean  # denormalize

    # Build date index
    dates = pd.date_range("2026-01-01", "2026-12-31", freq="B")[:N_DAYS]

    results = pd.DataFrame(preds_ohlc, columns=OHLC_COLS, index=dates)
    results.index.name = 'date'

    # Statistics
    print("\n" + "=" * 60)
    print("2026 OHLC Forecast Summary")
    print("=" * 60)

    for col in OHLC_COLS:
        s = results[col]
        print(f"  {col:>5s}: start={s.iloc[0]:.5f}  end={s.iloc[-1]:.5f}  "
              f"min={s.min():.5f}  max={s.max():.5f}  mean={s.mean():.5f}  std={s.std():.5f}")

    # Range check
    hilo = results['high'] - results['low']
    print(f"\n  Avg daily range: {hilo.mean():.6f} (min={hilo.min():.6f}, max={hilo.max():.6f})")
    print(f"  Trend (close):   {results['close'].iloc[-1] - results['close'].iloc[0]:.6f}")

    # Backtest: compare model's prediction of the last known day vs actual
    # Use the second-to-last sequence to predict the actual last 5 known days
    print("\n[3] Backtest accuracy on last 5 known trading days...", flush=True)
    test_ctx = torch.FloatTensor(normalized[-65:-5]).unsqueeze(0).to(DEVICE)  # days t_0 ... t_59
    with torch.no_grad():
        test_preds_norm = model(test_ctx)[:, -5:, :].cpu().numpy()
    test_preds_raw = test_preds_norm * ohlc_std + ohlc_mean
    test_actual = vals[-5:, :4] * ohlc_std + ohlc_mean
    test_dates = df['date'].values[-5:]

    errors = np.abs(test_preds_raw[0] - test_actual)
    print(f"  {'Date':<12s} {'Pred Close':>10s} {'Actual':>10s} {'Error':>10s}")
    for i in range(5):
        d = pd.Timestamp(test_dates[i]).date()
        print(f"  {str(d):<12s} {test_preds_raw[0, i, 3]:>10.5f} {test_actual[i, 3]:>10.5f} {errors[i, 3]:>10.5f}")
    print(f"  Mean absolute error: {errors.mean():.6f}")
    print(f"  Close MAE:          {errors[:, 3].mean():.6f}")

    # Save
    out_csv = CKPT_DIR / "predictions_2026_v2.csv"
    results.to_csv(out_csv)

    # Also save with last known context for plotting
    last_actual_df = pd.DataFrame(last_raw, columns=OHLC_COLS,
                                   index=pd.DatetimeIndex(df['date'].values[-60:]))
    combined = pd.concat([last_actual_df.iloc[-20:], results.iloc[:20]])
    combined.to_csv(CKPT_DIR / "prediction_with_context.csv")

    print(f"\n  Saved: {out_csv}")
    print(f"  Saved (w/context): {CKPT_DIR / 'prediction_with_context.csv'}")
    print(f"\n✅ Forecast complete")


if __name__ == "__main__":
    main()
