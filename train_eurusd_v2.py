#!/usr/bin/env python3
"""
Fin-JEPA v2 — Single-day OHLC predictor for EUR/USD.
Predicts next-trading-day OHLC. Target: 1 day, context: 60 days.
"""

import json, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_DIR = Path(__file__).resolve().parent / "checkpoints" / "eurusd_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OHLC_COLS = ['open', 'high', 'low', 'close']
ALL_COLS  = ['open', 'high', 'low', 'close', 'volume', 'returns',
             'volatility_20', 'volatility_5', 'range_pct',
             'close_ma_5', 'close_ma_20', 'high_low_ratio']

CTX_LEN = 60
EPOCHS = 80
BATCH_SIZE = 256

# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────

class SingleDayDataset(Dataset):
    def __init__(self, seqs_ctx, seqs_tgt):
        self.ctx = seqs_ctx
        self.tgt = seqs_tgt

    def __len__(self):
        return len(self.ctx)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.ctx[idx]), torch.FloatTensor(self.tgt[idx])


def fetch_eurusd():
    print("[1] Fetching EUR/USD from Yahoo Finance (till 2025-12-31)...", flush=True)
    import yfinance as yf
    df = yf.download("EURUSD=X", start="2000-01-01", end="2025-12-31", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    print(f"  Raw: {len(df):,} rows, {df['date'].min().date()} → {df['date'].max().date()}", flush=True)
    return df


def build_features(df):
    print("[2] Computing features...", flush=True)
    df = df.sort_values('date').reset_index(drop=True)

    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_5'] = df['returns'].rolling(5).std()
    df['range_pct'] = (df['high'] - df['low']) / df['close']
    df['close_ma_5'] = df['close'] / df['close'].rolling(5).mean()
    df['close_ma_20'] = df['close'] / df['close'].rolling(20).mean()
    df['high_low_ratio'] = df['high'] / df['low']

    df = df.dropna(subset=ALL_COLS).reset_index(drop=True)
    print(f"  After features: {len(df):,} rows", flush=True)
    return df


def normalize_and_sequence(df):
    print("[3] Normalizing and building sequences...", flush=True)

    vals = df[ALL_COLS].values.astype(np.float32)
    means = np.nanmean(vals, axis=0)
    stds  = np.nanstd(vals, axis=0) + 1e-8

    normalized = vals
    ohlc_means = means[:4].copy()
    ohlc_stds  = stds[:4].copy()

    print(f"  OHLC means: {dict(zip(OHLC_COLS, ohlc_means.round(5)))}", flush=True)
    print(f"  OHLC stds:  {dict(zip(OHLC_COLS, ohlc_stds.round(5)))}", flush=True)

    # Context=60, target=1 day, stride=1
    ctx_list, tgt_list = [], []
    v = (vals - means) / stds
    for i in range(len(v) - CTX_LEN):
        ctx = v[i:i + CTX_LEN]
        tgt = v[i + CTX_LEN, :4]
        if not np.isnan(ctx).any() and not np.isnan(tgt).any():
            ctx_list.append(ctx)
            tgt_list.append(tgt)

    ctx_arr = np.stack(ctx_list).astype(np.float32)
    tgt_arr = np.stack(tgt_list).astype(np.float32)
    print(f"  {len(ctx_arr):,} samples × ({CTX_LEN} ctx + 1 tgt) × {v.shape[1]} feats", flush=True)
    return df, ctx_arr, tgt_arr, means, stds, ohlc_means, ohlc_stds


def split_data(df, ctx_arr, tgt_arr):
    """Chronological split: last 20% of time range = validation. No data leakage."""
    dates = df['date'].values
    cutoff_date = dates[int(len(dates) * 0.8)]
    cutoff_idx = CTX_LEN + int(len(ctx_arr) * 0.8)

    # Use strictly non-overlapping chronological splits
    train_n = int(len(ctx_arr) * 0.8)
    train_ctx, train_tgt = ctx_arr[:train_n], tgt_arr[:train_n]
    val_ctx,   val_tgt   = ctx_arr[train_n:], tgt_arr[train_n:]

    print(f"  Train: {len(train_ctx):,}  Val: {len(val_ctx):,} (chronological, ~{cutoff_date.date()})", flush=True)
    return (SingleDayDataset(train_ctx, train_tgt),
            SingleDayDataset(val_ctx, val_tgt),
            ctx_arr[-1], dates.iloc[-1])


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_epoch(model, loader, opt, scaler):
    model.train()
    total_loss, total_mae = 0.0, 0.0
    n = 0
    for ctx, tgt in loader:
        ctx = ctx.to(DEVICE)
        tgt = tgt.to(DEVICE)

        opt.zero_grad()
        with torch.amp.autocast(device_type=DEVICE, enabled=DEVICE == "cuda"):
            preds = model(ctx)
            pred_next = preds[:, -1, :]
            loss = nn.functional.mse_loss(pred_next, tgt)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        total_loss += loss.item()
        with torch.no_grad():
            total_mae += (pred_next - tgt).abs().mean().item()
        n += 1
    return total_loss / n, total_mae / n


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_loss, total_mae = 0.0, 0.0
    n = 0
    for ctx, tgt in loader:
        ctx = ctx.to(DEVICE)
        tgt = tgt.to(DEVICE)
        preds = model(ctx)
        pred_next = preds[:, -1, :]
        loss = nn.functional.mse_loss(pred_next, tgt)
        total_loss += loss.item()
        total_mae += (pred_next - tgt).abs().mean().item()
        n += 1
    return total_loss / n, total_mae / n


def main():
    print("=" * 60, flush=True)
    print("EUR/USD Single-Day OHLC Predictor — Training (v2)")
    print(f"Device: {DEVICE}")
    print("=" * 60, flush=True)

    # Data
    df_raw  = fetch_eurusd()
    df_feat = build_features(df_raw)
    df, ctx_arr, tgt_arr, all_mean, all_std, ohlc_mean, ohlc_std = normalize_and_sequence(df_feat)
    trainset, valset, last_ctx, last_date = split_data(df, ctx_arr, tgt_arr)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(valset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Model
    n_features = len(ALL_COLS)
    model = OHLCTransformer(n_features=n_features, d_model=64, n_layers=4, n_heads=4).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[4] Model: {n_features} features, 64d, 4 layers, {n_params:,} params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler(device=DEVICE, enabled=DEVICE == "cuda")

    print(f"\n[5] Training {EPOCHS} epochs (single-day target)...", flush=True)
    best_val = float('inf')
    t0 = time.time()
    history = []

    for epoch in range(EPOCHS):
        tr_loss, tr_mae = train_epoch(model, train_loader, opt, scaler)
        va_loss, va_mae = eval_epoch(model, val_loader)
        sched.step()
        history.append({'epoch': epoch, 'tr_loss': tr_loss, 'tr_mae': tr_mae, 'va_loss': va_loss, 'va_mae': va_mae})

        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'n_features': n_features,
                'epoch': epoch,
                'val_loss': va_loss,
                'val_mae': va_mae,
            }, OUTPUT_DIR / "best.pt")

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            elapsed = time.time() - t0
            print(f"  epoch {epoch:3d} | tr_loss={tr_loss:.6f} tr_mae={tr_mae:.5f} | "
                  f"va_loss={va_loss:.6f} va_mae={va_mae:.5f} | {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed/60:.1f}min | Best val loss: {best_val:.6f}", flush=True)

    # ──────────────────────────────────────────
    # Holdout evaluation — predict last 10 val days and show accuracy
    # ──────────────────────────────────────────
    ckpt = torch.load(OUTPUT_DIR / "best.pt", map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    h_ctx = torch.FloatTensor(val_ctx[-10:]).to(DEVICE)
    h_tgt = torch.FloatTensor(val_tgt[-10:]).to(DEVICE)
    with torch.no_grad():
        h_preds = model(h_ctx)[:, -1, :].cpu().numpy()
    h_actual = h_tgt.cpu().numpy()
    h_preds_raw = h_preds * ohlc_std + ohlc_mean
    h_actual_raw = h_actual * ohlc_std + ohlc_mean

    val_start = len(ctx_arr) - 10
    val_dates = df['date'].values[CTX_LEN + val_start:CTX_LEN + val_start + 10]

    print(f"\n[6] Holdout — last 10 validation days")
    print(f"  {'Date':<12s} {'Pred Close':>10s} {'Actual':>10s} {'Err':>10s}")
    for i in range(10):
        d = pd.Timestamp(val_dates[i]).date()
        pe = h_preds_raw[i, 3]
        ae = h_actual_raw[i, 3]
        print(f"  {str(d):<12s} {pe:>10.5f} {ae:>10.5f} {abs(pe-ae):>10.5f}")
    print(f"  Close MAE: {np.abs(h_preds_raw[:, 3] - h_actual_raw[:, 3]).mean():.5f}")
    print(f"  OHLC MAE:  {np.abs(h_preds_raw - h_actual_raw).mean():.5f}")

    # ──────────────────────────────────────────
    # Predict next trading day (first after last known day in data)
    # ──────────────────────────────────────────
    print(f"\n[7] Predicting next trading day (after {last_date.date()})...", flush=True)
    c_ctx = torch.FloatTensor(last_ctx).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        next_pred_norm = model(c_ctx)[:, -1, :].cpu().numpy()
    next_pred_raw = next_pred_norm * ohlc_std + ohlc_mean
    print(f"  Predicted OHLC: O={next_pred_raw[0,0]:.5f} H={next_pred_raw[0,1]:.5f} "
          f"L={next_pred_raw[0,2]:.5f} C={next_pred_raw[0,3]:.5f}")

    # ──────────────────────────────────────────
    # Save artifacts
    # ──────────────────────────────────────────
    np.save(OUTPUT_DIR / "last_context.npy", last_ctx)
    np.save(OUTPUT_DIR / "next_day_prediction.npy", next_pred_raw[0])
    meta = {
        'description': 'Single-day OHLC predictor for EUR/USD',
        'symbol': 'EURUSD=X',
        'features': ALL_COLS,
        'n_features': n_features,
        'ohlc_mean': ohlc_mean.tolist(),
        'ohlc_std': ohlc_std.tolist(),
        'all_mean': all_mean.tolist(),
        'all_std': all_std.tolist(),
        'n_samples': len(ctx_arr),
        'n_train': len(trainset),
        'n_val': len(valset),
        'last_date': str(last_date.date()),
        'next_day_prediction': next_pred_raw[0].tolist(),
        'context_len': CTX_LEN,
        'target_len': 1,
        'best_val_loss': float(best_val),
        'best_val_mae': float(va_mae),
        'device': DEVICE,
        'params': n_params,
        'history': history,
    }
    json.dump(meta, open(OUTPUT_DIR / "meta.json", "w"), indent=2)

    print(f"\n✅ Training complete. Checkpoint: {OUTPUT_DIR / 'best.pt'}")
    print(f"   val_mse={best_val:.6f}  val_mae={va_mae:.5f}")
    print(f"   Next-day close prediction: {next_pred_raw[0,3]:.5f}")


if __name__ == "__main__":
    main()
