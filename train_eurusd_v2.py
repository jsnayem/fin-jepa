#!/usr/bin/env python3
"""
Fin-JEPA v2 — Direct OHLC autoregressive predictor for EUR/USD.
Predicts next-day OHLC directly, minimizing MSE on the 4 price features.
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

# ──────────────────────────────────────────────
# Model: Causal Transformer → direct OHLC
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
            nn.Linear(d_model, 4),  # open, high, low, close
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
        preds = self.head(x)  # (B, T, 4)
        return preds

    def autoregressive(self, ctx, n_steps, threshold=None):
        """Roll out n_steps OHLC predictions, enforcing high>=close etc."""
        current = ctx  # (1, T_ctx, F)
        preds_all = []
        for _ in range(n_steps):
            if current.shape[1] > 100:
                current = current[:, -100:]
            out = self.forward(current)  # (1, T, 4)
            ohlc_next = out[:, -1:, :]   # (1, 1, 4)

            # Enforce: high >= max(open, close) and low <= min(open, close)
            ohlc_next = self._ohlc_constraint(ohlc_next)

            preds_all.append(ohlc_next.squeeze(1).detach())

            # Build next step features from predicted OHLC
            next_feat = self._build_features(ohlc_next.squeeze(1), current[:, -1, :])
            current = torch.cat([current, next_feat.unsqueeze(1)], dim=1)

        return torch.stack(preds_all, dim=0)  # (n_steps, 4)

    @staticmethod
    def _ohlc_constraint(ohlc):
        """Enforce high >= max(open,close) and low <= min(open,close)."""
        o, h, l, c = ohlc[..., 0:1], ohlc[..., 1:2], ohlc[..., 2:3], ohlc[..., 3:4]
        h = torch.maximum(h, torch.maximum(o, c))
        l = torch.minimum(l, torch.minimum(o, c))
        return torch.cat([o, h, l, c], dim=-1)

    @staticmethod
    def _build_features(ohlc, prev_feat):
        """Build next-step feature vector from predicted OHLC + previous context."""
        o, h, l, c = ohlc[..., 0:1], ohlc[..., 1:2], ohlc[..., 2:3], ohlc[..., 3:4]
        prev_c = prev_feat[..., 3:4]  # previous close
        returns = (c - prev_c) / (prev_c.abs() + 1e-8)

        # Use previous features for rolling stats as approximation
        vol = prev_feat[..., 4:5]  # carry volume
        rp = (h - l) / (c.abs() + 1e-8)  # new range_pct

        # Rolling stats: decay previous values
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
# Data
# ──────────────────────────────────────────────

class OHLCSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.data = sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]  # (65, F)
        ctx = torch.FloatTensor(seq[:60])
        tgt = torch.FloatTensor(seq[60:65, :4])  # only OHLC
        return {'ctx': ctx, 'tgt': tgt, 'ctx_ohlc': ctx[:, :4]}


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

    normalized = (vals - means) / stds
    df_norm = df.copy()
    for i, c in enumerate(ALL_COLS):
        df_norm[c] = normalized[:, i]

    ohlc_means = means[:4].copy()
    ohlc_stds  = stds[:4].copy()

    print(f"  OHLC means: {dict(zip(OHLC_COLS, ohlc_means.round(5)))}", flush=True)
    print(f"  OHLC stds:  {dict(zip(OHLC_COLS, ohlc_stds.round(5)))}", flush=True)

    # Context=60, target=5, stride=1 for maximum overlap (more training signal)
    seqs = []
    v = df_norm[ALL_COLS].values.astype(np.float32)
    for i in range(0, max(1, len(v) - 65 + 1), 1):
        chunk = v[i:i+65]
        if len(chunk) == 65 and not np.isnan(chunk).any():
            seqs.append(chunk)

    seq = np.stack(seqs).astype(np.float32)
    print(f"  {len(seq):,} sequences × {seq.shape[1]} steps × {seq.shape[2]} features", flush=True)
    return seq, means, stds, ohlc_means, ohlc_stds


def split_data(sequences):
    n = len(sequences)
    split_idx = int(n * 0.85)
    train_seq = sequences[:split_idx]
    val_seq   = sequences[split_idx:]
    last_ctx  = sequences[-1, :60]
    print(f"  Train: {len(train_seq):,}  Val: {len(val_seq):,}", flush=True)
    return OHLCSequenceDataset(train_seq), OHLCSequenceDataset(val_seq), last_ctx


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_epoch(model, loader, opt, scaler):
    model.train()
    total_loss, total_mae = 0, 0
    n = 0
    for batch in loader:
        ctx = batch['ctx'].to(DEVICE)
        tgt = batch['tgt'].to(DEVICE)  # (B, 5, 4)

        opt.zero_grad()
        with torch.amp.autocast(device_type=DEVICE, enabled=DEVICE == "cuda"):
            preds = model(ctx)  # (B, 60, 4)
            pred_ohlc = preds[:, -5:, :]  # last 5 predictions
            loss = nn.functional.mse_loss(pred_ohlc, tgt)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        total_loss += loss.item()
        with torch.no_grad():
            total_mae += (pred_ohlc - tgt).abs().mean().item()
        n += 1

    return total_loss / n, total_mae / n


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_loss, total_mae = 0, 0
    n = 0
    for batch in loader:
        ctx = batch['ctx'].to(DEVICE)
        tgt = batch['tgt'].to(DEVICE)
        preds = model(ctx)
        pred_ohlc = preds[:, -5:, :]
        loss = nn.functional.mse_loss(pred_ohlc, tgt)
        total_loss += loss.item()
        total_mae += (pred_ohlc - tgt).abs().mean().item()
        n += 1
    return total_loss / n, total_mae / n


def main():
    print("=" * 60, flush=True)
    print("EUR/USD OHLC Predictor — Training (v2)")
    print(f"Device: {DEVICE}")
    print("=" * 60, flush=True)

    # Data
    df_raw  = fetch_eurusd()
    df_feat = build_features(df_raw)
    seqs, means, stds, ohlc_means, ohlc_stds = normalize_and_sequence(df_feat)
    trainset, valset, last_ctx = split_data(seqs)

    BS = 256
    train_loader = DataLoader(trainset, batch_size=BS, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(valset,   batch_size=BS, shuffle=False, num_workers=0)

    # Model
    n_features = len(ALL_COLS)
    model = OHLCTransformer(n_features=n_features, d_model=64, n_layers=4, n_heads=4).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[4] Model: {n_features} features, 64d, 4 layers, {n_params:,} params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    scaler = torch.amp.GradScaler(device=DEVICE, enabled=DEVICE == "cuda")

    print(f"\n[5] Training 200 epochs...", flush=True)
    best_val = float('inf')
    t0 = time.time()

    for epoch in range(200):
        tr_loss, tr_mae = train_epoch(model, train_loader, opt, scaler)
        va_loss, va_mae = eval_epoch(model, val_loader)
        sched.step()

        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'n_features': n_features,
                'epoch': epoch,
                'val_loss': va_loss,
                'val_mae': va_mae,
            }, OUTPUT_DIR / "best.pt")

        if epoch % 20 == 0 or epoch == 199:
            elapsed = time.time() - t0
            print(f"  epoch {epoch:3d} | tr_loss={tr_loss:.6f} tr_mae={tr_mae:.5f} | "
                  f"va_loss={va_loss:.6f} va_mae={va_mae:.5f} | {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed/60:.1f}min | Best val loss: {best_val:.6f}", flush=True)

    # Evaluate on held-out data: predict last 5 days, denormalize, compute actual error
    ckpt = torch.load(OUTPUT_DIR / "best.pt", map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Take a fresh holdout window from the very end
    holdout_ctx = torch.FloatTensor(seqs[-2:, :60]).to(DEVICE)  # last 2 sequences for a sanity check
    with torch.no_grad():
        holdout_preds_norm = model(holdout_ctx)[:, -5:, :].cpu().numpy()
    holdout_preds = holdout_preds_norm * ohlc_stds[:4] + ohlc_means[:4]

    print(f"\n[6] Quick holdout sanity check (last prediction):")
    print(f"  Predicted OHLC (denormed): {dict(zip(OHLC_COLS, holdout_preds[-1, -1].round(5)))}")

    # Save all artifacts
    np.save(OUTPUT_DIR / "last_context.npy", last_ctx)
    meta = {
        'description': 'Direct OHLC autoregressive predictor for EUR/USD',
        'symbol': 'EURUSD=X',
        'features': ALL_COLS,
        'n_features': n_features,
        'ohlc_mean': ohlc_means.tolist(),
        'ohlc_std':  ohlc_stds.tolist(),
        'all_mean':  means.tolist(),
        'all_std':   stds.tolist(),
        'n_sequences': len(seqs),
        'n_train': len(trainset),
        'n_val':   len(valset),
        'context_len': 60,
        'target_len': 5,
        'best_val_loss': float(best_val),
        'best_val_mae':  float(va_mae),
        'device': DEVICE,
        'params': n_params,
    }
    json.dump(meta, open(OUTPUT_DIR / "meta.json", "w"), indent=2)

    print(f"\n✅ Training complete. Checkpoint: {OUTPUT_DIR / 'best.pt'}")
    print(f"   val_mse={best_val:.6f}  val_mae={va_mae:.5f}")


if __name__ == "__main__":
    main()
