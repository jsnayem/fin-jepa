#!/usr/bin/env python3
"""
Fin-JEPA — Predict EUR/USD 2026 using trained model.
Loads checkpoint, takes last 2025 context, autoregressively rolls out
latent predictions for ~260 trading days in 2026, decodes via linear probe.
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints" / "eurusd"

FEATURE_COLS = ['open','high','low','close','volume','returns','vwap',
                'volatility_20','vol_ma_20','close_ma_20','range_pct']

# ──────────────────────────────────────────────
# Model (same as training script)
# ──────────────────────────────────────────────

class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=512):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class PriceEncoder(nn.Module):
    def __init__(self, n_features, embed_dim=64, hidden_dim=128, n_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, embed_dim)
        self.encoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.projector = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        x = self.input_proj(x.squeeze(1))
        x = self.encoder(x)
        x = self.projector(x)
        return x


class TransformerPredictor(nn.Module):
    def __init__(self, embed_dim=64, n_layers=4, n_heads=4, mlp_scale=4, dropout=0.1, max_seq_len=256):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads,
                dim_feedforward=embed_dim * mlp_scale,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.pred_proj = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim),
            nn.GELU(), nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, z_seq):
        B, T, D = z_seq.shape
        x = z_seq + self.pos_embed[:, :T, :]
        x = self.dropout(x)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=z_seq.device), diagonal=1)
        for block in self.blocks:
            x = block(x, src_mask=mask, is_causal=False)
        x = self.norm(x)
        x = self.pred_proj(x)
        return x


class FinJEPA(nn.Module):
    def __init__(self, n_features, embed_dim=64, encoder_layers=3,
                 predictor_layers=4, predictor_heads=4, sigreg_proj=512, sigreg_lambda=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.sigreg_lambda = sigreg_lambda
        self.encoder = PriceEncoder(n_features, embed_dim, n_layers=encoder_layers)
        self.predictor = TransformerPredictor(embed_dim, predictor_layers, predictor_heads)
        self.sigreg = SIGReg(num_proj=sigreg_proj)

    def encode_batch(self, seq):
        B, T, F = seq.shape
        z_all = self.encoder(seq.reshape(-1, 1, F))
        return z_all.reshape(B, T, -1)

    def forward(self, ctx, tgt=None):
        B, T_ctx, F = ctx.shape
        z_ctx = self.encode_batch(ctx)
        z_pred = self.predictor(z_ctx)
        output = {'emb': z_ctx}
        if tgt is not None:
            z_tgt = self.encode_batch(tgt)
            n_compare = min(T_ctx, tgt.shape[1])
            pred_loss = nn.functional.mse_loss(z_pred[:, :n_compare], z_tgt[:, :n_compare])
            output['pred_loss'] = pred_loss
            full_emb = torch.cat([z_ctx, z_tgt], dim=1)
            sigreg_loss = self.sigreg(full_emb.permute(1, 0, 2))
            output['sigreg_loss'] = sigreg_loss
            output['loss'] = pred_loss + self.sigreg_lambda * sigreg_loss
        return output

    def predict_future(self, ctx, n_steps=10, max_seq=250):
        z_ctx = self.encode_batch(ctx)
        current = z_ctx
        preds = []
        for _ in range(n_steps):
            if current.shape[1] > max_seq:
                current = current[:, -max_seq:]
            z_pred_all = self.predictor(current)
            z_next = z_pred_all[:, -1:]
            preds.append(z_next)
            current = torch.cat([current, z_next], dim=1)
        return torch.cat(preds, dim=1)


# ──────────────────────────────────────────────
# Decoder probe for latent → price
# ──────────────────────────────────────────────

def train_decoder_probe(model, sequences, embed_dim, means, stds):
    """Train a linear+MLP decoder: latent_embedding → all 11 price features."""
    print("\n  Training latent→price decoder probe...", flush=True)

    model.eval()
    # Collect latent embeddings and targets from training data
    latents = []
    targets = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, min(len(sequences), 2000), batch_size):
            batch = sequences[i:i+batch_size]
            seq_t = torch.FloatTensor(batch).to(DEVICE)
            z = model.encode_batch(seq_t)  # (B, 65, D)
            latents.append(z.reshape(-1, embed_dim).cpu())
            targets.append(seq_t.reshape(-1, len(FEATURE_COLS)).cpu())

    Z_lat = torch.cat(latents, dim=0)   # (N, D)
    Y_tgt = torch.cat(targets, dim=0)   # (N, 11)

    # Decoder: 2-layer MLP
    decoder = nn.Sequential(
        nn.Linear(embed_dim, 128),
        nn.GELU(),
        nn.Linear(128, 64),
        nn.GELU(),
        nn.Linear(64, len(FEATURE_COLS)),
    ).to(DEVICE)

    opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    n_train = int(len(Z_lat) * 0.8)
    Z_tr, Z_va = Z_lat[:n_train].to(DEVICE), Z_lat[n_train:].to(DEVICE)
    Y_tr, Y_va = Y_tgt[:n_train].to(DEVICE), Y_tgt[n_train:].to(DEVICE)

    best_loss = float('inf')
    for epoch in range(200):
        decoder.train()
        opt.zero_grad()
        loss = nn.functional.mse_loss(decoder(Z_tr), Y_tr)
        loss.backward()
        opt.step()

        if epoch % 50 == 0 or epoch == 199:
            decoder.eval()
            with torch.no_grad():
                va_loss = nn.functional.mse_loss(decoder(Z_va), Y_va).item()
            if va_loss < best_loss:
                best_loss = va_loss
                best_state = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}

    decoder.load_state_dict(best_state)
    print(f"  Decoder probe val MSE: {best_loss:.6f}", flush=True)
    return decoder


def decode_latents(decoder, latents, means, stds):
    """Convert latent embeddings back to original price scale."""
    with torch.no_grad():
        pred_norm = decoder(latents.to(DEVICE)).cpu().numpy()
    # Denormalize
    pred_raw = pred_norm * stds + means
    return pred_raw


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60, flush=True)
    print("Fin-JEPA — EUR/USD 2026 Forecast")
    print("=" * 60, flush=True)

    # Check artifacts exist
    ckpt_path = CKPT_DIR / "best.pt"
    ctx_path = CKPT_DIR / "last_context.npy"
    meta_path = CKPT_DIR / "meta.json"

    for p in [ckpt_path, ctx_path, meta_path]:
        if not p.exists():
            print(f"ERROR: Missing {p}. Run train_eurusd.py first.")
            return

    meta = json.load(open(meta_path))
    embed_dim = meta['embed_dim']
    means = np.array(meta['normalizer_mean'], dtype=np.float32)
    stds = np.array(meta['normalizer_std'], dtype=np.float32)

    print(f"  Model: {embed_dim}d | Features: {meta['n_features']} | Val loss: {meta['best_val_loss']:.4f}")

    # Load model
    print("\n[1] Loading checkpoint...", flush=True)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = FinJEPA(
        n_features=len(FEATURE_COLS),
        embed_dim=ckpt['embed_dim'],
        encoder_layers=ckpt['enc_layers'],
        predictor_layers=ckpt['pred_layers'],
        sigreg_proj=ckpt['sigreg_proj'],
        sigreg_lambda=ckpt['sigreg_lambda'],
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Loaded epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})", flush=True)

    # Load last context
    print("\n[2] Loading last context window (2025 data)...", flush=True)
    last_ctx = np.load(ctx_path)  # (60, 11)
    ctx_tensor = torch.FloatTensor(last_ctx).unsqueeze(0).to(DEVICE)  # (1, 60, 11)
    print(f"  Context shape: {ctx_tensor.shape}", flush=True)

    # Predict 260 trading days for 2026
    N_DAYS = 260
    print(f"\n[3] Autoregressive rollout: {N_DAYS} steps...", flush=True)

    with torch.no_grad():
        future_latents = model.predict_future(ctx_tensor, n_steps=N_DAYS)
    # future_latents: (1, 260, D)
    future_latents = future_latents.squeeze(0)  # (260, D)
    print(f"  Predicted latents: {future_latents.shape}", flush=True)

    # Train decoder probe on historical data
    print("\n[4] Building decoder probe from historical latents...", flush=True)
    # Re-run data pipeline to get all historical sequences
    import yfinance as yf
    df = yf.download("EURUSD=X", start="2000-01-01", end="2025-12-31", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_values('date').reset_index(drop=True)

    # Features
    df['returns'] = df['close'].pct_change()
    df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['vol_ma_20'] = df['volume'] / df['volume'].rolling(20).mean()
    df['vol_ma_20'] = df['vol_ma_20'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df['close_ma_20'] = df['close'] / df['close'].rolling(20).mean()
    df['range_pct'] = (df['high'] - df['low']) / df['close']
    df = df.dropna(subset=FEATURE_COLS)

    vals = df[FEATURE_COLS].values.astype(np.float32)
    normalized = (vals - means) / stds

    # Build sequences
    seqs = []
    for i in range(0, max(1, len(normalized) - 65 + 1), 5):
        chunk = normalized[i:i+65]
        if len(chunk) == 65 and not np.isnan(chunk).any():
            seqs.append(chunk)
    seqs = np.array(seqs, dtype=np.float32)

    decoder = train_decoder_probe(model, seqs, embed_dim, means, stds)

    # Decode predicted latents to price features
    print("\n[5] Decoding predicted latents to price features...", flush=True)
    decoded = decode_latents(decoder, future_latents, means, stds)  # (260, 11)

    # Build date index for 2026 (skip weekends)
    dates = pd.date_range("2026-01-01", "2026-12-31", freq="B")
    dates = dates[:N_DAYS]

    # Create output DataFrame
    results = pd.DataFrame(decoded, columns=FEATURE_COLS, index=dates)
    results.index.name = 'date'

    # Also save raw latents
    latent_df = pd.DataFrame(
        future_latents.cpu().numpy(),
        columns=[f'latent_{i}' for i in range(embed_dim)],
        index=dates,
    )
    latent_df.index.name = 'date'

    # Statistics
    print("\n" + "=" * 60, flush=True)
    print("2026 Forecast Summary (decoded close prices)")
    print("=" * 60, flush=True)
    close_pred = results['close']
    print(f"  Start (2026-01-01): {close_pred.iloc[0]:.5f}")
    print(f"  End   (2026-12-31): {close_pred.iloc[-1]:.5f}")
    print(f"  Min:               {close_pred.min():.5f} ({close_pred.idxmin().strftime('%Y-%m-%d')})")
    print(f"  Max:               {close_pred.max():.5f} ({close_pred.idxmax().strftime('%Y-%m-%d')})")
    print(f"  Mean:              {close_pred.mean():.5f}")
    print(f"  Std:               {close_pred.std():.5f}")
    print(f"  Trend (last-first): {close_pred.iloc[-1] - close_pred.iloc[0]:.6f}")

    # Save
    out_csv = CKPT_DIR / "predictions_2026.csv"
    latent_csv = CKPT_DIR / "latents_2026.csv"
    results.to_csv(out_csv)
    latent_df.to_csv(latent_csv)

    print(f"\n  Predictions saved to: {out_csv}")
    print(f"  Latent embeddings:    {latent_csv}")
    print(f"\n{'='*60}", flush=True)
    print("✅ Prediction complete")
    print(f"{'='*60}", flush=True)

    # Show first/last 5 rows
    print("\nFirst 5 days:")
    print(results.head(5).to_string())
    print("\nLast 5 days:")
    print(results.tail(5).to_string())


if __name__ == "__main__":
    main()
