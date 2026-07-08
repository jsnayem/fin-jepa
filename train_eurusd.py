#!/usr/bin/env python3
"""
Fin-JEPA — Train on EUR/USD daily forex (Yahoo Finance, till 2025-12-31).
Output: checkpoints/eurusd/best.pt + meta.json + context.npy
"""

import math, json, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_DIR = Path(__file__).resolve().parent / "checkpoints" / "eurusd"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = ['open','high','low','close','volume','returns','vwap',
                'volatility_20','vol_ma_20','close_ma_20','range_pct']

# ──────────────────────────────────────────────
# Model (self-contained copy from model.py)
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
# Data
# ──────────────────────────────────────────────

class ForexDataset(Dataset):
    def __init__(self, sequences):
        self.data = sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return {
            'ctx': torch.FloatTensor(seq[:60]),
            'tgt': torch.FloatTensor(seq[60:65]),
        }


def fetch_eurusd():
    print("[1] Fetching EUR/USD daily data from Yahoo Finance (till 2025-12-31)...", flush=True)
    import yfinance as yf
    df = yf.download("EURUSD=X", start="2000-01-01", end="2025-12-31", progress=False)
    if df.empty:
        raise RuntimeError("No data returned from Yahoo Finance. Check internet connection.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    print(f"  Raw: {len(df):,} rows, {df['date'].min().date()} → {df['date'].max().date()}", flush=True)
    return df


def build_features(df):
    print("[2] Computing technical features...", flush=True)
    df = df.sort_values('date').reset_index(drop=True)
    df['returns'] = df['close'].pct_change()
    df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['vol_ma_20'] = df['volume'] / df['volume'].rolling(20).mean()
    df['close_ma_20'] = df['close'] / df['close'].rolling(20).mean()
    df['range_pct'] = (df['high'] - df['low']) / df['close']

    # Handle NaN/Inf from division by zero (forex volume is often 0)
    df['vol_ma_20'] = df['vol_ma_20'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df = df.dropna(subset=FEATURE_COLS)
    print(f"  After feature computation: {len(df):,} rows", flush=True)
    return df


def normalize_and_sequence(df):
    print("[3] Normalizing and building sequences...", flush=True)

    vals = df[FEATURE_COLS].values.astype(np.float32)
    means = np.nanmean(vals, axis=0)
    stds = np.nanstd(vals, axis=0) + 1e-8

    normalized = (vals - means) / stds
    df_norm = df.copy()
    for i, c in enumerate(FEATURE_COLS):
        df_norm[c] = normalized[:, i]

    print(f"  Means:  {dict(zip(FEATURE_COLS, means.round(4)))}", flush=True)
    print(f"  Stds:   {dict(zip(FEATURE_COLS, stds.round(4)))}", flush=True)

    # Build sliding windows (ctx=60, tgt=5, stride=5)
    seqs = []
    v = df_norm[FEATURE_COLS].values.astype(np.float32)
    for i in tqdm(range(0, max(1, len(v) - 65 + 1), 5), desc="  Sequencing"):
        chunk = v[i:i+65]
        if len(chunk) == 65 and not np.isnan(chunk).any():
            seqs.append(chunk)

    seq = np.stack(seqs).astype(np.float32)
    print(f"  {len(seq):,} sequences × {seq.shape[1]} steps × {seq.shape[2]} features", flush=True)

    return seq, means, stds, df_norm


def split_data(sequences):
    n = len(sequences)
    split_idx = int(n * 0.80)
    train_seq = sequences[:split_idx]
    val_seq = sequences[split_idx:]
    print(f"  Train: {len(train_seq):,}  Val: {len(val_seq):,}", flush=True)

    # Save last context window for prediction
    last_ctx = sequences[-1, :60]
    trainset = ForexDataset(train_seq) if len(train_seq) > 0 else None
    valset = ForexDataset(val_seq) if len(val_seq) > 0 else None
    return trainset, valset, last_ctx


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_epoch(model, loader, opt, scaler, use_amp, desc="train"):
    model.train()
    total = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        ctx, tgt = batch['ctx'].to(DEVICE), batch['tgt'].to(DEVICE)
        opt.zero_grad()
        with torch.amp.autocast(device_type=DEVICE, enabled=use_amp):
            out = model(ctx, tgt)
        scaler.scale(out['loss']).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        total += out['loss'].item()
        pbar.set_postfix(loss=f"{out['loss'].item():.4f}")
    return total / len(loader)


@torch.no_grad()
def eval_loss(model, loader, desc="val"):
    model.eval()
    total = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        ctx, tgt = batch['ctx'].to(DEVICE), batch['tgt'].to(DEVICE)
        out = model(ctx, tgt)
        total += out['loss'].item()
        pbar.set_postfix(loss=f"{out['loss'].item():.4f}")
    return total / len(loader)


def train_model(train_loader, val_loader, epochs=200, embed_dim=64,
                enc_layers=4, pred_layers=6, sigreg_proj=128,
                sigreg_lambda=0.1, lr=1e-3, batch_size=128):
    n_features = len(FEATURE_COLS)
    print(f"\n[4] Building model: {n_features}→{embed_dim}d, enc={enc_layers}layers, pred={pred_layers}layers", flush=True)

    model = FinJEPA(
        n_features, embed_dim,
        encoder_layers=enc_layers,
        predictor_layers=pred_layers,
        sigreg_proj=sigreg_proj,
        sigreg_lambda=sigreg_lambda,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,} | Device: {DEVICE}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler(device=DEVICE, enabled=use_amp)

    print(f"\n[5] Training {epochs} epochs (bs={batch_size}, lr={lr})...", flush=True)
    history = {'train': [], 'val': []}
    best_val = float('inf')
    t0 = time.time()

    epoch_pbar = tqdm(range(epochs), desc="Epochs")
    for epoch in epoch_pbar:
        tr_l = train_epoch(model, train_loader, opt, scaler, use_amp, desc=f"  epoch {epoch:3d} train")
        va_l = eval_loss(model, val_loader, desc=f"  epoch {epoch:3d} val  ")
        history['train'].append(tr_l)
        history['val'].append(va_l)
        sched.step()

        if va_l < best_val:
            best_val = va_l
            torch.save({
                'model_state_dict': model.state_dict(),
                'embed_dim': embed_dim,
                'enc_layers': enc_layers,
                'pred_layers': pred_layers,
                'sigreg_proj': sigreg_proj,
                'sigreg_lambda': sigreg_lambda,
                'epoch': epoch,
                'val_loss': va_l,
            }, OUTPUT_DIR / "best.pt")

        epoch_pbar.set_postfix(tr=f"{tr_l:.4f}", va=f"{va_l:.4f}", best=f"{best_val:.4f}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed/60:.1f}min | Best val loss: {best_val:.4f}", flush=True)
    return model, best_val, history


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60, flush=True)
    print("Fin-JEPA — EUR/USD Forex Training")
    print(f"Device: {DEVICE}")
    print("=" * 60, flush=True)

    # 1. Fetch data
    df_raw = fetch_eurusd()

    # 2. Features
    df_feat = build_features(df_raw)

    # 3. Normalize + sequences
    sequences, means, stds, df_norm = normalize_and_sequence(df_feat)

    # 4. Split
    trainset, valset, last_ctx = split_data(sequences)

    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=0)

    # 5. Train
    model, best_val, history = train_model(
        train_loader, val_loader,
        epochs=100, embed_dim=32, enc_layers=3, pred_layers=4,
        sigreg_proj=64, sigreg_lambda=0.1, lr=1e-3, batch_size=128,
    )

    # 6. Save artifacts
    print("\n[6] Saving artifacts...", flush=True)

    np.save(OUTPUT_DIR / "last_context.npy", last_ctx)

    meta = {
        'description': 'Fin-JEPA trained on EUR/USD daily forex (till 2025-12-31)',
        'symbol': 'EURUSD=X',
        'features': FEATURE_COLS,
        'n_features': len(FEATURE_COLS),
        'normalizer_mean': means.tolist(),
        'normalizer_std': stds.tolist(),
        'n_sequences': len(sequences),
        'n_train': len(trainset),
        'n_val': len(valset),
        'context_len': 60,
        'target_len': 5,
        'stride': 5,
        'embed_dim': 32,
        'encoder_layers': 3,
        'predictor_layers': 4,
        'best_val_loss': float(best_val),
        'device': DEVICE,
    }
    json.dump(meta, open(OUTPUT_DIR / "meta.json", "w"), indent=2)
    json.dump(history, open(OUTPUT_DIR / "history.json", "w"), indent=2)

    print(f"  Checkpoint: {OUTPUT_DIR / 'best.pt'}")
    print(f"  Metadata:   {OUTPUT_DIR / 'meta.json'}")
    print(f"  Context:    {OUTPUT_DIR / 'last_context.npy'} (shape={last_ctx.shape})")
    print(f"\n{'='*60}", flush=True)
    print("✅ Training complete. Ready for prediction.")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
