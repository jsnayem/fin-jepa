"""Fin-JEPA Architecture Comparison v2 — encoder variants + density sweep."""
import os, sys, time, json, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.expanduser("~/dev/chan-jepa"))
from data import download_hs20, add_features, JEPADataset

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BASE = Path(__file__).resolve().parent / "output"

# ── Encoder variants ──

class PriceEncoder_MLPDeep(nn.Module):
    """MLP encoder that actually uses n_layers."""
    def __init__(self, n_features, embed_dim=64, n_layers=3):
        super().__init__()
        layers = [nn.Linear(n_features, embed_dim), nn.LayerNorm(embed_dim)]
        for _ in range(n_layers):
            layers += [nn.Linear(embed_dim, embed_dim*2), nn.GELU(), nn.Linear(embed_dim*2, embed_dim), nn.LayerNorm(embed_dim)]
        self.net = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, x):
        return self.proj(self.net(x.squeeze(1)))


class PriceEncoder_CNN(nn.Module):
    """1D-CNN encoder (V-JEPA style backbone)."""
    def __init__(self, n_features, embed_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, embed_dim//2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(embed_dim//2, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, x):
        # x: (B, 1, F) → (B, F) → (B, 1, F) for conv1d
        feat = x.squeeze(1).unsqueeze(-1)  # (B, F, 1) — treating each feature as channel
        c = self.conv(feat).squeeze(-1)  # (B, D)
        return self.proj(c)


class PriceEncoder_LSTM(nn.Module):
    """LSTM encoder."""
    def __init__(self, n_features, embed_dim=64, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, embed_dim, n_layers, batch_first=True, bidirectional=False)
        self.proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, x):
        # x: (B, 1, F) — need to expand to sequence... this won't work with single step
        # Instead: each "day" is a single vector, LSTM doesn't apply
        # So LSTM encoder doesn't make sense for per-day encoding
        # Keep as MLP fallback
        return self.proj(x.squeeze(1))


# ── Predictor variants ──

class MLPPredictor(nn.Module):
    """Simple MLP predictor (no attention)."""
    def __init__(self, embed_dim=64, n_layers=4):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers += [nn.Linear(embed_dim, embed_dim*2), nn.GELU(), nn.Linear(embed_dim*2, embed_dim)]
        self.net = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, z):
        B, T, D = z.shape
        x = z.reshape(B*T, D)
        x = self.net(x).reshape(B, T, D)
        return self.proj(x)


# ── SIGReg ──

class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=128):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3/(knots-1)
        w = torch.full((knots,), 2*dt); w[[0,-1]] = dt
        window = torch.exp(-t.square()/2)
        self.register_buffer("t", t); self.register_buffer("phi", window)
        self.register_buffer("weights", w*window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A.div_(A.norm(p=2, dim=0))
        x_t = (proj@A).unsqueeze(-1)*self.t
        err = (x_t.cos().mean(-3)-self.phi).square() + x_t.sin().mean(-3).square()
        return (err@self.weights).mean()*proj.size(-2)


# ── Predictor (Transformer, from model.py) ──

class TransformerPredictor(nn.Module):
    def __init__(self, embed_dim=64, n_layers=4, n_heads=4):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, 256, embed_dim)*0.02)
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, n_heads, embed_dim*4, 0.1, 'gelu', batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, z):
        B, T, D = z.shape
        x = z + self.pos[:, :T]
        x = self.drop(x)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=z.device), diagonal=1)
        for b in self.blocks:
            x = b(x, src_mask=mask, is_causal=False)
        return self.proj(self.norm(x))


# ── JEPA Model ──

class JEPA(nn.Module):
    def __init__(self, encoder, predictor, embed_dim=64, sigreg_proj=128, sigreg_lambda=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.lam = sigreg_lambda
        self.encoder = encoder
        self.predictor = predictor
        self.sigreg = SIGReg(num_proj=sigreg_proj)

    def encode(self, seq):
        B, T, F = seq.shape
        z = self.encoder(seq.reshape(-1, 1, F).squeeze(1))
        return z.reshape(B, T, -1)

    def forward(self, ctx, tgt=None):
        B, T, F = ctx.shape
        z_ctx = self.encode(ctx)
        z_pred = self.predictor(z_ctx)
        out = {'emb': z_ctx}
        if tgt is not None:
            z_tgt = self.encode(tgt)
            n = min(T, tgt.shape[1])
            pl = nn.functional.mse_loss(z_pred[:, :n], z_tgt[:, :n])
            fe = torch.cat([z_ctx, z_tgt], 1)
            sl = self.sigreg(fe.permute(1, 0, 2))
            out.update(pred_loss=pl.item(), sigreg_loss=sl.item(), loss=pl + self.lam * sl)
        return out


# ── Variants ──

def make_encoder(name):
    """Return (encoder_fn, desc) or None."""
    m = {
        # -- Density variants (standard architecture, different d) --
        # Uses the original model.py's approach: shallow MLP + TransformerPred
        'density_d48': lambda nf=11, d=48: (PriceEncoder_MLPDeep(nf, d, n_layers=1), TransformerPredictor(d, 4), d, 64),
        'density_d80': lambda nf=11, d=80: (PriceEncoder_MLPDeep(nf, d, n_layers=1), TransformerPredictor(d, 4), d, 128),

        # -- Deep encoder variants --
        'enc_deep6':  lambda nf=11: (PriceEncoder_MLPDeep(nf, 64, n_layers=6), TransformerPredictor(64, 4), 64, 128),
        'enc_deep8':  lambda nf=11: (PriceEncoder_MLPDeep(nf, 64, n_layers=8), TransformerPredictor(64, 4), 64, 128),

        # -- Predictor depth --
        'pred_l3':  lambda nf=11: (PriceEncoder_MLPDeep(nf, 64, n_layers=1), TransformerPredictor(64, 3), 64, 128),
        'pred_l5':  lambda nf=11: (PriceEncoder_MLPDeep(nf, 64, n_layers=1), TransformerPredictor(64, 5), 64, 128),
        'pred_l8':  lambda nf=11: (PriceEncoder_MLPDeep(nf, 64, n_layers=1), TransformerPredictor(64, 8), 64, 128),

        # -- CNN encoder --
        'cnn_enc':  lambda nf=11: (PriceEncoder_CNN(nf, 64), TransformerPredictor(64, 4), 64, 128),

        # -- MLP predictor (no attention) --
        'mlp_pred': lambda nf=11: (PriceEncoder_MLPDeep(nf, 64, n_layers=1), MLPPredictor(64, 4), 64, 128),

        # -- Encoder-heavy: deep encoder + shallow predictor --
        'enc6_pred2': lambda nf=11: (PriceEncoder_MLPDeep(nf, 64, n_layers=6), TransformerPredictor(64, 2), 64, 128),

        # -- CNN encoder + deeper predictor --
        'cnn_pred6': lambda nf=11: (PriceEncoder_CNN(nf, 64), TransformerPredictor(64, 6), 64, 128),
    }
    return m.get(name)


VARIANTS = [
    # Density
    'density_d48', 'density_d80',
    # Deep encoder
    'enc_deep6', 'enc_deep8',
    # Predictor depth
    'pred_l3', 'pred_l5', 'pred_l8',
    # CNN encoder
    'cnn_enc',
    # MLP predictor
    'mlp_pred',
    # Encoder-heavy
    'enc6_pred2',
    # CNN + deep predictor
    'cnn_pred6',
]


def run_variant(name, n_epochs=50):
    exp_dir = BASE / f"arch_{name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    if (exp_dir / "meta.json").exists():
        print(f"  Skipping {name} (already exists)")
        return json.load(open(exp_dir / "meta.json"))

    builder = make_encoder(name)
    enc, pred, D, nproj = builder()

    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print(f"  encoder={type(enc).__name__} predictor={type(pred).__name__} D={D} proj={nproj}")

    # Data
    df = download_hs20(); df = add_features(df).dropna()
    dates = sorted(df['date'].unique()); n = len(dates)
    tr_df = df[df['date'] < dates[int(n*0.80)]]
    va_df = df[(df['date'] >= dates[int(n*0.80)]) & (df['date'] < dates[int(n*0.95)])]
    tr_ds = JEPADataset(tr_df, 60, 5, 5)
    va_ds = JEPADataset(va_df, 60, 5, 5, normalizer=(tr_ds.mean, tr_ds.std))
    tr_loader = DataLoader(tr_ds, 64, True, num_workers=0)
    va_loader = DataLoader(va_ds, 64, False, num_workers=0)

    # Model
    nf = tr_ds.n_features
    # Rebuild with correct n_features
    if 'cnn' in name:
        enc = PriceEncoder_CNN(nf, D)
    elif 'mlp' in name:
        enc = PriceEncoder_MLPDeep(nf, D, n_layers=1)
    else:
        # For density/enc variants, rebuild with the right encoder
        pass  # Already built with default nf=11

    # Actually, the lambda captures nf=11 by default. Let me just set n_features.
    # This is getting complicated. Let me simplify by rebuilding.
    enc_map = {
        'density_d48': PriceEncoder_MLPDeep(nf, 48, 1),
        'density_d80': PriceEncoder_MLPDeep(nf, 80, 1),
        'enc_deep6': PriceEncoder_MLPDeep(nf, 64, 6),
        'enc_deep8': PriceEncoder_MLPDeep(nf, 64, 8),
        'pred_l3': PriceEncoder_MLPDeep(nf, 64, 1),
        'pred_l5': PriceEncoder_MLPDeep(nf, 64, 1),
        'pred_l8': PriceEncoder_MLPDeep(nf, 64, 1),
        'cnn_enc': PriceEncoder_CNN(nf, 64),
        'mlp_pred': PriceEncoder_MLPDeep(nf, 64, 1),
        'enc6_pred2': PriceEncoder_MLPDeep(nf, 64, 6),
        'cnn_pred6': PriceEncoder_CNN(nf, 64),
    }
    pred_map = {
        'density_d48': TransformerPredictor(48, 4),
        'density_d80': TransformerPredictor(80, 4),
        'enc_deep6': TransformerPredictor(64, 4),
        'enc_deep8': TransformerPredictor(64, 4),
        'pred_l3': TransformerPredictor(64, 3),
        'pred_l5': TransformerPredictor(64, 5),
        'pred_l8': TransformerPredictor(64, 8),
        'cnn_enc': TransformerPredictor(64, 4),
        'mlp_pred': MLPPredictor(64, 4),
        'enc6_pred2': TransformerPredictor(64, 2),
        'cnn_pred6': TransformerPredictor(64, 6),
    }
    D_map = {'density_d48': 48, 'density_d80': 80}
    D = D_map.get(name, 64)
    proj_map = {'density_d48': 64, 'density_d80': 128}
    nproj = proj_map.get(name, 128)

    model = JEPA(enc_map[name], pred_map[name], D, nproj).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # Train
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    best_va = float('inf')
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        tr_l = 0
        for batch in tr_loader:
            ctx, tgt = batch['ctx'].to(DEVICE), batch['tgt'].to(DEVICE)
            opt.zero_grad()
            out = model(ctx, tgt)
            out['loss'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_l += out['loss'].item()
        sched.step()

        model.eval()
        va_l = 0
        with torch.no_grad():
            for batch in va_loader:
                ctx, tgt = batch['ctx'].to(DEVICE), batch['tgt'].to(DEVICE)
                out = model(ctx, tgt)
                va_l += out['loss'].item()

        if va_l < best_va:
            best_va = va_l
            torch.save(model.state_dict(), exp_dir / "best.pt")

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  {epoch:3d}/{n_epochs} | tr={tr_l/len(tr_loader):.4f} va={va_l/len(va_loader):.4f} | {time.time()-t0:.0f}s")

    elapsed = time.time() - t0
    print(f"  Done: {elapsed:.0f}s ({elapsed/n_epochs:.1f}s/epoch)")

    meta = {
        'name': name,
        'encoder': type(enc_map[name]).__name__,
        'predictor': type(pred_map[name]).__name__,
        'embed_dim': D, 'sigreg_proj': nproj,
        'params': n_params, 'epochs': n_epochs,
        'elapsed': elapsed, 'best_va_loss': best_va/len(va_loader),
    }
    json.dump(meta, open(exp_dir / "meta.json", "w"), indent=2)
    return meta


if __name__ == "__main__":
    print("═"*60)
    print("Architecture Comparison v2")
    print("═"*60)
    print(f"Device: {DEVICE}")
    print(f"Variants: {len(VARIANTS)}")
    for v in VARIANTS:
        print(f"  - {v}: {make_encoder(v).__doc__ or ''}")

    results = []
    for v in VARIANTS:
        meta = run_variant(v, n_epochs=50)
        results.append(meta)

    print("\n" + "═"*60)
    print("Results Summary v2")
    print("═"*60)
    for r in results:
        print(f"  {r['name']:15s} | {r['encoder']:20s} {r['predictor']:20s} | "
              f"D={r['embed_dim']:2d} | params={r['params']:>7,} | "
              f"va={r['best_va_loss']:.4f} | {r['elapsed']:.0f}s")
