"""
Fin-JEPA — adapted from LeWorldModel (Maes+ 2026) for financial time series.
"""
import math
import torch
import torch.nn as nn
from einops import rearrange


# SIGReg — Sketched Isotropic Gaussian Regularizer
class SIGReg(nn.Module):
    def __init__(self, embed_dim, knots=17, num_proj=512):
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
        # Fixed random projection (sampled ONCE, not resampled every call) so the
        # regularizer gradient is stable. Previously torch.randn each forward made
        # the gradient noisy AND it was scaled by proj.size(-2) (~144x), dominating
        # the loss. Both issues are fixed here.
        g = torch.Generator().manual_seed(0)
        A = torch.randn(embed_dim, num_proj, generator=g)
        self.register_buffer("A", A.div_(A.norm(p=2, dim=0)))

    def forward(self, proj):
        x_t = (proj @ self.A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights)  # no * proj.size(-2) scaling
        return statistic.mean()


# Encoder — per-day feature extractor (like LeWM's ViT processes each frame)
class PriceEncoder(nn.Module):
    """Maps (B, 1, F) day features → (B, D) embedding."""
    def __init__(self, n_features, embed_dim=64, hidden_dim=128, n_layers=3, n_heads=3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, embed_dim)
        # Simple MLP encoder (1D conv would work too)
        self.encoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.projector = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        """x: (B, 1, F) → (B, D)"""
        x = self.input_proj(x.squeeze(1))  # (B, D)
        x = self.encoder(x)
        x = self.projector(x)
        return x


# Transformer Predictor (LeWM-style autoregressive)
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
        """
        z_seq: (B, T, D)
        Returns: (B, T, D) — shifted-by-1 predictions
        """
        B, T, D = z_seq.shape
        x = z_seq + self.pos_embed[:, :T, :]
        x = self.dropout(x)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=z_seq.device), diagonal=1)
        for block in self.blocks:
            x = block(x, src_mask=mask, is_causal=False)
        x = self.norm(x)
        x = self.pred_proj(x)
        return x


# Full Fin-JEPA
class FinJEPA(nn.Module):
    def __init__(self, n_features, embed_dim=64, encoder_layers=3, encoder_heads=3,
                 predictor_layers=4, predictor_heads=4, sigreg_proj=512, sigreg_lambda=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.sigreg_lambda = sigreg_lambda
        self.encoder = PriceEncoder(n_features, embed_dim, n_layers=encoder_layers, n_heads=encoder_heads)
        self.predictor = TransformerPredictor(embed_dim, predictor_layers, predictor_heads)
        self.sigreg = SIGReg(embed_dim, num_proj=sigreg_proj)

    def encode_batch(self, seq):
        """seq: (B, T, F) → (B, T, D) per-timestep embeddings (batched)"""
        B, T, F = seq.shape
        # Flatten batch & time: process all timesteps at once
        z_all = self.encoder(seq.reshape(-1, 1, F).squeeze(1))  # (B*T, D)
        return z_all.reshape(B, T, -1)  # (B, T, D)

    def forward(self, ctx, tgt=None):
        """ctx: (B, T_ctx, F), tgt: (B, T_tgt, F) → dict with losses.

        JEPA objective: encode the JOINT context+target sequence and have the
        causal predictor forecast the FUTURE latents (the target region), not the
        same-timestep latent (in-painting). This is what makes the encoder learn
        temporal structure.
        """
        z_ctx = self.encode_batch(ctx)
        output = {'emb': z_ctx}

        if tgt is not None:
            z_full = self.encode_batch(torch.cat([ctx, tgt], dim=1))  # (B, T_ctx+T_tgt, D)
            # Prevent collapse structurally: standardize embeddings to zero-mean,
            # unit-variance (per-feature, over the whole batch). SIGReg enforces
            # isotropy, so together they target N(0,I); a constant (collapsed)
            # embedding would have std->0 and blow up under this normalization,
            # so the encoder is forced to keep informative, varying latents.
            mu = z_full.mean(dim=(0, 1), keepdim=True)
            sd = z_full.std(dim=(0, 1), keepdim=True) + 1e-6
            z_full = (z_full - mu) / sd
            T = ctx.shape[1]
            z_ctx = z_full[:, :T]
            z_pred = self.predictor(z_full)                            # (B, T_ctx+T_tgt, D)
            output['emb'] = z_ctx
            output['pred'] = z_pred
            output['z_tgt'] = z_full[:, T:]                             # standardized actual future latents
            pred_loss = torch.nn.functional.mse_loss(z_pred[:, T:], z_full[:, T:])
            output['pred_loss'] = pred_loss
            sigreg_loss = self.sigreg(z_full.permute(1, 0, 2))
            output['sigreg_loss'] = sigreg_loss
            output['loss'] = pred_loss + self.sigreg_lambda * sigreg_loss

        return output

    def predict_future(self, ctx, n_steps=10):
        """Autoregressive rollout."""
        z_ctx = self.encode_batch(ctx)
        current = z_ctx
        preds = []
        for _ in range(n_steps):
            z_pred_all = self.predictor(current)
            z_next = z_pred_all[:, -1:]
            preds.append(z_next)
            current = torch.cat([current, z_next], dim=1)
        return torch.cat(preds, dim=1)


# Probing heads
class LinearProbe(nn.Module):
    def __init__(self, embed_dim, output_dim=1):
        super().__init__()
        self.probe = nn.Linear(embed_dim, output_dim)

    def forward(self, z):
        return self.probe(z)

class MLPProbe(nn.Module):
    def __init__(self, embed_dim, hidden_dim=32, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


if __name__ == "__main__":
    model = FinJEPA(n_features=11, embed_dim=64)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    ctx = torch.randn(4, 60, 11)
    tgt = torch.randn(4, 5, 11)
    out = model(ctx, tgt)
    print(f"Forward: loss={out['loss']:.4f} (pred={out['pred_loss']:.4f}, sigreg={out['sigreg_loss']:.4f})")
    fut = model.predict_future(ctx, 10)
    print(f"Rollout: {fut.shape}")

    sigreg = SIGReg(num_proj=128)
    print(f"SIGReg random={sigreg(torch.randn(65,4,64)):.4f} gaussian={sigreg(torch.randn(65,4,64)):.4f}")
    print("✅ Model OK")
