"""LeWM-aligned JEPA for financial time series.
Architecture copied from lucas-maes/le-wm with stock-appropriate modifications.

Key differences from original le-wm:
- No action encoder (no actions in stock data)
- AdaLN-zero blocks in predictor use self-conditioning (prev embedding) instead of actions
- Embedder uses per-stock normalization instead of BatchNorm
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ── SIGReg: Sketched Isotropic Gaussian Regularizer ──

class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU).
    
    Source: lucas-maes/le-wm/module.py
    """
    def __init__(self, knots=17, num_proj=1024):
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
        """proj: (T, B, D) — time-first for sequence-level regularization"""
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


# ── Helper: AdaLN modulation ──

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


# ── FeedForward ──

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


# ── Attention ──

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, causal=True):
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


# ── ConditionalBlock: AdaLN-zero (key LeWM innovation) ──

class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning.
    
    The conditioning input c provides adaptive scale/shift/gate for each sub-layer.
    All modulation parameters are zero-initialized, so each block starts as identity.
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        """x: (B, T, D), c: (B, T, D) — conditioning from embeddings"""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ── Standard Block (no conditioning) ──

class Block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ── Transformer ──

class Transformer(nn.Module):
    """Transformer with support for both standard and Conditional blocks."""
    def __init__(self, input_dim, hidden_dim, output_dim, depth, heads,
                 dim_head=64, mlp_dim=None, dropout=0.0, conditional=True):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        block_class = ConditionalBlock if conditional else Block
        mlp_dim = mlp_dim or hidden_dim * 4
        
        self.input_proj = nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Identity() if hidden_dim == output_dim else nn.Linear(hidden_dim, output_dim)
        self.layers = nn.ModuleList([
            block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])

    def forward(self, x, c=None):
        x = self.input_proj(x)
        for layer in self.layers:
            if isinstance(layer, ConditionalBlock):
                # If no conditioning provided, use x itself as conditioning
                x = layer(x, c if c is not None else x)
            else:
                x = layer(x)
        x = self.norm(x)
        return self.output_proj(x)


# ── ARPredictor ──

class ARPredictor(nn.Module):
    """Autoregressive predictor with AdaLN-zero conditioning.
    
    Source: lucas-maes/le-wm/module.py ARPredictor
    For stock data, conditioning c = prev embedding (self-conditioning)
    """
    def __init__(self, num_frames, input_dim, hidden_dim, output_dim=None,
                 depth=6, heads=16, dim_head=64, mlp_dim=None, dropout=0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(0.0)  # emb_dropout
        self.transformer = Transformer(
            input_dim, hidden_dim, output_dim or input_dim,
            depth, heads, dim_head, mlp_dim, dropout, conditional=True,
        )

    def forward(self, x, c=None):
        """x: (B, T, D), c: (B, T, D) — conditioning (optional)"""
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


# ── Embedder (low-dim observation embedder) ──

class Embedder(nn.Module):
    """Conv1d + MLP embedder for low-dimensional observations.
    
    Source: lucas-maes/le-wm/module.py Embedder
    """
    def __init__(self, input_dim, emb_dim=192, mlp_scale=4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, emb_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(emb_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """x: (B, T, F)"""
        x = x.float()
        x = x.permute(0, 2, 1)       # (B, F, T)
        x = self.patch_embed(x)       # (B, D, T)
        x = x.permute(0, 2, 1)        # (B, T, D)
        x = self.embed(x)
        return x


# ── MLP Projector ──

class MLPProj(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, norm_fn=None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim * 4
        output_dim = output_dim or input_dim
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)


# ── Full LeWM-style JEPA for Stocks ──

class Fin-JEPA(nn.Module):
    """JEPA adapted for stock time series.
    
    Architecture: lucas-maes/le-wm JEPA
    - Encoder: Embedder (Conv1d + MLP) per-timestep
    - Predictor: ARPredictor with AdaLN-zero
    - SIGReg regularization
    - No action encoder (stocks have no actions)
    """
    def __init__(self, n_features=11, embed_dim=192, 
                 encoder_mlp_scale=4,
                 predictor_depth=6, predictor_heads=16, predictor_mlp_dim=None,
                 dim_head=64, dropout=0.1,
                 sigreg_proj=1024, sigreg_lambda=0.09,
                 history_size=3, pred_steps=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.sigreg_lambda = sigreg_lambda
        self.history_size = history_size
        self.pred_steps = pred_steps
        
        # Encoder: Embed features into latent space
        self.encoder = Embedder(n_features, embed_dim, encoder_mlp_scale)
        
        # Projector (applied after encoder, before predictor)
        self.projector = MLPProj(embed_dim, hidden_dim=embed_dim*4, norm_fn=nn.BatchNorm1d)
        
        # Predictor with AdaLN-zero
        pred_mlp = predictor_mlp_dim or embed_dim * 4
        self.predictor = ARPredictor(
            num_frames=history_size,
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            depth=predictor_depth,
            heads=predictor_heads,
            dim_head=dim_head,
            mlp_dim=pred_mlp,
            dropout=dropout,
        )
        
        # Prediction projection
        self.pred_proj = MLPProj(embed_dim)
        
        # SIGReg regularizer
        self.sigreg = SIGReg(num_proj=sigreg_proj)

    def encode(self, seq):
        """seq: (B, T, F) → (B, T, D)"""
        return self.encoder(seq)

    def forward(self, ctx, tgt=None):
        """ctx: (B, T, F), tgt: (B, pred, F)"""
        B, T, nF = ctx.shape
        
        # Encode all timesteps
        z = self.encoder(ctx)                 # (B, T, D)
        emb = self.projector(z.reshape(-1, self.embed_dim)).reshape(z.shape)  # (B, T, D)
        
        output = {'emb': emb}
        
        # Split into context and target
        ctx_emb = emb[:, :self.history_size]   # (B, H, D)
        
        if tgt is not None:
            z_tgt = self.encoder(tgt)
            tgt_emb = self.projector(z_tgt.reshape(-1, self.embed_dim)).reshape(z_tgt.shape)  # (B, pred, D)
            
            # Predict (self-conditioning: use ctx_emb as conditioning)
            pred_emb = self.predictor(ctx_emb, c=ctx_emb)     # (B, H, D)
            pred_emb = self.pred_proj(pred_emb.reshape(-1, self.embed_dim)).reshape(pred_emb.shape)
            
            # Compare predictions to targets
            n = min(self.history_size, tgt_emb.shape[1])
            pred_loss = F.mse_loss(pred_emb[:, :n], tgt_emb[:, :n])
            
            # SIGReg on all embeddings
            all_emb = torch.cat([emb, tgt_emb], dim=1)        # (B, T+pred, D)
            sigreg_loss = self.sigreg(all_emb.transpose(0, 1))  # (T+pred, B, D)
            
            output['pred_loss'] = pred_loss
            output['sigreg_loss'] = sigreg_loss
            output['loss'] = pred_loss + self.sigreg_lambda * sigreg_loss
        
        return output

    def predict_future(self, ctx, n_steps=10):
        """Autoregressive rollout."""
        B, T, F = ctx.shape
        z = self.encoder(ctx)
        emb = self.projector(z.reshape(-1, self.embed_dim)).reshape(z.shape)
        
        current = emb
        preds = []
        for _ in range(n_steps):
            ctx_emb = current[:, -self.history_size:] if current.shape[1] >= self.history_size else current
            z_pred = self.predictor(ctx_emb, c=ctx_emb)
            z_next = z_pred[:, -1:]  # (B, 1, D)
            z_next = self.pred_proj(z_next.reshape(-1, self.embed_dim)).reshape(z_next.shape)
            preds.append(z_next)
            current = torch.cat([current, z_next], dim=1)
        
        return torch.cat(preds, dim=1)


# ── Quick test ──
if __name__ == "__main__":
    model = Fin-JEPA(n_features=11, embed_dim=192, history_size=3, pred_steps=1)
    print(f"Fin-JEPA params: {sum(p.numel() for p in model.parameters()):,}")
    
    ctx = torch.randn(4, 3, 11)
    tgt = torch.randn(4, 1, 11)
    out = model(ctx, tgt)
    print(f"Forward: loss={out['loss']:.4f} (pred={out['pred_loss']:.4f}, sigreg={out['sigreg_loss']:.4f})")
    
    fut = model.predict_future(ctx, 5)
    print(f"Rollout: {fut.shape}")
    print("✅ Fin-JEPA OK")
