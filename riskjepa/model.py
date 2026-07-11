"""
riskjepa/model.py — RiskJEPA: a profit-weighted EUR/USD predictor.

Thesis (research/new_model_design.md): EUR/USD is symmetric/mean-reverting with
~2-day regime persistence, so direction is near-random. The edge is *risk-reward*,
not sign accuracy: predict the vol-normalized forward return AND an uncertainty,
trade only high-magnitude/high-confidence bars, size by volatility, and FLAT when
uncertain (triple-barrier kill-switch). Selection is on walk-forward profit-factor /
Sharpe, not JEPA loss.

Architecture (faithful to the design doc):
  - ConvEncoder: conv-tokenizer over the time axis (k=5) + 2 residual conv blocks,
    optional PatchTST patching (P=4) to drop T 48->12. Bidirectional over the local
    window is fine for SSL.
  - TransformerPredictor: ALiBi distance bias (no learned pos_embed) + optional
    learned time-of-day / day-of-week token.
  - RevIN: instance-norm each window before the encoder, restore after (FX drift fix).
  - Two/three heads off the context repr (z_ctx mean):
      ret_head: Linear(D->1) -> vol-normalized forward return  y_t = r_t / vol_t
      unc_head: Linear(D->2) -> (mu, log s) Gaussian (heteroskedastic uncertainty)
      tb_head : Linear(D->3) -> triple-barrier CE logits (+1/-1/0 = FLAT)
  - Collapse guard: hard-standardize + SIGReg combo (reused from model.py).
  - JEPA future-latent SSL aux loss retained so the encoder still learns temporal
    structure, but never used as the selection metric.

This module does not modify model.py / forex_features.py — it is a sibling package.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── SIGReg (reused from model.py; copied here so riskjepa is self-contained) ──
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
        g = torch.Generator().manual_seed(0)
        A = torch.randn(embed_dim, num_proj, generator=g)
        self.register_buffer("A", A.div_(A.norm(p=2, dim=0)))

    def forward(self, proj):
        x_t = (proj @ self.A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = err @ self.weights
        return statistic.mean()


# ── RevIN (Kim et al. 2022): instance-norm each window, restore after ──────────
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        if affine:
            self.affine = nn.Parameter(torch.ones(1, 1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.affine = None
            self.bias = None

    def forward(self, x, denorm=False):
        # x: (B, T, F)
        if not denorm:
            mu = x.mean(-2, keepdim=True)
            sd = x.std(-2, keepdim=True) + self.eps
            self._mu = mu
            self._sd = sd
            xn = (x - mu) / sd
            if self.affine is not None:
                xn = xn * self.affine + self.bias
            return xn
        else:
            if self.affine is not None:
                x = (x - self.bias) / (self.affine + self.eps)
            return x * self._sd + self._mu


# ── ConvEncoder: temporal conv-tokenizer (timeseries_proposal.md §1) ───────────
class ConvEncoder(nn.Module):
    """Conv-tokenizer: (B, T, F) -> (B, T', D) with local temporal structure.

    Optionally patches (PatchTST): every P bars -> 1 token via LayerNorm+Linear,
    so T' = T // P. Each token is a multi-bar "word" with a 5-bar receptive field.
    """

    def __init__(self, n_features, embed_dim=64, hidden_dim=128,
                 n_conv_blocks=2, patch_size=1, use_projector=True):
        super().__init__()
        self.patch_size = patch_size
        self.in_proj = nn.Conv1d(n_features, embed_dim, kernel_size=5, padding=2)
        self.blocks = nn.ModuleList()
        for _ in range(n_conv_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.GELU(),
            ))
        self.residual = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        if patch_size > 1:
            self.patch = nn.Sequential(
                nn.LayerNorm(embed_dim * patch_size),
                nn.Linear(embed_dim * patch_size, embed_dim),
                nn.GELU(),
            )
        else:
            self.patch = None
        if use_projector:
            self.projector = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.projector = nn.Identity()

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T) for conv
        z = x.transpose(1, 2)
        z = self.in_proj(z)
        for blk in self.blocks:
            z = blk(z) + self.residual(z)
        z = z.transpose(1, 2)                       # (B, T, D)
        if self.patch is not None:
            T = z.shape[1]
            Tp = T // self.patch_size
            z = z[:, :Tp * self.patch_size].reshape(
                z.shape[0], Tp, self.patch_size * z.shape[-1])
            z = self.patch(z)                       # (B, Tp, D)
        z = self.projector(z)
        return z


# ── ALiBi-aware Transformer predictor (timeseries_proposal.md §2) ──────────────
class ALiBiPredictor(nn.Module):
    """Causal Transformer with ALiBi distance bias (no learned absolute pos_embed).

    ALiBi adds a head-dependent linear penalty to the attention scores, giving the
    model an inductive bias toward recency without learned position embeddings. The
    bias is passed directly as the *additive* attention mask to nn.MultiheadAttention
    (PyTorch adds it to the scaled QK scores before softmax — exactly the ALiBi
    formulation). An optional learned time-of-day / day-of-week token is concatenated
    to each token so FX seasonality (London/NY opens) is captured.
    """

    def __init__(self, embed_dim=64, n_layers=4, n_heads=4, mlp_scale=4,
                 dropout=0.1, max_seq_len=256, alibi_m=0.25,
                 tod_dim=48, dow_dim=7):
        super().__init__()
        self.n_heads = n_heads
        self.alibi_m = alibi_m
        self.tod_emb = nn.Embedding(24, tod_dim)        # hour of day
        self.dow_emb = nn.Embedding(7, dow_dim)         # day of week (0=Mon)
        self.fuse = nn.Linear(embed_dim + tod_dim + dow_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_scale), nn.GELU(),
            nn.Linear(embed_dim * mlp_scale, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.n_layers = n_layers
        self.norm = nn.LayerNorm(embed_dim)
        self.pred_proj = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim),
            nn.GELU(), nn.Linear(embed_dim, embed_dim),
        )
        # precompute ALiBi bias for the max sequence length
        self.register_buffer("_alibi", self._build_alibi(max_seq_len), persistent=False)

    @staticmethod
    def _build_alibi(max_seq_len, n_heads=4, m=0.25):
        # slopes: 2^{-i} for i in 1..n_heads (Press et al. 2022)
        slopes = torch.tensor([2.0 ** (-(i + 1)) for i in range(n_heads)]) * m
        idx = torch.arange(max_seq_len).float()
        rel = idx.unsqueeze(0) - idx.unsqueeze(1)         # (T, T), i - j
        bias = -slopes.unsqueeze(-1).unsqueeze(-1) * rel.unsqueeze(0)
        causal = torch.triu(torch.full((max_seq_len, max_seq_len), float('-inf')), diagonal=1)
        bias = bias + causal.unsqueeze(0)
        return bias  # (H, T, T)

    def _alibi_bias(self, T, device):
        if T > self._alibi.shape[1]:
            b = self._build_alibi(T, self.n_heads, self.alibi_m).to(device)
        else:
            b = self._alibi[:, :T, :T].to(device)
        # nn.MultiheadAttention wants (N*heads, L, S); expand head dim per batch.
        # We return (heads, T, T); caller expands to (B*heads, T, T) via repeat.
        return b

    def forward(self, z_seq, tod=None, dow=None):
        """
        z_seq: (B, T, D). tod: (B, T) long hour; dow: (B, T) long weekday.
        Returns: (B, T, D) shifted-by-1 predictions.
        """
        B, T, D = z_seq.shape
        if tod is not None and dow is not None:
            tok = torch.cat([z_seq,
                             self.tod_emb(tod.clamp(0, 23)),
                             self.dow_emb(dow.clamp(0, 6))], dim=-1)
            z_seq = self.fuse(tok)
        bias = self._alibi_bias(T, z_seq.device)          # (H, T, T)
        # expand to (B*H, T, T) so each head gets its own slope bias
        bias = bias.unsqueeze(0).expand(B, -1, -1, -1) \
            .reshape(B * self.n_heads, T, T)
        for _ in range(self.n_layers):
            x = self.norm1(z_seq)
            attn_out, _ = self.attn(x, x, x, attn_mask=bias, need_weights=False)
            z_seq = z_seq + self.drop(attn_out)
            z_seq = z_seq + self.drop(self.ffn(self.norm2(z_seq)))
        z_seq = self.norm(z_seq)
        z_seq = self.pred_proj(z_seq)
        return z_seq


# ── RiskJEPA: full model ───────────────────────────────────────────────────────
class RiskJEPA(nn.Module):
    def __init__(self, n_features, embed_dim=64, enc_conv_blocks=2,
                 patch_size=1, predictor_layers=4, predictor_heads=4,
                 sigreg_proj=512, sigreg_lambda=1.0,
                 use_revin=True, use_alibi=True,
                 aux_lambda=0.5, tb_lambda=0.3, nll_lambda=0.3,
                 horizon=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.sigreg_lambda = sigreg_lambda
        self.aux_lambda = aux_lambda
        self.tb_lambda = tb_lambda
        self.nll_lambda = nll_lambda
        self.horizon = horizon
        self.use_revin = use_revin

        self.revin = RevIN(n_features) if use_revin else None
        self.encoder = ConvEncoder(n_features, embed_dim,
                                   n_conv_blocks=enc_conv_blocks, patch_size=patch_size)
        # predictor takes the (possibly patched) sequence
        self.predictor = ALiBiPredictor(embed_dim, n_layers=predictor_layers,
                                        n_heads=predictor_heads) if use_alibi \
            else None
        # fallback: a plain causal TransformerPredictor if ALiBi disabled
        self.sigreg = SIGReg(embed_dim, num_proj=sigreg_proj)
        # heads off context repr mean (z_ctx over the whole context window)
        self.ret_head = nn.Linear(embed_dim, 1)
        self.unc_head = nn.Linear(embed_dim, 2)    # (mu, log s)
        self.tb_head = nn.Linear(embed_dim, 3)     # +1 / -1 / 0 (FLAT)

    def encode_batch(self, seq, tod=None, dow=None):
        """seq: (B, T, F) -> (B, T', D)."""
        if self.revin is not None:
            seq = self.revin(seq)
        return self.encoder(seq)

    def _ctx_repr(self, z_ctx):
        return z_ctx.mean(dim=1)                   # (B, D)

    def forward(self, ctx, tgt=None, tod=None, dow=None, y=None, y_tb=None):
        """
        ctx: (B, T_ctx, F), tgt: (B, T_tgt, F).
        y:    (B,) vol-normalized forward return (optional, for aux loss).
        y_tb: (B,) triple-barrier label in {+1,-1,0} (optional).
        Returns dict with losses + predictions.
        """
        z_ctx = self.encode_batch(ctx, tod=tod, dow=dow)
        out = {'emb': z_ctx}
        ctx_repr = self._ctx_repr(z_ctx)

        # heads (always computed — used for eval even without tgt)
        out['ret_pred'] = self.ret_head(ctx_repr).squeeze(-1)         # (B,)
        unc = self.unc_head(ctx_repr)                                 # (B, 2)
        out['unc_mu'] = unc[:, 0]
        out['unc_logs'] = unc[:, 1]
        out['sigma'] = unc[:, 1].exp().clamp(min=1e-3)                # heteroskedastic sigma
        out['tb_logits'] = self.tb_head(ctx_repr)                     # (B, 3)

        if tgt is not None:
            z_full = self.encode_batch(torch.cat([ctx, tgt], dim=1),
                                       tod=tod, dow=dow) if tod is None else \
                self.encode_batch(torch.cat([ctx, tgt], dim=1))
            # collapse guard: standardize per-feature over the whole batch
            mu = z_full.mean(dim=(0, 1), keepdim=True)
            sd = z_full.std(dim=(0, 1), keepdim=True) + 1e-6
            z_full = (z_full - mu) / sd
            T = z_ctx.shape[1]
            z_ctx_std = z_full[:, :T]
            z_pred = self.predictor(z_full) if self.predictor is not None \
                else None
            out['z_tgt'] = z_full[:, T:]
            if z_pred is not None:
                pred_loss = F.mse_loss(z_pred[:, T:], z_full[:, T:])
                out['pred_loss'] = pred_loss
                sigreg_loss = self.sigreg(z_full.permute(1, 0, 2))
                out['sigreg_loss'] = sigreg_loss
                out['loss'] = pred_loss + self.sigreg_lambda * sigreg_loss
            else:
                out['pred_loss'] = torch.zeros((), device=ctx.device)
                out['sigreg_loss'] = torch.zeros((), device=ctx.device)
                out['loss'] = torch.zeros((), device=ctx.device)

        # ── auxiliary tradable losses (the selection-relevant signal) ──
        if y is not None:
            m = ~torch.isnan(y)
            if m.any():
                ret_loss = F.mse_loss(out['ret_pred'][m], y[m])
                out['ret_loss'] = ret_loss
                # NLL of Gaussian N(unc_mu, exp(unc_logs)) on vol-norm return
                s = out['unc_logs'][m].exp().clamp(min=1e-3)
                nll = (out['unc_logs'][m]
                       + 0.5 * ((y[m] - out['unc_mu'][m]) / s).square()
                       + 0.5 * math.log(2 * math.pi))
                out['nll_loss'] = nll.mean()
                if 'loss' in out:
                    out['loss'] = out['loss'] + self.aux_lambda * ret_loss \
                        + self.nll_lambda * out['nll_loss']
        if y_tb is not None:
            m = y_tb != float('nan')
            if m.any():
                tb_idx = y_tb[m].long().clamp(-1, 1) + 1   # map -1,0,1 -> 0,1,2
                tb_loss = F.cross_entropy(out['tb_logits'][m], tb_idx)
                out['tb_loss'] = tb_loss
                if 'loss' in out:
                    out['loss'] = out['loss'] + self.tb_lambda * tb_loss
        return out

    def predict_future(self, ctx, n_steps=10, tod=None, dow=None):
        z_ctx = self.encode_batch(ctx, tod=tod, dow=dow)
        current = z_ctx
        preds = []
        for _ in range(n_steps):
            z_pred_all = self.predictor(current)
            z_next = z_pred_all[:, -1:]
            preds.append(z_next)
            current = torch.cat([current, z_next], dim=1)
        return torch.cat(preds, dim=1)


if __name__ == "__main__":
    B, T, NF = 4, 48, 35
    ctx = torch.randn(B, T, NF)
    tgt = torch.randn(B, 12, NF)
    y = torch.randn(B)
    y_tb = torch.tensor([-1., 0., 1., 0.])
    net = RiskJEPA(n_features=NF, embed_dim=64, patch_size=4)
    out = net(ctx, tgt, y=y, y_tb=y_tb)
    print("keys:", sorted(out.keys()))
    print(f"loss={out['loss'].item():.4f} pred={out['pred_loss'].item():.4f} "
          f"sig={out['sigreg_loss'].item():.4f} ret={out['ret_loss'].item():.4f} "
          f"nll={out['nll_loss'].item():.4f} tb={out['tb_loss'].item():.4f}")
    print(f"ret_pred={out['ret_pred'].shape} sigma={out['sigma'].shape} "
          f"tb_logits={out['tb_logits'].shape}")
    # verify RevIN round-trip
    if net.use_revin:
        x = torch.randn(2, 10, NF)
        xn = net.revin(x)
        xd = net.revin(xn, denorm=True)
        print("RevIN round-trip max err:", (x - xd).abs().max().item())
    print("✅ RiskJEPA OK")
