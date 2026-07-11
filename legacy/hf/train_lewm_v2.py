"""LeWM-style two-phase training for Fin-JEPA on HF Jobs.

Experiment B: Two-phase paradigm
  Phase 1: Train encoder only (SIGReg loss, no predictor)
  Phase 2: Load Phase 1 checkpoint, freeze encoder, train predictor only

Based on train_lewm.py (end-to-end baseline).
"""
# /// script
# dependencies = [
#   "torch>=2.0",
#   "huggingface_hub",
#   "numpy",
#   "einops",
# ]
# ///

import os, sys, json, time, warnings, math, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from argparse import ArgumentParser
from huggingface_hub import snapshot_download, upload_folder
from einops import rearrange
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DATA_REPO = "cedwyh/fin-jepa-h3"
BASE = Path("/tmp/fin-jepa-lewm")
BASE.mkdir(exist_ok=True)


# ═══ Model: LeWM-aligned Fin-JEPA ═══

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=1024):
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


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head*heads; self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
    def forward(self, x, causal=True):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q,k,v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.to_out(rearrange(out, "b h t d -> b t (h d)"))


class ConditionalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6*dim))
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)
    def forward(self, x, c):
        s1,s2,g1,s3,s4,g2 = self.adaLN(c).chunk(6, dim=-1)
        x = x + g1 * self.attn(modulate(self.norm1(x), s1, s2))
        x = x + g2 * self.mlp(modulate(self.norm2(x), s3, s4))
        return x


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, heads,
                 dim_head=64, mlp_dim=None, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        mlp_dim = mlp_dim or hidden_dim*4
        self.input_proj = nn.Identity() if input_dim==hidden_dim else nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Identity() if hidden_dim==output_dim else nn.Linear(hidden_dim, output_dim)
        self.layers = nn.ModuleList([ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)])
    def forward(self, x, c=None):
        x = self.input_proj(x)
        for layer in self.layers: x = layer(x, c if c is not None else x)
        return self.output_proj(self.norm(x))


class ARPredictor(nn.Module):
    def __init__(self, num_frames, input_dim, hidden_dim, output_dim=None, depth=6, heads=16, dim_head=64, mlp_dim=None, dropout=0.1):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, num_frames, input_dim)*0.02)
        self.drop = nn.Dropout(0.0)
        self.transformer = Transformer(input_dim, hidden_dim, output_dim or input_dim, depth, heads, dim_head, mlp_dim, dropout)
    def forward(self, x, c=None):
        x = x + self.pos[:, :x.size(1)]
        return self.transformer(self.drop(x), c)


class Embedder(nn.Module):
    def __init__(self, input_dim, emb_dim=192, scale=4):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, emb_dim, 1)
        self.mlp = nn.Sequential(nn.Linear(emb_dim, scale*emb_dim), nn.SiLU(), nn.Linear(scale*emb_dim, emb_dim))
    def forward(self, x):
        return self.mlp(self.conv(x.permute(0,2,1)).permute(0,2,1))


class MLPProj(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim*4; output_dim = output_dim or input_dim
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))
    def forward(self, x): return self.net(x)


class Fin-JEPA(nn.Module):
    def __init__(self, n_features=11, embed_dim=192, pred_depth=6, pred_heads=16,
                 sigreg_proj=1024, sigreg_lambda=0.09, history_size=3,
                 target_ln=True, smooth_l1=True, self_cond=True):
        super().__init__()
        self.embed_dim = embed_dim; self.sigreg_lambda = sigreg_lambda; self.history_size = history_size
        self.target_ln = target_ln; self.smooth_l1 = smooth_l1; self.self_cond = self_cond
        self.encoder = Embedder(n_features, embed_dim)
        self.projector = MLPProj(embed_dim)
        self.predictor = ARPredictor(history_size, embed_dim, embed_dim, depth=pred_depth, heads=pred_heads)
        self.pred_proj = MLPProj(embed_dim)
        self.sigreg = SIGReg(num_proj=sigreg_proj)

    def encode(self, seq):
        z = self.encoder(seq)
        return self.projector(z.reshape(-1,self.embed_dim)).reshape(z.shape)

    def forward_encoder_only(self, ctx, tgt=None):
        """Phase 1 forward: encoder + SIGReg only, no predictor.

        Compatible with standard training loop signature (ctx, tgt).
        tgt is ignored when provided.
        """
        emb = self.encode(ctx)
        # SIGReg on encoder embeddings to prevent collapse
        sl = self.sigreg(emb.transpose(0, 1))
        return {'emb': emb, 'sigreg_loss': sl, 'loss': self.sigreg_lambda * sl}

    def forward(self, ctx, tgt=None):
        B, T = ctx.shape[:2]
        emb = self.encode(ctx)
        out = {'emb': emb}
        if tgt is not None:
            tgt_emb = self.encode(tgt)
            # Target LayerNorm (vjepa2 trick): normalize target embeddings
            if self.target_ln:
                tgt_emb = F.layer_norm(tgt_emb, (tgt_emb.shape[-1],))

            pred = self.pred_proj(
                self.predictor(emb[:,:self.history_size],
                    emb[:,:self.history_size] if self.self_cond else None)
                .reshape(-1,self.embed_dim)
            ).reshape(emb[:,:self.history_size].shape)

            n = min(self.history_size, tgt_emb.shape[1])
            if self.smooth_l1:
                pl = F.smooth_l1_loss(pred[:,:n], tgt_emb[:,:n], beta=1.0)
            else:
                pl = F.mse_loss(pred[:,:n], tgt_emb[:,:n])

            sl = self.sigreg(torch.cat([emb,tgt_emb],1).transpose(0,1))
            out.update(pred_loss=pl, sigreg_loss=sl, loss=pl+self.sigreg_lambda*sl)
        return out


# ═══ Dataset (proper chronological split) ═══

class SeqDataset(Dataset):
    def __init__(self, data_dir, prefix="shard", history_size=3):
        """prefix='shard' for train, prefix='val' for validation"""
        meta = json.load(open(Path(data_dir)/"meta.json"))
        shard_files = sorted(Path(data_dir).glob(f"{prefix}_*.npy"))
        self.data = np.concatenate([np.load(f).astype(np.float32) for f in shard_files], axis=0)
        self.nf = self.data.shape[-1]
        self.hist = history_size
        total_steps = self.data.shape[1]
        print(f"  {prefix}: {len(self.data):,} seqs x {total_steps} steps x {self.nf} features | hist={history_size}", flush=True)
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        s = self.data[i]
        return {'ctx': torch.FloatTensor(s[:self.hist]), 'tgt': torch.FloatTensor(s[self.hist:self.hist+1])}


# ═══ Training ═══

class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps; self.max_steps = max_steps
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return [base_lr * step / max(1, self.warmup_steps) for base_lr in self.base_lrs]
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        return [base_lr * 0.5 * (1 + math.cos(math.pi * progress)) for base_lr in self.base_lrs]


def param_groups(model, wd=1e-3):
    """Split params into high-dim (with decay) and low-dim+bias (no decay).
    From keon/jepa leworldmodel.py
    """
    d = []; nd = []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim < 2 or n.endswith("bias"):
            nd.append(p)
        else:
            d.append(p)
    return [
        {"params": d, "weight_decay": wd},
        {"params": nd, "weight_decay": 0.0},
    ]


def collapse_metrics(emb):
    """Monitor for collapse: std, effective rank, norm.
    From H-JEPA trainer.py
    """
    flat = emb.reshape(-1, emb.shape[-1])  # (B*T, D)
    std = flat.std(dim=0).mean().item()
    norms = flat.norm(dim=1)
    avg_norm = norms.mean().item()
    # Effective rank: exp of entropy of normalized singular values
    try:
        s = torch.linalg.svdvals(flat)
        s_norm = s / (s.sum() + 1e-10)
        eff_rank = torch.exp(-(s_norm * torch.log(s_norm + 1e-10)).sum()).item()
    except:
        eff_rank = 0.0
    return {'std': std, 'eff_rank': eff_rank, 'avg_norm': avg_norm}


def run_exp(config):
    phase = config.get('phase', 1)
    load_path = config.get('load', None)

    print(f"\n{'='*50}", flush=True)
    print(f"[{config['tag']}] Phase {phase} | D={config['D']} L={config['L']} lr={config['lr']}", flush=True)
    if phase == 1:
        print(f"  Encoder-only training (SIGReg, no predictor)", flush=True)
    elif phase == 2:
        print(f"  Predictor-only training (frozen encoder)", flush=True)
        print(f"  Load checkpoint: {load_path}", flush=True)
    print(f"  target_ln={config.get('target_ln',True)} smooth_l1={config.get('smooth_l1',True)}", flush=True)
    print('='*50, flush=True)

    out = BASE / config['tag']; out.mkdir(exist_ok=True)

    # Data
    data_dir = BASE/"data"
    if not (data_dir/"meta.json").exists():
        snapshot_download(DATA_REPO, repo_type="dataset", local_dir=str(data_dir), local_dir_use_symlinks=False)
    tr_ds = SeqDataset(data_dir, "shard", history_size=config.get('hist',3))
    va_ds = SeqDataset(data_dir, "val", history_size=config.get('hist',3))
    tr_loader = DataLoader(tr_ds, config.get('bs',256), True, num_workers=2)
    va_loader = DataLoader(va_ds, config.get('bs',256), False, num_workers=2)

    # Model
    model = Fin-JEPA(
        tr_ds.nf, config['D'], pred_depth=config['L'],
        sigreg_proj=config.get('nproj',1024), sigreg_lambda=config.get('lam',0.09),
        target_ln=config.get('target_ln',True), smooth_l1=config.get('smooth_l1',True),
        history_size=config.get('hist',3), self_cond=not config.get('no_self_cond',False),
    ).to(DEVICE)

    # ── Phase-specific setup ──
    if phase == 1:
        # Phase 1: train encoder only — freeze predictor parts
        for p in model.predictor.parameters(): p.requires_grad_(False)
        for p in model.pred_proj.parameters(): p.requires_grad_(False)
        forward_fn = lambda ctx, tgt: model.forward_encoder_only(ctx)
        print("  [Phase 1] Predictor & pred_proj frozen — encoder-only training", flush=True)

    elif phase == 2:
        # Phase 2: load checkpoint, freeze encoder, train predictor
        if load_path:
            ckpt_path = Path(load_path)
            if not ckpt_path.exists():
                # Try relative to BASE/tag
                ckpt_path = BASE / load_path / "best.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Cannot find checkpoint: {load_path} (also tried {ckpt_path})")
            state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state, strict=False)
            print(f"  Loaded Phase 1 checkpoint: {ckpt_path}", flush=True)
        else:
            print(f"  ⚠️ No checkpoint specified for Phase 2 — training from scratch!", flush=True)

        # Freeze encoder + projector
        for p in model.encoder.parameters(): p.requires_grad_(False)
        for p in model.projector.parameters(): p.requires_grad_(False)
        forward_fn = model.forward  # standard forward with predictor
        print("  [Phase 2] Encoder & projector frozen — predictor-only training", flush=True)

    else:
        # Default: standard end-to-end training (backwards compatible)
        forward_fn = model.forward

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {n_params:,} | Trainable: {n_trainable:,} | Device: {DEVICE}", flush=True)

    opt = torch.optim.AdamW(param_groups(model), lr=config['lr'])
    steps_per_epoch = len(tr_loader)
    total_steps = steps_per_epoch * config['epochs']
    warmup_steps = max(1, total_steps // 100)
    sched = LinearWarmupCosineAnnealingLR(opt, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler(enabled=(DEVICE=='cuda'))
    best_loss, t0 = float('inf'), time.time()

    for ep in range(config['epochs']):
        model.train()
        tr_l, cm_tr = 0, {'std':0,'eff_rank':0,'avg_norm':0}
        for batch in tr_loader:
            ctx, tgt = batch['ctx'].to(DEVICE), batch['tgt'].to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=(DEVICE=='cuda'), dtype=torch.bfloat16):
                o = forward_fn(ctx, tgt)
            scaler.scale(o['loss']).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            sched.step()
            tr_l += o['loss'].item()
            cm_tr = collapse_metrics(o['emb'].detach())

        model.eval()
        va_l, cm_va = 0, {'std':0,'eff_rank':0,'avg_norm':0}
        with torch.no_grad():
            for batch in va_loader:
                ctx, tgt = batch['ctx'].to(DEVICE), batch['tgt'].to(DEVICE)
                with torch.amp.autocast(device_type='cuda', enabled=(DEVICE=='cuda'), dtype=torch.bfloat16):
                    o = forward_fn(ctx, tgt)
                va_l += o['loss'].item()
                cm_va = collapse_metrics(o['emb'].detach())
        va_l /= len(va_loader)

        if va_l < best_loss:
            best_loss = va_l
            torch.save(model.state_dict(), out/"best.pt")
            # Upload checkpoint immediately (version management)
            try:
                from huggingface_hub import upload_file
                upload_file(
                    path_or_fileobj=str(out/"best.pt"),
                    path_in_repo=f"checkpoints/{config['tag']}/best.pt",
                    repo_id=DATA_REPO, repo_type="dataset",
                )
            except Exception as e:
                print(f"  ⚠️ Checkpoint upload failed (non-fatal): {e}", flush=True)

        if ep%5==0 or ep==config['epochs']-1:
            gm = torch.cuda.max_memory_allocated()/1024**3 if DEVICE=='cuda' else 0
            # Build loss string depending on available keys
            loss_parts = []
            loss_parts.append(f"tr={tr_l/len(tr_loader):.4f} va={va_l:.4f}")
            if 'pred_loss' in o:
                loss_parts.append(f"pred={o['pred_loss'].item():.4f}")
            if 'sigreg_loss' in o:
                loss_parts.append(f"sig={o['sigreg_loss'].item():.4f}")
            print(f"  {ep:3d}/{config['epochs']} | {' | '.join(loss_parts)} | "
                  f"lr={sched.get_last_lr()[0]:.2e} | {time.time()-t0:.0f}s | gpu={gm:.1f}GB | "
                  f"std={cm_va['std']:.4f} rank={cm_va['eff_rank']:.1f}", flush=True)

    result = {'tag':config['tag'],'phase':phase,'D':config['D'],'L':config['L'],
              'params':n_params,'trainable':n_trainable,'best_va':best_loss,'epochs':config['epochs'],
              'elapsed':time.time()-t0, 'target_ln':config.get('target_ln',True),
              'smooth_l1':config.get('smooth_l1',True)}
    json.dump(result, open(out/"result.json","w"), indent=2)
    print(f"  Done: {result['elapsed']:.0f}s | best_va={best_loss:.4f}", flush=True)
    return result


def main():
    parser = ArgumentParser(description="Experiment B: Two-phase LeWM Fin-JEPA training")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="Training phase: 1 (encoder only) or 2 (predictor only)")
    parser.add_argument("--tag", default="phase1-exp", help="Experiment tag (also output dir name under /tmp)")
    parser.add_argument("--load", default=None,
                        help="Phase 2 only: path to Phase 1 checkpoint. Can be absolute path, "
                             "or tag name (looks for /tmp/fin-jepa-lewm/<tag>/best.pt)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--D", type=int, default=192, help="Embedding dimension")
    parser.add_argument("--L", type=int, default=6, help="Predictor depth")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--bs", type=int, default=256, help="Batch size")
    parser.add_argument("--hist", type=int, default=3, help="History (context) size")
    parser.add_argument("--nproj", type=int, default=1024, help="SIGReg num projections")
    parser.add_argument("--lam", type=float, default=0.09, help="SIGReg lambda weight")
    parser.add_argument("--no-target-ln", dest="target_ln", action="store_false",
                        help="Disable target LayerNorm")
    parser.add_argument("--no-smooth-l1", dest="smooth_l1", action="store_false",
                        help="Disable SmoothL1 (use MSE)")
    parser.add_argument("--no-self-cond", dest="self_cond", action="store_false",
                        help="Disable self-conditioning")
    parser.set_defaults(target_ln=True, smooth_l1=True, self_cond=True)
    args = parser.parse_args()

    if args.phase == 2 and not args.load:
        print("⚠️  Phase 2 requires --load to specify Phase 1 checkpoint path or tag.")
        print("   Example: --load phase1-test  (looks in /tmp/fin-jepa-lewm/phase1-test/best.pt)")
        print("   Example: --load /absolute/path/to/checkpoint.pt")
        sys.exit(1)

    print("═"*60, flush=True)
    print(f"Experiment B: Two-phase LeWM Fin-JEPA | {DEVICE}", flush=True)
    print(f"Phase: {args.phase} | Tag: {args.tag} | Epochs: {args.epochs}", flush=True)
    print(f"Data: {DATA_REPO} (proper per-stock chronological split)", flush=True)
    print("═"*60, flush=True)

    config = {
        'tag': args.tag,
        'phase': args.phase,
        'load': args.load,
        'D': args.D,
        'L': args.L,
        'lr': args.lr,
        'bs': args.bs,
        'hist': args.hist,
        'nproj': args.nproj,
        'lam': args.lam,
        'epochs': args.epochs,
        'target_ln': args.target_ln,
        'smooth_l1': args.smooth_l1,
        'self_cond': args.self_cond,
        'no_self_cond': not args.self_cond,
    }

    r = run_exp(config)
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n✅ Phase {args.phase} complete: {r['tag']} | best_va={r['best_va']:.4f} | {r['elapsed']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
