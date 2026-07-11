#!/usr/bin/env python3
"""Phase 2: Load frozen Phase 1 encoder, train predictor on top."""

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
from huggingface_hub import snapshot_download, upload_folder, hf_hub_download, upload_file
from einops import rearrange
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DATA_REPO = "cedwyh/fin-jepa-h3"
BASE = Path("/tmp/fin-jepa-phase2")
BASE.mkdir(exist_ok=True)

# ── Model Components (same as train_lewm.py) ──

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
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return (err @ self.weights).mean() * proj.size(-2)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
    def forward(self, x, causal=False):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.to_out(rearrange(out, "b h t d -> b t (h d)"))

class ConditionalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + g1 * self.attn(modulate(x, s1, sc1))
        x = x + g2 * self.mlp(modulate(x, s2, sc2))
        return x

class Embedder(nn.Module):
    def __init__(self, in_dim=11, dim=192, depth=6, heads=6, mlp_ratio=4):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, dim, bias=True)
        self.blocks = nn.ModuleList([ConditionalBlock(dim, heads, dim // heads, dim * mlp_ratio) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
    def forward(self, x):
        x = self.proj_in(x)
        c = x
        for blk in self.blocks:
            x = blk(x, c)
        return self.norm_out(x)

class ARPredictor(nn.Module):
    def __init__(self, history_size, input_dim, output_dim, depth=6, heads=16):
        super().__init__()
        self.history_size = history_size
        self.pos_emb = nn.Parameter(torch.randn(history_size, input_dim) * 0.02)
        self.pred_tok = nn.Parameter(torch.zeros(1, 1, output_dim))
        self.blocks = nn.ModuleList([ConditionalBlock(input_dim, heads, input_dim // heads, input_dim * 4) for _ in range(depth)])
    def forward(self, hist, cond=None):
        B, T, D = hist.shape
        x = hist + self.pos_emb[:T]
        c = hist if cond is None else cond
        for blk in self.blocks:
            x = blk(x, c)
        return x[:, -1:]

class MLPProj(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim*4; output_dim = output_dim or input_dim
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))
    def forward(self, x): return self.net(x)

# ── Phase 2 Model: frozen encoder + trainable predictor ──

class Phase2Model(nn.Module):
    """Encoder frozen from Phase 1, only predictor is trainable."""
    def __init__(self, encoder, embed_dim=192, pred_depth=6, history_size=3, 
                 sigreg_lambda=0.09, smooth_l1=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.sigreg_lambda = sigreg_lambda
        self.smooth_l1 = smooth_l1
        self.history_size = history_size
        
        # Freeze encoder
        self.encoder = encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        
        # Trainable projector (for SIGReg)
        self.projector = MLPProj(embed_dim)
        
        # Trainable predictor + projection
        self.predictor = ARPredictor(history_size, embed_dim, embed_dim, depth=pred_depth, heads=16)
        self.pred_proj = MLPProj(embed_dim)
        
        # SIGReg (trainable)
        self.sigreg = SIGReg(num_proj=1024)
    
    def encode(self, seq):
        with torch.no_grad():
            z = self.encoder(seq)
        return self.projector(z.reshape(-1, self.embed_dim)).reshape(z.shape)
    
    def forward(self, ctx, tgt=None):
        B, T = ctx.shape[:2]
        emb = self.encode(ctx)
        out = {'emb': emb}
        if tgt is not None:
            tgt_emb = self.encode(tgt)
            tgt_emb = F.layer_norm(tgt_emb, (tgt_emb.shape[-1],))
            
            pred = self.pred_proj(
                self.predictor(emb[:, :self.history_size], None)
                .reshape(-1, self.embed_dim)
            ).reshape(-1, 1, self.embed_dim)
            
            n = min(1, tgt_emb.shape[1])
            if self.smooth_l1:
                pl = F.smooth_l1_loss(pred[:, :n], tgt_emb[:, :n], beta=1.0)
            else:
                pl = F.mse_loss(pred[:, :n], tgt_emb[:, :n])
            
            sl = self.sigreg(torch.cat([emb, tgt_emb], 1).transpose(0, 1))
            out.update(pred_loss=pl, sigreg_loss=sl, loss=pl + self.sigreg_lambda * sl)
        return out


# ── Dataset ──

class SeqDataset(Dataset):
    def __init__(self, data_dir, prefix="shard", history_size=3):
        meta = json.load(open(Path(data_dir)/"meta.json"))
        shard_files = sorted(Path(data_dir).glob(f"{prefix}_*.npy"))
        self.data = np.concatenate([np.load(f).astype(np.float32) for f in shard_files], axis=0)
        self.nf = self.data.shape[-1]
        self.hist = history_size
        print(f"  {prefix}: {len(self.data):,} seqs x {self.data.shape[1]} steps | hist={history_size}", flush=True)
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        s = self.data[i]
        return {'ctx': torch.FloatTensor(s[:self.hist]), 'tgt': torch.FloatTensor(s[self.hist:self.hist+1])}


# ── Training ──

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
    d, nd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim < 2 or n.endswith("bias"):
            nd.append(p)
        else:
            d.append(p)
    return [{"params": d, "weight_decay": wd}, {"params": nd, "weight_decay": 0.0}]

def collapse_metrics(emb):
    flat = emb.reshape(-1, emb.shape[-1])
    std = flat.std(dim=0).mean().item()
    norms = flat.norm(dim=1)
    avg_norm = norms.mean().item()
    try:
        s = torch.linalg.svdvals(flat)
        s_norm = s / (s.sum() + 1e-10)
        eff_rank = torch.exp(-(s_norm * torch.log(s_norm + 1e-10)).sum()).item()
    except:
        eff_rank = 0.0
    return {'std': std, 'eff_rank': eff_rank, 'avg_norm': avg_norm}


def run_phase2(config):
    tag = config['tag']
    ckpt_tag = config['phase1_ckpt']
    print(f"\n{'='*50}", flush=True)
    print(f"[Phase 2] {tag} | D={config['D']} pred_depth={config['L']} ", flush=True)
    print(f"  Loading encoder from: {ckpt_tag}", flush=True)
    print('='*50, flush=True)
    
    out = BASE / tag; out.mkdir(exist_ok=True)
    
    # Download Phase 1 checkpoint
    ckpt_path = hf_hub_download(DATA_REPO, f"checkpoints/{ckpt_tag}/best.pt", repo_type="dataset")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    print(f"  Phase 1 checkpoint loaded: {os.path.getsize(ckpt_path)/1024**2:.1f}MB", flush=True)
    
    # Build encoder from saved state
    encoder = Embedder(in_dim=11, dim=config['D'], depth=6, heads=6)
    # Filter encoder keys from the state dict
    enc_state = {k.replace('encoder.', ''): v for k, v in state.items() if k.startswith('encoder.')}
    encoder.load_state_dict(enc_state, strict=True)
    print(f"  Encoder loaded, frozen ({sum(p.numel() for p in encoder.parameters()):,} params)", flush=True)
    del state, enc_state; gc.collect()
    
    # Data
    data_dir = BASE/"data"
    if not (data_dir/"meta.json").exists():
        snapshot_download(DATA_REPO, repo_type="dataset", local_dir=str(data_dir), local_dir_use_symlinks=False)
    tr_ds = SeqDataset(data_dir, "shard", history_size=config.get('hist', 3))
    va_ds = SeqDataset(data_dir, "val", history_size=config.get('hist', 3))
    tr_loader = DataLoader(tr_ds, config.get('bs', 512), True, num_workers=2)  # Larger batch since encoder is frozen
    va_loader = DataLoader(va_ds, config.get('bs', 512), False, num_workers=2)
    
    # Phase 2 model
    model = Phase2Model(encoder, embed_dim=config['D'], pred_depth=config['L'],
                        history_size=config.get('hist', 3), smooth_l1=config.get('smooth_l1', True))
    model = model.to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} params | Device: {DEVICE}", flush=True)
    
    opt = torch.optim.AdamW(param_groups(model), lr=config['lr'])
    steps_per_epoch = len(tr_loader)
    total_steps = steps_per_epoch * config['epochs']
    warmup_steps = max(1, total_steps // 100)
    sched = LinearWarmupCosineAnnealingLR(opt, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == 'cuda'))
    best_loss, t0 = float('inf'), time.time()
    
    for ep in range(config['epochs']):
        model.train()
        tr_l, cm_tr = 0, {'std': 0, 'eff_rank': 0, 'avg_norm': 0}
        for batch in tr_loader:
            ctx, tgt = batch['ctx'].to(DEVICE), batch['tgt'].to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=(DEVICE=='cuda'), dtype=torch.bfloat16):
                o = model(ctx, tgt)
            scaler.scale(o['loss']).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            sched.step()
            tr_l += o['loss'].item()
            cm_tr = collapse_metrics(o['emb'].detach())
        
        model.eval()
        va_l, cm_va = 0, {'std': 0, 'eff_rank': 0, 'avg_norm': 0}
        with torch.no_grad():
            for batch in va_loader:
                ctx, tgt = batch['ctx'].to(DEVICE), batch['tgt'].to(DEVICE)
                with torch.amp.autocast(device_type='cuda', enabled=(DEVICE=='cuda'), dtype=torch.bfloat16):
                    o = model(ctx, tgt)
                va_l += o['loss'].item()
        va_l /= len(va_loader)
        
        if va_l < best_loss:
            best_loss = va_l
            torch.save(model.state_dict(), out/"best.pt")
            try:
                upload_file(path_or_fileobj=str(out/"best.pt"),
                          path_in_repo=f"checkpoints/{tag}/best.pt",
                          repo_id=DATA_REPO, repo_type="dataset")
            except Exception as e:
                print(f"  ⚠️ Upload failed: {e}", flush=True)
        
        if ep % 5 == 0 or ep == config['epochs'] - 1:
            gm = torch.cuda.max_memory_allocated()/1024**3 if DEVICE=='cuda' else 0
            print(f"  {ep:3d}/{config['epochs']} | tr={tr_l/len(tr_loader):.4f} va={va_l:.4f} | "
                  f"lr={sched.get_last_lr()[0]:.2e} | {time.time()-t0:.0f}s | gpu={gm:.1f}GB | "
                  f"std={cm_va['std']:.4f}", flush=True)
    
    result = {'tag': tag, 'D': config['D'], 'L': config['L'],
              'phase1_ckpt': ckpt_tag,
              'params_trainable': trainable, 'params_total': total,
              'best_va': best_loss, 'epochs': config['epochs'],
              'elapsed': time.time() - t0}
    json.dump(result, open(out/"result.json", "w"), indent=2)
    print(f"\n  Phase 2 Done: {result['elapsed']:.0f}s | best_va={best_loss:.4f}", flush=True)
    
    try:
        upload_folder(folder_path=str(out), repo_id=DATA_REPO, repo_type="dataset",
                     commit_message=f"Phase 2 {tag}: {result['best_va']:.4f}",
                     path_in_repo=f"sweeps/{tag}")
    except Exception as e:
        print(f"  ⚠️ Upload failed: {e}", flush=True)
    
    return result


def main():
    parser = ArgumentParser()
    parser.add_argument("--tag", default="phase2-v1")
    parser.add_argument("--phase1-ckpt", default="phase1-v1")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--hist", type=int, default=3)
    args = parser.parse_args()
    
    config = {
        'tag': args.tag, 'D': args.dim, 'L': args.depth, 'lr': args.lr,
        'phase1_ckpt': args.phase1_ckpt, 'epochs': args.epochs, 'bs': 512,
        'hist': args.hist, 'smooth_l1': True,
    }
    r = run_phase2(config)
    print(f"\n✅ Phase 2 {args.tag} complete: {r['best_va']:.4f} in {r['elapsed']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
