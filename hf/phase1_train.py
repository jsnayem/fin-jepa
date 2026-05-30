#!/usr/bin/env python3
"""Phase 1: Encoder-only pre-training via SIGReg (no predictor).
Self-contained for HF Jobs."""

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
PHASE1_REPO = "cedwyh/fin-jepa-h3"  # upload phase1 ckpt here
BASE = Path("/tmp/fin-jepa-phase1")
BASE.mkdir(exist_ok=True)

# ── Model Components (subset of Fin-JEPA — encoder only) ──

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

class MLPProj(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim*4; output_dim = output_dim or input_dim
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))
    def forward(self, x): return self.net(x)

class EncoderJEPA(nn.Module):
    """Encoder-only JEPA: produces Z vectors with SIGReg regularization."""
    def __init__(self, n_features=11, embed_dim=192, sigreg_proj=1024, sigreg_lambda=0.09):
        super().__init__()
        self.embed_dim = embed_dim
        self.sigreg_lambda = sigreg_lambda
        self.encoder = Embedder(n_features, embed_dim)
        self.projector = MLPProj(embed_dim)
        self.sigreg = SIGReg(num_proj=sigreg_proj)
    
    def encode(self, seq):
        z = self.encoder(seq)
        return self.projector(z.reshape(-1, self.embed_dim)).reshape(z.shape)
    
    def forward(self, seq):
        emb = self.encode(seq)
        sl = self.sigreg(emb.transpose(0, 1))
        return {'emb': emb, 'sigreg_loss': sl, 'loss': self.sigreg_lambda * sl}

# ── Dataset ──

class SeqDataset(Dataset):
    def __init__(self, data_dir, prefix="shard", seq_len=65):
        meta = json.load(open(Path(data_dir)/"meta.json"))
        shard_files = sorted(Path(data_dir).glob(f"{prefix}_*.npy"))
        self.data = np.concatenate([np.load(f).astype(np.float32) for f in shard_files], axis=0)
        self.nf = self.data.shape[-1]
        self.seq_len = seq_len
        print(f"  {prefix}: {len(self.data):,} seqs x {seq_len} steps x {self.nf} features", flush=True)
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        s = self.data[i]
        return torch.FloatTensor(s)  # (65, 11)

# ── Training utilities ──

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

# ── Main ──

def run_phase1(config):
    tag = config['tag']
    print(f"\n{'='*50}", flush=True)
    print(f"[Phase 1] {tag} | D={config['D']} lr={config['lr']} ep={config['epochs']}", flush=True)
    print('='*50, flush=True)
    
    out = BASE / tag; out.mkdir(exist_ok=True)
    
    # Data
    data_dir = BASE/"data"
    if not (data_dir/"meta.json").exists():
        snapshot_download(DATA_REPO, repo_type="dataset", local_dir=str(data_dir), local_dir_use_symlinks=False)
    tr_ds = SeqDataset(data_dir, "shard")
    va_ds = SeqDataset(data_dir, "val")
    tr_loader = DataLoader(tr_ds, config.get('bs', 256), True, num_workers=2)
    va_loader = DataLoader(va_ds, config.get('bs', 256), False, num_workers=2)
    
    # Model
    model = EncoderJEPA(embed_dim=config['D'], sigreg_proj=config.get('nproj', 1024), sigreg_lambda=config.get('lam', 0.09))
    n_params = sum(p.numel() for p in model.parameters())
    model = model.to(DEVICE)
    print(f"  Params: {n_params:,} | Device: {DEVICE}", flush=True)
    
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
            batch = batch.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=(DEVICE=='cuda'), dtype=torch.bfloat16):
                o = model(batch)
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
                batch = batch.to(DEVICE)
                with torch.amp.autocast(device_type='cuda', enabled=(DEVICE=='cuda'), dtype=torch.bfloat16):
                    o = model(batch)
                va_l += o['loss'].item()
                cm_va = collapse_metrics(o['emb'].detach())
        va_l /= len(va_loader)
        
        if va_l < best_loss:
            best_loss = va_l
            torch.save(model.state_dict(), out/"best.pt")
            try:
                upload_file(path_or_fileobj=str(out/"best.pt"), 
                          path_in_repo=f"checkpoints/{tag}/best.pt",
                          repo_id=PHASE1_REPO, repo_type="dataset")
            except Exception as e:
                print(f"  ⚠️ Upload failed: {e}", flush=True)
        
        if ep % 5 == 0 or ep == config['epochs'] - 1:
            gm = torch.cuda.max_memory_allocated()/1024**3 if DEVICE=='cuda' else 0
            print(f"  {ep:3d}/{config['epochs']} | tr={tr_l/len(tr_loader):.4f} va={va_l:.4f} | "
                  f"lr={sched.get_last_lr()[0]:.2e} | {time.time()-t0:.0f}s | gpu={gm:.1f}GB | "
                  f"std={cm_va['std']:.4f} rank={cm_va['eff_rank']:.1f}", flush=True)
    
    result = {'tag': tag, 'D': config['D'], 'params': n_params,
              'best_va': best_loss, 'epochs': config['epochs'], 'elapsed': time.time() - t0}
    json.dump(result, open(out/"result.json", "w"), indent=2)
    print(f"\n  Phase 1 Done: {result['elapsed']:.0f}s | best_va={best_loss:.4f}", flush=True)
    
    # Upload results
    try:
        upload_folder(folder_path=str(out), repo_id=PHASE1_REPO, repo_type="dataset",
                     commit_message=f"Phase 1 {tag}: {result['best_va']:.4f}",
                     path_in_repo=f"sweeps/{tag}")
    except Exception as e:
        print(f"  ⚠️ Upload failed: {e}", flush=True)
    
    return result


def main():
    parser = ArgumentParser()
    parser.add_argument("--tag", default="phase1-v1")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dim", type=int, default=192)
    args = parser.parse_args()
    
    config = {
        'tag': args.tag, 'D': args.dim, 'lr': args.lr,
        'nproj': 1024, 'lam': 0.09, 'epochs': args.epochs, 'bs': 256,
    }
    r = run_phase1(config)
    print(f"\n✅ Phase 1 {args.tag} complete: {r['best_va']:.4f} in {r['elapsed']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
