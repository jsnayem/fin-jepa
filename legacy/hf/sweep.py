"""Fin-JEPA Full Experiment Sweep — runs on HF Jobs with GPU.

Usage:
    hf jobs uv run --resource t4-small sweep.py --tag sweep1 --variants 4
    
This script:
  1. Downloads the preprocessed dataset from HF
  2. Trains multiple JEPA variants
  3. Runs VoE analysis on best model
  4. Uploads results to HF
"""
# /// script
# dependencies = [
#   "torch>=2.0",
#   "huggingface_hub",
#   "numpy",
#   "pandas",
#   "scikit-learn",
# ]
# ///

import os, sys, json, time, warnings, gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from argparse import ArgumentParser
from huggingface_hub import hf_hub_download, snapshot_download, upload_folder
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DATA_REPO = "cedwyh/fin-jepa-data"
BASE = Path("/tmp/fin-jepa-sweep")
BASE.mkdir(exist_ok=True)

# ── JEPA Model (light clone — no local file dependency) ──
class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=128):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        w = torch.full((knots,), 2 * dt)
        w[[0,-1]] = dt
        window = torch.exp(-t.square()/2)
        self.register_buffer("t", t); self.register_buffer("phi", window)
        self.register_buffer("weights", w*window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A.div_(A.norm(p=2, dim=0))
        x_t = (proj@A).unsqueeze(-1)*self.t
        err = (x_t.cos().mean(-3)-self.phi).square() + x_t.sin().mean(-3).square()
        return (err@self.weights).mean()*proj.size(-2)

class PriceEncoder(nn.Module):
    def __init__(self, nf, D=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nf,D), nn.LayerNorm(D), nn.Linear(D,D*2), nn.GELU(),
                                 nn.Linear(D*2,D*2), nn.GELU(), nn.Linear(D*2,D),
                                 nn.LayerNorm(D), nn.Linear(D,D), nn.GELU(), nn.Linear(D,D))

    def forward(self, x):
        return self.net(x)  # (B, F) → (B, D)

class TransformerPred(nn.Module):
    def __init__(self, D=64, L=6, H=4, max_T=256):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, max_T, D)*0.02)
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(D, H, D*4, 0.1, 'gelu', batch_first=True, norm_first=True) for _ in range(L)])
        self.norm = nn.LayerNorm(D)
        self.proj = nn.Sequential(nn.LayerNorm(D), nn.Linear(D,D), nn.GELU(), nn.Linear(D,D))

    def forward(self, z):
        B,T,D = z.shape
        x = z + self.pos[:,:T]
        x = self.drop(x)
        mask = torch.triu(torch.full((T,T),float('-inf'),device=z.device),diagonal=1)
        for b in self.blocks: x = b(x, src_mask=mask, is_causal=False)
        return self.proj(self.norm(x))

class JEPA(nn.Module):
    def __init__(self, nf=11, D=64, L=6, nproj=128, lam=0.1):
        super().__init__()
        self.D, self.lam = D, lam
        self.enc = PriceEncoder(nf, D)
        self.pred = TransformerPred(D, L)
        self.sigreg = SIGReg(num_proj=nproj) if nproj > 0 else None

    def encode(self, seq):
        B,T,F = seq.shape
        return self.enc(seq.reshape(-1,F)).reshape(B,T,self.D)

    def forward(self, ctx, tgt):
        z_ctx, B, T = self.encode(ctx), *ctx.shape[:2]
        z_pred = self.pred(z_ctx)
        out = {'emb': z_ctx}
        if tgt is not None:
            z_tgt = self.encode(tgt)
            n = min(T, tgt.shape[1])
            pl = nn.functional.mse_loss(z_pred[:,:n], z_tgt[:,:n])
            fe = torch.cat([z_ctx,z_tgt],1)
            if self.sigreg:
                sl = self.sigreg(fe.permute(1,0,2))
                out.update(pred_loss=pl, sigreg_loss=sl, loss=pl+self.lam*sl)
            else:
                out.update(pred_loss=pl, sigreg_loss=0.0, loss=pl)
        return out


# ── Dataset ──
class SeqDataset(Dataset):
    def __init__(self, data_dir, split="train", val_ratio=0.1):
        meta = json.load(open(Path(data_dir)/"meta.json"))
        all_data = [np.load(f).astype(np.float32) for f in sorted(Path(data_dir).glob("shard_*.npy"))]
        data = np.concatenate(all_data, axis=0)
        split_i = int(len(data)*(1-val_ratio))
        self.data = data[:split_i] if split=="train" else data[split_i:]
        self.nf = data.shape[-1]
        print(f"  {split}: {len(self.data):,} seqs", flush=True)

    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        s = self.data[i]
        return {'ctx':torch.FloatTensor(s[:60]), 'tgt':torch.FloatTensor(s[60:65])}


# ── Training ──
def run_exp(config):
    print(f"\n{'='*50}\n[{config['tag']}] D={config['D']} L={config['L']} nproj={config['nproj']}\n{'='*50}", flush=True)
    out = BASE / config['tag']; out.mkdir(exist_ok=True)

    # Data
    data_dir = BASE/"data"
    if not (data_dir/"meta.json").exists():
        snapshot_download(DATA_REPO, repo_type="dataset", local_dir=str(data_dir), local_dir_use_symlinks=False)
    tr_ds = SeqDataset(data_dir, "train"); va_ds = SeqDataset(data_dir, "val")
    tr_loader = DataLoader(tr_ds, config.get('bs',512), True, num_workers=2)
    va_loader = DataLoader(va_ds, config.get('bs',512), False, num_workers=2)

    # Model
    model = JEPA(tr_ds.nf, config['D'], config['L'], config['nproj'], config.get('lam',0.1)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,} | Device: {DEVICE}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=config.get('lr',1e-3), weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['epochs'])
    scaler = torch.amp.GradScaler(enabled=DEVICE=='cuda')
    best_loss, t0 = float('inf'), time.time()

    for ep in range(config['epochs']):
        model.train()
        tr_l = 0
        for b in tr_loader:
            ctx,tgt = b['ctx'].to(DEVICE),b['tgt'].to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=DEVICE=='cuda'):
                o = model(ctx,tgt)
            scaler.scale(o['loss']).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()
            tr_l += o['loss'].item()
        sched.step()

        model.eval()
        va_l = 0
        with torch.no_grad():
            for b in va_loader:
                ctx,tgt = b['ctx'].to(DEVICE),b['tgt'].to(DEVICE)
                va_l += model(ctx,tgt)['loss'].item()
        va_l /= len(va_loader)

        if va_l < best_loss:
            best_loss = va_l
            torch.save(model.state_dict(), out/"best.pt")

        if ep%10==0 or ep==config['epochs']-1:
            gm = torch.cuda.max_memory_allocated()/1024**3 if DEVICE=='cuda' else 0
            print(f"  {ep:3d} | tr={tr_l/len(tr_loader):.4f} va={va_l:.4f} | {time.time()-t0:.0f}s | gpu={gm:.1f}GB", flush=True)

    # Probe (subset)
    model.eval()
    zs,ys = [],[]
    with torch.no_grad():
        for i,b in enumerate(va_loader):
            if i*config.get('bs',512)>50000: break
            ctx,tgt = b['ctx'].to(DEVICE),b['tgt'].to(DEVICE)
            zs.append(model.encode(ctx)[:,-1,:].cpu().numpy())
            ys.append(tgt[:,-1,5].cpu().numpy())
    Z,Y = np.concatenate(zs),np.concatenate(ys)
    probe = nn.Linear(config['D'],1).to(DEVICE)
    popt = torch.optim.Adam(probe.parameters(),lr=1e-3)
    zt,yt = torch.FloatTensor(Z[:len(Z)//2]).to(DEVICE),torch.FloatTensor(Y[:len(Y)//2]).to(DEVICE).unsqueeze(1)
    ze,ye = torch.FloatTensor(Z[len(Z)//2:]).to(DEVICE),torch.FloatTensor(Y[len(Y)//2:]).to(DEVICE).unsqueeze(1)
    for _ in range(50):
        popt.zero_grad(); nn.functional.mse_loss(probe(zt),yt).backward(); popt.step()
    with torch.no_grad():
        r2 = 1 - nn.functional.mse_loss(probe(ze),ye).item()/ye.var().item()
        ric = torch.corrcoef(torch.stack([probe(ze).squeeze().argsort().float(),ye.squeeze().argsort().float()]))[0,1].item()

    result = {'tag':config['tag'],'D':config['D'],'L':config['L'],'nproj':config['nproj'],
              'params':n_params,'best_va':best_loss,'r2':r2,'rank_ic':ric,'epochs':config['epochs'],
              'elapsed':time.time()-t0}
    json.dump(result,open(out/"result.json","w"),indent=2)
    print(f"  R²={r2:.4f} RankIC={ric:.4f} | {result['elapsed']:.0f}s", flush=True)
    return result


# ── Sweep Configs ──
VARIANTS = [
    # (tag, D, L, nproj, epochs, lr)
    # Arch comparison (same as local but on full data)
    {"tag":"v1_tiny_d32",   "D":32,  "L":2, "nproj":64,  "epochs":50,  "lr":1e-3},
    {"tag":"v2_base_d64",   "D":64,  "L":4, "nproj":128, "epochs":50,  "lr":1e-3},
    {"tag":"v3_large_d128", "D":128, "L":4, "nproj":256, "epochs":50,  "lr":1e-3},
    {"tag":"v4_deep_d64",   "D":64,  "L":6, "nproj":128, "epochs":50,  "lr":1e-3},
    # Long runs
    {"tag":"v4_deep_long",  "D":64,  "L":6, "nproj":128, "epochs":200, "lr":1e-3},
    # SIGReg ablation
    {"tag":"v4_nosigreg",   "D":64,  "L":6, "nproj":0,   "epochs":50,  "lr":1e-3, "lam":0.0},
    # Higher lr
    {"tag":"v4_lr3e-3",     "D":64,  "L":6, "nproj":128, "epochs":50,  "lr":3e-3},
    {"tag":"v4_lr1e-4",     "D":64,  "L":6, "nproj":128, "epochs":50,  "lr":1e-4},
]


def main():
    parser = ArgumentParser()
    parser.add_argument("--tag", default="sweep1", help="Sweep name")
    parser.add_argument("--variants", type=int, default=0, help="Number of variants to run (0=all)")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    args = parser.parse_args()

    print("═"*60)
    print(f"Fin-JEPA HF Sweep | {DEVICE}")
    print(f"Tag: {args.tag} | Variants: {args.variants or len(VARIANTS)}")
    print(f"Data: {DATA_REPO}")
    print("═"*60)

    variants = VARIANTS[args.start:]
    if args.variants: variants = variants[:args.variants]

    results = []
    for i, cv in enumerate(variants):
        print(f"\n[{i+1}/{len(variants)}] {cv['tag']}", flush=True)
        r = run_exp(cv)
        results.append(r)
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("Sweep Results")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['tag']:20s} | D={r['D']:3d} L={r['L']} | params={r['params']:>7,} | "
              f"va={r['best_va']:.4f} | R²={r['r2']:.4f} | RankIC={r['rank_ic']:.4f} | {r['elapsed']:.0f}s")

    # Upload
    json.dump(results, open(BASE/f"results_{args.tag}.json","w"), indent=2)
    try:
        upload_folder(folder_path=str(BASE), repo_id=DATA_REPO, repo_type="dataset",
                      commit_message=f"Sweep {args.tag}: {len(results)} variants",
                      path_in_repo=f"sweeps/{args.tag}")
        print(f"\nResults uploaded to {DATA_REPO}/sweeps/{args.tag}", flush=True)
    except Exception as e:
        print(f"\nUpload failed: {e}", flush=True)
        print(f"Results at {BASE}", flush=True)

    print(f"\n✅ Sweep {args.tag} complete", flush=True)


if __name__ == "__main__":
    main()
