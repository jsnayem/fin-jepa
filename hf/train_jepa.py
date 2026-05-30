"""Fin-JEPA — HF GPU Training Script

Usage:
    python train_jepa.py --dataset cedwyh/fin-jepa-data --embed-dim 64 --epochs 200

Runs on HF Space (GPU) or any CUDA machine.
"""
import os, sys, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from argparse import ArgumentParser
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Fin-JEPA, LinearProbe

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class NPSequenceDataset(Dataset):
    """Load precomputed NPY shards from HF Dataset."""
    def __init__(self, data_dir, split="train", val_ratio=0.1):
        import json
        meta = json.load(open(Path(data_dir) / "meta.json"))
        
        # Load all shards
        shard_files = sorted(Path(data_dir).glob("shard_*.npy"))
        if not shard_files:
            raise FileNotFoundError(f"No shard_*.npy files in {data_dir}")
        
        all_data = [np.load(f).astype(np.float32) for f in shard_files]
        data = np.concatenate(all_data, axis=0)
        
        n = len(data)
        split_idx = int(n * (1 - val_ratio))
        
        if split == "train":
            self.data = data[:split_idx]
        else:
            self.data = data[split_idx:]
        
        self.feat_dim = data.shape[-1]
        print(f"  {split}: {len(self.data):,} sequences, {self.feat_dim} features", flush=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        return {
            'ctx': torch.FloatTensor(seq[:60]),
            'tgt': torch.FloatTensor(seq[60:65]),
        }


def train_epoch(model, loader, opt, scaler):
    model.train()
    total = 0
    for batch in loader:
        ctx, tgt = batch['ctx'].to(DEVICE), batch['tgt'].to(DEVICE)
        opt.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=DEVICE == 'cuda'):
            out = model(ctx, tgt)
        scaler.scale(out['loss']).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        total += out['loss'].item()
    return total / len(loader)


@torch.no_grad()
def eval_loss(model, loader):
    model.eval()
    total = 0
    for batch in loader:
        ctx, tgt = batch['ctx'].to(DEVICE), batch['tgt'].to(DEVICE)
        out = model(ctx, tgt)
        total += out['loss'].item()
    return total / len(loader)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="cedwyh/fin-jepa-data")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--enc-layers", type=int, default=4)
    parser.add_argument("--pred-layers", type=int, default=6)
    parser.add_argument("--sigreg-proj", type=int, default=128)
    parser.add_argument("--sigreg-lambda", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output", default="./checkpoints")
    parser.add_argument("--tag", default="exp1")
    args = parser.parse_args()
    
    output_dir = Path(args.output) / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("═" * 60)
    print(f"Fin-JEPA — HF GPU Training | {DEVICE}")
    print(f"Tag: {args.tag} | Embed: {args.embed_dim} | Pred layers: {args.pred_layers}")
    print("═" * 60)
    t0 = time.time()
    
    # 1. Load dataset
    print("\n[1] Loading dataset...", flush=True)
    from huggingface_hub import snapshot_download
    data_dir = Path("/tmp/fin-jepa-data")
    data_dir.mkdir(exist_ok=True)
    
    snapshot_download(
        repo_id=args.dataset,
        repo_type="dataset",
        local_dir=str(data_dir),
        local_dir_use_symlinks=False,
    )
    
    train_ds = NPSequenceDataset(data_dir, "train")
    val_ds = NPSequenceDataset(data_dir, "val")
    n_features = train_ds.feat_dim
    
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # 2. Create model
    print(f"\n[2] Model ({n_features} → {args.embed_dim}d)...", flush=True)
    model = Fin-JEPA(
        n_features, args.embed_dim,
        encoder_layers=args.enc_layers,
        predictor_layers=args.pred_layers,
        sigreg_proj=args.sigreg_proj,
        sigreg_lambda=args.sigreg_lambda,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}", flush=True)
    
    # 3. Train
    print(f"\n[3] Training {args.epochs} epochs...", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device='cuda', enabled=DEVICE == 'cuda')
    
    history = {'train': [], 'val': []}
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        tr_l = train_epoch(model, train_loader, opt, scaler)
        va_l = eval_loss(model, val_loader)
        history['train'].append(tr_l)
        history['val'].append(va_l)
        sched.step()
        
        if va_l < best_loss:
            best_loss = va_l
            torch.save({
                'model': model.state_dict(),
                'args': vars(args),
                'epoch': epoch,
                'val_loss': va_l,
            }, output_dir / "best.pt")
        
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**3 if DEVICE == 'cuda' else 0
            print(f"  {epoch:3d} | tr={tr_l:.4f} va={va_l:.4f} | {time.time()-t0:.0f}s | gpu={gpu_mem:.1f}GB", flush=True)
    
    json.dump(history, open(output_dir / "history.json", "w"))
    print(f"  Done: {(time.time()-t0)/60:.1f}min", flush=True)
    
    # 4. Probe (on a subset — 50K samples for speed)
    print(f"\n[4] Probing (50K subset)...", flush=True)
    ckpt = torch.load(output_dir / "best.pt", map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    # Extract embeddings
    probe_loader = DataLoader(val_ds, 512, shuffle=False)
    zs, ys = [], []
    with torch.no_grad():
        for i, batch in enumerate(probe_loader):
            if i * 512 > 50000: break
            ctx, tgt = batch['ctx'].to(DEVICE), batch['tgt'].to(DEVICE)
            z = model.encode_batch(ctx)[:, -1, :].cpu().numpy()
            y = tgt[:, -1, 5].cpu().numpy()  # forward return
            zs.append(z); ys.append(y)
    
    Z = np.concatenate(zs); Y = np.concatenate(ys)
    
    # Probe
    probe = LinearProbe(args.embed_dim).to(DEVICE)
    popt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    zt = torch.FloatTensor(Z[:len(Z)//2]).to(DEVICE)
    yt = torch.FloatTensor(Y[:len(Y)//2]).to(DEVICE).unsqueeze(1)
    ze = torch.FloatTensor(Z[len(Z)//2:]).to(DEVICE)
    ye = torch.FloatTensor(Y[len(Y)//2:]).to(DEVICE).unsqueeze(1)
    for _ in range(50):
        popt.zero_grad()
        nn.functional.mse_loss(probe(zt), yt).backward()
        popt.step()
    
    with torch.no_grad():
        pred = probe(ze)
        r2 = 1 - nn.functional.mse_loss(pred, ye).item() / ye.var().item()
        ric = torch.corrcoef(torch.stack([
            pred.squeeze().argsort().float(), ye.squeeze().argsort().float()
        ]))[0, 1].item()
    
    print(f"  R²={r2:.4f}  RankIC={ric:.4f}", flush=True)
    
    # 5. Save results & upload
    result = {
        'tag': args.tag, 'n_params': n_params, 'best_val_loss': best_loss,
        'r2': r2, 'rank_ic': ric, 'epochs': args.epochs, 'elapsed': time.time()-t0,
        'config': vars(args), 'DEVICE': DEVICE,
    }
    json.dump(result, open(output_dir / "result.json", "w"), indent=2)
    
    print(f"\n{'='*60}", flush=True)
    print(f"✅ Training complete", flush=True)
    print(f"   Results: {output_dir}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
