"""
Fin-JEPA Architecture Comparison — trains multiple variants and collects results.
"""
import os, sys, time, json, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.expanduser("~/dev/chan-jepa"))
from model import Fin-JEPA, PriceEncoder
from data import download_hs20, add_features, JEPADataset

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BASE = Path(__file__).resolve().parent / "output"

VARIANTS = [
    # (name, embed_dim, enc_layers, pred_layers, sigreg_proj)
    ("v1_tiny_d32",  32,  2, 2, 64),
    ("v2_base_d64",  64,  3, 4, 128),
    ("v3_large_d128", 128, 3, 4, 256),
    ("v4_deep_d64",  64,  4, 6, 128),
]


def make_model(name, embed_dim, enc_layers, pred_layers, sigreg_proj, n_features=11):
    return Fin-JEPA(n_features, embed_dim,
                    encoder_layers=enc_layers, predictor_layers=pred_layers,
                    sigreg_proj=sigreg_proj, sigreg_lambda=0.1)


def run_variant(variant, n_epochs=50):
    name, embed_dim, enc_l, pred_l, sigreg_p = variant
    exp_dir = BASE / f"arch_{name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print(f"  embed_dim={embed_dim} enc_layers={enc_l} pred_layers={pred_l} sigreg_proj={sigreg_p}")
    
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
    model = make_model(*variant).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    best_va = float('inf')
    t0 = time.time()
    
    for epoch in range(n_epochs):
        # Train
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
        
        # Validate
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
        'name': name, 'embed_dim': embed_dim, 'enc_layers': enc_l, 'pred_layers': pred_l,
        'sigreg_proj': sigreg_p, 'params': n_params, 'epochs': n_epochs,
        'elapsed': elapsed, 'best_va_loss': best_va/len(va_loader),
    }
    json.dump(meta, open(exp_dir / "meta.json", "w"), indent=2)
    return meta


if __name__ == "__main__":
    print("═"*60)
    print("Architecture Comparison")
    print("═"*60)
    
    results = []
    for v in VARIANTS:
        meta = run_variant(v, n_epochs=50)
        results.append(meta)
    
    print("\n" + "═"*60)
    print("Results Summary")
    print("═"*60)
    for r in results:
        print(f"  {r['name']:20s} | params={r['params']:>8,} | best_va={r['best_va_loss']:.4f} | {r['elapsed']:.0f}s")
