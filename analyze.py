"""
Fin-JEPA Analysis Suite — runs after training, produces comprehensive report.
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from pathlib import Path
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.expanduser("~/dev/chan-jepa"))
from model import Fin-JEPA, LinearProbe, MLPProbe
from data import download_hs20, add_features, JEPADataset, FEATURE_COLS

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BASE = Path(__file__).resolve().parent / "output"


def load_best(subdir="exp1_opt"):
    """Load the best model from given experiment."""
    model_dir = BASE / subdir
    model_path = model_dir / "best.pt"
    if not model_path.exists():
        # Try latest
        items = list(BASE.glob("*/best.pt"))
        if not items:
            print(f"No trained model found in {BASE}")
            return None, None
        model_path = items[-1]
        model_dir = model_path.parent
    
    print(f"Loading model from {model_dir.name}...")
    meta_path = model_dir / "meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    
    model = Fin-JEPA(11, meta.get('embed_dim', 64)).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model, model_dir


@torch.no_grad()
def compute_voe_single(model, df, stock, seq_len=60, max_steps=500):
    """Compute VoE curve for one stock."""
    sd = df[df['ticker'] == stock].sort_values('date').copy()
    vals = sd[FEATURE_COLS].values.astype(np.float32)
    if len(vals) < seq_len + 10:
        return pd.DataFrame()
    
    errors = []
    n = min(len(vals) - seq_len, max_steps)
    for i in range(n):
        ctx = torch.FloatTensor(vals[i:i+seq_len]).unsqueeze(0).to(DEVICE)
        tgt = torch.FloatTensor(vals[i+1:i+seq_len+6]).unsqueeze(0).to(DEVICE)  # 5 pred steps + 1
        if len(tgt[0]) < 5: continue
        out = model(ctx, tgt)
        errors.append({
            'date': sd.iloc[i+seq_len-1]['date'],
            'pred_loss': out['pred_loss'].item(),
            'ticker': stock,
        })
    return pd.DataFrame(errors)


def compute_anomaly_labels(df, stock):
    """Simple anomaly labels for comparison."""
    sd = df[df['ticker'] == stock].sort_values('date').copy()
    sd['ret'] = sd['close'].pct_change()
    sd['ret_fwd'] = sd['ret'].shift(-1)
    sd['vol_ratio'] = sd['volume'] / sd['volume'].rolling(20).mean()
    sd['range'] = (sd['high'] - sd['low']) / sd['close']
    ret_std = sd['ret'].std()
    sd['reversal'] = (sd['ret'].abs() > 2*ret_std) & (sd['ret_fwd'].abs() > ret_std)
    sd['vcp'] = sd['vol_ratio'] < 0.5
    sd['gap'] = (sd['open']/sd['close'].shift(1)-1).abs() > 0.03
    sd['any_anom'] = sd['reversal'] | sd['vcp'] | sd['gap']
    return sd[['date', 'ticker', 'any_anom', 'reversal', 'vcp', 'gap', 'ret', 'vol_ratio']]


def probe_factor_exposure(model, df):
    """Probe latent space for known factor exposures."""
    print("\n[Factor Exposure]")
    ds = JEPADataset(df, 60, 5, 5)
    loader = torch.utils.data.DataLoader(ds, 128, False, num_workers=0)
    z_all, info = [], []
    for batch in loader:
        ctx = batch['ctx'].to(DEVICE)
        z = model.encode_batch(ctx)[:, -1, :].cpu().numpy()
        z_all.append(z)
    Z = np.concatenate(z_all)
    
    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=5)
    Z_pca = pca.fit_transform(Z)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"  Top-5 components explain {pca.explained_variance_ratio_.sum():.1%}")
    
    # Check if first PC correlates with known factors
    # PC1 direction = most variance
    if Z.shape[0] > 100:
        from scipy.stats import pearsonr
        # Is PC1 correlated with market beta? (We can estimate from raw returns)
        raw_ret = np.stack([s[-1, 5] for s in ds.samples[:len(Z)]])  # returns at last day
        if len(raw_ret) < len(Z):
            Z = Z[:len(raw_ret)]
        pc1 = Z_pca[:len(Z), 0]
        corr, pv = pearsonr(pc1, raw_ret)
        print(f"  PC1 × return correlation: r={corr:.4f} (p={pv:.4f})")
    
    return Z


def main():
    print("═" * 60)
    print("Fin-JEPA Analysis Suite")
    print("═" * 60)
    
    # Load best model
    model, model_dir = load_best()
    if model is None:
        return
    
    # Load data
    print("\n[1] Data...")
    df = download_hs20(); df = add_features(df).dropna()
    stocks = df['ticker'].unique()
    print(f"  {len(df)} rows, {len(stocks)} stocks")
    
    # VoE Analysis
    print("\n[2] VoE Analysis (5 stocks)...")
    all_voe = []
    for s in stocks[:5]:
        voe = compute_voe_single(model, df, s)
        if len(voe):
            all_voe.append(voe)
            print(f"  {s}: {len(voe)} days")
    
    if all_voe:
        voe_full = pd.concat(all_voe)
        anom = pd.concat([compute_anomaly_labels(df, s) for s in stocks[:5]])
        merged = voe_full.merge(anom, on=['date','ticker'])
        merged['voe_z'] = (merged['pred_loss'] - merged['pred_loss'].rolling(60).mean()) / merged['pred_loss'].rolling(60).std()
        merged = merged.dropna()
        
        if merged['any_anom'].sum() > 0:
            auc = roc_auc_score(merged['any_anom'], merged['voe_z'])
            print(f"\n  VoE → Anomaly AUC: {auc:.3f}")
            for col, name in [('reversal','Reversal'),('vcp','VCP'),('gap','Gap')]:
                if merged[col].sum() > 0:
                    auc_t = roc_auc_score(merged[col], merged['voe_z'])
                    print(f"  VoE → {name}: AUC={auc_t:.3f}")
        
        # Top anomalies
        print("\n  Top-10 highest VoE days:")
        for _, row in merged.nlargest(10, 'pred_loss').iterrows():
            print(f"    {row['date']} {row['ticker']:10s} | loss={row['pred_loss']:.4f} "
                  f"rev={row['reversal']} vcp={row['vcp']} gap={row['gap']}")
    
    # Factor exposure probing
    Z = probe_factor_exposure(model, df)
    
    # Latent space structure
    print("\n[3] Latent Structure")
    from sklearn.manifold import TSNE
    Z_sample = Z[:1000] if len(Z) > 1000 else Z
    Z_tsne = TSNE(n_components=2, random_state=42).fit_transform(Z_sample)
    print(f"  t-SNE shape: {Z_tsne.shape}")
    # Check by stock
    print(f"  Active dimensions (std>0.01): {(Z.std(0) > 0.01).sum()}/{Z.shape[1]}")
    
    print(f"\n{'═'*60}\n✅ Analysis complete\n{'═'*60}")


if __name__ == "__main__":
    main()
