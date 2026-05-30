#!/usr/bin/env python3
"""Run: python prepare_dataset.py"""
import os, sys, gc, json, time, zipfile, io, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
warnings.filterwarnings("ignore")

FEATURE_COLS = ['open','high','low','close','volume','returns','vwap',
                'volatility_20','vol_ma_20','close_ma_20','range_pct']
N_WORKERS, TARGET, OUTPUT_REPO = 4, 800, "cedwyh/fin-jepa-data"

def process_stock(args):
    i, (sname, csv_bytes) = args
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
        dc = [c for c in df.columns if 'Unnamed' in c]
        if dc: df = df.rename(columns={dc[0]:'date'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df['ticker'] = sname.replace('.csv','')
        if 'factor' in df.columns:
            for c in ['open','high','low','close']:
                df[c] = df[c] * df['factor']
        df['returns'] = df['close'].pct_change()
        df['vwap'] = (df['high']+df['low']+df['close'])/3
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['vol_ma_20'] = df['volume']/df['volume'].rolling(20).mean()
        df['close_ma_20'] = df['close']/df['close'].rolling(20).mean()
        df['range_pct'] = (df['high']-df['low'])/df['close']
        df = df.dropna(subset=FEATURE_COLS)
        if len(df) < 120: return None, i
        return df[['date','ticker']+FEATURE_COLS], i
    except Exception as e:
        return None, i

def main():
    from huggingface_hub import hf_hub_download, create_repo, upload_folder
    
    t0 = time.time()
    print("="*60, flush=True); print("Fin-JEPA Data Prep", flush=True); print("="*60, flush=True)

    # Download
    print("\n[1] Downloading...", flush=True)
    zpath = hf_hub_download("perctrix/Stock-China-daily", "daily.zip", repo_type="dataset")
    zsize = os.path.getsize(zpath)/1024/1024
    print(f"  {zpath} ({zsize:.0f}MB)", flush=True)

    # Scan & select
    print("\n[2] Reading stocks...", flush=True)
    with zipfile.ZipFile(zpath) as z:
        all_stocks = sorted([f for f in z.namelist() if f.endswith('.csv')])
        stocks = all_stocks[:TARGET]
        csv_data = [(s, z.read(s)) for s in stocks]
    print(f"  {len(csv_data)} stocks loaded ({time.time()-t0:.0f}s)", flush=True)

    # Process
    print(f"\n[3] Processing ({N_WORKERS} workers)...", flush=True)
    dfs = []
    with Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(process_stock, enumerate(csv_data))):
            df_i, idx = r
            if df_i is not None: dfs.append(df_i)
            if (i+1) % 200 == 0:
                print(f"  {i+1}/{len(stocks)} kept={len(dfs)} ({time.time()-t0:.0f}s)", flush=True)

    combined = pd.concat(dfs, ignore_index=True)
    del dfs; gc.collect()
    n_stocks = combined['ticker'].nunique()
    print(f"\n  Result: {len(combined):,} rows, {n_stocks} stocks", flush=True)
    print(f"  Date: {combined['date'].min()} → {combined['date'].max()}", flush=True)

    # Normalize
    print(f"\n[4] Normalizing...", flush=True)
    vals = combined[FEATURE_COLS].values.astype(np.float32)
    means = np.nanmean(vals, axis=0)
    stds = np.nanstd(vals, axis=0) + 1e-8
    for i, c in enumerate(FEATURE_COLS):
        combined[c] = (combined[c].astype(np.float32) - means[i]) / stds[i]
    print(f"  Means={means.round(3)}", flush=True)
    print(f"  Stds={stds.round(3)}", flush=True)

    # Sequences
    print(f"\n[5] Building sequences...", flush=True)
    seqs = []
    for ticker, grp in combined.groupby('ticker'):
        v = grp.sort_values('date')[FEATURE_COLS].values.astype(np.float32)
        for i in range(0, max(1, len(v) - 65 + 1), 5):
            c = v[i:i+65]
            if len(c) == 65 and not np.isnan(c).any():
                seqs.append(c)

    seq = np.stack(seqs).astype(np.float32)
    print(f"  {len(seq):,} sequences × {seq.shape[1]} steps × {seq.shape[2]} features", flush=True)
    print(f"  Size: {seq.nbytes/1024/1024:.0f}MB", flush=True)

    # Save
    print(f"\n[6] Saving shards...", flush=True)
    outdir = Path("/tmp/fin-jepa-dataset")
    outdir.mkdir(exist_ok=True)
    shard_size = 100000
    n_shards = (len(seq) + shard_size - 1) // shard_size
    for si in range(n_shards):
        start = si * shard_size
        end = min(start + shard_size, len(seq))
        np.save(outdir / f"shard_{si:04d}.npy", seq[start:end])
        print(f"  shard {si+1}/{n_shards}: {start:,}→{end:,} ({end-start:,})", flush=True)
    
    meta = {
        'n_sequences': len(seq), 'seq_len': 60, 'pred_steps': 5,
        'n_features': len(FEATURE_COLS), 'feature_names': FEATURE_COLS,
        'n_stocks': n_stocks,
        'date_range': [str(combined['date'].min()), str(combined['date'].max())],
        'normalizer_mean': means.tolist(), 'normalizer_std': stds.tolist(),
    }
    json.dump(meta, open(outdir / "meta.json", "w"), indent=2)

    # Upload
    print(f"\n[7] Uploading to HF: {OUTPUT_REPO}...", flush=True)
    create_repo(OUTPUT_REPO, repo_type="dataset", exist_ok=True)
    upload_folder(
        str(outdir), OUTPUT_REPO, repo_type="dataset",
        commit_message=f"Fin-JEPA JEPA dataset: {len(seq):,} seqs, {n_stocks} stocks, {combined['date'].min()}–{combined['date'].max()}",
    )
    
    elapsed = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print(f"✅ Done: {elapsed/60:.1f}min", flush=True)
    print(f"   {len(seq):,} sequences, {n_stocks} stocks", flush=True)
    print(f"   Uploaded to: {OUTPUT_REPO}", flush=True)
    print(f"{'='*60}", flush=True)

if __name__ == "__main__":
    main()
