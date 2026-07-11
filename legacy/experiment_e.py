"""
Experiment E: JEPA Predictor Error as Trading Signal (Highly Optimized)

Hypothesis: When predictor error is HIGH, the market is in an unusual state 
(regime change, high volatility) - this itself could be a signal.

Tests:
1. Spearman correlation between predictor error and future volatility
2. AUC for predicting next-day direction using predictor error as feature
3. Whether high-error regimes (top 20% of error distribution) have 
   significantly different future returns
"""

import os, sys, json, time, gc, zipfile, io
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F

os.environ['HF_TOKEN'] = open(os.path.expanduser('~/.cache/huggingface/token')).read().strip()
from huggingface_hub import hf_hub_download

# ── Config ──
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
BATCH_SIZE = 256
MAX_STOCKS = 300  # Validate regime analysis
HIST = 3  # history size for model

print("=" * 60)
print("Experiment E: JEPA Predictor Error as Trading Signal")
print(f"Device: {DEVICE} | Max Stocks: {MAX_STOCKS}")
print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

t0 = time.time()

# ── Load Model from evaluate.py ──
print("\nLoading model components from evaluate.py...")
exec(open('evaluate.py').read().split("if __name__")[0])

# Load best model (A: lewm_v2_lr5e-3_full)
model = Fin-JEPA(pred_depth=6, history_size=HIST, self_cond=True).to(DEVICE)
ckpt_path = hf_hub_download('cedwyh/fin-jepa-h3', 'checkpoints/lewm_v2_lr5e-3_full/best.pt', repo_type='dataset')
state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
model.load_state_dict(state, strict=False)
model.eval()
print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
del state; gc.collect()


# ── Load Jinjing Features ──
print("\nLoading Jinjing features...")
jj = pd.read_parquet(hf_hub_download('cedwyh/jinjing-shared-data', 'all_features_v2.parquet', repo_type='dataset'))
jj['date'] = pd.to_datetime(jj['date']).dt.date
jj_syms = set(jj['symbol'].unique())
print(f"  {len(jj):,} rows, {len(jj_syms)} symbols")


# ── Load Stock Data from ZIP ──
print("\nLoading stock data from ZIP...")
zpath = hf_hub_download('perctrix/Stock-China-daily', 'daily.zip', repo_type='dataset')

FEATS = ['open', 'high', 'low', 'close', 'volume', 'returns', 'vwap',
         'volatility_20', 'vol_ma_20', 'close_ma_20', 'range_pct']

stock_data = []  # list of (symbol, date, window_65)

with zipfile.ZipFile(zpath) as z:
    csv_files = [f for f in z.namelist() if f.endswith('.csv')]
    print(f"  Scanning {len(csv_files)} CSVs...")

    done = 0
    for fname in sorted(csv_files):
        if not fname.endswith('.csv'):
            continue
        code = fname.replace('.csv', '')
        sym = (f'sh{code.split(".")[0]}' if '.XSHG' in code else
               f'sz{code.split(".")[0]}' if '.XSHE' in code else None)
        if not sym or sym not in jj_syms:
            continue

        csv_bytes = z.read(fname)
        g = pd.read_csv(io.BytesIO(csv_bytes))
        dc = [c for c in g.columns if 'Unnamed' in c]
        if dc:
            g = g.rename(columns={dc[0]: 'date'})
        g['date'] = pd.to_datetime(g['date'])
        g = g.sort_values('date')

        if 'factor' in g.columns:
            for c in ['open', 'high', 'low', 'close']:
                g[c] *= g['factor']

        g['returns'] = g['close'].pct_change()
        g['vwap'] = (g['high'] + g['low'] + g['close']) / 3
        g['volatility_20'] = g['returns'].rolling(20).std()
        g['vol_ma_20'] = g['volume'] / g['volume'].rolling(20).mean()
        g['close_ma_20'] = g['close'] / g['close'].rolling(20).mean()
        g['range_pct'] = (g['high'] - g['low']) / g['close']
        g = g.dropna(subset=FEATS)

        if len(g) < 66:
            continue

        vals = g[FEATS].values.astype(np.float32)
        mn = vals.mean(0)
        sd = vals.std(0) + 1e-8
        vn = (vals - mn) / sd

        jj_dates = set(jj[jj['symbol'] == sym]['date'])
        
        # Take last valid window for speed (representative sample)
        for i in range(len(vn) - 65):
            ed = g.iloc[i + 64]['date'].date()
            if ed in jj_dates:
                stock_data.append((sym, ed, vn[i:i + 65]))

        done += 1
        if done >= MAX_STOCKS:
            break

        if done % 10 == 0:
            print(f"    Processed {done} stocks ({time.time()-t0:.0f}s)...")

print(f"  Total windows: {len(stock_data)}")

if len(stock_data) < 50:
    print(f"ERROR: Only {len(stock_data)} windows found")
    sys.exit(1)


# ── Batched Predictor Error Computation (k=1 only for speed) ──
print("\nComputing k=1 predictor error (batched)...")

RET_IDX = 5  # index of 'returns' feature
CLOSE_IDX = 3  # index of 'close' feature

@torch.no_grad()
def compute_k1_errors_batch(windows, batch_size=BATCH_SIZE):
    """Compute k=1 predictor errors in batches."""
    n = len(windows)
    errors = np.zeros(n, dtype=np.float32)
    
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_tensor = torch.from_numpy(np.stack(windows[batch_start:batch_end])).to(DEVICE)
        
        # Encode context and target
        ctx = batch_tensor[:, :HIST]  # (B, 3, 11)
        tgt = batch_tensor[:, HIST:HIST+1]  # (B, 1, 11)
        
        ctx_emb = model.encode(ctx)
        tgt_emb = model.encode(tgt)
        tgt_emb = F.layer_norm(tgt_emb, (tgt_emb.shape[-1],))
        
        # Predict
        pred = model.pred_proj(
            model.predictor(ctx_emb, ctx_emb if model.self_cond else None)
            .reshape(-1, 192)
        ).reshape(ctx_emb.shape)
        
        # MSE at k=1
        loss = F.mse_loss(pred[:, 0], tgt_emb[:, 0], reduction='none').mean(dim=1)
        errors[batch_start:batch_end] = loss.cpu().numpy()
        
        if (batch_end) % 10000 == 0 or batch_end == n:
            print(f"    Processed {batch_end}/{n} windows ({time.time()-t0:.0f}s)...")
    
    return errors

# Extract windows
windows = [w for _, _, w in stock_data]
errors_k1 = compute_k1_errors_batch(windows)

# Compute k=5 and k=20 for a subset for speed
print("\nComputing k=5,20 errors for subset...")
subset_size = min(5000, len(windows))
subset_idx = np.random.choice(len(windows), subset_size, replace=False)
subset_windows = [windows[i] for i in subset_idx]

@torch.no_grad()
def compute_ar_errors(windows_list, k):
    """Compute autoregressive errors at k steps for a small list."""
    losses = []
    for win in windows_list:
        seq = torch.from_numpy(win).float().unsqueeze(0).to(DEVICE)
        ctx = seq[:, :HIST]
        tgt = seq[:, HIST:HIST+k]
        
        if tgt.shape[1] < k:
            losses.append(np.nan)
            continue
        
        ctx_emb = model.encode(ctx)
        tgt_emb = model.encode(tgt)
        tgt_emb = F.layer_norm(tgt_emb, (tgt_emb.shape[-1],))
        
        # Autoregressive rollout
        current = ctx_emb
        for step in range(k):
            ctx_step = current[:, -HIST:] if current.shape[1] >= HIST else current
            z_next = model.predictor(ctx_step, ctx_step if model.self_cond else None)
            z_next = model.pred_proj(z_next[:, -1:].reshape(-1, 192)).reshape(-1, 1, 192)
            current = torch.cat([current, z_next], dim=1)
        
        pred_at_k = current[:, -1]
        actual_at_k = tgt_emb[:, -1]
        loss = F.mse_loss(pred_at_k, actual_at_k).item()
        losses.append(loss)
    
    return np.array(losses)

errors_k5 = np.full(len(windows), np.nan)
errors_k20 = np.full(len(windows), np.nan)

errors_k5_subset = compute_ar_errors(subset_windows, k=5)
errors_k20_subset = compute_ar_errors(subset_windows, k=20)

errors_k5[subset_idx] = errors_k5_subset
errors_k20[subset_idx] = errors_k20_subset

print(f"  Computed k=5,20 for {subset_size} windows")

# Free memory
del windows, subset_windows; gc.collect()
torch.mps.empty_cache() if DEVICE == 'mps' else None


# ── Create Results DataFrame ──
df_results = pd.DataFrame([
    {'symbol': s, 'date': d} for s, d, _ in stock_data
])
df_results['error_k1'] = errors_k1
df_results['error_k5'] = errors_k5
df_results['error_k20'] = errors_k20

print(f"\n  k=1: mean={np.nanmean(errors_k1):.6f}, std={np.nanstd(errors_k1):.6f}")
print(f"  k=5: mean={np.nanmean(errors_k5):.6f}, std={np.nanstd(errors_k5):.6f}")
print(f"  k=20: mean={np.nanmean(errors_k20):.6f}, std={np.nanstd(errors_k20):.6f}")


# ── Compute Future Metrics from Window Data ──
print("\nComputing future metrics from window data...")

future_volatility = []
next_day_return = []
max_drawdown_5d = []

for idx, (sym, dt, win) in enumerate(stock_data):
    returns = win[:, RET_IDX]
    fut_returns = returns[60:]
    
    fut_vol = np.nanstd(fut_returns) if len(fut_returns) >= 5 else np.nan
    future_volatility.append(fut_vol)
    
    next_ret = fut_returns[0] if len(fut_returns) > 0 else np.nan
    next_day_return.append(next_ret)
    
    cumret = np.nancumsum(fut_returns)
    peak = np.maximum.accumulate(cumret)
    dd = cumret - peak
    max_dd = np.min(dd) if len(dd) > 0 else np.nan
    max_drawdown_5d.append(max_dd)

df_results['future_volatility'] = future_volatility
df_results['next_day_return'] = next_day_return
df_results['max_drawdown_5d'] = max_drawdown_5d
df_results['next_day_direction'] = (df_results['next_day_return'] > 0).astype(int)

print(f"  Future volatility: mean={np.nanmean(future_volatility):.6f}")
print(f"  Next-day return: mean={np.nanmean(next_day_return):.6f}")

del stock_data; gc.collect()


# ── Analysis 1: Spearman Correlation ──
print("\n" + "=" * 60)
print("Analysis 1: Spearman Correlation")
print("=" * 60)

results = {}

for k_name, k_val in [('k1', 'error_k1'), ('k5', 'error_k5'), ('k20', 'error_k20')]:
    valid = df_results[[k_val, 'future_volatility']].dropna()
    if len(valid) > 30:
        corr, pval = stats.spearmanr(valid[k_val], valid['future_volatility'])
        print(f"  {k_name}: Spearman r = {corr:.4f} (p = {pval:.4e}), n = {len(valid)}")
        results[f'spearman_{k_name}'] = {'r': float(corr), 'p': float(pval), 'n': int(len(valid))}
    else:
        print(f"  {k_name}: Not enough samples")
        results[f'spearman_{k_name}'] = None


# ── Analysis 2: AUC for Direction Prediction ──
print("\n" + "=" * 60)
print("Analysis 2: AUC for Direction Prediction")
print("=" * 60)

for k_name, k_val in [('k1', 'error_k1'), ('k5', 'error_k5'), ('k20', 'error_k20')]:
    valid = df_results[[k_val, 'next_day_direction']].dropna()
    if len(valid) > 30:
        try:
            auc = roc_auc_score(valid['next_day_direction'], -valid[k_val])
            print(f"  {k_name}: AUC = {auc:.4f}, n = {len(valid)}")
            results[f'auc_{k_name}'] = {'auc': float(auc), 'n': int(len(valid))}
        except Exception as e:
            print(f"  {k_name}: Error computing AUC: {e}")
            results[f'auc_{k_name}'] = None
    else:
        print(f"  {k_name}: Not enough samples")
        results[f'auc_{k_name}'] = None


# ── Analysis 3: High-Error Regime Analysis ──
print("\n" + "=" * 60)
print("Analysis 3: High-Error Regime Analysis")
print("=" * 60)

df_results['error_combined'] = (
    df_results['error_k1'] / df_results['error_k1'].median() +
    df_results['error_k5'] / df_results['error_k5'].median() +
    df_results['error_k20'] / df_results['error_k20'].median()
)

high_threshold = df_results['error_combined'].quantile(0.80)
low_threshold = df_results['error_combined'].quantile(0.20)

high_error = df_results[df_results['error_combined'] >= high_threshold]
low_error = df_results[df_results['error_combined'] <= low_threshold]

print(f"  High error (>= 80th pctl): n={len(high_error)}")
print(f"  Low error (<= 20th pctl): n={len(low_error)}")

if len(high_error) > 10 and len(low_error) > 10:
    high_ret = high_error['next_day_return'].mean()
    low_ret = low_error['next_day_return'].mean()
    high_vol = high_error['future_volatility'].mean()
    low_vol = low_error['future_volatility'].mean()
    
    t_stat_ret, p_val_ret = stats.ttest_ind(high_error['next_day_return'].dropna(), 
                                             low_error['next_day_return'].dropna())
    
    print(f"\n  High-Error: ret={high_ret:.6f}, vol={high_vol:.6f}")
    print(f"  Low-Error: ret={low_ret:.6f}, vol={low_vol:.6f}")
    print(f"  Diff (High-Low): ret={high_ret - low_ret:.6f} (p={p_val_ret:.4e})")
    
    results['regime_analysis'] = {
        'high_error_n': int(len(high_error)),
        'low_error_n': int(len(low_error)),
        'high_error_mean_return': float(high_ret),
        'low_error_mean_return': float(low_ret),
        'return_diff': float(high_ret - low_ret),
        'return_p_value': float(p_val_ret),
        'high_error_mean_volatility': float(high_vol),
        'low_error_mean_volatility': float(low_vol),
        'volatility_diff': float(high_vol - low_vol),
    }
else:
    print("  Not enough samples for regime analysis")
    results['regime_analysis'] = None


# ── Summary ──
elapsed = time.time() - t0

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total stocks: {MAX_STOCKS}")
print(f"Total windows: {len(df_results)}")
print(f"Elapsed time: {elapsed:.1f}s")
print("\nKey Findings:")
for k in ['k1', 'k5', 'k20']:
    if f'spearman_{k}' in results and results[f'spearman_{k}']:
        r = results[f'spearman_{k}']['r']
        direction = "positive" if r > 0 else "negative"
        strength = "strong" if abs(r) > 0.3 else "moderate" if abs(r) > 0.1 else "weak"
        print(f"  {k}: {strength} {direction} correlation (r={r:.4f})")
    if f'auc_{k}' in results and results[f'auc_{k}']:
        auc = results[f'auc_{k}']['auc']
        predictive = "predictive" if abs(auc - 0.5) > 0.05 else "random"
        print(f"  {k}: AUC={auc:.4f} ({predictive})")

if 'regime_analysis' in results and results['regime_analysis']:
    ra = results['regime_analysis']
    sig = "SIGNIFICANT" if ra['return_p_value'] < 0.05 else "not significant"
    print(f"\nRegime: Return diff is {sig} (p={ra['return_p_value']:.4e})")


# ── Save Results ──
output = {
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'elapsed_seconds': round(elapsed, 1),
    'config': {
        'device': DEVICE,
        'batch_size': BATCH_SIZE,
        'max_stocks': MAX_STOCKS,
        'history_size': HIST,
    },
    'data_summary': {
        'n_windows': int(len(df_results)),
        'n_stocks': MAX_STOCKS,
    },
    'error_stats': {
        'k1': {'mean': float(np.nanmean(df_results['error_k1'])), 'std': float(np.nanstd(df_results['error_k1']))},
        'k5': {'mean': float(np.nanmean(df_results['error_k5'])), 'std': float(np.nanstd(df_results['error_k5']))},
        'k20': {'mean': float(np.nanmean(df_results['error_k20'])), 'std': float(np.nanstd(df_results['error_k20']))},
    },
    'results': results,
}

out_path = '/Users/hermes/dev/fin-jepa/output/exp_e_results.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: {out_path}")
print(f"Total runtime: {elapsed:.1f}s")
print("\nDone! ✅")
