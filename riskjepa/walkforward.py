"""
riskjepa/walkforward.py — Purged walk-forward (expanding-window) evaluation.

Implements the evaluation recipe from research/new_model_design.md + §4/P1
(paper_grounding_and_plan.md): instead of a single 90/10 split (one fragile,
leakage-prone estimate), split the timeline into K expanding folds. Each fold
trains on data up to a cutoff and evaluates on a held-out future window, with a
TGT-bar EMBARGO between the train end and the val start so context/label leakage
across the fold boundary is impossible.

This module is data-agnostic: it operates on a RiskJEPADataset (riskjepa.features)
and a forward function `predict(ds, idx, net, device)` supplied by the caller
(the trainer / probe). It returns per-fold cost-aware backtest metrics and an
aggregated mean±std table.

Walk-forward design (expanding windows, purged + embargoed):
  - time-ordered window starts -> sorted by index.
  - K fold boundaries partition the trainable range into K contiguous segments.
  - fold k: train = starts[:cut_k], val = starts[cut_k:] (expanding), then purge
    val windows within EMBARGO bars of cut_k, and also drop val windows that
    overlap the next fold's train (purge) so each val window is tested exactly
    once against the model trained on strictly earlier data.
"""
import numpy as np

import riskjepa.metrics as rjm


def fold_splits(starts, n_folds=5, embargo=12):
    """Return list of (train_idx, val_idx) index-arrays into `starts`.

    `starts` is the array of window start positions (already sorted by time, which
    make_dataset guarantees since it appends in increasing `i`). The timeline is
    divided into `n_folds` contiguous val segments. Fold k (0-indexed) uses val =
    segment k and train = ALL earlier windows (expanding window). Folds with no
    training data (the first) are dropped by the caller; this yields `n_folds-1`
    evaluated folds each tested once against a model trained only on strictly
    earlier data. Every val window is additionally purged by `embargo` bars from
    the train cutoff so a val window's context can never read train data.
    """
    starts = np.asarray(starts)
    n = len(starts)
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    folds = []
    for k in range(n_folds):
        va = np.arange(edges[k], edges[k + 1])          # this fold's val segment
        tr = np.arange(0, edges[k])                      # all earlier windows
        if va.size == 0:
            continue
        # embargo: drop val windows whose START is within `embargo` bars of the
        # last training window (prevents context leakage across the boundary).
        last_train = starts[tr[-1]] if tr.size else -10**9
        if tr.size:
            keep = starts[va] > last_train + embargo
            va = va[keep]
        folds.append((tr, va))
    return folds


def run_fold(net, ds, val_idx, device, predict_fn, spread_bars=0.10,
             c1_grid=None, use_tb=True, use_sigma=False):
    """Run the probe over `val_idx` windows, backtest, return (best_row, full_sweep).

    predict_fn(ds, idx_array, net, device) -> dict with keys:
        'y_pred' (N,) vol-normalized forward return prediction
        'y_true' (N,) realized vol-normalized forward return
        'tb'     (N,) triple-barrier sign (+1/-1/0)
        'sigma'  (N,) optional per-sample uncertainty (only if use_sigma)
    """
    out = predict_fn(ds, np.asarray(val_idx), net, device)
    y_pred = np.asarray(out['y_pred'], dtype=float)
    y_true = np.asarray(out['y_true'], dtype=float)
    tb = np.asarray(out['tb'], dtype=float) if use_tb else None
    sigma = np.asarray(out['sigma'], dtype=float) if (use_sigma and 'sigma' in out) else None

    sweep = rjm.sweep_c1(y_pred, y_true, tb=tb, c1_grid=c1_grid,
                         spread_bars=spread_bars, sigma=sigma)
    # pick the c1 with the best profit factor (the selection metric)
    best = max(sweep, key=lambda r: (r['profit_factor'] if np.isfinite(r['profit_factor']) else -1))
    return best, sweep


def walk_forward(net, ds, predict_fn, device='cpu', n_folds=5, embargo=None,
                 spread_bars=0.10, c1_grid=None, use_tb=True, use_sigma=False):
    """Run purged walk-forward over the dataset and aggregate metrics.

    Returns dict:
      'folds':  list of per-fold best rows
      'agg':    mean±std over folds for the key metrics
      'sweeps': list of full per-fold sweeps
    """
    if embargo is None:
        embargo = getattr(ds, 'tgt', 12)
    folds = fold_splits(ds.starts, n_folds=n_folds, embargo=embargo)

    fold_rows = []
    sweeps = []
    for k, (tr, va) in enumerate(folds):
        if tr.size == 0 or va.size == 0:
            print(f"  fold {k}: no train or val (skipped)")
            continue
        best, sweep = run_fold(net, ds, va, device, predict_fn,
                               spread_bars=spread_bars, c1_grid=c1_grid,
                               use_tb=use_tb, use_sigma=use_sigma)
        best = dict(best); best['fold'] = k
        best['n_train'] = int(tr.size); best['n_val'] = int(va.size)
        fold_rows.append(best)
        sweeps.append(sweep)
        print(f"  fold {k}: n_val={va.size} c1={best['c1']:.2f} "
              f"win%={best['winrate']*100:5.1f} PF={best['profit_factor']:.3f} "
              f"Sharpe={best['sharpe']:6.2f} %flat={best['pct_flat']*100:5.1f} "
              f"rankIC={best['rankIC']:+.3f}")

    if not fold_rows:
        return {'folds': [], 'agg': {}, 'sweeps': []}

    keys = ['n_trade', 'pct_flat', 'winrate', 'profit_factor', 'sharpe', 'rankIC', 'mean_pos']
    agg = {}
    for key in keys:
        vals = np.array([r[key] for r in fold_rows], dtype=float)
        # guard inf profit factor in the mean
        finite = np.isfinite(vals)
        agg[key] = {
            'mean': float(np.mean(vals[finite])) if finite.any() else float('nan'),
            'std': float(np.std(vals[finite])) if finite.any() else float('nan'),
            'n_finite': int(finite.sum()),
        }
    return {'folds': fold_rows, 'agg': agg, 'sweeps': sweeps}


if __name__ == "__main__":
    # quick self-test: synthetic dataset + linear predictor to confirm the
    # walk-forward plumbing (splits, per-fold backtest, aggregation) runs.
    import numpy as np
    from riskjepa.features import RiskJEPADataset

    rng = np.random.default_rng(0)
    n = 6000
    F = 4
    feat = rng.standard_normal((n, F)).astype(np.float32)
    vret = rng.standard_normal(n).astype(np.float32) * 0.01
    tb = np.where(np.abs(vret) > 0.3 * np.std(vret), np.sign(vret), 0.0).astype(np.float32)
    starts = np.arange(120, n - 12 + 1)
    split = np.array(['train'] * len(starts))
    ds = RiskJEPADataset(feat, vret, tb, starts, split, ctx=48, tgt=12)

    # a constant weak predictor: y_pred = mean of first feature over context
    def predict_fn(ds, idx, net, device):
        ctxs = np.stack([ds.feat[i - ds.ctx:i].mean(0) for i in idx])
        y_pred = ctxs[:, 0].copy()
        y_true = np.array([ds.vret[i + ds.tgt - 1] for i in idx])
        tbv = np.array([ds.tb[i + ds.tgt - 1] for i in idx])
        return {'y_pred': y_pred, 'y_true': y_true, 'tb': tbv}

    res = walk_forward(None, ds, predict_fn, device='cpu', n_folds=5,
                       spread_bars=0.10, c1_grid=[0.5, 1.0, 2.0])
    print("agg:", {k: (round(v['mean'], 4), round(v['std'], 4)) for k, v in res['agg'].items()})
    print("✅ walk-forward plumbing OK")
