"""
riskjepa/probe.py — Validate the RiskJEPA (risk-reward) data path on real EUR/USD.

Two modes:
  1. --baseline  (DEFAULT, CPU): fit a tiny ridge/linear probe directly on the
     35-feature context (no SSL encoder) and run the cost-aware risk-reward
     backtest (riskjepa.metrics). Answers the key question NOW, on CPU: does
     the vol-normalized forward-return label + triple-barrier kill-switch give
     a positive profit factor on the validation split? This is the honest first
     test of the risk-reward thesis.

  2. frozen encoder: load a 35-feature RiskJEPA checkpoint (retrained on the
     riskjepa schema — GPU step, not yet available) and probe its context
     embedding the same way. Activated with --ckpt <path>.

Run:
  .venv/bin/python riskjepa/probe.py --baseline --data data/EURUSD_H1.csv
  .venv/bin/python riskjepa/probe.py --ckpt checkpoints/riskjepa/best.pt
"""
import argparse
import json
import os

import numpy as np
import torch

import forex_features as ff
import riskjepa.features as rjf
import riskjepa.metrics as rjm
import riskjepa.model as rjm_model
import riskjepa.walkforward as rjwf


# ── ranking metrics (no sklearn) ─────────────────────────────────────────────
def _rank(x):
    order = np.argsort(x)
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(1, len(x) + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    for u in np.where(counts > 1)[0]:
        r[inv == u] = r[inv == u].mean()
    return r


def spearman(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float('nan')


# ── feature baseline probe (CPU) ─────────────────────────────────────────────
def ridge_predict(Xtr, ytr, Xva, lam=1.0):
    """Closed-form ridge regression probe (fast, CPU, deterministic)."""
    Xt = np.concatenate([Xtr, np.ones((len(Xtr), 1))], -1)
    Xv = np.concatenate([Xva, np.ones((len(Xva), 1))], -1)
    A = Xt.T @ Xt + lam * np.eye(Xt.shape[1])
    w = np.linalg.solve(A, Xt.T @ ytr)
    return Xv @ w


def baseline_probe(df, tau, device, spread_bars, c1_grid):
    ds, info = rjf.make_dataset(df)
    print("riskjepa dataset:", {k: info[k] for k in
                                ('n_features', 'n_train', 'n_val', 'cutoff')})

    # build (ctx_flat, y, tb) arrays per split
    def collect(starts):
        ctxs, ys, tbs = [], [], []
        for i in starts:
            ctx = ds.feat[i - ds.ctx:i].ravel()      # (CTX*F,)
            ctxs.append(ctx)
            ys.append(ds.vret[i + ds.tgt - 1])
            tbs.append(ds.tb[i + ds.tgt - 1])
        return np.array(ctxs), np.array(ys), np.array(tbs)

    tr_s = ds.starts[ds.split == 'train']
    va_s = ds.starts[ds.split == 'val']
    Xtr, ytr, _ = collect(tr_s)
    Xva, yva, tbva = collect(va_s)

    pred_va = ridge_predict(Xtr, ytr, Xva, lam=1.0)

    ic = np.corrcoef(pred_va, yva)[0, 1]
    ric = spearman(pred_va, yva)
    dir_label = (yva > np.median(yva)).astype(int)
    # directional AUC via Mann-Whitney (sign of pred vs binary label)
    pos = pred_va[dir_label == 1]; neg = pred_va[dir_label == 0]
    order = np.argsort(np.argsort(np.concatenate([pos, neg])))
    auc = float((order[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                / (len(pos) * len(neg))) if len(pos) and len(neg) else float('nan')

    print(f"\n[baseline probe] val IC={ic:.4f} rankIC={ric:.4f} dirAUC={auc:.4f}")

    print("\n[risk-reward backtest sweep — val split, cost-aware]")
    rows = rjm.sweep_c1(pred_va, yva, tb=tbva,
                        c1_grid=c1_grid, spread_bars=spread_bars)
    hdr = f"{'c1':>5} {'n_trade':>8} {'%flat':>7} {'win%':>6} {'PF':>7} {'Sharpe':>7} {'rankIC':>7} {'mean|pos|':>9}"
    print(hdr)
    for r in rows:
        print(f"{r['c1']:>5.2f} {r['n_trade']:>8} {r['pct_flat']*100:>6.1f}% "
              f"{r['winrate']*100:>5.1f}% {r['profit_factor']:>7.3f} "
              f"{r['sharpe']:>7.2f} {r['rankIC']:>7.3f} {r['mean_pos']:>9.3f}")

    return {
        'mode': 'baseline',
        'tau': tau,
        'info': info,
        'probe_IC': ic, 'probe_rankIC': ric, 'probe_dirAUC': auc,
        'spread_bars': spread_bars,
        'backtest_sweep': rows,
    }


# ── frozen RiskJEPA probe + walk-forward backtest (GPU-trained checkpoint) ─────
def frozen_probe(df, ckpt, tau, device, spread_bars, c1_grid, n_folds=5, use_sigma=False):
    """Load a RiskJEPA checkpoint and run the purged walk-forward cost-aware

    backtest with profit-factor / Sharpe selection (the model's own selection
    metric). Replaces the old ridge-probe-on-encoder path — RiskJEPA has its own
    ret_head / unc_head / tb_head, so we evaluate those directly.
    """
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    ta = ck.get('args', {})
    # rebuild the exact architecture from the checkpoint's saved args
    net = rjm_model.RiskJEPA(
        n_features=ta.get('n_features', 35),
        embed_dim=ta.get('embed_dim', 64),
        enc_conv_blocks=ta.get('enc_conv_blocks', 2),
        patch_size=ta.get('patch_size', 1),
        predictor_layers=ta.get('pred_layers', 4),
        predictor_heads=ta.get('heads', 4),
        sigreg_proj=ta.get('sigreg_proj', 512),
        sigreg_lambda=ta.get('sigreg_lambda', 1.0),
        use_revin=ta.get('use_revin', True),
        aux_lambda=ta.get('aux_lambda', 0.5),
        tb_lambda=ta.get('tb_lambda', 0.3),
        nll_lambda=ta.get('nll_lambda', 0.3),
        horizon=tau,
    ).to(device)
    net.load_state_dict(ck['model_state'])
    net.eval()

    ds, info = rjf.make_dataset(df)
    print(f"riskjepa dataset: { {k: info[k] for k in ('n_features', 'n_train', 'n_val', 'cutoff')} }")

    embargo = int(ta.get('embago') or ds.tgt)
    print(f"\n[walk-forward backtest] {n_folds} folds, embargo={embargo}, "
          f"spread={spread_bars}, use_sigma={use_sigma}")
    res = rjwf.walk_forward(net, ds, rjwf_predict_for_probe, device=str(device),
                            n_folds=n_folds, embargo=embargo,
                            spread_bars=spread_bars, c1_grid=c1_grid,
                            use_tb=True, use_sigma=use_sigma)
    return {'mode': 'frozen', 'ckpt': ckpt, 'info': info, **res}


@torch.no_grad()
def rjwf_predict_for_probe(ds, idx, net, device):
    """Probe forward fn compatible with riskjepa.walkforward.run_fold."""
    idx = np.asarray(idx)
    C = torch.stack([
        torch.from_numpy(np.ascontiguousarray(ds.feat[i - ds.ctx:i])).float()
        for i in idx
    ]).to(device)
    o = net(C)
    y_true = np.array([ds.vret[i + ds.tgt - 1] for i in idx], dtype=float)
    tb = np.array([ds.tb[i + ds.tgt - 1] for i in idx], dtype=float)
    return {
        'y_pred': o['ret_pred'].detach().cpu().numpy().astype(float),
        'y_true': y_true,
        'tb': tb,
        'sigma': o['sigma'].detach().cpu().numpy().astype(float),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/EURUSD_H1.csv')
    ap.add_argument('--ckpt', default=None,
                    help='RiskJEPA checkpoint to evaluate via walk-forward backtest.')
    ap.add_argument('--tau', type=int, default=rjf.TGT,
                    help='forward horizon (bars) — defaults to riskjepa TGT=12')
    ap.add_argument('--baseline', action='store_true',
                    help='feature-baseline probe on CPU (default if no --ckpt)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--spread_bars', type=float, default=0.10,
                    help='round-turn cost in vol-normalized return units (~0.1=1.8pip RT on 0.18%% vol)')
    ap.add_argument('--c1', type=str, default='0.3,0.5,0.75,1.0,1.5,2.0',
                    help='conviction thresholds to sweep (comma-separated)')
    ap.add_argument('--n_folds', type=int, default=5,
                    help='number of walk-forward folds (frozen mode)')
    ap.add_argument('--use_sigma', action='store_true',
                    help='use the unc_head uncertainty for the |y|<c1*sigma kill-switch')
    ap.add_argument('--out', default='checkpoints/riskjepa/probe.json')
    args = ap.parse_args()

    c1_grid = [float(x) for x in args.c1.split(',')]
    device = args.device
    if args.ckpt:
        import torch  # noqa
    df = ff.load_eurusd_h1(args.data)

    if args.ckpt:
        res = frozen_probe(df, args.ckpt, args.tau, device,
                           args.spread_bars, c1_grid,
                           n_folds=args.n_folds, use_sigma=args.use_sigma)
        # pretty-print the aggregate
        agg = res.get('agg', {})
        print('\n[walk-forward aggregate — mean ± std across folds]')
        for k, v in agg.items():
            print(f"  {k:>10}: {v['mean']:+.4f} ± {v['std']:.4f} (n={v['n_finite']})")
    else:
        res = baseline_probe(df, args.tau, 'cpu', args.spread_bars, c1_grid)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
