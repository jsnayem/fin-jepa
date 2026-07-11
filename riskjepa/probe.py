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

import forex_features as ff
import riskjepa.features as rjf
import riskjepa.metrics as rjm


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


# ── frozen-encoder probe (35-feature RiskJEPA checkpoint) ─────────────────────
def frozen_probe(df, ckpt, tau, device, spread_bars, c1_grid):
    import torch
    import model  # NB: original model; RiskJEPA checkpoint schema differs — see note
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    ta = ck.get('args', {})
    # NOTE: a true RiskJEPA checkpoint is built in new_model.py (not yet written).
    # This path expects a checkpoint whose 'args' carries n_features=35 (the
    # riskjepa schema). It is the GPU-retraining step's output.
    net = model.FinJEPA(
        n_features=ta.get('n_features', 35), embed_dim=ta.get('embed_dim', 64),
        encoder_layers=ta.get('enc_layers', 4), encoder_heads=ta.get('heads', 4),
        predictor_layers=ta.get('pred_layers', 6), predictor_heads=ta.get('heads', 4),
        sigreg_proj=ta.get('sigreg_proj', 512), sigreg_lambda=ta.get('sigreg_lambda', 0.1),
    ).to(device)
    net.load_state_dict(ck['model_state'])
    net.eval()

    ds, info = rjf.make_dataset(df)
    va_s = ds.starts[ds.split == 'val']
    ctxs, ys, tbs = [], [], []
    with torch.no_grad():
        for i in va_s:
            c = torch.FloatTensor(ds.feat[i - ds.ctx:i]).unsqueeze(0).to(device)
            z = net.encode_batch(c)           # (1, CTX, D)
            ctxs.append(z.mean(1).cpu().numpy().ravel())
            ys.append(ds.vret[i + ds.tgt - 1])
            tbs.append(ds.tb[i + ds.tgt - 1])
    Xva = np.array(ctxs); yva = np.array(ys); tbva = np.array(tbs)

    # probe head on the representation
    Xtr_coll, ytr_coll = [], []
    tr_s = ds.starts[ds.split == 'train']
    with torch.no_grad():
        for i in tr_s:
            c = torch.FloatTensor(ds.feat[i - ds.ctx:i]).unsqueeze(0).to(device)
            z = net.encode_batch(c)
            Xtr_coll.append(z.mean(1).cpu().numpy().ravel())
            ytr_coll.append(ds.vret[i + ds.tgt - 1])
    Xtr = np.array(Xtr_coll); ytr = np.array(ytr_coll)
    pred_va = ridge_predict(Xtr, ytr, Xva, lam=1.0)

    rows = rjm.sweep_c1(pred_va, yva, tb=tbva, c1_grid=c1_grid, spread_bars=spread_bars)
    return {'mode': 'frozen', 'ckpt': ckpt, 'info': info, 'backtest_sweep': rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/EURUSD_H1.csv')
    ap.add_argument('--ckpt', default=None,
                    help='35-feature RiskJEPA checkpoint (frozen-encoder mode).')
    ap.add_argument('--tau', type=int, default=rjf.TGT,
                    help='forward horizon (bars) — defaults to riskjepa TGT=12')
    ap.add_argument('--baseline', action='store_true',
                    help='feature-baseline probe on CPU (default if no --ckpt)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--spread_bars', type=float, default=0.10,
                    help='round-turn cost in vol-normalized return units (~0.1=1.8pip RT on 0.18%% vol)')
    ap.add_argument('--c1', type=str, default='0.3,0.5,0.75,1.0,1.5,2.0',
                    help='conviction thresholds to sweep (comma-separated)')
    ap.add_argument('--out', default='checkpoints/riskjepa/probe.json')
    args = ap.parse_args()

    c1_grid = [float(x) for x in args.c1.split(',')]
    device = torch.device(args.device) if args.ckpt else 'cpu'
    if args.ckpt:
        import torch  # noqa
    df = ff.load_eurusd_h1(args.data)

    if args.ckpt:
        res = frozen_probe(df, args.ckpt, args.tau, str(device),
                           args.spread_bars, c1_grid)
    else:
        res = baseline_probe(df, args.tau, 'cpu', args.spread_bars, c1_grid)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
