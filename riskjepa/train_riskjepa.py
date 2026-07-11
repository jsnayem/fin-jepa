"""
riskjepa/train_riskjepa.py — Train RiskJEPA with profit-factor selection.

Implements the training + selection recipe from research/new_model_design.md:
  - Composite objective (model.RiskJEPA.forward already assembles it):
        L = mse(z_pred, z_tgt)            # JEPA future-latent SSL aux
          + λ_sig · sigreg                # collapse guard (isotropy)
          + β · NLL(μ, log s; r/σ)        # heteroskedastic return head
          + γ · triple_barrier_CE         # risk-reward kill-switch head
  - Walk-forward training: for each fold, train on the fold's (earlier) windows
    and evaluate the cost-aware backtest on the fold's val windows **every epoch**.
  - SELECTION METRIC = OOS profit-factor / Sharpe on a cost-aware backtest — NOT
    val_loss. We keep the checkpoint whose walk-forward PF is highest (best.pt),
    and also log val_loss for diagnosis.
  - Schedule: warmup (1/20 of steps) + cosine decay, lr 3e-4, AdamW, wd 1e-4
    (paper §4/P2). SIGReg λ ∈ {1.0, 2.0} (REPORT §17 FX regime).

The frozen-encoder probe (riskjepa.probe.frozen_probe) consumes the saved
checkpoints; best.pt is the profit-factor-selected model for deployment.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import riskjepa.features as rjf
import riskjepa.walkforward as rjwf
import riskjepa.model as rjm_model
import riskjepa.metrics as rjm_metrics
from riskjepa.features import RiskJEPADataset
import forex_features as ff  # for load_eurusd_h1 (the raw loader)


# ── probe forward fn (used both for training-time selection and eval) ──────────
@torch.no_grad()
def riskjepa_predict(ds, idx, net, device):
    """Forward the encoder over windows[idx]; return ret_pred / vret / tb / sigma.

    One batched forward over all windows (efficient); returns the vol-normalized
    return prediction (ret_pred), the realized vol-normalized return (vret), the
    triple-barrier sign (tb), and the per-sample uncertainty sigma (unc_head).
    """
    net.eval()
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


# ── per-epoch walk-forward selection proxy ─────────────────────────────────────
def val_backtest(net, ds, val_idx, device, spread_bars, c1_grid):
    out = riskjepa_predict(ds, val_idx, net, device)
    sweep = rjm_metrics.sweep_c1(out['y_pred'], out['y_true'], tb=out['tb'],
                                 c1_grid=c1_grid, spread_bars=spread_bars)
    best = max(sweep, key=lambda r: (r['profit_factor'] if np.isfinite(r['profit_factor']) else -1))
    return best, out


def cosine_warmup(optimizer, total_steps, warmup_steps, base_lr):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/EURUSD_H1.csv')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--embed_dim', type=int, default=64)
    ap.add_argument('--enc_conv_blocks', type=int, default=2)
    ap.add_argument('--patch_size', type=int, default=4)
    ap.add_argument('--pred_layers', type=int, default=4)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--sigreg_lambda', type=float, default=1.0)
    ap.add_argument('--sigreg_proj', type=int, default=512)
    ap.add_argument('--aux_lambda', type=float, default=0.5)
    ap.add_argument('--tb_lambda', type=float, default=0.3)
    ap.add_argument('--nll_lambda', type=float, default=0.3)
    ap.add_argument('--use_revin', action='store_true', default=True)
    ap.add_argument('--no_revin', dest='use_revin', action='store_false')
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--embago', type=int, default=None, help='override embargo (default = TGT)')
    ap.add_argument('--spread_bars', type=float, default=0.10)
    ap.add_argument('--c1_grid', type=str, default='0.5,1.0,2.0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--ckpt', default='checkpoints/riskjepa')
    ap.add_argument('--smoke', action='store_true',
                    help='tiny CPU check: 2 epochs, small batch, few windows')
    ap.add_argument('--fold', type=int, default=None,
                    help='train only this fold index (0-based); default = all folds')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.ckpt, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    c1_grid = [float(x) for x in args.c1_grid.split(',')]
    print('device:', device)

    df = ff.load_eurusd_h1(args.data)
    ds, info = rjf.make_dataset(df)
    print('dataset:', {k: info[k] for k in ('n_features', 'n_train', 'n_val', 'cutoff')})
    embargo = args.embago if args.embago is not None else ds.tgt
    folds = rjwf.fold_splits(ds.starts, n_folds=args.n_folds, embargo=embargo)
    print(f'{len(folds)} fold(s) (expanding, embargo={embargo})')

    if args.smoke:
        args.epochs = 2
        args.batch = 64
        # restrict to a small slice of windows for CPU test
        ds.starts = ds.starts[:2000]
        ds.split = np.array(['train'] * len(ds.starts))
        folds = [(np.arange(0, 1200), np.arange(1200, 2000))]
        print('SMOKE: small windows, epochs=2, batch=64')

    fold_list = [args.fold] if args.fold is not None else range(len(folds))

    all_results = {}
    for fk in fold_list:
        tr, va = folds[fk]
        if tr.size == 0 or va.size == 0:
            print(f'fold {fk}: skipped (empty train/val)')
            continue
        print(f'\n===== FOLD {fk}: n_train={tr.size} n_val={va.size} =====')
        tr_loader = DataLoader(Subset(ds, tr), batch_size=args.batch, shuffle=True,
                               num_workers=args.workers, drop_last=False)

        net = rjm_model.RiskJEPA(
            n_features=info['n_features'], embed_dim=args.embed_dim,
            enc_conv_blocks=args.enc_conv_blocks, patch_size=args.patch_size,
            predictor_layers=args.pred_layers, predictor_heads=args.heads,
            sigreg_proj=args.sigreg_proj, sigreg_lambda=args.sigreg_lambda,
            use_revin=args.use_revin, aux_lambda=args.aux_lambda,
            tb_lambda=args.tb_lambda, nll_lambda=args.nll_lambda,
            horizon=ds.tgt,
        ).to(device)
        n_params = sum(p.numel() for p in net.parameters())
        print('params:', f'{n_params:,}')

        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
        total_steps = max(1, args.epochs * len(tr_loader))
        sched = cosine_warmup(opt, total_steps, max(1, total_steps // 20), args.lr)

        fdir = os.path.join(args.ckpt, f'fold{fk}')
        os.makedirs(fdir, exist_ok=True)
        best_pf = -1.0
        best_ep = -1
        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            net.train()
            run = {'loss': 0., 'pred': 0., 'sig': 0., 'ret': 0., 'nll': 0., 'tb': 0.}
            steps = 0
            for b in tr_loader:
                ctx = b['ctx'].to(device)
                tgt = b['tgt'].to(device)
                y = b['y'].to(device=device, dtype=torch.float32)
                y_tb = b['y_tb'].to(device=device, dtype=torch.float32)
                out = net(ctx, tgt, y=y, y_tb=y_tb)
                loss = out['loss']
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
                run['loss'] += float(loss.detach())
                run['pred'] += float(out.get('pred_loss', torch.zeros(())).detach())
                run['sig'] += float(out.get('sigreg_loss', torch.zeros(())).detach())
                run['ret'] += float(out.get('ret_loss', torch.zeros(())).detach())
                run['nll'] += float(out.get('nll_loss', torch.zeros(())).detach())
                run['tb'] += float(out.get('tb_loss', torch.zeros(())).detach())
                steps += 1
            tr_loss = run['loss'] / max(1, steps)

            # ── selection: cost-aware backtest on the fold's val windows ──
            net.eval()
            best, _ = val_backtest(net, ds, va, device, args.spread_bars, c1_grid)
            dt = time.time() - t0
            print(f'  ep{ep} [{dt:.0f}s] tr_loss={tr_loss:.4f} '
                  f'(pred={run["pred"]/max(1,steps):.4f},sig={run["sig"]/max(1,steps):.4f},'
                  f'ret={run["ret"]/max(1,steps):.4f},nll={run["nll"]/max(1,steps):.4f},'
                  f'tb={run["tb"]/max(1,steps):.4f}) | '
                  f'VAL PF={best["profit_factor"]:.3f} Sharpe={best["sharpe"]:.2f} '
                  f'win%={best["winrate"]*100:.1f} %flat={best["pct_flat"]*100:.1f} '
                  f'rankIC={best["rankIC"]:+.3f}')

            # keep the profit-factor-selected checkpoint
            pf = best['profit_factor'] if np.isfinite(best['profit_factor']) else -1.0
            if pf > best_pf:
                best_pf = pf
                best_ep = ep
                torch.save({
                    'model_state': net.state_dict(),
                    'args': vars(args), 'fold': fk, 'epoch': ep,
                    'val_metrics': best, 'n_params': n_params,
                }, os.path.join(fdir, 'best.pt'))
                print('    -> saved best.pt (PF-selected)')

            # always keep last
            torch.save({
                'model_state': net.state_dict(), 'args': vars(args),
                'fold': fk, 'epoch': ep, 'n_params': n_params,
            }, os.path.join(fdir, 'last.pt'))

        all_results[fk] = {'best_val_pf': best_pf, 'best_epoch': best_ep,
                           'n_train': int(tr.size), 'n_val': int(va.size)}

    with open(os.path.join(args.ckpt, 'train_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print('\nfold results:', json.dumps(all_results, indent=2))
    print('done.')


if __name__ == '__main__':
    main()
