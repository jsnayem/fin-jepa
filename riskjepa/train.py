"""
riskjepa/train.py — FinJEPA pretraining on the RiskJEPA (risk-reward) data path.

Mirrors train_forex_h1.py but:
  - uses riskjepa.features.make_dataset (35 features: 15 base + 17 alpha + 3 MR,
    CTX=48/TGT=12, vol-normalized forward-return label, triple-barrier sign,
    embargo) instead of forex_features.make_dataset.
  - enables the retargeted auxiliary head (aux_lambda > 0) so the encoder is
    shaped to predict the vol-normalized forward return (the RiskJEPA target),
    not the mega-alpha.
  - selection metric stays val loss (JEPA objective) for this step; profit-factor
    selection is the riskjepa/probe + backtest step.

Run (local / Kaggle):
  .venv/bin/python riskjepa/train.py --epochs 40 --batch 256 --aux_lambda 0.5 --sigreg_lambda 2.0
Checkpoint (35-feature schema) -> checkpoints/riskjepa/best.pt
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import riskjepa.features as rjf
import forex_features as ff
import model


def effective_rank(z):
    x = z.detach().reshape(-1, z.size(-1))
    x = x - x.mean(0, keepdim=True)
    if x.size(0) < 2:
        return float('nan')
    cov = (x.t() @ x) / (x.size(0) - 1)
    cov = cov / (cov.trace() + 1e-12)
    e = torch.linalg.eigvalsh(cov.clamp_min(0))
    e = e.clamp_min(0)
    p = e / (e.sum() + 1e-12)
    ent = -(p * (p + 1e-12).log()).sum()
    return float(torch.exp(ent))


def stdz(z):
    x = z.detach().reshape(-1, z.size(-1))
    return float(x.std(0).mean())


def evaluate(net, loader, device, aux_lambda):
    net.eval()
    tot = {'loss': 0., 'pred': 0., 'sig': 0., 'aux': 0.}
    n = 0
    er_buf, sz_buf = [], []
    with torch.no_grad():
        for b in loader:
            ctx = b['ctx'].to(device)
            tgt = b['tgt'].to(device)
            out = net(ctx, tgt)
            tot['loss'] += float(out['loss']) * ctx.size(0)
            tot['pred'] += float(out['pred_loss']) * ctx.size(0)
            tot['sig'] += float(out['sigreg_loss']) * ctx.size(0)
            if getattr(net, 'return_head', None) is not None and 'y' in b:
                y = b['y'].to(device=device, dtype=torch.float32)
                m = ~torch.isnan(y)
                if m.any():
                    tot['aux'] += float(F.mse_loss(out['ret_pred'][m], y[m]).detach()) * int(m.sum())
            er_buf.append(effective_rank(out['emb']))
            sz_buf.append(stdz(out['emb']))
            n += ctx.size(0)
    return {k: v / n for k, v in tot.items()}, np.mean(er_buf), np.mean(sz_buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/EURUSD_H1.csv')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--ctx', type=int, default=rjf.CTX)
    ap.add_argument('--tgt', type=int, default=rjf.TGT)
    ap.add_argument('--embed_dim', type=int, default=64)
    ap.add_argument('--enc_layers', type=int, default=4)
    ap.add_argument('--pred_layers', type=int, default=6)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--sigreg_lambda', type=float, default=2.0)
    ap.add_argument('--sigreg_proj', type=int, default=512)
    ap.add_argument('--aux_lambda', type=float, default=0.5,
                    help='weight for auxiliary vol-normalized forward-return head')
    ap.add_argument('--ckpt', default='checkpoints/riskjepa')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--log_every', type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.ckpt, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('device:', device, '| cuda_available=', torch.cuda.is_available())

    df = ff.load_eurusd_h1(args.data)
    ds, info = rjf.make_dataset(df, ctx=args.ctx, tgt=args.tgt)
    print('riskjepa dataset:', {k: info[k] for k in
                                ('n', 'n_features', 'n_train', 'n_val', 'cutoff')})
    assert info['n_features'] == 35, f"expected 35 features, got {info['n_features']}"

    tr_idx = np.where(ds.split == 'train')[0]
    va_idx = np.where(ds.split == 'val')[0]
    tr_loader = DataLoader(Subset(ds, tr_idx), batch_size=args.batch, shuffle=True,
                           num_workers=args.workers, drop_last=True)
    va_loader = DataLoader(Subset(ds, va_idx), batch_size=args.batch, shuffle=False,
                           num_workers=args.workers, drop_last=False)
    print(f'train batches {len(tr_loader)} | val batches {len(va_loader)}')

    net = model.FinJEPA(
        n_features=info['n_features'], embed_dim=args.embed_dim,
        encoder_layers=args.enc_layers, encoder_heads=args.heads,
        predictor_layers=args.pred_layers, predictor_heads=args.heads,
        sigreg_proj=args.sigreg_proj, sigreg_lambda=args.sigreg_lambda,
        aux_lambda=args.aux_lambda,
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print('params:', f'{n_params:,}')

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(ck['model_state'])
        opt.load_state_dict(ck['optimizer_state'])
        start_epoch = ck.get('epoch', 0)
        if 'rng' in ck:
            torch.set_rng_state(torch.from_numpy(ck['rng']['torch']))
            np.random.set_state(ck['rng']['numpy'])
            random.setstate(ck['rng']['random'])
        for g in opt.param_groups:
            g['lr'] = args.lr
        print(f'resumed from {args.resume} at epoch {start_epoch}')

    best_val = float('inf')
    for ep in range(start_epoch + 1, args.epochs + 1):
        net.train()
        t0 = time.time()
        run = {'loss': 0., 'pred': 0., 'sig': 0., 'aux': 0.}
        steps = 0
        for it, b in enumerate(tr_loader, 1):
            ctx = b['ctx'].to(device)
            tgt = b['tgt'].to(device)
            out = net(ctx, tgt)
            loss = out['loss']
            if getattr(net, 'return_head', None) is not None and 'y' in b:
                y = b['y'].to(device=device, dtype=torch.float32)
                m = ~torch.isnan(y)
                if m.any():
                    ret_loss = F.mse_loss(out['ret_pred'][m], y[m])
                    loss = loss + args.aux_lambda * ret_loss
                    run['aux'] += float(ret_loss.detach()) * int(m.sum())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            run['loss'] += float(loss)
            run['pred'] += float(out['pred_loss'])
            run['sig'] += float(out['sigreg_loss'])
            steps += 1
            if it % args.log_every == 0:
                print(f'  ep{ep} it{it}/{len(tr_loader)} '
                      f'loss={run["loss"]/steps:.4f} '
                      f'pred={run["pred"]/steps:.4f} sig={run["sig"]/steps:.4f} '
                      f'aux={run["aux"]/steps:.4f}')
        tr_loss = run['loss'] / steps
        val, val_er, val_sz = evaluate(net, va_loader, device, aux_lambda=args.aux_lambda)
        dt = time.time() - t0
        print(f'ep{ep} [{dt:.0f}s] '
              f'train loss={tr_loss:.4f}(pred={run["pred"]/steps:.4f},sig={run["sig"]/steps:.4f},aux={run["aux"]/steps:.4f}) | '
              f'val loss={val["loss"]:.4f}(pred={val["pred"]:.4f},sig={val["sig"]:.4f},aux={val["aux"]:.4f}) | '
              f'effR={val_er:.2f} stdZ={val_sz:.3f}')

        if val['loss'] < best_val:
            best_val = val['loss']
            torch.save({'model_state': net.state_dict(), 'args': vars(args),
                        'n_params': n_params}, os.path.join(args.ckpt, 'best.pt'))
            meta = {'n_params': n_params, 'epoch': ep, 'best_val_loss': best_val,
                    'val_pred_loss': val['pred'], 'val_sigreg_loss': val['sig'],
                    'val_aux_loss': val['aux'], 'val_eff_rank': val_er,
                    'val_stdZ': val_sz, **vars(args)}
            with open(os.path.join(args.ckpt, 'meta.json'), 'w') as f:
                json.dump(meta, f, indent=2)
            print('  -> saved best')

        log_path = os.path.join(args.ckpt, 'train_log.jsonl')
        with open(log_path, 'w' if ep == 1 else 'a') as lf:
            lf.write(json.dumps({
                'epoch': ep, 'tr_loss': round(tr_loss, 5),
                'val_loss': round(val['loss'], 5), 'val_pred': round(val['pred'], 5),
                'val_sig': round(val['sig'], 5), 'val_aux': round(val['aux'], 5),
                'eff_rank': round(val_er, 3), 'stdZ': round(val_sz, 4),
                'best': bool(val['loss'] < best_val),
            }) + '\n')

        torch.save({
            'model_state': net.state_dict(),
            'optimizer_state': opt.state_dict(),
            'epoch': ep,
            'rng': {
                'torch': torch.get_rng_state().numpy(),
                'numpy': np.random.get_state(),
                'random': random.getstate(),
            },
            'args': vars(args),
            'n_params': n_params,
        }, os.path.join(args.ckpt, 'last.pt'))
        print(f'  -> saved last.pt (epoch {ep})')

    print('done. best_val_loss=%.4f' % best_val)
    # final summary for Kaggle logs
    print('SUMMARY_JSON ' + json.dumps({'best_val_loss': round(best_val, 5),
                                        'eff_rank': round(val_er, 3),
                                        'stdZ': round(val_sz, 4),
                                        'n_params': n_params,
                                        'n_features': info['n_features'],
                                        'ctx': args.ctx, 'tgt': args.tgt}))


if __name__ == '__main__':
    main()
