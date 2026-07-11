"""
Fin-JEPA downstream probe + Value-of-Explanation (VoE) on hourly EUR/USD.

After pretraining (`train_forex_h1.py`), this:
  1. extracts context embeddings from the frozen FinJEPA encoder,
  2. trains a small probe head to predict the forward mega-alpha label tau bars
     ahead, reporting IC / rank-IC / R^2 / directional AUC on the validation split,
  3. computes the Value-of-Explanation: whether the JEPA latent prediction error
     is informative about the (alpha-based) forward move — the paper's VoE
     revisited with a *discriminative* label instead of raw forward return.

Run:
  .venv/bin/python probe_forex_h1.py --ckpt checkpoints/forex_h1/best.pt --tau 24
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import forex_features as ff
import model


# ── metrics (no hard sklearn dependency) ──────────────────────────────────────
def _rank(x):
    # average ranks for ties
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    # handle ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    for u in np.where(counts > 1)[0]:
        m = np.mean(ranks[inv == u])
        ranks[inv == u] = m
    return ranks


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    denom = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else float('nan')


def spearman(a, b):
    return pearson(_rank(a), _rank(b))


def roc_auc(pred, label):
    """label in {0,1}; pred higher => more likely 1."""
    pos = pred[label == 1]; neg = pred[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    # Mann-Whitney U -> AUC
    order = np.argsort(np.argsort(pred))  # ranks
    return float((order[label == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


# ── embedding extraction ──────────────────────────────────────────────────────
@torch.no_grad()
def extract(net, ds, starts, tau, tgt, device, batch=512, repr_kind='mean'):
    """Returns X (N,D) repr, y_mega (N,), and latent_err (N,) per window start."""
    Xs, Ys, Es = [], [], []
    net.eval()
    for i in range(0, len(starts), batch):
        s = starts[i:i + batch]
        ctx = torch.FloatTensor(np.stack([ds.feat[sj - ds.ctx:sj] for sj in s])).to(device)
        tgtw = torch.FloatTensor(np.stack([ds.feat[sj:sj + tgt] for sj in s])).to(device)
        out = net(ctx, tgtw)
        z_ctx = out['emb']  # (B, CTX, D)
        if repr_kind == 'last':
            X = z_ctx[:, -1]
        elif repr_kind == 'pred':
            X = out.get('pred', z_ctx[:, -1])[:, -1]
        else:  # mean pool
            X = z_ctx.mean(1)
        Xs.append(X.cpu().numpy())

        y_idx = s + tau
        y = ds.mega[y_idx]
        y[~((y_idx >= 0) & (y_idx < len(ds.mega)))] = np.nan
        Ys.append(y.copy())

        z_tgt = net.encode_batch(tgtw)
        z_pred = out.get('pred', None)
        if z_pred is None:
            z_pred = net.predictor(z_ctx)
        n = min(z_pred.size(1), z_tgt.size(1))
        err = F.mse_loss(z_pred[:, :n], z_tgt[:, :n], reduction='none').mean((-1, -2)).cpu().numpy()
        Es.append(err)
    X = np.concatenate(Xs, 0)
    y = np.concatenate(Ys, 0)
    err = np.concatenate(Es, 0)
    ok = ~np.isnan(y)
    return X[ok], y[ok], err[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='FinJEPA best.pt from train_forex_h1.py')
    ap.add_argument('--data', default='data/EURUSD_H1.csv')
    ap.add_argument('--tau', type=int, default=24, help='forward horizon (bars) for the mega-alpha label')
    ap.add_argument('--probe', default='mlp', choices=['linear', 'mlp'])
    ap.add_argument('--probe_epochs', type=int, default=30)
    ap.add_argument('--probe_lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--repr', default='mean', choices=['mean', 'last', 'pred'])
    ap.add_argument('--out', default='checkpoints/forex_h1/probe.json')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ── load pretrained FinJEPA ──
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ta = ckpt.get('args', {})
    net = model.FinJEPA(
        n_features=ta.get('n_features', 32), embed_dim=ta.get('embed_dim', 64),
        encoder_layers=ta.get('enc_layers', 4), encoder_heads=ta.get('heads', 4),
        predictor_layers=ta.get('pred_layers', 6), predictor_heads=ta.get('heads', 4),
        sigreg_proj=ta.get('sigreg_proj', 512), sigreg_lambda=ta.get('sigreg_lambda', 0.1),
    ).to(device)
    net.load_state_dict(ckpt['model_state'])
    print('loaded FinJEPA from', args.ckpt, '| params', f"{ckpt.get('n_params','?'):,}")

    # ── dataset ──
    df = ff.load_eurusd_h1(args.data)
    ds, info = ff.make_dataset(df, ctx=ta.get('ctx', ff.CTX), tgt=ta.get('tgt', ff.TGT))
    tgt = ta.get('tgt', ff.TGT)
    tr_starts = ds.starts[ds.split == 'train']
    va_starts = ds.starts[ds.split == 'val']
    print(f"extracting embeddings: train {len(tr_starts)} | val {len(va_starts)}")

    Xtr, ytr, _ = extract(net, ds, tr_starts, args.tau, tgt, device, args.batch, args.repr)
    Xva, yva, err_va = extract(net, ds, va_starts, args.tau, tgt, device, args.batch, args.repr)
    D = Xtr.shape[1]
    print(f"repr Xtr {Xtr.shape}, ytr nan-frac {np.isnan(ytr).mean():.3f}, val {Xva.shape}")

    # ── train probe head ──
    Xtr_t = torch.FloatTensor(Xtr); ytr_t = torch.FloatTensor(ytr).unsqueeze(1)
    Xva_t = torch.FloatTensor(Xva)
    head = (model.MLPProbe(D) if args.probe == 'mlp' else model.LinearProbe(D)).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.probe_lr)
    dl = DataLoader(list(zip(Xtr_t, ytr_t)), batch_size=args.batch, shuffle=True)
    for _ in range(args.probe_epochs):
        head.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.mse_loss(head(xb), yb).backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        pred_va = head(Xva_t.to(device)).cpu().numpy().ravel()

    # ── metrics ──
    ic = pearson(pred_va, yva)
    ric = spearman(pred_va, yva)
    r2v = r2(pred_va, yva)
    dir_label = (yva > np.median(yva)).astype(int)
    auc = roc_auc(pred_va, dir_label)

    # ── Value-of-Explanation (alpha-label VoE) ──
    # Does latent prediction error distinguish large vs small forward moves?
    mag = np.abs(yva)
    voe_ic = spearman(-err_va, mag)                 # higher error -> larger move?
    # binary extreme-move label (top/bottom 20% by |mega|)
    thr = np.quantile(mag, 0.8)
    extreme = (mag >= thr).astype(int)
    voe_auc = roc_auc(-err_va, extreme)
    # raw forward-return VoE (paper's own check, expected ~random)
    raw = yva  # mega-alpha already forward-looking; also compare to sign
    voe_raw_auc = roc_auc(-err_va, (raw > 0).astype(int))

    res = {
        'ckpt': args.ckpt, 'tau': args.tau, 'repr': args.repr, 'probe': args.probe,
        'n_train': int(len(Xtr)), 'n_val': int(len(Xva)),
        'probe_IC': ic, 'probe_rankIC': ric, 'probe_R2': r2v, 'probe_dirAUC': auc,
        'VoE_alpha_label_IC': voe_ic, 'VoE_alpha_label_AUC': voe_auc,
        'VoE_rawdir_AUC': voe_raw_auc,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
