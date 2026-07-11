"""Benchmark JEPA models: inference speed for 5000 stocks."""
import os, sys, time, json, warnings
import numpy as np
import torch
from pathlib import Path
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.expanduser("~/dev/chan-jepa"))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BASE = Path(__file__).resolve().parent / "output"

BENCHMARK_BATCHES = [1, 128, 512, 1024, 2048, 4096]
SEQ_LEN, N_FEATURES = 60, 11

# Reuse model builders from compare_arch2
from compare_arch2 import (
    PriceEncoder_MLPDeep, PriceEncoder_CNN, MLPPredictor,
    TransformerPredictor, JEPA,
)

def build_model_from_meta(meta_path):
    meta = json.load(open(meta_path))
    name = meta['name']
    if name.startswith('arch_'): name = name[5:]

    # Original v1-v4
    if name in ['v1_tiny_d32', 'v2_base_d64', 'v3_large_d128', 'v4_deep_d64']:
        from model import Fin-JEPA as OrigJEPA
        model = OrigJEPA(N_FEATURES, meta['embed_dim'],
                         encoder_layers=meta.get('enc_layers', 3),
                         predictor_layers=meta.get('pred_layers', 4),
                         sigreg_proj=meta.get('sigreg_proj', meta['embed_dim']*2)).to(DEVICE)
        model.eval()
        return model, meta

    D = meta.get('embed_dim', 64)
    nproj = meta.get('sigreg_proj', 128)

    encoders = {
        'density_d48': PriceEncoder_MLPDeep(N_FEATURES, 48, 1),
        'density_d80': PriceEncoder_MLPDeep(N_FEATURES, 80, 1),
        'enc_deep6': PriceEncoder_MLPDeep(N_FEATURES, 64, 6),
        'enc_deep8': PriceEncoder_MLPDeep(N_FEATURES, 64, 8),
        'pred_l3': PriceEncoder_MLPDeep(N_FEATURES, 64, 1),
        'pred_l5': PriceEncoder_MLPDeep(N_FEATURES, 64, 1),
        'pred_l8': PriceEncoder_MLPDeep(N_FEATURES, 64, 1),
        'cnn_enc': PriceEncoder_CNN(N_FEATURES, 64),
        'cnn_pred6': PriceEncoder_CNN(N_FEATURES, 64),
        'mlp_pred': PriceEncoder_MLPDeep(N_FEATURES, 64, 1),
        'enc6_pred2': PriceEncoder_MLPDeep(N_FEATURES, 64, 6),
    }
    predictors = {
        'density_d48': TransformerPredictor(48, 4),
        'density_d80': TransformerPredictor(80, 4),
        'enc_deep6': TransformerPredictor(64, 4),
        'enc_deep8': TransformerPredictor(64, 4),
        'pred_l3': TransformerPredictor(64, 3),
        'pred_l5': TransformerPredictor(64, 5),
        'pred_l8': TransformerPredictor(64, 8),
        'cnn_enc': TransformerPredictor(64, 4),
        'cnn_pred6': TransformerPredictor(64, 6),
        'mlp_pred': MLPPredictor(64, 4),
        'enc6_pred2': TransformerPredictor(64, 2),
    }
    if name not in encoders:
        print(f"  ⚠ Skip {name}: unknown architecture")
        return None, meta

    model = JEPA(encoders[name], predictors[name], D, sigreg_proj=nproj).to(DEVICE)
    model.eval()
    return model, meta


def benchmark_model(model, name, n_warmup=5, n_runs=20):
    """Benchmark at multiple batch sizes. Returns {bs: metrics}."""
    results = {}
    for bs in BENCHMARK_BATCHES:
        ctx = torch.randn(bs, SEQ_LEN, N_FEATURES, device=DEVICE)
        try:
            with torch.no_grad():
                for _ in range(n_warmup):
                    model.encode(ctx)
            torch.mps.synchronize() if DEVICE == 'mps' else None

            # Benchmark encode only
            torch.mps.synchronize() if DEVICE == 'mps' else None
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_runs):
                    z = model.encode(ctx)
            torch.mps.synchronize() if DEVICE == 'mps' else None
            enc_time = (time.perf_counter() - t0) / n_runs

            # Benchmark full (encode + predict)
            torch.mps.synchronize() if DEVICE == 'mps' else None
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_runs):
                    z = model.encode(ctx)
                    model.predictor(z)
            torch.mps.synchronize() if DEVICE == 'mps' else None
            full_time = (time.perf_counter() - t0) / n_runs

            results[bs] = {
                'enc_ms': round(enc_time * 1000, 2),
                'full_ms': round(full_time * 1000, 2),
                'stocks_per_sec': round(bs / full_time, 1),
                'time_5000_s': round(5000 / (bs / full_time), 2),
            }
            print(f"    bs={bs:4d} | enc={enc_time*1000:6.2f}ms | full={full_time*1000:6.2f}ms | 5000x={results[bs]['time_5000_s']:.1f}s", flush=True)
        except Exception as e:
            print(f"    bs={bs:4d} | ERROR: {e}", flush=True)
            results[bs] = None
    return results


def main():
    print("═" * 70, flush=True)
    print(f"Fin-JEPA Inference Benchmark | Device: {DEVICE}", flush=True)
    print("═" * 70, flush=True)

    metas = sorted(BASE.glob("arch_*/meta.json"))
    if not metas:
        print("No models found!", flush=True); return

    all_results = []
    for meta_path in metas:
        name = meta_path.parent.name[5:]
        ckpt = meta_path.parent / "best.pt"
        if not ckpt.exists():
            print(f"\n  ⚠ {name}: no checkpoint, skip", flush=True); continue

        print(f"\n{'─'*70}", flush=True)
        print(f"Model: {name}", flush=True)
        model, meta = build_model_from_meta(meta_path)
        if model is None: continue

        try:
            state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state, strict=False)
            print(f"  Weights loaded | params={meta['params']:,} | val_loss={meta['best_va_loss']:.4f}", flush=True)
        except Exception as e:
            print(f"  Weight load FAILED: {e}", flush=True); continue

        perf = benchmark_model(model, name)

        # Pick optimal batch size (fastest time_5000)
        valid = {k: v for k, v in perf.items() if v is not None}
        if not valid: continue
        best_bs = max(valid, key=lambda k: valid[k]['stocks_per_sec'])
        best = valid[best_bs]

        all_results.append({
            'name': name,
            'params': meta['params'],
            'va_loss': meta['best_va_loss'],
            'enc_type': meta.get('encoder', '?'),
            'pred_type': meta.get('predictor', '?'),
            'embed_dim': meta.get('embed_dim', 64),
            'optimal_batch_size': best_bs,
            'time_5000_fastest_s': best['time_5000_s'],
            'stocks_per_sec': best['stocks_per_sec'],
            'performance': {str(k): v for k, v in valid.items()},
        })

    if not all_results:
        print("\nNo models benchmarked successfully!", flush=True); return

    # Summary
    all_results.sort(key=lambda r: r['time_5000_fastest_s'])
    print(f"\n{'='*70}", flush=True)
    print("Summary — sorted by speed (5000 stocks)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Name':15s} {'Params':>8s} {'Loss':8s} {'5000stk':8s} {'Stock/s':8s} {'BS':5s}", flush=True)
    print(f"{'─'*15} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*5}", flush=True)
    for r in all_results:
        print(f"{r['name']:15s} {r['params']:>8,} {r['va_loss']:.4f}  "
              f"{r['time_5000_fastest_s']:6.1f}s {r['stocks_per_sec']:>8.0f} {r['optimal_batch_size']:5d}", flush=True)

    # Pareto frontier
    print(f"\n{'='*70}", flush=True)
    print("Pareto frontier (no model is both faster AND more accurate):", flush=True)
    pareto = []
    for r in all_results:
        dominated = any(
            r2 is not r and r2['va_loss'] <= r['va_loss'] and r2['time_5000_fastest_s'] <= r['time_5000_fastest_s']
            and (r2['va_loss'] < r['va_loss'] or r2['time_5000_fastest_s'] < r['time_5000_fastest_s'])
            for r2 in all_results
        )
        if not dominated:
            pareto.append(r)
            print(f"  ✅ {r['name']:15s} | loss={r['va_loss']:.4f} | "
                  f"5000stk={r['time_5000_fastest_s']:.1f}s | {r['stocks_per_sec']:.0f} stk/s", flush=True)

    json.dump(all_results, open(BASE / "benchmark_results.json", "w"), indent=2)
    print(f"\nResults saved!", flush=True)


if __name__ == "__main__":
    main()
