# Fin-JEPA — Grounding in the Source Paper + Consolidated Plan

*This addendum reconciles the three research proposals (`jepa_proposal.md`,
`fin_jepa_proposal.md`, `timeseries_proposal.md`) against the actual source
spec: `docs/jepa_full_draft.md` (Wang, "Fin-JEPA", github.com/cedricwyh/fin-jepa).
It corrects a mischaracterization from the earlier repo analysis and produces a
single prioritized, paper-aware implementation plan.*

## 1. What the paper actually says (ground truth)

- **Domain:** DAILY equities (6,230 stocks, 2010–2026, 2.07M samples). 22 base
  features (11 used in ablations). NOT forex, NOT hourly.
- **Architecture (v4_deep_d64 — the validated best):** D=64, encoder = 4-layer
  MLP (Linear→GELU→Linear×, LayerNorm, no dropout), predictor = 6-layer causal
  Transformer (4 heads, mlp_scale=4, norm_first), SIGReg λ=0.1, 367K params.
  The repo's `model.py` defaults (enc=4, pred=6, heads=4, D=64) **match this
  faithfully.** ✅
- **Training (validated):** AdamW lr=5e-4, weight_decay=1e-5, batch=512,
  linear warmup 1 epoch + cosine decay, 50 epochs, A10G. CTX=30 / TGT=10.
- **Collapse metrics (healthy):** stdZ=1.35, eff_rank=37.8/64 (threshold >16).
- **Latent prediction:** beats identity baseline by 10.1% (Z_pred MSE 0.2922
  vs 0.3249) → "financial time series contain learnable temporal structure."
- **Downstream VoE (the key result):** AUC≈0.49–0.50, |Spearman r|<0.04 at
  horizons {1,5,20}d, regime p=0.22. **No tradable directional signal.** This is
  the paper's own finding — so the repo's dirAUC≈0.50 is NOT a bug, it is a
  faithful reproduction of the null result.
- **Explicit future work (paper §4.5/§6):** SIGReg ablation, PELT regime
  detection, multi-horizon training (τ∈{5,10,20}), baseline comparison,
  cross-sectional/cross-attention, pre-train+fine-tune.

## 2. Repo-vs-paper discrepancies (what the forex adaptation changed)

| Aspect | Paper (validated) | This repo (forex) | Note |
|---|---|---|---|
| Data | daily equities, multi-stock | hourly EUR/USD, single pair | domain shift; single-pair FX is intrinsically lower-rank |
| Features | 22 (11 used) | 32 (15 base + 17 alpha) | richer; many collinear |
| CTX/TGT | 30 / 10 | 120 / 24 | reasonable for hourly |
| lr | 5e-4 | 1e-4 | repo lower |
| weight_decay | 1e-5 | 1e-4 | repo 10× higher |
| batch | 512 | 256 | repo smaller |
| schedule | warmup 1 + cosine | constant | repo no schedule |
| epochs | 50 | 40 | close |
| SIGReg λ | 0.1 | 0.1→2.0 (sweep) | **λ needed raising on FX** |
| eff_rank @ λ=0.1 | 37.8/64 | 5.67/64 | FX intrinsically lower-rank |
| aux head | absent | present (§18) | experimental, paper-unvalidated |

Key reconciliation: the paper reaches eff_rank 37.8 at λ=0.1 because equities
span thousands of stocks (high intrinsic rank). A single FX pair cannot reach
that — REPORT §17's λ sweep (5.67→14.77/64) is the right FX regime, and the
proposals' eff_rank targets of 25–40/64 are likely **optimistic for one pair**;
15–25/64 (fin_jepa_proposal) is the more honest ceiling.

## 3. Correction on the legacy/ files

Earlier I called `ijepa.py / leworldmodel.py / model_lewm.py / compare_arch* /
experiment_e.py / hf/* / output/*` "orphaned foreign files." That was inaccurate.
Per the paper's reproducibility statement (§Reproducibility, §4.2), these ARE the
upstream Fin-JEPA reference assets (`cedricwyh/fin-jepa`): `compare_arch.py`
trains the 15 ablation variants; `output/arch_*/meta.json` are the ablation
results (Table 1 in the paper); `experiment_e.py` is the VoE downstream eval;
`hf/` is the full-scale HF training. They are the canonical reference
implementation, not junk. Moving them to `legacy/` is still fine (the forex
pipeline never imports them and should stand alone at the top level), but they
should be treated as the upstream reference, not deleted.

## 4. Consolidated prioritized implementation plan

Sequenced so each step is low-risk and independently evaluable. Targets are
FX-honest, not paper-equity-optimistic.

**P0 — Fix the label (highest leverage, lowest risk).** [fin_jepa_proposal §A]
- Replace the probe/aux target: instead of `compute_mega_alpha` (alphas.py:236–
  241, a contemporaneous volatility ratio), use a **volatility-normalized forward
   return** `y = Σ logret[i:i+TGT] / vol60[i]` computed in `build_feature_matrix`.
- Use a **triple-barrier / signed-label** for dirAUC instead of median split
  (probe_forex_h1.py:168).
- Add a **TGT-bar embargo** between train/val windows in `make_dataset`
  (forex_features.py:125–131) to kill leakage.
- Keep mega-alpha only as an auxiliary head.

**P1 — Fix evaluation (robustness).** [fin_jepa_proposal §B]
- Replace the single 90/10 split with **walk-forward / purged K-fold** (5 folds,
  embargo). Report mean±std of IC / dirAUC — one window is not evidence.

**P2 — Fix the training schedule to match validated recipe.** [paper §4.3.2 + jepa_proposal §C5]
- Add linear warmup (1 ep) + cosine decay. Bump lr toward 5e-4 (paper) or 3e-4
  (agent suggestion); set weight_decay to ~1e-4 (paper uses 1e-5 — stay low;
  do NOT use the 0.05 from the I-JEPA legacy, that is for image SSL).
- Keep batch 256 (GPU-limited) or 512 if mem allows. 50 epochs.
- Select `best.pt` on **validation IC**, not JEPA loss (train_forex_h1.py:187).

**P3 — Architecture: stop-gradient / EMA target + temporal encoder.** [jepa_proposal C1/C3 + timeseries §1]
- Add a frozen/EMA target encoder (deepcopy of PriceEncoder, requires_grad_=
  False, EMA momentum 0.996→1.0), encode the target region only through it
  (detached), apply LayerNorm to the target. This is the canonical JEPA
  stabilizer the repo currently replaces with per-batch standardization.
- **Optional temporal encoder:** conv-tokenizer / PatchTST-style patching
  (timeseries §1) to give the encoder within-bar structure and raise eff_rank.
  This is the highest-effort change — gate it behind P0–P2 clearing the noise
  floor.
- Drop the hacky per-batch hard-stdz (model.py:147–149) in favor of a learnable
  LayerNorm inside the encoder so train/eval latents match
  (jepa_proposal W2/C2). Keep SIGReg for isotropy; operate λ≈1.0–2.0 (FX
  regime), not 0.1.

**P4 — Aux head pointed at the NEW label.** [REPORT §18 + fin_jepa_proposal §C]
- Set `aux_lambda` (train_forex_h1.py:88) to 0.3–0.5 on the vol-normalized
  return label. This is the right mechanism the paper left for future work, now
  usable because the label is tradable.

## 5. Honest targets (FX, single pair, post-fix)

| Metric | Current (§17, λ=2.0) | Realistic | Stretch |
|---|---|---|---|
| probe_IC (vol-norm return) | 0.011–0.052 | 0.03–0.06 | 0.10 |
| probe_rankIC | −0.012→+0.027 | 0.03–0.06 | 0.10 |
| dirAUC (triple-barrier) | 0.482–0.507 | 0.52–0.53 | 0.55 |
| eff_rank | 14.8/64 | 15–25/64 | 30 |
| VoE_alpha_label_AUC | ~0.50 | 0.52–0.54 | — |

**Guardrail (fin_jepa_proposal):** anything > dirAUC 0.55 OOS on free FX data
should be treated as a leakage bug, not alpha. The paper's own AUC≈0.49 is the
sanity anchor — beating it to ~0.52–0.53 OOS with costs is a real, tradable
result; anything more is suspect.

## 6. What NOT to do

- Do not keep tuning λ past ~2.0 looking for direction — the gap is
  dimensional+label, not λ (REPORT §16/§17).
- Do not adopt I-JEPA's weight_decay=0.05 — wrong domain, will over-regularize.
- Do not expect eff_rank to reach the paper's 37.8 on a single FX pair.
- Do not delete the `legacy/` reference assets — they are the upstream paper code.
