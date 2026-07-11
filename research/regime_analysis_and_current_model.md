# Regime Analysis + Current Fin-JEPA — Tradability Improvement Plan

*Connects Rivin's cross-asset regime study (`MARKET REGIME ANALYSIS ACROSS ASSET
CLASSES.md`) to the EUR/USD Fin-JEPA in this repo, and proposes concrete,
low-risk changes to move from "no directional edge" toward a **tradable** signal
(winrate / risk-reward), not just a better dirAUC.*

## A. What the regime paper says about EUR/USD

Rivin runs UMAP + K-means on 2021–2024 intraday patterns. BTC is the outlier:
extreme asymmetry (75% small-bear / 25% large-bull days), 1.05% daily vol, clean
clusters (silhouette 0.83), year-to-year regime shifts — genuinely "trendy," and
that is *why* an 8-month lookback helps BTC (§5.1: the model needs long history
to learn the 75/25 asymmetry and regime transitions).

EUR/USD is the opposite on every axis. It is **symmetric** (52% up / 48% down,
+0.33% vs −0.36%), **low-vol** (0.18%, the lowest of the four assets), has
**fuzzy regimes** (silhouette 0.59, "significant overlap"), a **2-day median
regime persistence**, and is **structurally stable year to year** (50–52%
positive every year). Equities (AAPL/QQQ) sit in between but pattern with forex.

The paper's own conclusion (§5.2) is the thesis of this document: traditional
markets need **technical mean-reversion with a short lookback, not regime
detection**. For a predictor this implies (1) a **mean-reversion bias**, (2) a
**short horizon/context** (~2 days matches persistence — a 5+ day context adds
noise, not signal), (3) **no long-context regime machinery** (the regimes are
fuzzy and non-persistent — nothing to detect), and (4) that **direction is
near-random, so risk-reward / position-sizing off predicted *magnitude* is the
real lever**, not sign accuracy.

## B. Critique of the current Fin-JEPA through this lens

**dirAUC≈0.50 is exactly what a symmetric market predicts.** REPORT §17 shows
dirAUC pinned at 0.48–0.51 across every `sigreg_lambda`. That is not a bug — it
is the regime finding restated: EUR/USD is 52/48, so absent leakage the sign is
a coin flip. `paper_grounding_and_plan.md:121` correctly warns that dirAUC>0.55
OOS on free FX is a leakage tell, not alpha.

**The mega-alpha label carries almost no directional info.** `compute_mega_alpha`
(`alphas.py:236-241`) is the z-scored average of four *contemporaneous* candle/
volume ratios (`a101,a3,a43,a40`) — effectively a volatility/structure composite,
not a return. Forward magnitude is learnable (hence the small positive probe IC,
REPORT §17: 0.011→0.052), but its **sign has no monotone tie to price
direction**, so median-split dirAUC (`probe_forex_h1.py:168`) on it is
meaningless as a trade proxy. It is also drawn from the same feature family fed
to the encoder (`forex_features.py:79-87`) → self-referential IC risk.

**CTX=120 is too long for a 2-day-persistence market.** `forex_features.py:15-16`
uses CTX=120 (~5 trading days), TGT=24 (~1 day). The paper says forex regimes
last ~2 days; a 5-day context mostly feeds the encoder stale, decorrelated bars,
which is consistent with the stuck **eff_rank ~5/64** (REPORT §16) — there is
little persistent structure over that window to encode.

**Latent-only is the wrong target for a tradable signal.** The JEPA objective
optimizes future-latent MSE (`model.py:156`), which the paper's own VoE shows
does not encode direction. For a symmetric market the tradable quantity is
*magnitude + a sizing rule*, which the latent-only loss never sees.

## C. Concrete, low-risk changes (with file refs)

1. **Replace the label with a vol-normalized forward return + triple-barrier
   sign.** In `build_feature_matrix` (`forex_features.py:79-87`) compute
   `y_mag = Σ logret[i:i+TGT] / vol60[i]` and a triple-barrier sign `y_sign`.
   Feed `y_mag` as the probe/aux regression target and `y_sign` for dirAUC
   (replacing the median split at `probe_forex_h1.py:168`). Justification via the
   regime finding: the market is symmetric, so **sign is near-random and
   magnitude is the learnable, tradable quantity** — size positions by predicted
   `|y_mag|`. Add a **TGT-bar embargo** in the window loop
   (`forex_features.py:125-131`) to kill train/val leakage.

2. **Shorten context/horizon and add mean-reversion features.** Set CTX=48
   (~2 days, = median persistence) and TGT=12–24 (`forex_features.py:15-16`).
   Add a short MR feature block in `build_base_features`
   (`forex_features.py:52-71`): distance-from-VWAP (`(close-vwap)/vwap`, vwap
   already in `add_vwap_adv`), a short RSI(14), and an intraday-range reversion
   term (`(close-mid)/range`). These directly encode the "technical
   mean-reversion, short lookback" the paper prescribes, on top of (or trimming)
   the 17 alphas in `ALPHA_COLS` (`alphas.py:225-228`).

3. **Add a risk-reward / sizing head (magnitude + sign).** The aux head already
   exists (`model.py:120,164-166`) — point it at `y_mag` and add a second output
   for `y_sign`. Trade only when `|predicted y_mag| > threshold` (calibrate on
   train). This exploits the paper's point directly: returns are symmetric but
   magnitude is learnable, and gating trades on magnitude lifts winrate/PF even
   when raw dirAUC ≈ 0.52.

4. **Walk-forward eval with P&L metrics.** Replace the single 90/10 split
   (`forex_features.py:97-112`) with 5-fold purged walk-forward + embargo. In
   `probe_forex_h1.py` report **profit factor, Sharpe, and winrate** of the
   magnitude-thresholded strategy (with a spread/cost haircut) as the primary
   metric — dirAUC is secondary since the goal is profitability, not sign
   accuracy.

5. **Keep the aux head, retargeted.** `model.py:120` stays; just point it at the
   new `y_mag`/`y_sign` labels via `aux_lambda≈0.3–0.5` (`train_forex_h1.py:88`).
   Same mechanism, tradable target.

## D. Execution plan + honest targets

**Sequenced, CPU-feasible first (no GPU needed for steps 1–3):**
1. Add vol-norm forward return + triple-barrier label + embargo
   (`forex_features.py`); rerun the existing frozen-probe (`probe_forex_h1.py`)
   on the *current* checkpoint against the new label — CPU-cheap, tells us
   immediately whether magnitude IC survives a real return target.
2. Add MR features + set CTX=48/TGT=12; rebuild dataset, re-probe (still CPU).
3. Add walk-forward + P&L/Sharpe/PF metrics to the probe — pure eval, CPU.
4. *(GPU)* Retrain with the retargeted aux head (`aux_lambda=0.5`, `sigreg_lambda`
   ~1.0–2.0 per REPORT §17) on the short context; compare PF/Sharpe.

**Honest targets (symmetric, single-pair, free data):** direction stays hard —
aim for a **stable dirAUC 0.52–0.53 OOS** (anything >0.55 = leakage, per
`paper_grounding_and_plan.md:121`). The real deliverable is a **positive profit
factor (~1.1–1.3) and winrate ~52–55%** from magnitude-thresholded sizing, with
Sharpe ~0.5–1.0 gross before costs — and honesty that EUR/USD's 0.18% vol and
~1 pip+ spread may eat much of that net. Magnitude IC 0.03–0.06 (realistic),
0.10 stretch. Success = a small but *robust across folds* positive PF, not a
high dirAUC.
