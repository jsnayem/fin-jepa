# New Model Design: RiskJEPA — a Profit-Weighted EUR/USD Predictor

## Thesis
Latent-only Fin-JEPA reproduces the paper's null result on hourly EUR/USD
(REPORT §16–17: `dirAUC≈0.50`, `eff_rank≈5–15/64`, probe IC≤0.05). As the
Rivin study confirms, EUR/USD is *symmetric* (~52/48), *mean-reverting*,
low-vol (0.18%), with fuzzy regimes (silhouette 0.59) — **direction is
near-random and not learnable at the sign level**. A profitable model here
must stop betting on direction and instead capture *risk-adjusted asymmetry*:
trade only high-magnitude/high-confidence bars, size by volatility, and **flat
when uncertain**. I propose **RiskJEPA**: an encoder–predictor with a direct
return-regression head + an uncertainty head, trained and *selected* on
walk-forward P&L/sharpe/profit-factor, not JEPA loss.

## Why latent-only JEPA is structurally unprofitable
- `FinJEPA.forward` (model.py:129–168) only predicts future *latents*; the
  signal is recovered by a frozen probe (probe_forex_h1.py:147–162) that sees
  no cost, no sizing, no abstention. The probe optimizes IC — not profit.
- The paper's own VoE finding (paper_grounding_and_plan.md §1) is AUC≈0.49–0.50
  at horizons {1,5,20}d: forward *direction* is not encoded. We accept
  `dirAUC≈0.50` and stop chasing it.
- The latent has **no native sizing or kill-switch**: every bar gets a position,
  so spread cost (~0.5–1.0 pip on EUR/USD) bleeds any marginal IC to death.

## Reframe: risk-reward, not direction
Let `r_t = Σ_{k=1}^{τ} logret[t+k]` be the τ-bar forward return and
`σ_t = vol60[t]` (forex_features.py:62). Predict:
1. **Vol-normalized return** `ŷ_t = r_t / σ_t` (scaled, stationary — the
   honest FX target from paper_grounding_and_plan.md §4/P0).
2. **Uncertainty** `σ̂_t = std of the predictive distribution** (learn a
   heteroskedastic head, see below).

**Sizing rule (the lever):**
```
pos_t = 0                                  if |ŷ_t| < c1·σ̂_t   (flat / kill-switch)
pos_t = sign(ŷ_t) · tanh(|ŷ_t| / σ̂_t)      otherwise, vol-scaled, capped at 1.0
```
`|ŷ| < noise` ⇒ no trade. This converts a ~50% hit-rate into `profit factor > 1`
because we (a) only pay spread when conviction is real, (b) scale size to
volatility so a 1-pip mean-reversion pays off proportionately, (c) sit flat in
choppy/low-magnitude regimes where the Rivin study says there is no edge.

## Architecture — RiskJEPA
Reuse the repo, add heads + a triple-barrier label.

- **Encoder (conv-tokenizer)** — replace `PriceEncoder` MLP (model.py:39–65)
  per timeseries_proposal.md §1: `Conv1d(F, D, k=5, pad=2)` + 2 residual conv
  blocks over the time axis, optional PatchTST patching (P=4) to drop T 120→30.
  Bidirectional over local window is fine for SSL.
- **Predictor** — `TransformerPredictor` (model.py:69–100) kept, but swap the
  learned `pos_embed` for **ALiBi** distance bias (timeseries_proposal.md §2)
  and add a learned time-of-day/day-of-week token (forex gaps are weekly).
- **RevIN** instance-norm each window (timeseries_proposal.md §3) before the
  encoder, restore after — cheap fix for FX drift.
- **Two prediction heads off the last/mean token** (`z_ctx`, model.py:153):
  - `ret_head`: `Linear(D→1)` predicting `ŷ_t` (vol-normalized forward return at
    τ∈{12,24}). Same slot as the existing experimental `return_head`
    (model.py:120) but retargeted at the *raw* return, not mega-alpha.
  - `unc_head`: `Linear(D→2)` → (μ, log s) for a Gaussian; use the *ensemble
    or dropout* variance, or just `log s`, as `σ̂_t`.
- **Collapse guard** — keep the hard-standardize + SIGReg combo (model.py:147–
  159); operate `sigreg_lambda∈{1.0,2.0}` (REPORT §17: rank rose to 14.8/64).

## Triple-barrier labeling (the flat lever)
In `build_feature_matrix` (forex_features.py:79–87) add: for each window start,
place three barriers on `r_t`: upper `+h·σ_t`, lower `−h·σ_t`, and a vertical
`τ`-bar cap. Label `L_t ∈ {+1,−1,0}` by which barrier hits first
(`0` = sideways/timeout ⇒ maps to FLAT). Thresholds `h≈0.8`, `τ∈{12,24}`. The
`0` class is the data-driven kill-switch: train `ret_head` with an L1/quantile
penalty so predictions inside the sideways band are pushed toward 0, reinforcing
`pos_t=0`. This is exactly the risk-reward lever a symmetric market needs and
that latent-only JEPA lacks entirely.

## Training + selection recipe
- **Objective:** `L = mse(ŷ_pred, ŷ_true) + λ_sig·sigreg + β·NLL(μ,log s; r_t/σ_t)
  + γ·triple_barrier_CE`. Keep JEPA future-latent loss (model.py:156) as an
  auxiliary SSL term so the encoder still learns temporal structure — but
  **do not select on it**.
- **Selection metric = profit, not loss.** Use **walk-forward / purged K-fold**
  (paper_grounding_and_plan.md §4/P1): 5 expanding windows with a TGT-bar
  embargo (forex_features.py:125–131) to kill leakage. Pick the checkpoint by
  **mean OOS profit-factor / Sharpe** on a cost-aware backtest, not `val_loss`
  (train_forex_h1.py:187).
- **Costs:** realistic EUR/USD spread 0.5–1.0 pip + round-turn commission; each
  trade pays 2× spread. Per-trade sizing `pos_t` in lots, not binary.
- **Schedule:** warmup 1 ep + cosine decay, lr 3e-4, AdamW, wd 1e-4
  (paper_grounding_and_plan.md §4/P2).

## Why this beats latent-only Fin-JEPA on the *profitability* goal
| Property | Latent-only FinJEPA | RiskJEPA |
|---|---|---|
| Native sizing | none (probe→binary) | vol-scaled `tanh(\|ŷ\|/σ̂)` |
| Abstention / kill-switch | none | `\|ŷ\|<c·σ̂` ⇒ FLAT (triple-barrier 0) |
| Vol-awareness | none | `σ_t` in target + `unc_head` |
| Selection metric | JEPA loss | OOS profit-factor/Sharpe |
| Cost handling | none | spread+commission in backtest |

Direction hit-rate stays ~50% (accepted). Profit factor > 1 comes from (1)
trading only the top conviction/magnitude decile, (2) correct vol scaling, (3)
flat-when-uncertain. Latent-only JEPA has no mechanism for any of these.

## Implementation plan (new files, no changes to existing code)
- **`new_model.py`** — `RiskJEPA`: copy `SIGReg` (model.py:11–35) + `TransformerPredictor`
  (swap `pos_embed`→ALiBi, model.py:69–100); add `ConvEncoder` (timeseries_proposal §1);
  `ret_head` + `unc_head`; optional RevIN wrapper. Reuse `FinJEPA.forward`'s
  joint-encode + future-latent loss (model.py:140–160) as SSL aux.
- **`train_new.py`** — reuse `forex_features.make_dataset` (forex_features.py:97),
  add triple-barrier labels in `build_feature_matrix` (forex_features.py:79–87);
  reuse `effective_rank`/`stdz` (train_forex_h1.py:25–46); new loss; **walk-forward
  loop with purged K-fold + embargo**; select best on backtest profit-factor.
- **`backtest.py`** — given `ŷ_t, σ̂_t` over the val windows, apply the sizing rule,
  charge spread+commission, report winrate, profit-factor, Sharpe, %-flat,
  per-fold mean±std. Reuse metrics style from probe_forex_h1.py:42–63.

## Honest targets (hourly EUR/USD, after costs)
- **Winrate:** 50–53% (hit-rate is still ~random — no claim otherwise).
- **Profit factor:** 1.05–1.25.
- **Sharpe:** 0.3–0.8.
- **% flat:** 40–70% (most bars should be no-trade in a symmetric market).
- **Guardrail:** anything dramatically better OOS (Sharpe > 1.5, profit factor >
  1.6, winrate > 56%) is almost certainly **leakage** (look-ahead in features,
  missing embargo, or spread not charged) — treat as a bug, not alpha. The Rivin
  study's 52/48 symmetry is the anchor: real edge here is *selection + sizing*,
  not prediction.
