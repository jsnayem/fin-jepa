# Fin-JEPA on EUR/USD H1 — Proposal to Fix the Directional Signal

## 1. What the literature actually says

**Fin-JEPA (Wang, SSRN 6855118)** — the paper this repo faithfully reproduces — itself
reports the failure we are seeing: on 6,230 equities its latent predictions beat an
identity baseline by ~10% MSE, *yet the downstream VoE signal was AUC≈0.49, Spearman≈0*.
The paper never demonstrated a tradable directional edge; it explicitly punted regime
detection and multi-horizon training to "future work." So **dirAUC≈0.50 is the expected
outcome of a paper-faithful reproduction, not a bug in this repo.** Independent
replications (e.g. the "JEPA-Trader" experiments) reach the same conclusion: JEPA learns
valid representations but no net edge on free data.

The broader SSL-for-time-series literature (TS2Vec, TF-C, the 2306.10125 survey) is more
encouraging but with a crucial caveat: those methods are validated on **classification /
forecasting benchmarks with strong autocorrelation and high SNR** (UCR, human-activity,
ECG). FX hourly returns have a signal-to-noise ratio near zero and are close to a
martingale. The known failure modes all apply here:

- **Low effective rank / dimensional collapse** — exactly the `eff_rank ~5-15/64`
  documented in REPORT §16–17. SIGReg enforces isotropy but not *usefulness*; raising λ
  spreads rank and lifts `probe_IC` (0.011→0.052) but not `dirAUC`. That is the tell that
  the extra dimensions encode *variance*, not *direction*.
- **The rank-vs-direction gap** — IC/rankIC are learnable because the label has heavy
  tails and the model captures *magnitude/volatility* structure; the *sign* is what stays
  at chance. This is generic to noisy financial series.
- **Non-stationarity & look-ahead** — a single 90/10 time split leaks regime information
  and gives one non-robust estimate.

## 2. Critique of this repo's financial adaptation

**The label is the primary problem, not the architecture.** `compute_mega_alpha`
(`alphas.py:236-241`) averages four *contemporaneous* alpha signals (`a101,a3,a43,a40`)
and z-scores them. Three issues:

1. **It is not a return.** `a101 = (c-o)/(h-l)` (`alphas.py:194`) etc. are structural
   candle/volume ratios, largely mean-reverting and volatility-dominated. Their forward
   value is *more predictable than a return* (hence IC>0) but has almost no monotone
   relationship to price direction — so `dirAUC` on this label is meaningless as a trading
   proxy. The probe is optimizing a target that is not what a trader wants.
2. **Self-referential leakage risk.** The mega-alpha is built from the *same feature
   family* fed into the encoder (`build_feature_matrix`, `forex_features.py:79-87`), so a
   positive IC can be the model reading its own input, not forecasting.
3. **Median-split dirAUC** (`probe_forex_h1.py:168`) turns a symmetric z-scored blob into
   a coin-flip label; even a perfect magnitude predictor scores 0.50.

**The horizon/context are defensible but not multi-scale.** CTX=120/TGT=24
(`forex_features.py:15-16`) is reasonable, but a single scale can't separate microstructure
mean-reversion (1–4h) from trend (1–5d).

**The 90/10 split** (`forex_features.py:97,111`) is a single fragile estimate; the probe
evaluating only the last-10% window (`probe_forex_h1.py:139`) conflates model quality with
one regime.

**Is the aux head (REPORT §18, `model.py:120,164-166`) the right fix?** Partially. Forcing
the latent to decode a forward label is the correct *mechanism*, but pointed at the
mega-alpha it will just make the encoder better at predicting an untradable target. **Fix
the label first, then the aux head becomes the right lever.**

## 3. Proposal — concrete changes

### A. Fix the label (highest leverage)
- Replace the probe/aux target with a **volatility-normalized forward return**:
  `y = sum(logret[i:i+TGT]) / vol60[i]`, computed in `build_feature_matrix`
  (`forex_features.py:79`). This is what a directional model must predict and makes
  `dirAUC` interpretable. Keep the mega-alpha as an *auxiliary* head only.
- Use a **triple-barrier / sign label** for dirAUC (up/down before a vol-scaled barrier),
  not a median split (`probe_forex_h1.py:168`).
- **Purge leakage:** exclude same-family alpha features from the target, and add an
  embargo of TGT bars between train and val windows in `make_dataset`
  (`forex_features.py:125-131`).

### B. Fix evaluation
- Replace the single 90/10 split with **walk-forward / purged K-fold** (5 folds,
  TGT-bar embargo). Report mean±std IC and dirAUC across folds — one number from one
  window is not evidence.

### C. Architecture / features
- **Multi-scale context:** feed two encoders (e.g. CTX=48 and CTX=240) and concatenate
  reprs before the probe. Cheap and directly targets the trend/reversion mixture.
- **Keep λ≈1.0** (REPORT §17 sweet spot: eff_rank 12, best IC) — do not push λ=2.0, its
  extra rank is variance not signal.
- **Aux head weight:** set `aux_lambda` (`train_forex_h1.py`) to 0.3–0.5 pointed at the
  *new vol-normalized return label*, and select `best.pt` on validation **IC**, not JEPA
  loss (`train_forex_h1.py` currently selects on pred loss — change the selection metric).

### D. Realistic targets (be honest about FX)
On hourly EUR/USD, spread ≈ 0.5–1.0 pip vs typical hourly range ≈ 8–12 pips, so a costed
edge needs dirAUC ≳ 0.52 *out-of-sample and after costs*.

| Metric | Current | Realistic target | Stretch |
|---|---|---|---|
| probe IC (vol-norm return) | ~0.01–0.05 | 0.02–0.05 | 0.08 |
| probe rankIC | ~0.02 | 0.03–0.06 | 0.10 |
| dirAUC (triple-barrier) | 0.50 | 0.52–0.53 | 0.55 |
| eff_rank | ~12/64 | 15–25 | 30 |

**Anything above dirAUC 0.55 OOS on free FX data should be treated as a leakage bug, not
a discovery** — that is the single most important guardrail. Success = a *stable* 0.52
across all walk-forward folds, which is enough to be tradable at H1 with disciplined cost
control.

### Sequenced plan
1. Vol-normalized return label + embargo (§A). Re-probe λ=1.0 baseline.
2. Walk-forward eval (§B) — establish honest mean±std.
3. Aux head @ 0.3 on the new label, select on IC (§C).
4. Only if 1–3 clear the noise floor: add multi-scale context (§C).
