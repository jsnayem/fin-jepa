# Fin-JEPA Project Journal

> A running log of **what went right, what went wrong, and what we learned** while
> training a paper-faithful latent-only Fin-JEPA on hourly EUR/USD. Written for
> future reflection — read before starting a new direction so we don't re-litigate
> or re-break things.
>
> Companion to `REPORT.md` (the technical hand-off). This file is the *narrative*;
> REPORT is the *state*. New session? Skim §3 (Lessons) first.

---

## 1. How to use this journal
- Append a new dated entry under §2 whenever we hit a milestone, a dead end, or a
  non-obvious finding. One block per run/decision.
- Keep §3 (Lessons: Wins / Pitfalls) a curated, de-duplicated list — move insights
  there once they're confirmed, so the timeline stays readable.
- Be specific about *why* something worked or broke. Vague "tried X, didn't work"
  is useless in 3 months.

---

## 2. Timeline

### 2026-07-11 — Session A: plumbing, bugs, first divergence
- **Built** the full pipeline: `alphas.py` (17 formulaic alpha features),
  `forex_features.py` (gap-split 90/10, CTX=120/TGT=24/GAP_H=120), `model.py`
  (`FinJEPA`, 368,640 params at D=64/enc4/pred6/heads4), `train_forex_h1.py`,
  `probe_forex_h1.py`.
- **Bugs fixed (don't re-fix):** CSV had 7 tab fields but a 6-name header →
  mis-parse; column case mismatch (lowercase vs capitalized); `ts_argmax` crash on
  warmup NaNs; `GAP_H=2` too small (weekend FX closures → 0 windows) → set 120.
- **First runs (3 & 40 ep) were identical** because `best.pt` saved at epoch 3.
  Probe was random (IC≈0.03). Root cause found (see Session B).

### 2026-07-11 — Session B: two real bugs found and fixed
- **WRONG #1 — SIGReg scaling bug.** `SIGReg.forward` multiplied the statistic by
  `proj.size(-2)` (sequence len 144), making `sigreg_loss` ~144× too big; at λ=0.1 it
  dominated and destabilized training (val_sig exploded to ~20).
- **WRONG #2 — in-painting objective.** `forward` compared `z_pred[:, :n]` with
  `z_tgt[:, :n]` (SAME timestep), not the future. True JEPA predicts the *future*
  latent. Fixed to `pred_loss = mse(z_pred[:, T_ctx:], z_full[:, T_ctx:])`.
- **WRONG #3 — embedding collapse.** After the above, latents shrank to a tiny ball
  (stdZ→0.008) to game `pred_loss`. Fixed by **hard-standardizing** `z_full`
  (zero-mean/unit-var per feature over batch) before SIGReg + pred-loss. stdZ→1.0.
- After fixes, Stage 1 (10 ep) trained cleanly: stdZ≈1.0, val_pred 0.22→0.008.

### 2026-07-11 — Session C: 40-epoch run, discovered the real bottleneck
- Extended to 40 ep (fresh Colab T4, single `colab run ... 40`). Ran clean.
- **Finding:** objective is well-fit (val_pred 0.008) but **effective rank stuck at
  ~5/64** and downstream probe stayed noise (IC 0.011, dirAUC 0.482). The bottleneck
  is NOT training stability — it's that the encoder packs everything into ~5 dims.
- Hypothesis logged: raise `sigreg_lambda` to spread dimensionality.

### 2026-07-11 — Session D: λ sweep + live progress monitor
- **Added `FINJEPA_LAMBDA` (env or 2nd argv) to `colab_run_train.py`** so λ is tunable
  from the Colab CLI without code churn. Committed + pushed.
- **WRONG #4 — Colab one-assignment limit.** Launching a 2nd `--session` while the
  prior was kept alive threw `TooManyAssignmentsError`. Fix: `colab stop -s <old>`
  before launching the next; use a distinct `--session` name per run.
- **Made progress observable.** `train_log.jsonl` is flushed per epoch (append-mode
  close) and `colab download` works during BUSY — so we can poll it. Added
  `watch_progress.py` (live `n/N %` bar + metrics) instead of blocking on `DONE`.
- **λ=1.0 run (session `finjepa-l1`):** eff_rank **5.67 → 12.28**, probe_IC
  **0.011 → 0.045**, rankIC −0.012 → +0.023, dirAUC 0.505. Confirmed the hypothesis:
  low effective rank was the culprit; higher λ widens latents and unlocks signal.

---

## 3. Lessons

### ✅ What went right (worth repeating)
- **Hard-standardization > soft variance penalty** for collapse. Structurally forbids
  the shrink-to-ball cheat; stdZ pinned at 1.0 with zero extra hyperparams.
- **Treat `checkpoints/forex_h1/*.pt` as the only durable backup.** The Colab CLI has
  NO `upload` and re-assign on a kept session is rejected — if the VM is evicted, the
  local `last.pt` can't be re-uploaded, so you re-run Stage 1. Keep local copies.
- **A single `colab run --keep <script> <epochs>` beats the staged plan** now that
  training is stable. No need for the fragile exec-detach dance.
- **Poll `train_log.jsonl`, not stdout.** Subprocess stdout is block-buffered; the
  JSONL (append + close per epoch) is the real telemetry channel.
- **Effective rank is the key diagnostic** for "is the encoder actually learning
  structure?" Loss going down is necessary but not sufficient.

### ❌ Pitfalls (don't repeat)
- **Don't trust `best_val_loss` as a signal of *good* features.** Epoch-3-best masked a
  broken objective for two whole runs.
- **SIGReg scaling by sequence length is a trap** — it silently makes λ enormous.
  Always sanity-check the magnitude of each loss term at init.
- **Same-timestep "prediction" is in-painting, not JEPA.** Compare predicted-future
  vs actual-future latents or you learn nothing temporal.
- **One Colab assignment per account.** Stop the old session before launching a new
  one, or you waste a launch on `TooManyAssignmentsError`.
- **Low effective rank ≈ no downstream signal.** If rank is ~5/64, the probe will be
  noise no matter how long you train. Fix dimensionality (λ) before adding epochs.
- **`--timeout` default is 30s** in this CLI and WILL kill your run. Always pass
  `--timeout 5400` (or ≥ training time).

### 🧪 Open questions / next levers
- Optimal λ for this data: 1.0 clearly beats 0.1; sweep {0.5, 2.0} to find the peak
  (watch probe_dirAUC crossing 0.52 and val_eff_rank vs best_val_loss trade-off).
- If dirAUC stays <0.52 even at high λ → low rank may be intrinsic to latent-only FX
  (consistent with the paper's own VoE finding, §2.6). Then try: a return-prediction
  auxiliary loss, or a longer context/horizon.
- Consider committing the small JSON evidence files (`meta.json`, `probe.json`,
  `train_log.jsonl`) per run; keep `*.pt` out of git.

---

## 4. Run ledger (quick reference)
| date | session | λ | epochs | eff_rank | probe_IC | probe_dirAUC | notes |
|---|---|---|---|---|---|---|---|
| 2026-07-11 | finjepa | 0.1 | 40 | 5.67 | 0.011 | 0.482 | baseline; low-rank bottleneck |
| 2026-07-11 | finjepa-l1 | 1.0 | 40 | 12.28 | 0.045 | 0.505 | λ fix works; signal emerges |
| | | | | | | | |
