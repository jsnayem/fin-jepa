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
- **Never commit run-output JSONs (`meta.json`/`probe.json`/`train_log.jsonl`) into
  the repo.** A fresh Colab clone starts with them, and (a) the live monitor reads a
  stale 40-line `train_log.jsonl` and falsely reports "100% done" before training
  even starts, and (b) you download last run's `probe.json` thinking it's new. Fix:
  gitignore `checkpoints/**/*.json` and open `train_log.jsonl` with **truncate** on
  epoch 1 (was append → polluted files). Also: if the monitor says 100% suspiciously
  fast, the run hasn't actually finished — check `colab status -s <name>` for BUSY.
- **Label dtype:** numpy labels are float64 → DataLoader collates to Double, which
  makes an aux MSE loss float64 while the model is float32 and `backward()` dies with
  "Found dtype Double but expected Float". Always cast labels to `float32`.
- **`colab exec -f script.py` does NOT forward extra argv** — it errors on them and
  the run never starts. Use `colab run script.py args...` (forwards argv) or bake args
  into the script. `colab run` = `new`+`exec`+`stop`.
- **GPU (T4) can return `Service Unavailable` (503)** at allocation — transient
  backend/quota, not our bug. Retry; or fall back to **CPU** (`colab run` without
  `--gpu`). CPU Colab RAM is small: the run **OOM-killed (SIGKILL)** at batch 256 →
  use batch 64 (or smaller). CPU is ~30× slower than T4, so only use it for smoke
  tests, not full 40-epoch runs.
- **Session model (COLAB_SKILL.md):** a session is a Jupyter kernel kept alive by a
  detached daemon, independent of your shell. `--keep` persists it. Aborting *during*
  allocation cancels; once allocated, the daemon keeps it and aborting a `colab exec`
  client does NOT stop the kernel. Debug failures with `colab log -s <name>`.

### 🧪 Open questions / next levers
- Optimal λ for this data: 1.0 clearly beats 0.1; sweep {0.5, 2.0} to find the peak
  (watch probe_dirAUC crossing 0.52 and val_eff_rank vs best_val_loss trade-off).
- If dirAUC stays <0.52 even at high λ → low rank may be intrinsic to latent-only FX
  (consistent with the paper's own VoE finding, §2.6). Then try: a return-prediction
  auxiliary loss, or a longer context/horizon.
- Consider committing the small JSON evidence files (`meta.json`, `probe.json`,
  `train_log.jsonl`) per run; keep `*.pt` out of git.

### 2026-07-11 — Session E: auxiliary-label head + Colab ops schooling
- **Added the aux head** to attack the directional gap: `FinJEPA.return_head`
  (`nn.Linear(D,1)`, only when `aux_lambda>0`) predicts the forward mega-alpha from
  the context-embedding mean (the same repr the probe uses). Training combines
  `loss = jepa_loss + aux_lambda·mse(ret_pred, y)`. Code committed; baselines
  (aux=0) byte-identical.
- **Bug #5 — Double/Float dtype crash.** First Colab aux run died with
  `RuntimeError: Found dtype Double but expected Float` at `backward()`. The label `y`
  is numpy float64 → collated to Double → aux loss became float64 while the model is
  float32. Fix: cast `y` to `float32`, detach `ret_loss` for logging. A 1-epoch local
  CPU run would have caught this instantly — add that as a pre-flight smoke test.
- **Bug #6 — wrapper swallowed the traceback.** `subprocess.run(check=True)` raised
  `CalledProcessError` and hid the real error. Fix: wrapper now writes training output
  to `train_run.log` on the VM and always dumps it.
- **Bug #7 — `colab exec -f script.py` doesn't forward argv.** A launch
  `colab exec ... colab_run_train.py 10 2.0 0.5` errored on the extra args, so the run
  never started (kernel stayed IDLE). `colab run` forwards argv; use that, or bake args
  into a script.
- **Colab environment schooling (user added `COLAB_SKILL.md`):** a session is a
  Jupyter kernel kept alive by a **detached daemon** — independent of the local shell.
  `colab run` = `new`+`exec`+`stop`; `--keep` persists it. **GPU (T4) allocation threw
  `Service Unavailable` (503)** repeatedly (transient); fell back to **CPU**. On CPU
  the run **OOM-killed (SIGKILL)** at batch 256 → relaunched batch 64 (added `argv[4]`
  batch to the wrapper) and it ran, but CPU is ~30× slower (epoch 1 >26 min). The aux
  experiment is **blocked on GPU**, not failed — code is ready.
- **Monitor improved:** `watch_progress.py` is now phase-aware (shows the
  clone/install/feature-build phase + elapsed time, not "stuck") and has `--once` for
  quick on-demand polls. SOP updated to match the accurate session model + new+exec
  resilient pattern + `colab log` debugging.

## 5. Future plan (next sessions)
1. **P0 — finish the aux-label experiment on GPU/T4** (retry the 503; it's transient):
   `colab run --gpu T4 --session finjepa-aux05 --keep --timeout 5400 \
    colab_run_train.py 40 2.0 0.5`, monitor, download to `checkpoints/forex_h1_aux05/`.
   Success = `probe_dirAUC > 0.52`. If still flat: sweep `aux_lambda`∈{1.0,2.0}, or
   predict raw forward return, or lengthen `TGT`/`CTX`.
2. **P1 — pre-flight smoke test:** run `train_forex_h1.py --epochs 1 --aux_lambda 0.5`
   locally (tiny batch) before every Colab launch, to catch dtype/crash bugs fast.
   Also make the probe step in the wrapper non-fatal.
3. **P2 — if directional gap persists:** return-aux on longer horizon; richer encoder;
   accept paper §2.6 (latent-only FX may not encode forward direction — the aux head
   is the direct test).
4. **P3 — docs/housekeeping:** keep JOURNAL/REPORT/SOP current; stop Colab sessions
   when done (one assignment per account).

---

## 4. Run ledger (quick reference)
| date | session | λ | epochs | eff_rank | probe_IC | probe_dirAUC | notes |
|---|---|---|---|---|---|---|---|
| 2026-07-11 | finjepa | 0.1 | 40 | 5.67 | 0.011 | 0.482 | baseline; low-rank bottleneck |
| 2026-07-11 | finjepa-l05 | 0.5 | 40 | 11.37 | 0.041 | 0.507 | rank climbs with λ |
| 2026-07-11 | finjepa-l1 | 1.0 | 40 | 12.28 | 0.045 | 0.505 | λ fix works; signal emerges |
| 2026-07-11 | finjepa-l2 | 2.0 | 40 | 14.77 | 0.052 | 0.502 | best IC; dirAUC still ~0.50 |
| | | | | | | | |

### 2026-07-12 — RiskJEPA pivot: Kaggle GPU retrain (Colab quota spent)
- **Pivot:** Colab GPU quota exhausted for the day → moved compute to **Kaggle GPU**
  (user `nayem939`, ACCESS_TOKEN auth, `kaggle` CLI 2.2.3 in `.venv`). Same proven
  wrapper pattern as `colab_run_train.py`, pointed at the RiskJEPA (risk-reward) path.
- **Built `riskjepa/` (committed 5619426):** regime-aware data path (CTX=48/TGT=12,
  35 feats = 15 base + 17 alpha + 3 MR, vol-normalized forward-return label,
  triple-barrier sign 0=FLAT, embargo) + cost-aware backtest (`metrics.py`) +
  probe (`probe.py`, CPU baseline + frozen-encoder mode).
- **Built `riskjepa/train.py` (committed 259e470):** FinJEPA pretrain on the 35-feat
  schema with `aux_lambda>0` so `return_head` predicts the vol-normalized return
  (the RiskJEPA target, replacing mega-alpha). Config chosen: `sigreg_lambda=2.0`,
  `aux_lambda=0.5` (the §17 sweep's best rank + modest IC, retargeted).
- **Kaggle kernel `nayem939/fin-jepa-riskjepa-train`** (GPU script, private): entry
  `kaggle_run_train.py` clones the repo, `pip install einops`, trains 40 ep on cuda,
  runs `riskjepa/probe.py` (frozen-encoder + risk-reward backtest), copies
  `riskjepa_*.pt/json` to `/kaggle/working/`. Pushed v2, **RUNNING**.
- **Pre-flight smoke (local CPU):** `python -m riskjepa.train --epochs 3 --batch 64`
  runs (import + 35-feat loop OK; CPU slow, ~30× vs GPU per JOURNAL §E). Full
  validation deferred to the Kaggle GPU run.
- **Honest expectation (from research/new_model_design.md):** winrate 50–53%, profit
  factor 1.05–1.25, Sharpe 0.3–0.8, %-flat 40–70%. Anything dramatically better OOS
  (Sharpe>1.5, PF>1.6, winrate>56%) = leakage, treat as bug.
- **Lessons carried over from Colab runs:** cast labels to float32 (Double/Float crash
  bug); `einops` must be pip-installed on remote; `train_log.jsonl` truncates on ep1;
  `*.pt` + run JSONs are gitignored (clone pollution). Kernel re-runs clone fresh main.

**Sweep read-out:** `probe_IC` and `val_eff_rank` rise monotonically with λ
(0.1→2.0). `best_val_loss` also rises (0.011→0.045) — expected, since higher λ
trades `pred` fidelity for isotropy. `probe_dirAUC` stays ~0.50–0.51 across all λ,
i.e. rank correlation (IC) is real but **directional** hit-rate is still chance.
Next lever if we want dirAUC>0.52: return-prediction aux loss, or longer horizon.
