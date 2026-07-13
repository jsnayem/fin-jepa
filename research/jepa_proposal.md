# Fin-JEPA Architecture & Training Improvement Proposal

*Scope: concrete fixes to the **fin-jepa** repo (this directory), grounded in the canonical JEPA family (I-JEPA, V-JEPA, LeWM, BYOL/EMA teachers) and the repo's own legacy code.*

## 1. JEPA core principles (what keeps it stable)

The JEPA family learns representations by predicting the **embedding of a target** from the embedding of a **context**, in latent space — not by reconstructing inputs. Three mechanisms make this stable:

- **Asymmetric target construction.** In I-JEPA (`legacy/ijepa.py:127-153`) the target embeddings come from a *separate* EMA-updated encoder (`tgt_enc`, `requires_grad_(False)`), and the loss is taken against `LN(tgt_enc(x))` under `no_grad`. The BYOL/EMA teacher provides a slowly-moving, **stop-gradient** anchor so the predictor cannot collapse by simply copying its own output. V-JEPA and the Apple "frozen teacher" variant use the same asymmetry (freeze a teacher, train a student to predict it).
- **Context ≠ in-painting.** The target is a *held-out future/block* (`sample_ijepa_masks`, `legacy/ijepa.py:100-117`), never the same timestep. Predicting the future from the past is what forces temporal structure.
- **Collapse prevention via asymmetry + target normalization**, *not* via batch standardization. I-JEPA applies **LayerNorm to the target** (`LN(s_y)`, `legacy/ijepa.py:143`) and uses smooth-L1, not MSE on raw latents.

LeWM (`legacy/leworldmodel.py`, `legacy/model_lewm.py`) is the closest relative: end-to-end next-embedding MSE + SIGReg, AdaLN-zero predictor, and crucially a **separate encoding path for the target** (`z_tgt = self.encoder(tgt)`, `model_lewm.py:306`) fed to the predictor as ground truth — still asymmetric because the target is a distinct forward of the future.

## 2. Critique of current FinJEPA

The current model (`model.py`) is "latent-only JEPA" but deviates from the canonical recipe in ways that explain the stagnation in `REPORT.md` §17: `eff_rank` 5→15/64 with λ, but `probe_dirAUC` glued at ~0.50 regardless of λ (the λ sweep in §17 shows dirAUC flat 0.482–0.507 across λ∈{0.1,2.0}).

**W1 — No stop-gradient / EMA target encoder (most important).** `forward` (model.py:140-160) builds the target as `z_full[:, T:] = encode_batch(cat(ctx,tgt))[:, T:]` — i.e. the target latents are produced by the **same trainable encoder** that processes context, and gradients flow into them (only `pred_loss` is detached from the predictor, not from the target encoder). There is no teacher/student asymmetry. Canonical JEPA's stability comes precisely from the *frozen, moving* target. The repo substitutes hard batch-standardization (model.py:147-149) for that asymmetry — a hack that also creates a train/eval mismatch (see W2).

**W2 — Hard standardization train/probe mismatch.** `z_full` is standardized per-feature over the *current batch* (model.py:147-149) and `output['emb'] = z_ctx` returns the standardized context. But at probe time the same standardization runs over a *different* batch (probe_forex_h1.py:74-76 calls `net(ctx, tgtw)`), and `predict_future` (model.py:170-180) applies none at all. So the latent the probe consumes depends on batch composition → representation drift between pretrain and eval, and a latent space whose scale is defined by batch statistics rather than the encoder.

**W3 — Per-timestep MLP encoder has no temporal mixing.** `PriceEncoder` (model.py:39-65) is a plain `(B,1,F)→(B,D)` MLP applied independently to each bar; there is no conv/attention over time. All dynamics are delegated to the 6-layer causal predictor. LeWM's `Embedder` uses `Conv1d` over the time axis (`model_lewm.py:204-218`) to inject local temporal structure before the predictor. FinJEPA's encoder is structurally blind to time.

**W4 — Single contiguous target, raw-MSE, no target LN.** Canonical I-JEPA uses **multiple target blocks** + **LayerNorm on targets** + **smooth-L1** (`legacy/ijepa.py:143-150`). FinJEPA uses one future block and MSE on standardized raw latents (model.py:156). MSE on isotropic-standardized latents is sensitive to outliers and gives no multi-horizon signal.

**W5 — No schedule / no EMA momentum.** `train_forex_h1.py:78,125` uses constant `lr=1e-4`, no warmup, no cosine decay, no EMA on any weights. I-JEPA uses `lr_warmup_cosine` + EMA `0.996→1.0` (`legacy/ijepa.py:43-46,152`). Small data + constant LR under-exploits the representation.

**W6 — Effective rank is partly data-limited.** With 32 inputs (15 base + 17 alphas, many collinear) a 64-D MLP output cannot exceed ~rank 32, and the observed 5–15 reflects redundant features. λ mainly trades pred-fidelity for isotropy (§17) and cannot create dimensions the inputs don't have.

## 3. Proposed changes

**C1 — Add a stop-gradient target encoder (highest priority).** Introduce `self.target_encoder = deepcopy(PriceEncoder)` with `requires_grad_(False)` (mirror `legacy/ijepa.py:127-128`). Encode the target region *only* through it, detached. EMA-update it each step with momentum `m` ramped `0.996→1.0` (`legacy/ijepa.py:37-40,152`). Apply **LayerNorm to the target** before the loss (`legacy/ijepa.py:143`). Replace the symmetric `cat(ctx,tgt)` (model.py:141) with two independent forward passes; target latents are now a stable anchor, removing the need for the hacky hard-stdz.

**C2 — Remove batch-dependent standardization; use deterministic normalization.** Per-feature batch standardization (model.py:147-149) is the train/eval mismatch source. Replace with a learnable per-timestep **LayerNorm inside the encoder/projector** (as LeWM `Projector`, `model_lewm.py:68-82`) so the latent scale is encoded, identical at train and probe. Keep SIGReg for isotropy only.

**C3 — Temporal encoder.** Give `PriceEncoder` a `Conv1d` or small temporal attention over the T axis (LeWM `Embedder`, `model_lewm.py:204-218`) so the encoder sees time; keep the causal predictor for forecasting. This directly attacks the low `eff_rank`.

**C4 — Multi-horizon, LN + smooth-L1 targets.** Predict at τ∈{1,6,12,24} (like I-JEPA's M target blocks, `legacy/ijepa.py:100-150`), each target LayerNorm'd, loss = mean smooth-L1. More supervision signal, more stable.

**C5 — Training schedule.** Add `--warmup_epochs` + cosine decay (`legacy/ijepa.py:43-46`), base `lr=3e-4`, weight-decay `0.05` on ≥2-D params (`param_groups`, `legacy/ijepa.py:30-34`), EMA momentum schedule, and longer runs (100–200 epochs; data is small, ~27k windows). Keep `--aux_lambda` (directional lever, model.py:120,164) and tune it (0.5–2.0) — it is the right tool for `dirAUC`.

**C6 — Feature decorrelation.** Drop the most collinear base features (e.g. `ma5/20/60`, `vol5/20/60` overlap heavily) or PCA-whiten inputs, to lift the intrinsic rank ceiling behind W6.

## 4. Healthy training plan

1. **Smoke (10 ep, λ=0.5, no EMA yet):** confirm C2/C3 stop the train/eval mismatch; watch `stdZ≈1`, `val_pred_loss` falling, `eff_rank` rising off 5.
2. **Full (150 ep):** `lr=3e-4`, warmup 5 ep → cosine, wd=0.05, EMA `0.996→1.0`, target LN + smooth-L1, multi-τ, `sigreg_lambda∈{0.5,1.0}` (fixed per run, no mid-run sweep), `aux_lambda∈{0.5,1.0}`. Batch 256, `grad_clip=1.0` (already at train_forex_h1.py:168).
3. **λ/aux grid (each 150 ep):** {(λ=0.5,aux=0.5),(1.0,0.5),(1.0,1.0),(0.5,1.0)} — track `eff_rank`, `probe_IC`, `probe_dirAUC` jointly.
4. **Probe:** frozen encoder → `MLPProbe`, `--tau 24` (probe_forex_h1.py:108), report IC/rankIC/R²/dirAUC + alpha-label VoE.

## 5. Realistic targets

Given FX difficulty and the §17 noise floor (|IC|≈0.02), treat these as success thresholds, not guarantees:

| Metric | Current (§17, best λ=2.0) | Target |
|---|---|---|
| `val_eff_rank` | 14.8/64 | **25–40/64** (C1+C3+C6) |
| `probe_IC` | 0.052 | **0.08–0.15** |
| `probe_rankIC` | +0.027 | **+0.05–0.12** |
| `probe_dirAUC` | 0.502 | **0.54–0.58** (the key gap; C1+C5 aux) |
| `VoE_alpha_label_AUC` | 0.504 | **0.53–0.57** |
| `val_pred_loss` | 0.004 | keep <0.01 (don't over-trade rank for it) |

If `dirAUC` still sits at ~0.50 after C1–C5, the bottleneck is intrinsic (latent-only JEPA on hourly FX does not encode *direction* — consistent with the paper's own VoE finding, REPORT §2.6), and the next lever is a **stronger aux directional head** or a **return-decoding projection head**, not further λ tuning.
