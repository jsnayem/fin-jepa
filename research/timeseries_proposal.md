# Time-Series Modeling Improvements for Fin-JEPA

## 0. Diagnosis (why the encoder underperforms)

The current `PriceEncoder` (`model.py:39-65`) is a **per-timestep MLP**: `encode_batch`
(`model.py:122-127`) flattens `(B,T,F)`→`(B*T,F)` and feeds each bar independently
through `input_proj` (`model.py:43`) + a 3-layer MLP. Three consequences:

1. **No within-bar temporal structure.** The 32 features per bar (OHLC shapes,
   volatilities, ma-ratios, alphas — `forex_features.py:74-76`) are collapsed to `D=64`
   by a single `Linear(F→D)` then mixed *within* the bar, but the encoder never sees the
   bar's position relative to its neighbors. All cross-bar correlation is delegated to the
   `TransformerPredictor` (`model.py:69`). This is the inverse of every strong TS encoder
   (PatchTST, TimesNet, TS2Vec), where the *tokenizer* itself captures local temporal
   patterns.
2. **Low effective rank is partly an encoder artifact.** The encoder emits a near-constant
   function of a bar's static feature vector; only the predictor's attention injects
   variation. That structurally caps `eff_rank` (stuck 5–12/64 across λ sweeps, REPORT §16-17).
3. **`pos_embed` is a single learned lookup** (`model.py:72`), identical for every window,
   and the predictor sees `ctx`+`tgt` as one 144-token sequence. There is no notion of
   *where in the 5-day context* a token sits beyond an absolute index, and no handling of
   FX regime/seasonality (London/NY opens, weekend lull).

## 1. Encoder redesign — conv-tokenizer + local patching

Replace the point-wise MLP with a **temporal conv-tokenizer** that ingests a short local
window of bars per token, so the encoder itself learns shape/microstructure:

- Input `(B,1,T,F)` → `Conv1d(F, D, kernel=5, stride=1, pad=2)` over the time axis
  (`forex_features.py:15`, CTX=120). Each output token is a *5-bar* receptive field instead
  of one bar. Stack 2–3 residual conv blocks (`[Conv-GELU-Conv, residual]`) → `(B,D,T)`.
- Optionally **patch** (PatchTST, Nie et al. 2023): aggregate every `P=4` bars into a patch
  token via `LayerNorm + Linear(5·F→D)`, reducing T to 30 tokens. This (a) cuts attention
  cost `120²→30²`, (b) gives each token a multi-bar "word", and (c) raises the information
  per token so the latent can spread across more dims.
- Keep a light MLP `projector` (`model.py:53-58`) as a non-linear post-token mixer.

This makes the encoder **bidirectional** over the local window (it may read future bars in
its receptive field) — acceptable for SSL pretraining; at inference the predictor still runs
causally on tokens (`predict_future`, `model.py:170`).

## 2. Positional encoding — relative, not absolute

The single learned `pos_embed` (`model.py:72`) does not extrapolate and carries no
direction/distance bias. Switch the predictor to **ALiBi** distance penalties
(Press et al. 2022): add a head-dependent linear bias `m·|i−j|` into attention scores in
`TransformerPredictor.forward` (`model.py:87-100`), drop `pos_embed`, and inject a
**learned per-token time-of-day / day-of-week embedding** (derived from the bar timestamp in
`forex_features.py:25-34`) concatenated to each token — this is the standard fix for FX
regime/seasonality that absolute index PE cannot capture.

## 3. Context / horizon tuning

CTX=120 (5d), TGT=24 (1d) (`forex_features.py:15-16`) is reasonable but the encoder only
"sees" 120 bars through the predictor. Propose:
- **CTX=240 (10d), TGT=24/48**: more lookback for the conv-tokenizer; keep TGT=24 for the
  probe (`probe_forex_h1.py:108`) but also pretrain a TGT=48 head.
- **Non-stationarity**: wrap the feature tensor in **RevIN** (Kim et al. 2022) —
  instance-normalize each window in `make_dataset` (`forex_features.py:97`), restore after
  the predictor. FX vol/mean drift badly; RevIN is the cheapest, highest-ROI fix for
  distribution shift and is orthogonal to the encoder change.

## 4. Collapse avoidance (keep, sharpen)

The hard standardization + SIGReg combo (`model.py:147-159`) is correct and should stay; it
is what stopped the §13 collapse. Two refinements:
- **Raise the operating λ to ~1.0–2.0** (REPORT §17: rank rose monotonically to 14.8/64 at
  λ=2.0, probe_IC to 0.052). The current default `0.1` (`train_forex_h1.py:86`) is too low.
- Add a **variance floor / batched covariance objective** (TS2Vec-style) as a softer
  complement so very high λ doesn't over-spread and inflate `val_loss` (REPORT §17 tail).

## 5. Training plan (healthy, staged)

1. Freeze the objective fixes already in (§13/§16). Add conv-tokenizer + ALiBi + RevIN.
2. Sweep: `CTX∈{120,240}`, `enc=conv3`, `pred_layers=6`, `λ∈{1.0,2.0}`, `aux_lambda∈{0,0.5}`
   (`train_forex_h1.py:88`) on Colab T4, 40 ep each. Track `val_eff_rank` *and*
   `probe_dirAUC` jointly.
3. Use the auxiliary forward-label head (`model.py:120`, `return_head`) at `aux_lambda=0.5`
   — REPORT §18 shows directional signal is the missing lever; it injects gradient toward
   the probe's exact target without disturbing the JEPA pretask.
4. LR 1e-4, AdamW, grad-clip 1.0 (`train_forex_h1.py:125,168`), seed 0 for comparability.

## 6. Realistic targets

Given λ=2.0 already reached eff_rank 14.8, probe_IC 0.052, the richer encoder + RevIN +
ALiBi should push:
- **eff_rank ≥ 25–35 / 64** (conv patches spread dimensionality),
- **probe_IC ≥ 0.06–0.10**, rankIC ≥ 0.04,
- **probe_dirAUC ≥ 0.53–0.55** (the current ~0.50 floor; the aux head is key),
- **VoE_alpha_label_AUC ≥ 0.52**.

If dirAUC still fails to clear 0.53 after these, per REPORT §16/§18 the latent-only JEPA on
hourly FX is near its ceiling and a return-prediction *decoder* head (not just probe) becomes
the correct next lever — but the encoder changes above address the dominant bottleneck
(low-rank, no within-bar structure) first.
