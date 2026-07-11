# Fin-JEPA on Hourly EUR/USD — Session Report

> Purpose: hand off full context to the next session. Read top-to-bottom; the
> "CURRENT BLOCKER" and "NEXT STEPS" sections are the actionable part.

---

## 1. Objective
Train a **paper-faithful (latent-only) Fin-JEPA** on hourly EUR/USD forex data,
enriched with Formulaic Alpha features and a probe head for a downstream tradable
signal. Source paper: `docs/jepa_full_draft.md` (Fin-JEPA = JEPA for financial
time series; encoder `PriceEncoder` MLP+GELU+LayerNorm + causal
`TransformerPredictor`; best config **v4_deep_d64**: D=64, enc 4 layers, pred 6
layers, heads=4, SIGReg λ=0.1). Latent-only (no price output head).

## 2. Key user decisions (do not re-litigate)
1. Latent-only, paper-faithful.
2. Use real data already in repo: `data/EURUSD_H1.csv` (hourly EUR/USD).
3. Features = price (O/H/L/C) + real Volume; **drop the unlabeled 7th column**
   (it is the spread, values 1–15, degenerate).
4. Include Opt1 alpha features + Opt2 alpha-label VoE + Opt3 probe head.
5. VWAP computed as a **20-bar VWAP proxy from the volume column** (true VWAP
   absent in the data).
6. Paper's own finding: latent error vs raw forward return is ~random
   (AUC≈0.49–0.50); alpha-based labels expected to be more discriminative.
7. Scale of Colab run: "few epochs first" → then 40 epochs (see §6/§7).

## 3. Repository state
- Branch `main`, remote `git@github.com:jsnayem/fin-jepa.git` (public).
- Everything below is **pushed to GitHub** (commits `7e5db85`, `02551a5`,
  `797e9bc`, `fe8ea4c`).
- `.gitignore` ignores `*.pt`, `.venv/`, `data/raw/`, `output/results/` — so the
  CSV `data/EURUSD_H1.csv` IS tracked, but `checkpoints/*.pt` are NOT.
- **Unstaged (deliberately left out of scope):** deletions of old `ctrader/*`
  scripts. They are unrelated cleanup; safe to leave or stage later.

## 4. Important files
| File | Role |
|---|---|
| `docs/jepa_full_draft.md` | paper spec — source of truth for faithfulness |
| `data/EURUSD_H1.csv` | 97,797 hourly bars, 2010-06-29 → 2026-07-10, 7 tab-sep fields (DateTime, O, H, L, C, Volume, Spread) but only 6-name header |
| `alphas.py` | Formulaic alpha toolkit + 17 curated alpha features + mega-alpha. Uses **lowercase** cols `open/high/low/close/volume/vwap/returns/adv20/adv60`. `ALPHA_COLS` list. `RANK_W=120` substitutes cross-sectional `rank()` with rolling `ts_rank`. |
| `forex_features.py` | `load_eurusd_h1`, `add_vwap_adv`, `build_base_features` (15 base feats), `build_feature_matrix`, `make_dataset` (gap-split, 90/10 time split, z-score on train only, **CTX=120, TGT=24, GAP_H=120**), `ForexH1Dataset` + `probe_pairs`. |
| `model.py` | `FinJEPA`, `SIGReg`, `PriceEncoder`, `TransformerPredictor`, `LinearProbe`, `MLPProbe`. **368,640 params** at D=64/enc4/pred6/heads4. |
| `train_forex_h1.py` | JEPA pretrain; logs stdZ + effective_rank; saves `best.pt` + `meta.json` (now with `val_pred_loss`, `val_sigreg_loss`). |
| `probe_forex_h1.py` | frozen-embedding probe head (IC/rankIC/R²/dirAUC) + alpha-label VoE (IC/AUC) + raw-dir VoE AUC. |
| `colab_run_train.py` | wrapper run ON the Colab VM: clones repo, `pip install einops`, runs `train_forex_h1.py` (default 40 epochs) + `probe_forex_h1.py`. |
| `colab_train.ipynb` | Colab notebook (clone-or-upload → train → probe). |
| `checkpoints/forex_h1/` | local copy of last downloaded artifacts (`best.pt`, `meta.json`, `probe.json`). |

## 5. Bugs fixed this session (so you don't re-fix)
1. **CSV misparse:** file has 7 tab fields but a 6-name header; pandas dropped
   the datetime and mis-shifted columns. Fixed `load_eurusd_h1` to
   `pd.read_csv(path, sep='\t', header=None, skiprows=1, names=[DateTime,Open,High,Low,Close,Volume,Spread])` then parse `DateTime`.
2. **Column case mismatch:** `alphas.py` expected lowercase `open/...`, df was
   capitalized. Fixed `_build_alpha_features` to use a case-insensitive dict.
3. **`ts_argmax`/`ts_argmin` crash** on all-NaN front rows (warmup). Added
   `_safe_argmax/_safe_argmin`; also `warnings.filterwarnings('ignore', RuntimeWarning)`.
4. **`GAP_H=2` too small:** hourly FX closes on weekends, so each run was only
   ~120 bars (< the 144-bar ctx+tgt window) → 0 valid windows. Set `GAP_H=120`
   (only genuine >5-day outages split; weekly ~48–75h closures are NOT split).
5. **`model.py` syntax error:** class was named `Fin-JEPA` (illegal hyphen) →
   renamed to `FinJEPA`. Also `forward` now returns `'pred'` (avoids recompute).

## 6. Local validation (CPU, `.venv`)
- Feature pipeline: 27,251 train / 3,238 val windows, 32 features
  (15 base + 17 alphas), probe labels clean (nan frac 0).
- `FinJEPA` instantiates at 368,640 params; forward runs.
- CPU training is slow (~18+ min for probe extract over 27k windows) — use Colab GPU.
- Full `probe_forex_h1.py` validated on a 3k-window CPU subset (random-init:
  IC≈0.10, VoE IC≈0.08) — code path correct.

## 7. Colab GPU run (google-colab-cli) — HOW TO
- Tool: **`google-colab-cli` 0.6.0** installed in **`.colab-venv`** (NOT the
  `colab-cli` sync-only package — that one can't run anything). Binary:
  `.colab-venv/bin/colab`.
- Auth already done (OAuth cached in `~/.config/colab-cli/`). Requires a browser
  only the first time.
- Launch training (detached so the local command returns; poll the log):
  ```
  cd /home/nayem/Projects/jepa/fin-jepa
  setsid stdbuf -oL -eL .colab-venv/bin/colab run --gpu T4 --session finjepa-train \
      --keep --timeout 5400 colab_run_train.py < /dev/null > /tmp/colab_train.log 2>&1 & disown
  ```
  (The inner `--timeout` MUST be > training time; default is 30s and it WILL kill
  the run. 40 epochs on T4 ≈ 25–35 min — use ≥3600, 5400 is safe.)
- Poll: `tail -30 /tmp/colab_train.log`. Training epoch prints are block-buffered
  and only flush at completion; the wrapper `>>` echoes appear live.
- Check liveness: `.colab-venv/bin/colab sessions` / `colab status -s finjepa-train`
  (status shows BUSY while training; `colab exec` queues behind the busy kernel
  and will time out, so don't rely on it mid-run).
- **Download artifacts** (note the absolute remote path — the script `chdir`s
  into `fin-jepa`, and the session cwd is `/content`):
  ```
  for f in best.pt meta.json probe.json; do
    .colab-venv/bin/colab download -s finjepa-train \
      /content/fin-jepa/checkpoints/forex_h1/$f checkpoints/forex_h1/$f
  done
  ```
- Stop session: `.colab-venv/bin/colab stop -s finjepa-train`.

## 8. RESULTS so far (both 3-epoch and 40-epoch runs)
Identical, because the **best checkpoint was saved at epoch 3** in both cases:
```
n_params 368640 | epoch 3 | best_val_loss 2.758
val_pred_loss   0.759
val_sigreg_loss 19.99
val_eff_rank    24.28 / 64
val_stdZ        0.836
probe_IC        0.027   probe_rankIC 0.040   probe_R2 -0.01   probe_dirAUC 0.508
VoE_alpha_label_IC  -0.099   VoE_alpha_label_AUC 0.427   VoE_rawdir_AUC 0.506
```
Probe signal is **essentially random** (IC≈0.03, dirAUC≈0.51).

## 9. CURRENT BLOCKER (the real problem)
Training **diverges after epoch 3** rather than improving. Evidence:
`val_pred_loss` rose from ~0.09 (init) to 0.759 and `val_sigreg_loss` exploded
from ~0.86 to ~20. Two root causes in `model.py` / `train_forex_h1.py`:

1. **SIGReg dominates the loss.** In `SIGReg.forward`, `statistic` is multiplied
   by `proj.size(-2)` = sequence length (144 = ctx+tgt). So `sigreg_loss` is
   scaled ~144×; at λ=0.1 it contributes ~2.0 vs `pred_loss` 0.76, forcing
   embeddings isotropic Gaussian and destabilizing training.
2. **`pred_loss` is degenerate (in-painting).** `forward` compares
   `z_pred[:, :n_compare]` with `z_tgt[:, :n_compare]` — i.e. the SAME timestep,
   not the future. True JEPA predicts the **future** latent `z_tgt[t+k]` from
   `z_ctx[≤t]`. A same-index target learns little temporal structure, which is
   why the encoder embeddings stay near-random and the probe is flat.

Net: running more epochs does NOT help because the saved `best.pt` is the
epoch-3 model in both runs.

## 10. NEXT STEPS (proposed — not yet executed; needs user go-ahead)
A. **Fix the objective** (faithful JEPA):
   - In `model.FinJEPA.forward`, shift the target so the predictor predicts the
     next/future latent: compare `z_pred[:, :-k]` with `z_tgt[:, k:]` (k=1, or
     predict the whole TGT horizon ahead). Currently it's same-index.
   - In `SIGReg.forward`, remove the `× proj.size(-2)` scaling (or set
     `sigreg_lambda` much lower / drop SIGReg) so the encoder can actually learn.
   - Re-verify param count stays 368,640 (paper-faithful) after edits.
B. **Optional diagnostic:** run a short **unbuffered** training
   (`python -u train_forex_h1.py --epochs 10 --log_every 1`) locally or on Colab
   to print the per-epoch `pred_loss`/`sigreg_loss` trajectory and confirm the
   divergence is fixed before committing to 40 epochs.
C. After fix: re-run 40 epochs on Colab T4, download, and re-evaluate probe +
   VoE. Expect probe IC/rankIC to rise above noise if the encoder learns.
D. (Stretch) Add loss/eff-rank curve plotting and a script to export latent
   embeddings for the cTrader side.

## 11. Useful commands
Local pretrain (CPU, slow):
```
.venv/bin/python train_forex_h1.py --epochs 40 --batch 256
.venv/bin/python probe_forex_h1.py --ckpt checkpoints/forex_h1/best.pt --tau 24
```
Colab (see §7 for full flow). Smoke test that GPU is available:
```
.colab-venv/bin/colab run --gpu T4 /tmp/colab_smoke.py   # prints Tesla T4 + CUDA True
```

## 12. Open questions / things to confirm
- Is the v4_deep_d64 paper config (D=64, enc4, pred6, heads4, λ=0.1) the right
  target, given SIGReg scaling makes λ=0.1 effectively huge? May need λ tuning.
- Confirm `data/EURUSD_H1.csv` is the intended dataset (committed; 97,797 rows).
- Decide whether to also stage/commit the `ctrader/*` deletions.

---

## 13. UPDATE — fixes applied + resumable staged training

**Root causes from §9 are FIXED in code (committed & pushed):**
- `model.py` SIGReg: projection matrix `A` is now a **fixed buffer** (sampled once,
  seeded) instead of `torch.randn` every call; removed the `* proj.size(-2)` (×144)
  scaling → `val_sigreg_loss` dropped from ~20 to ~0.42 at init (stable).
- `model.py` FinJEPA.forward: encodes the **joint** `cat(ctx,tgt)` and predicts the
  **future** latents (`pred_loss = mse(z_pred[:, T_ctx:], z_full[:, T_ctx:])`) instead
  of same-timestep in-painting. True JEPA objective. Exposes `z_tgt` (standardized
  target latents) for consistent VoE.
- `probe_forex_h1.py`: VoE latent error now compares predicted-future vs actual target
  latents in the SAME (standardized) space.
- `train_forex_h1.py`: added `--resume`, LR default → `1e-4`, saves `last.pt` every
  epoch (model + optimizer + RNG states); writes `train_log.jsonl` per-epoch.

**Second fix — EMBEDDING COLLAPSE (committed `6c52f13`):** after §9 fixes, Stage 1
still collapsed: `stdZ` → 0.008 because SIGReg only enforces *isotropy*, not *scale*
(the model shrinks latents to a tiny ball and gets `pred_loss`≈0 for free). Fix: in
`FinJEPA.forward`, **hard-standardize** `z_full` to zero-mean/unit-variance per feature
(over the batch) before SIGReg + pred-loss. This structurally forbids collapse. Dropped
the weak soft `var_beta` penalty. After fix `stdZ` ≈ 1.0 (healthy).

**Staged training flow (CORRECTED — no `upload` command exists in this CLI):**
- `colab run --keep` can be issued **once** per session. A second `colab run` on the
  same kept session fails with `TooManyAssignmentsError` (Google rejects re-assign).
- There is **NO `colab upload`**. To continue past Stage 1 you must reuse the **kept
  VM's own `last.pt`** via `colab exec -f <local_script.py>` (exec uploads+runs a LOCAL
  file on the live VM; cwd is `/content`, repo at `/content/fin-jepa`).
- `colab exec` does **not** stream long-running subprocess output. Launch training
  **detached**, redirect to a log, then poll with `colab download`:
  ```
  # local /tmp/stage2.py:
  import os, sys, subprocess
  os.chdir("/content/fin-jepa")
  subprocess.Popen([sys.executable, "-u", "colab_run_train.py", "10"],
                   stdout=open("stage2.log","w"), stderr=subprocess.STDOUT)
  ```
  `colab exec -s finjepa -f /tmp/stage2.py --timeout 60`  → then
  `colab download -s finjepa /content/fin-jepa/checkpoints/forex_h1/train_log.jsonl .`
- **If the kept session is evicted/lost** (happens on `stop` or idle timeout): the VM
  `last.pt` is gone and cannot be re-uploaded. The ONLY recovery is the **local**
  `checkpoints/forex_h1/last.pt` backup — but it also can't be uploaded, so you must
  **re-run Stage 1 fresh** (`colab run --keep colab_run_train.py 10`) and continue via
  `exec`. Lesson: treat local `checkpoints/forex_h1/*.pt` as the durable backup; the
  staged plan's "upload" step is impossible with this CLI.

## 14. RESULTS — Stage 1 (epochs 1–10), post-collapse-fix (commit `6c52f13`)

Training is now **healthy** (collapse gone):
```
epoch  tr_loss  val_loss  val_pred  val_sig  eff_rank  stdZ
1       0.577    0.238     0.220     0.181    5.44      1.000
5       0.027    0.035     0.0167    0.183    4.55      1.000
10      0.018    0.0203    0.0079    0.124    5.33      1.000
best_val_loss 0.0203; val_eff_rank 5.33/64; val_stdZ 1.000
```
- `stdZ` stable ≈ 1.0 (no collapse). `val_pred_loss` 0.22 → 0.008 (learns future latents).
- `val_eff_rank` stays **low (~5/64)** — embeddings are isotropic but occupy a low-dim
  subspace; SIGReg (λ=0.1) is not spreading dimensionality.
- **Probe still ~noise** (best.pt): `probe_IC -0.0048`, `probe_rankIC -0.034`,
  `probe_dirAUC 0.469`; `VoE_alpha_label_IC 0.016`, `VoE_alpha_label_AUC 0.518`,
  `VoE_rawdir_AUC 0.514`. No tradable downstream signal at 10 epochs.

## 15. WRAP-UP (end of session — progress saved)

**Done & pushed:** CSV/feature/alpha/model plumbing; §9 divergence fixes (fb557ab);
hard-standardization collapse fix (6c52f13); latent-only JEPA runs cleanly on Colab T4.
**Done locally (backed up, NOT committed):** Stage-1 artifacts in `checkpoints/forex_h1/`
(`last.pt` 4.6MB, `best.pt`, `meta.json`, `probe.json`, `train_log.jsonl`). Weights are
gitignored (`*.pt`); the JSON logs are small evidence files.

**Paused at:** 10 / 40 epochs. Colab session `finjepa` **terminated** (GPU freed). Local
Stage-1 backups intact.

**Open / next session:**
1. Resume to 40 epochs via the corrected flow (§13): fresh `colab run --keep
   colab_run_train.py 10` (Stage 1 again, ~2 min), then `colab exec -f` stages 2–4 with
   the detached-logging pattern, polling `train_log.jsonl`. Re-evaluate probe IC.
2. If probe IC stays ~0 at 40 epochs, the likely culprit is **low effective rank
   (~5/64)**: raise `sigreg_lambda` (e.g. 0.5–1.0) to force wider/richer latents, or
   accept that a latent-only JEPA on FX may not encode forward returns (consistent with
   the paper's own VoE finding in §2.6). Also try a longer context/horizon or a
   return-prediction auxiliary loss.
3. Consider committing the small JSON result files (`meta.json`, `probe.json`,
   `train_log.jsonl`) as evidence; keep `*.pt` out of git.

**How to resume quickly next time:**
```
.colab-venv/bin/colab run --gpu T4 --session finjepa --keep --timeout 1200 colab_run_train.py 10
# then for each further 10-epoch stage, exec the detached logger (§13) and poll:
.colab-venv/bin/colab download -s finjepa /content/fin-jepa/checkpoints/forex_h1/train_log.jsonl .
.colab-venv/bin/colab download -s finjepa /content/fin-jepa/checkpoints/forex_h1/probe.json .
```
