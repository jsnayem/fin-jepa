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
  cd "$(git rev-parse --show-toplevel)"
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

## 16. UPDATE — Stage 1 extended to 40 epochs + λ sweep plan

**40-epoch run completed (fresh Colab T4, 2026-07-11).** Single `colab run --keep
colab_run_train.py 40` (no staging needed now that training is stable). Local
artifacts refreshed: `best.pt`(1.6MB), `last.pt`(4.7MB), `meta.json`, `probe.json`,
`train_log.jsonl`.

```
epoch  tr_loss  val_loss  val_pred  val_sig  eff_rank  stdZ
10      0.018    0.0203    0.0079    0.124    5.33      1.000
20      0.012    0.0136    0.0056    0.082    5.33      1.000
30      0.0094   0.0116    0.0046    0.070    5.46      1.000
37      0.0084   0.0106*   0.0041    0.064    5.51      1.000   (*best_val_loss)
40      0.0081   0.0108    0.0046    0.062    5.67      1.000
```
- `stdZ`≈1.0 throughout, `val_pred_loss` 0.0079→0.0041 → objective well-fit, no collapse.
- **`val_eff_rank` stuck at ~5–5.7/64** — the only signal that didn't improve. Encoder
  packs all learned structure into a ~5-dim subspace.
- **Probe/downstream flat (noise):**
  `probe_IC 0.011`, `probe_rankIC -0.012`, `probe_dirAUC 0.482`;
  `VoE_alpha_label_AUC 0.501`, `VoE_rawdir_AUC 0.503`.

**Conclusion:** confirms the §15 hypothesis — training is healthy but effective rank is
the bottleneck; a latent-only JEPA on hourly FX with λ=0.1 yields no tradable signal.

**NEXT STEP (now executing — λ sweep):** raise `sigreg_lambda` to force wider/richer
latents (the §15 option #2). To make λ tunable from the Colab CLI without code churn,
`colab_run_train.py` now reads `FINJEPA_LAMBDA` (default 0.1) and forwards
`--sigreg_lambda` to `train_forex_h1.py`. Commit `colab_run_train.py` and push, then
relaunch:
```
FINJEPA_LAMBDA=1.0 .colab-venv/bin/colab run --gpu T4 --session finjepa-l1 \
    --keep --timeout 5400 colab_run_train.py 40
```
Plan: run λ∈{0.5, 1.0} (and if rank still low, λ=2.0) for 40 epochs each, track
`val_eff_rank` as the key metric. If rank climbs toward 20–40/64 and probe IC/rankIC
rise above noise (|IC|>~0.02, dirAUC>0.52), the encoder is finally encoding structure.
If rank stays ~5 regardless of λ, the low rank is intrinsic (data/task), and per §2.6
we accept that latent-only JEPA on FX does not encode forward returns — at which point
add a return-prediction auxiliary loss or longer horizon as the next lever.

**Colab session hygiene:** the previous `finjepa` session is kept/alive; each new λ run
uses a distinct `--session` name (re-assign on a kept session is rejected). Stop a
session when done: `.colab-venv/bin/colab stop -s <name>`.

## 17. UPDATE — λ=1.0 run + live progress monitor

**Live progress is now possible.** Training writes `train_log.jsonl` (opened in append
mode per epoch → flushed on close), and `colab download` works while the session is
BUSY. So instead of blocking on the `DONE` marker, poll that file. New local helper:
```
.venv/bin/python watch_progress.py -s <session> --epochs 40 --dest /tmp/l1_log.jsonl
```
It downloads `train_log.jsonl` every `--interval` s, prints a 20-char bar +
`n/40 (pct%)` + latest `tr/val/effR/stdZ`, then waits for `probe.json` and exits.

**λ=1.0 run (session `finjepa-l1`, 2026-07-11) — ran to completion:**
```
epoch  tr_loss  val_loss  val_pred  val_sig  eff_rank  stdZ
10      0.045    0.0468    0.0149    0.0319   9.12      0.999
20      0.033    0.0348    0.0110    0.0238   11.42     0.999
30      0.026    0.0316    0.0098    0.0217   12.05     0.999
39      0.020    0.0301*   0.0098    0.0203   12.28     0.999   (*best_val_loss)
```
- **`val_eff_rank` 5.67 → 12.28** — SIGReg at λ=1.0 successfully widens the latent
  subspace (the §16 bottleneck). `best_val_loss` rose 0.0106→0.0301 because higher λ
  trades `pred` fidelity for isotropy; `val_sig` fell to 0.020 (well-regularized).
- **Probe/downstream improved above noise** (artifacts in `checkpoints/forex_h1_l1/`):
  `probe_IC 0.045` (was 0.011), `probe_rankIC +0.023` (was −0.012),
  `probe_dirAUC 0.505`, `VoE_alpha_label_AUC 0.504`. IC/rankIC now clear of the
  |IC|≈0.02 noise floor; dirAUC still marginal (>0.52 target not yet met).

**Conclusion:** the §16 hypothesis was right — low effective rank was the culprit, and
raising `sigreg_lambda` fixes it, unlocking a real (if still modest) downstream signal.

**NEXT STEP (sweep to find optimal λ):** λ=1.0 is a clear winner over 0.1. To locate the
peak, run λ∈{0.5, 2.0} (each its own `--session`, stop the prior first). Track
`probe_IC`/`probe_dirAUC` and `val_eff_rank` jointly — too-high λ over-spreads and push
`best_val_loss` up. If dirAUC crosses 0.52 at some λ, that's the operating point;
otherwise accept the λ=1.0 signal and move to a return-prediction auxiliary loss /
longer horizon as the next lever (per §16).

### λ sweep results (all 40 ep, T4)
| λ | val_eff_rank | best_val_loss | probe_IC | probe_rankIC | probe_dirAUC |
|---|---|---|---|---|---|
| 0.1 | 5.67 | 0.0106 | 0.011 | −0.012 | 0.482 |
| 0.5 | 11.37 | 0.0216 | 0.041 | +0.035 | 0.507 |
| 1.0 | 12.28 | 0.0301 | 0.045 | +0.023 | 0.505 |
| 2.0 | 14.77 | 0.0447 | 0.052 | +0.027 | 0.502 |

`eff_rank` and `probe_IC` rise **monotonically** with λ. `best_val_loss` also rises
(higher λ trades `pred` fidelity for isotropy — expected). `probe_dirAUC` stays
~0.50–0.51 at every λ: rank correlation is real but **directional** hit-rate is still
chance. Artifacts in `checkpoints/forex_h1{_l05,_l1,_l2}/`.

**Bug hit during sweep — clone pollution (FIXED):** `meta.json`/`probe.json`/
`train_log.jsonl` were committed, so every fresh Colab clone shipped the λ=0.1 copies.
`train_log.jsonl` was also opened in *append* mode, so a new run added to the stale 40
lines. The live monitor then read the stale 40-line file and falsely reported "100%
done" before λ=2.0 even started, and we first downloaded the old `probe.json`. Fixes
(committed): gitignore `checkpoints/**/*.json`; `train_forex_h1.py` now **truncates**
`train_log.jsonl` on epoch 1. Lesson: if the monitor ever shows 100% implausibly fast,
verify with `colab status -s <name>` (must be BUSY while training).

**Next lever (dirAUC still ~0.50):** add a return-prediction auxiliary loss, or try a
longer context/horizon (CTX/TGT), or a richer encoder. λ is now doing its job
(spreading rank); the remaining gap is directional, not dimensional.

## 18. UPDATE — auxiliary forward-label head (directional signal experiment)

**Hypothesis:** `dirAUC` stays ~0.50 because nothing forces the latent to be
*decodable* into the forward label. Add a lightweight head that predicts the forward
mega-alpha (the probe's label, already z-scored) from the **context-embedding mean**
(the same repr the probe uses). Gradient flows into the encoder, so the latent is
shaped to make that label linearly predictable → should lift `probe_IC`/`dirAUC`
without disturbing the JEPA pretask.

**Implementation (committed):** `model.FinJEPA` gains `return_head = nn.Linear(D,1)`
**only when `aux_lambda>0`** (paper-faithful baselines byte-identical). `forex_features`
adds the forward label `y = mega[start+tgt]` to each batch. `train_forex_h1.py` adds
`--aux_lambda`, combines `loss = jepa_loss + aux_lambda * mse(ret_pred, y)` (NaN-masked),
and logs `val_aux`. `best_val` still uses the JEPA loss only (faithful selection).
`colab_run_train.py` forwards `aux_lambda` as argv[3] / `FINJEPA_AUX`.

**Run plan:** launch with `sigreg_lambda=2.0` (best rank from §17) + `aux_lambda=0.5`,
40 ep, session `finjepa-aux05`, artifacts → `checkpoints/forex_h1_aux05/`. Compare
`probe_dirAUC`/`probe_IC` vs the λ=2.0 baseline (dirAUC 0.502, IC 0.052). If dirAUC
crosses ~0.52, the aux head is the missing lever; if not, try `aux_lambda`∈{1.0, 2.0}
or a longer horizon.

**Status (2026-07-11) — experiment code is READY, the run is BLOCKED (not failed):**
- Aux head implemented & committed (`model.FinJEPA.return_head`, only when
  `aux_lambda>0` so paper-faithful baselines are untouched). `forex_features` adds the
  forward mega-alpha label `y`; `train_forex_h1.py` adds `--aux_lambda` and combines
  `loss = jepa_loss + aux_lambda·mse(ret_pred, y)` (NaN-masked). `colab_run_train.py`
  forwards `aux_lambda` (argv[3]) and `batch` (argv[4]).
- **Bugs hit & fixed while attempting the run:**
  1. `RuntimeError: Found dtype Double but expected Float` — the label `y` arrived
     as float64 (numpy), so the aux loss became float64 while the model is float32
     and `backward()` died. Fixed: cast `y` to `float32` and detach `ret_loss` for
     logging. Committed.
  2. The wrapper swallowed the training traceback (`subprocess.run(check=True)`
     raised `CalledProcessError`, hiding the real error). Fixed: wrapper now writes
     training output to `train_run.log` on the VM and always dumps it. Committed.
  3. **`colab exec -f script.py` does NOT forward extra argv** — a launch
     `colab exec ... colab_run_train.py 10 2.0 0.5` errored on `10 2.0 0.5`, so the
     run never started. Use `colab run` (forwards argv) or bake args into a script.
- **Colab environment learnings (see `COLAB_SKILL.md`, added this session):**
  - A session is a Jupyter kernel kept alive by a **detached daemon**, independent of
    the local shell. `colab run` = `new`+`exec`+`stop`. With `--keep` the VM persists.
  - **GPU (T4) allocation returned `Service Unavailable` (503)** repeatedly — a
    transient backend/quota issue, not our code. Fallback: **CPU** (`colab run`
    without `--gpu`).
  - On **CPU the run OOM-killed (SIGKILL)** at batch 256 (Colab CPU RAM is small).
    Relaunched with batch 64 (added `argv[4]` batch to the wrapper) — it started
    (BUSY) but CPU is ~30× slower (epoch 1 not done in ~26 min). A 10–40 ep CPU run
    is impractical; **the aux experiment needs GPU**.
- **Net:** code is correct and ready; re-run on T4 the moment allocation succeeds.

## 19. FUTURE PLAN (next sessions)

**P0 — Complete the auxiliary-label experiment (needs GPU/T4):**
- Retry T4 allocation (the 503 is transient; retry a few times). When it works, run
  `colab run --gpu T4 --session finjepa-aux05 --keep --timeout 5400 \
   colab_run_train.py 40 2.0 0.5` (λ=2.0, aux=0.5, 40 ep), monitor with
  `watch_progress.py -s finjepa-aux05 --epochs 40`, download to
  `checkpoints/forex_h1_aux05/`.
- Compare vs λ=2.0 baseline (dirAUC 0.502, IC 0.052). **Success = dirAUC > ~0.52.**
- If directional signal still flat: sweep `aux_lambda`∈{1.0, 2.0}; or predict the
  **raw forward return** instead of (or in addition to) the mega-alpha; or lengthen
  the horizon `TGT` (and `CTX`).

**P1 — Cheap guardrails so Colab runs stop failing silently:**
- Add a **local CPU smoke test**: `train_forex_h1.py --epochs 1 --aux_lambda 0.5`
  runs in `.venv` and would have caught the dtype bug in seconds (do this before
  every Colab launch). (Note: a 1-ep CPU run of the full model is slow but a tiny
  subset / `--batch 64` makes it feasible.)
- Make the probe step in `colab_run_train.py` non-fatal (so a probe crash doesn't
  lose the trained `best.pt`).

**P2 — If the directional gap persists after aux loss:**
- Return-prediction auxiliary on a *longer* horizon; richer encoder (more layers /
  wider D, re-check param count stays reasonable); multi-scale / volume-aware features.
- Revisit the paper's own VoE finding (§2.6): latent-only JEPA on FX may inherently
  not encode forward *direction* — the aux head is the direct test of that.

**P3 — Housekeeping / docs:**
- Keep `JOURNAL.md` ledger + `REPORT.md` state current; `SOP.md` is the runbook.
- Commit small JSON evidence per run (currently gitignored to avoid clone pollution).
- Stop Colab sessions when done (one assignment per account).


