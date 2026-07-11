# Fin-JEPA — Standard Operating Procedure (SOP)

> Agent-facing runbook for working with this project. Captures the Colab workflow
> and the non-obvious gotchas so we don't re-learn them each session. Companion to
> `REPORT.md` (state) and `JOURNAL.md` (narrative/lessons). When in doubt, the
> journal's §3 (pitfalls) is the short version of this file.

---

## 0. The 30-second version
1. Edit code → **commit & push** (the Colab VM *clones the repo*, it does not see
   un-pushed local changes).
2. Launch training with `colab run --keep` (+ `--timeout 5400`), a **distinct**
   `--session` name, and pass λ as the 2nd script arg.
3. Monitor live with `watch_progress.py` (polls `train_log.jsonl`).
4. When done, **download artifacts to a unique local dir**, then **stop the session**
   (one GPU assignment per account).
5. Record results in `JOURNAL.md` §4 ledger and update `REPORT.md`.

---

## 1. Colab session lifecycle (CRITICAL)
- **One assignment per account.** Launching a 2nd `--session` while a prior is kept
  alive throws `TooManyAssignmentsError`. **Always `colab stop -s <old>` before
  launching the next run.**
- **Distinct `--session` name per run** (e.g. `finjepa-l05`, `finjepa-l2`). Re-using a
  name on a kept session is rejected.
- **`--keep`** keeps the VM alive after the script so you can attach/download later.
  Without it, the VM is released the moment the script ends (you'd lose `last.pt`).
- **`--timeout` defaults to 30s and WILL kill your run.** Always pass
  `--timeout 5400` (40 ep on T4 ≈ 25–35 min; 5400 is safe).
- **No `upload` command exists** in this CLI. You cannot push a local `last.pt` back
  to a fresh VM. If a kept session is evicted, the only recovery is the **local**
  `checkpoints/forex_h1/*.pt` backup — but that also can't be re-uploaded, so you must
  re-run Stage 1. **Treat local `*.pt` as the durable backup.**
- **`colab exec` queues behind a BUSY kernel and times out** — don't rely on it to
  stream output mid-run. Use `colab download` instead (works while BUSY).
- **Housekeeping — close the session when the work is done.** After you've downloaded
  the artifacts, run `colab stop -s <name>`. There is only **one** GPU assignment per
  account; a forgotten `--keep` session blocks every future launch with
  `TooManyAssignmentsError` and burns GPU quota. Stop it even if you think you might
  resume later — a fresh `colab run` is cheaper than a stuck launch.
- **🚨 NEVER ABORT A COLAB COMMAND.** This CLI ties the remote VM's lifetime to the
  local `colab run` client process. If you abort (or otherwise kill) the command that
  launched the run, the client dies and **the remote session is torn down with it** —
  the run is lost and you must relaunch from scratch. Let the command return on its
  own. If a command seems stuck, it almost certainly is NOT — see §9.
- **If a run appears "stuck" on the monitor, it is the VM preparing.** The first
  ~2–4 minutes of every run are the VM cloning the repo, `pip install`ing, and
  building the feature matrix; `train_log.jsonl` does not exist yet, so the monitor
  prints a `[pre-training]` phase line — that is normal, NOT an error, NOT a hang.
  Do not abort; just wait (or run an on-demand `--once` poll, see §9).

## 2. Launch a training run
```bash
cd /home/nayem/Projects/jepa/fin-jepa
# stop any prior session first (one assignment per account)
.colab-venv/bin/colab stop -s finjepa-l1 2>/dev/null || true
# launch detached so the local command returns; poll via watch_progress.py
setsid stdbuf -oL -eL .colab-venv/bin/colab run --gpu T4 --session finjepa-l2 \
    --keep --timeout 5400 colab_run_train.py 40 2.0 \
    < /dev/null > /tmp/colab_finjepa-l2.log 2>&1 & disown
```
- `colab_run_train.py` args: `argv[1]` = **cumulative epoch total**, `argv[2]` =
  **`sigreg_lambda`** (falls back to `FINJEPA_LAMBDA` env, then 0.1).
- It auto-resumes from `last.pt` if present on the VM; on a fresh clone there is none,
  so it trains from scratch. A single `colab run ... 40` beats the old staged plan now
  that training is stable.

## 3. Monitor progress (don't wait on the DONE marker)
```bash
.venv/bin/python watch_progress.py -s finjepa-l2 --epochs 40 --dest /tmp/l2_log.jsonl
```
- Prints a live `n/40 (pct%)` bar + latest `tr/val/effR/stdZ`, then waits for
  `probe.json` and exits.
- **Why this works:** `train_log.jsonl` is flushed per epoch (opened per epoch, closed
  on `with`), and `colab download` works while the session is BUSY. Subprocess *stdout*
  is block-buffered and only flushes at completion — ignore it.
- **Sanity check:** if the monitor reports 100% suspiciously fast, the run has NOT
  finished — verify with `colab status -s <name>` (must say BUSY while training). A
  false 100% means a *stale* `train_log.jsonl` was read (see §5).

## 4. After training is done — download & clean up
```bash
mkdir -p checkpoints/forex_h1_l2
for f in best.pt meta.json probe.json train_log.jsonl last.pt; do
  .colab-venv/bin/colab download -s finjepa-l2 \
    /content/fin-jepa/checkpoints/forex_h1/$f checkpoints/forex_h1_l2/$f
done
# free the GPU (one assignment per account)
.colab-venv/bin/colab stop -s finjepa-l2
```
- **Always download to a UNIQUE local dir per run** (`_l05`, `_l1`, `_l2`, …) so runs
  don't overwrite each other. `*.pt` are gitignored; the JSONs are now gitignored too.
- If a run used the old append-mode log on a polluted clone, the jsonl may have 80
  lines (stale + new). Keep only the last 40: `tail -40 in.jsonl > out.jsonl`.

## 5. Repo hygiene (things that bit us)
- **Never commit run-output JSONs** (`meta.json`, `probe.json`, `train_log.jsonl`).
  They are now gitignored (`checkpoints/**/*.json`). A committed copy ships in every
  fresh clone and (a) fools the monitor into a false "100% done" and (b) gets
  downloaded as if it were the new run's results.
- `train_forex_h1.py` **truncates** `train_log.jsonl` on epoch 1 (was append → polluted
  files on re-runs). Don't change it back to append unless resuming within one VM.
- **Push code before launching.** The VM clones `main`; local-only edits are invisible.
- Keep `REPORT.md` (state) and `JOURNAL.md` (ledger/lessons) updated per run.

## 6. Diagnostic rules of thumb
- **Effective rank is the key signal.** A latent-only JEPA can have a beautifully low
  `val_loss` yet learn nothing useful if `val_eff_rank` is ~5/64 — the encoder packs
  everything into a tiny subspace and the probe is noise. **Low rank → raise
  `sigreg_lambda`** (0.1→0.5→1.0→2.0). In this project rank rose 5.7→14.8 with λ, and
  `probe_IC` rose with it (0.01→0.05).
- **`stdZ ≈ 1.0`** means no embedding collapse (hard-standardization in
  `FinJEPA.forward` enforces this). If it drops toward 0, collapse is back.
- **`probe_dirAUC`** stayed ~0.50 even at high λ here: rank correlation (IC) is real
  but **directional** accuracy is still chance. That's a different problem — fix with a
  return-prediction auxiliary loss or a longer horizon, not more λ.

## 7. Local vs Colab
- Local CPU training is slow (~18+ min just for probe extraction over 27k windows).
  **Use the Colab T4 for anything ≥ a few epochs.**
- Local commands (for reference):
  ```
  .venv/bin/python train_forex_h1.py --epochs 40 --batch 256
  .venv/bin/python probe_forex_h1.py --ckpt checkpoints/forex_h1/best.pt --tau 24
  ```

## 9. Monitoring without it looking stuck (percentage progress)
The monitor (`watch_progress.py`) is the live progress view. Two ways to use it:

1. **Block to completion** (best when you can let it run):
   ```
   .venv/bin/python watch_progress.py -s finjepa-aux05 --epochs 40
   ```
   It prints a `[pre-training]` phase line (with elapsed time) during the VM's
   clone/install/feature-build, then a live `n/40 (pct%)` bar + metrics, then waits
   for `probe.json` and exits. Do NOT abort — see §1.

2. **Launch-and-return + on-demand polls** (preferred to avoid a 30-min block the
   user might abort):
   - Launch detached and return immediately (no blocking monitor in that command):
     ```
     setsid stdbuf -oL -eL .colab-venv/bin/colab run --gpu T4 --session finjepa-aux05 \
         --keep --timeout 5400 colab_run_train.py 40 2.0 0.5 \
         < /dev/null > /tmp/colab_finjepa-aux05.log 2>&1 & disown
     echo launched
     ```
   - Check progress any time with a quick one-shot (returns in ~5s, shows the
     percentage; safe-ish, but still: don't abort the LAUNCH command):
     ```
     .venv/bin/python watch_progress.py -s finjepa-aux05 --epochs 40 --once
     ```
   - When the `--once` poll shows `n == epochs`, download artifacts (§4) and stop
     the session (§1 housekeeping).

**Rule of thumb:** if a command is going to block >~2 min, prefer the launch-and-return
pattern and report progress via `--once` polls on request, rather than a long blocking
monitor the user may be tempted to abort.

## 8. End-of-run checklist
- [ ] Code changes committed & **pushed** (VM clones `main`).
- [ ] Run launched with distinct `--session`, `--keep`, `--timeout 5400`, λ as argv[2].
- [ ] Monitored live with `watch_progress.py` (confirmed BUSY, not false 100%).
- [ ] Artifacts downloaded to a **unique** local dir (`checkpoints/forex_h1_<tag>/`).
- [ ] Session **stopped** (housekeeping — free the single GPU assignment).
- [ ] `JOURNAL.md` §4 ledger + `REPORT.md` updated with the new row.
