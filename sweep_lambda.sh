#!/usr/bin/env bash
# Sweep sigreg_lambda values, monitoring each live via watch_progress.py.
# Stops the prior session before each launch (one Colab assignment per account).
set -u
cd /home/nayem/Projects/jepa/fin-jepa

run_sweep() {
  local LAM=$1 SES=$2 DESTDIR=$3
  echo "===== λ=$LAM  session=$SES  dest=$DESTDIR ====="
  # stop any prior session still alive
  .colab-venv/bin/colab stop -s finjepa-l1 2>/dev/null || true
  .colab-venv/bin/colab stop -s finjepa-l05 2>/dev/null || true
  .colab-venv/bin/colab stop -s finjepa-l2  2>/dev/null || true
  # launch detached
  setsid stdbuf -oL -eL .colab-venv/bin/colab run --gpu T4 --session "$SES" \
      --keep --timeout 5400 colab_run_train.py 40 "$LAM" \
      < /dev/null > "/tmp/colab_$SES.log" 2>&1 & disown
  # monitor to completion (blocks ~30 min)
  .venv/bin/python watch_progress.py -s "$SES" --epochs 40 --dest "/tmp/${SES}_log.jsonl"
  # download artifacts to a distinct local dir
  mkdir -p "$DESTDIR"
  for f in best.pt meta.json probe.json train_log.jsonl last.pt; do
    .colab-venv/bin/colab download -s "$SES" \
      /content/fin-jepa/checkpoints/forex_h1/$f "$DESTDIR/$f" 2>&1 | tail -1
  done
  echo "----- $SES results -----"
  cat "$DESTDIR/probe.json"
  echo
}

run_sweep 0.5 finjepa-l05 checkpoints/forex_h1_l05
run_sweep 2.0 finjepa-l2  checkpoints/forex_h1_l2
echo "===== SWEEP DONE ====="
