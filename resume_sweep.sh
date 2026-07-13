#!/usr/bin/env bash
# Resume sweep: λ=0.5 is already running (finjepa-l05). Monitor it, download,
# then run λ=2.0 (stopping l05 first — one Colab assignment per account).
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

monitor_and_fetch() {
  local SES=$1 DESTDIR=$2
  .venv/bin/python watch_progress.py -s "$SES" --epochs 40 --dest "/tmp/${SES}_log.jsonl"
  mkdir -p "$DESTDIR"
  for f in best.pt meta.json probe.json train_log.jsonl last.pt; do
    .colab-venv/bin/colab download -s "$SES" \
      /content/fin-jepa/checkpoints/forex_h1/$f "$DESTDIR/$f" 2>&1 | tail -1
  done
  echo "----- $SES results -----"; cat "$DESTDIR/probe.json"; echo
}

# 1) λ=0.5 already running
monitor_and_fetch finjepa-l05 checkpoints/forex_h1_l05

# 2) stop it, launch λ=2.0
.colab-venv/bin/colab stop -s finjepa-l05 2>/dev/null || true
setsid stdbuf -oL -eL .colab-venv/bin/colab run --gpu T4 --session finjepa-l2 \
    --keep --timeout 5400 colab_run_train.py 40 2.0 \
    < /dev/null > /tmp/colab_finjepa-l2.log 2>&1 & disown
monitor_and_fetch finjepa-l2 checkpoints/forex_h1_l2

echo "===== SWEEP DONE ====="
