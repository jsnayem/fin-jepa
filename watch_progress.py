#!/usr/bin/env python3
"""
Local poller: stream per-epoch progress of a Colab JEPA training run.

Training writes checkpoints/forex_h1/train_log.jsonl (one flushed line per epoch)
on the VM. `colab download` works even while the session is BUSY, so we can poll
it for a live percentage + latest metrics, instead of blocking on the DONE marker.

Usage:
  python watch_progress.py -s finjepa-l1 --epochs 40 [--dest /tmp/l1.jsonl] [--interval 30]
"""
import argparse
import json
import os
import subprocess
import sys
import time

COLAB = ".colab-venv/bin/colab"
REMOTE_LOG = "/content/fin-jepa/checkpoints/forex_h1/train_log.jsonl"
REMOTE_PROBE = "/content/fin-jepa/checkpoints/forex_h1/probe.json"


def download(session, remote, local):
    r = subprocess.run(
        [COLAB, "download", "-s", session, remote, local],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--session", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--dest", default="/tmp/colab_train_log.jsonl")
    ap.add_argument("--interval", type=int, default=30)
    args = ap.parse_args()

    prev = 0
    while True:
        if not download(args.session, REMOTE_LOG, args.dest):
            print(f"[poll] download failed (session busy? sleeping {args.interval}s)",
                  file=sys.stderr)
            time.sleep(args.interval)
            continue

        rows = []
        try:
            with open(args.dest) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except FileNotFoundError:
            rows = []

        n = len(rows)
        if n != prev:
            prev = n
            pct = 100.0 * n / args.epochs
            last = rows[-1] if rows else {}
            bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
            print(f"[{bar}] {n:>2}/{args.epochs} ({pct:5.1f}%)  "
                  f"tr={last.get('tr_loss','?'):.4f} val={last.get('val_loss','?'):.4f} "
                  f"effR={last.get('eff_rank','?'):.2f} stdZ={last.get('stdZ','?'):.3f}")
            sys.stdout.flush()

        if n >= args.epochs:
            print("[poll] training complete — waiting for probe eval...")
            for _ in range(40):  # up to ~interval*40s for probe.json
                if download(args.session, REMOTE_PROBE, args.dest + ".probe"):
                    print(f"[poll] probe.json ready: {args.dest}.probe")
                    break
                time.sleep(args.interval)
            else:
                print("[poll] probe.json not found after wait.", file=sys.stderr)
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
