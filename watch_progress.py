#!/usr/bin/env python3
"""
Local poller: stream per-epoch progress of a Colab JEPA training run.

Training writes checkpoints/forex_h1/train_log.jsonl (one flushed line per epoch)
on the VM. `colab download` works even while the session is BUSY, so we poll it
for a live percentage + latest metrics.

Why this exists: the *first ~2-4 minutes* of a run are the VM cloning the repo,
installing deps, and building the feature matrix — during that time train_log.jsonl
does NOT exist yet, so the poller shows a PHASE line (not an error, not "stuck").
Once epoch 1 is written the percentage bar appears.

Usage:
  # block until done, live bar + phase:
  python watch_progress.py -s finjepa-aux05 --epochs 40
  # quick one-shot progress check (for on-demand polling without blocking):
  python watch_progress.py -s finjepa-aux05 --epochs 40 --once
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
    r = subprocess.run([COLAB, "download", "-s", session, remote, local],
                       capture_output=True, text=True)
    return r.returncode == 0


def session_status(session):
    r = subprocess.run([COLAB, "status", "-s", session],
                       capture_output=True, text=True)
    if "not found" in r.stdout.lower():
        return "GONE"
    if "BUSY" in r.stdout:
        return "BUSY"
    if "IDLE" in r.stdout or "READY" in r.stdout:
        return "IDLE"
    return "?"


def progress_line(session, epochs, dest):
    """Return (msg, done_bool). msg is a human progress string."""
    if not download(session, REMOTE_LOG, dest):
        st = session_status(session)
        if st == "GONE":
            return ("[!] session '%s' not found — likely killed (e.g. the "
                    "launching command was aborted). Relaunch." % session, True)
        # pre-training phase: normal, ~2-4 min
        return (f"[pre-training] VM {st} — cloning repo / installing deps / "
                f"building features. Normal; epoch 1 appears in ~2-4 min.", False)
    rows = []
    try:
        with open(dest) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except FileNotFoundError:
        return (f"[pre-training] train_log.jsonl not present yet — VM still "
                f"cloning/installing/building features (~2-4 min).", False)
    n = len(rows)
    if n == 0:
        return "[pre-training] log empty, waiting for epoch 1...", False
    pct = 100.0 * n / epochs
    last = rows[-1]
    bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
    msg = (f"[{bar}] {n:>2}/{epochs} ({pct:5.1f}%)  "
           f"tr={last.get('tr_loss','?'):.4f} val={last.get('val_loss','?'):.4f} "
           f"aux={last.get('val_aux','?'):.4f} effR={last.get('eff_rank','?'):.2f} "
           f"stdZ={last.get('stdZ','?'):.3f}")
    return msg, n >= epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--session", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--dest", default="/tmp/colab_train_log.jsonl")
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--once", action="store_true",
                    help="print current progress once and exit (no blocking)")
    args = ap.parse_args()

    t0 = time.time()
    if args.once:
        msg, done = progress_line(args.session, args.epochs, args.dest)
        el = int(time.time() - t0)
        print(f"[+{el}s] {msg}")
        sys.exit(0)

    prev = 0
    while True:
        msg, done = progress_line(args.session, args.epochs, args.dest)
        el = int(time.time() - t0)
        # only reprint when the percentage changes (avoid spam), but always show
        # a heartbeat during the pre-training phase so it never looks frozen.
        if msg != getattr(progress_line, "_last", "") or "pre-training" in msg:
            print(f"[+{el}s] {msg}", flush=True)
            progress_line._last = msg
        if done:
            if "GONE" in msg:
                break
            print("[poll] training complete — waiting for probe eval...")
            for _ in range(40):
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
