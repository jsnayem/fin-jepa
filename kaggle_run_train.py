"""
kaggle_run_train.py — executed INSIDE a Kaggle GPU notebook/kernel.

Mirrors the proven colab_run_train.py flow but targets the RiskJEPA (risk-reward)
data path:
  - clones the fin-jepa repo (contains data/EURUSD_H1.csv + riskjepa/),
  - pip installs einops (model.py imports it; Kaggle base image may lack it),
  - trains riskjepa/train.py on GPU (sigreg_lambda=2.0, aux_lambda=0.5 — the
    config that gave the best effective rank + a real (if modest) probe IC in the
    latent-only sweep, retargeted now at the vol-normalized return label),
  - runs the risk-reward probe + cost-aware backtest so the kernel log/OUTPUT
    carries the real P&L verdict (profit factor, Sharpe, winrate, %-flat).

Artifacts are written to /kaggle/working/ (downloadable from the kernel) and the
final verdict is printed as a JSON line for easy scraping.
"""
import os
import subprocess
import sys

REPO = "https://github.com/jsnayem/fin-jepa.git"
EPOCHS = int(os.environ.get("RJ_EPOCHS", "40"))
BATCH = int(os.environ.get("RJ_BATCH", "256"))
SIGREG = os.environ.get("RJ_SIGREG", "2.0")
AUX = os.environ.get("RJ_AUX", "0.5")
TGT = os.environ.get("RJ_TGT", "12")
CTX = os.environ.get("RJ_CTX", "48")
SPREAD = os.environ.get("RJ_SPREAD", "0.10")   # round-turn cost (vol-normalized units)
WORK = os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working")


def sh(cmd, capture=False):
    print(">>", cmd, flush=True)
    if capture:
        return subprocess.run(cmd, shell=True, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True).stdout
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    # Kaggle drops the repo at /kaggle/working or we clone it.
    here = os.getcwd()
    if not os.path.isdir("fin-jepa"):
        sh(f"git clone {REPO}")
    os.chdir("fin-jepa")

    sh(f"{sys.executable} -m pip install -q einops")
    sh(f"{sys.executable} -c \"import torch; print('CUDA', torch.cuda.is_available(), "
       f"torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\"")

    ckpt_dir = "checkpoints/riskjepa"
    os.makedirs(ckpt_dir, exist_ok=True)
    resume = f"{ckpt_dir}/last.pt" if os.path.exists(f"{ckpt_dir}/last.pt") else None

    train_cmd = [
        sys.executable, "-u", "riskjepa/train.py",
        "--epochs", str(EPOCHS), "--batch", str(BATCH),
        "--ctx", str(CTX), "--tgt", str(TGT),
        "--sigreg_lambda", SIGREG, "--aux_lambda", AUX,
        "--device", "cuda", "--ckpt", ckpt_dir,
    ]
    if resume:
        train_cmd += ["--resume", resume]
    print(">>", " ".join(train_cmd), flush=True)
    with open("train_run.log", "w") as tl:
        try:
            subprocess.run(train_cmd, check=True, stdout=tl, stderr=subprocess.STDOUT)
        finally:
            tl.flush()
    sh("cat train_run.log")

    # Risk-reward probe + cost-aware backtest on the val split.
    probe_out = os.path.join(ckpt_dir, "probe.json")
    sh(f"{sys.executable} -m riskjepa.probe --ckpt {ckpt_dir}/best.pt "
       f"--data data/EURUSD_H1.csv --spread_bars {SPREAD} --out {probe_out}")

    # Copy artifacts to the Kaggle output dir so they're downloadable.
    os.makedirs(WORK, exist_ok=True)
    for f in ("best.pt", "last.pt", "meta.json", "probe.json", "train_log.jsonl"):
        src = os.path.join(ckpt_dir, f)
        if os.path.exists(src):
            import shutil
            shutil.copy(src, os.path.join(WORK, f"riskjepa_{f}"))
    print("DONE. Artifacts in", WORK, "(riskjepa_best.pt, riskjepa_probe.json, ...)")
