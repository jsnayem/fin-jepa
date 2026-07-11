"""
kaggle_run_train.py — executed INSIDE a Kaggle GPU script kernel.

Locates the fin-jepa repo, then trains the RiskJEPA (risk-reward) path:
  - Prefers a fresh `git clone` of main (needs internet; gets the latest code).
  - Falls back to an attached dataset mount (`fin-jepa-train-bundle`, repo under
    `fin-jepa/`) if the clone fails (no/limited internet).
  - Runs riskjepa/train.py on GPU (sigreg_lambda=2.0, aux_lambda=0.5 — the config
    with best effective rank + a real probe IC in the latent-only sweep,
    retargeted at the vol-normalized return label), then runs riskjepa/probe.py
    (frozen-encoder) + the cost-aware risk-reward backtest so the kernel log /
    OUTPUT carries the real P&L verdict. Artifacts copied to /kaggle/working.

NOTE: model.py's unused einops import was removed, so torch alone is required
(present in the Kaggle PyTorch base image) — no pip install needed.
"""
import os
import shutil
import subprocess
import sys

DATASET_MOUNT = "/kaggle/input/fin-jepa-train-bundle/fin-jepa"
WORK = os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working")

EPOCHS = int(os.environ.get("RJ_EPOCHS", "40"))
BATCH = int(os.environ.get("RJ_BATCH", "256"))
SIGREG = os.environ.get("RJ_SIGREG", "2.0")
AUX = os.environ.get("RJ_AUX", "0.5")
TGT = os.environ.get("RJ_TGT", "12")
CTX = os.environ.get("RJ_CTX", "48")
SPREAD = os.environ.get("RJ_SPREAD", "0.10")   # round-turn cost (vol-normalized units)


def sh(cmd, capture=False):
    print(">>", cmd, flush=True)
    if capture:
        return subprocess.run(cmd, shell=True, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True).stdout
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    # Locate the repo. Prefer a fresh git clone (needs internet, gives latest
    # main); fall back to an attached dataset mount (no internet); else cwd.
    repo = None
    if not os.path.isdir("fin-jepa"):
        print(">> attempting git clone (needs internet)...")
        try:
            sh("git clone https://github.com/jsnayem/fin-jepa.git")
            if os.path.isdir("fin-jepa"):
                repo = "fin-jepa"
        except Exception as e:
            print(">> clone failed:", e)
    if repo is None and os.path.isdir(DATASET_MOUNT):
        repo = os.path.join(WORK, "fin-jepa")
        if os.path.isdir(repo):
            shutil.rmtree(repo)
        shutil.copytree(DATASET_MOUNT, repo)
        print(f">> copied dataset repo -> {repo}")
    elif repo is None and os.path.isdir("fin-jepa"):
        repo = "fin-jepa"
    if repo is None:
        repo = os.getcwd()
    os.chdir(repo)

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
            shutil.copy(src, os.path.join(WORK, f"riskjepa_{f}"))
    print("DONE. Artifacts in", WORK, "(riskjepa_best.pt, riskjepa_probe.json, ...)")
