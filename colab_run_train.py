"""
Wrapper executed ON the Colab VM by `colab run`.

Staged / resumable training:
  - Clones the repo only if missing (so a reused session isn't re-cloned).
  - Resumes from checkpoints/forex_h1/last.pt if present (pass --resume).
  - EPOCHS (cumulative total) comes from argv[1] (default 10). Stage N passes N*10.
  - Training runs unbuffered (-u) so epoch logs stream to the Colab log.
  - After training, runs the probe + VoE eval.
"""
import os
import subprocess
import sys

REPO = "https://github.com/jsnayem/fin-jepa.git"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
BATCH = int(os.environ.get("FINJEPA_BATCH", "256"))
LAMBDA = os.environ.get("FINJEPA_LAMBDA", "0.1")
CKPT = "checkpoints/forex_h1"


def sh(cmd):
    print(">>", cmd)
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    if not os.path.isdir("fin-jepa"):
        sh(f"git clone {REPO}")
    os.chdir("fin-jepa")
    sh(f"{sys.executable} -m pip install -q einops")
    sh(f"{sys.executable} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\"")

    resume = f"{CKPT}/last.pt" if os.path.exists(f"{CKPT}/last.pt") else None
    train_cmd = [sys.executable, "-u", "train_forex_h1.py",
                  "--epochs", str(EPOCHS), "--batch", str(BATCH),
                  "--sigreg_lambda", LAMBDA,
                  "--device", "cuda", "--ckpt", CKPT]
    if resume:
        train_cmd += ["--resume", resume]
    print(">>", " ".join(train_cmd))
    subprocess.run(train_cmd, check=True)

    sh(f"{sys.executable} probe_forex_h1.py --ckpt {CKPT}/best.pt --tau 24 "
       f"--device cuda --out {CKPT}/probe.json")
    print("DONE. Artifacts in", CKPT, "(best.pt, last.pt, meta.json, probe.json)")
