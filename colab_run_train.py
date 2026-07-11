"""
Wrapper executed ON the Colab VM by `colab run`.

It bootstraps the repo + deps and runs the Fin-JEPA pretraining, then leaves
the session alive (--keep) so the checkpoint can be pulled back with
`colab download`. Training config: a few GPU epochs (edit EPOCHS below).
"""
import os
import subprocess
import sys

REPO = "https://github.com/jsnayem/fin-jepa.git"
EPOCHS = int(os.environ.get("FINJEPA_EPOCHS", "3"))
BATCH = int(os.environ.get("FINJEPA_BATCH", "256"))


def sh(cmd):
    print(">>", cmd)
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    sh(f"git clone {REPO}")
    os.chdir("fin-jepa")
    sh(f"{sys.executable} -m pip install -q einops")
    sh(f"{sys.executable} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\"")
    sh(f"{sys.executable} train_forex_h1.py --epochs {EPOCHS} --batch {BATCH} "
       f"--device cuda --ckpt checkpoints/forex_h1")
    sh(f"{sys.executable} probe_forex_h1.py --ckpt checkpoints/forex_h1/best.pt "
       f"--tau 24 --device cuda --out checkpoints/forex_h1/probe.json")
    print("DONE. Artifacts in checkpoints/forex_h1/ (best.pt, meta.json, probe.json)")
