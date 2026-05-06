"""Run all 7 experiments sequentially. Logs each run's status; continues on failure."""
import subprocess
import sys
import time
from pathlib import Path


EXPERIMENTS = [
    {"name": "pca_dt",
     "cmd": [sys.executable, "-m", "src.train_pca_dt", "--name", "pca_dt"]},

    {"name": "resnet18_pretrained",
     "cmd": [sys.executable, "-m", "src.train", "--model", "resnet18", "--pretrained",
             "--name", "resnet18_pretrained", "--epochs", "30"]},
    {"name": "resnet18_scratch",
     "cmd": [sys.executable, "-m", "src.train", "--model", "resnet18",
             "--name", "resnet18_scratch", "--epochs", "30"]},

    {"name": "resnet50_pretrained",
     "cmd": [sys.executable, "-m", "src.train", "--model", "resnet50", "--pretrained",
             "--name", "resnet50_pretrained", "--epochs", "30"]},
    {"name": "resnet50_scratch",
     "cmd": [sys.executable, "-m", "src.train", "--model", "resnet50",
             "--name", "resnet50_scratch", "--epochs", "30"]},

    {"name": "vit_pretrained",
     "cmd": [sys.executable, "-m", "src.train", "--model", "vit_base", "--pretrained",
             "--name", "vit_pretrained", "--epochs", "30"]},
    {"name": "vit_scratch",
     "cmd": [sys.executable, "-m", "src.train", "--model", "vit_base",
             "--name", "vit_scratch", "--epochs", "30"]},
]


def main() -> None:
    Path("runs").mkdir(exist_ok=True)
    summary: list[dict] = []
    for exp in EXPERIMENTS:
        print(f"\n{'=' * 60}\n[run_all] starting: {exp['name']}\n{'=' * 60}")
        t0 = time.time()
        try:
            subprocess.run(exp["cmd"], check=True)
            status = "ok"
            err = ""
        except subprocess.CalledProcessError as e:
            status = "failed"
            err = str(e)
            print(f"[run_all] {exp['name']} FAILED: {err}")
        dt = time.time() - t0
        summary.append({"name": exp["name"], "status": status,
                        "elapsed_sec": round(dt, 1), "error": err})

    print("\n" + "=" * 60)
    print("[run_all] summary")
    for row in summary:
        print(f"  {row['name']:<24} {row['status']:<8} {row['elapsed_sec']:>8.1f}s "
              f"{row['error']}")


if __name__ == "__main__":
    main()
