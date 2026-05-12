"""Augmentation ablation runs.

Base model = ResNet18 pretrained (cheapest deep run, ~7.5 min each).
Progressive levels:
    none      -> resize only
    hflip     -> + RandomHorizontalFlip
    hflip_rot -> + RandomRotation(15)
    full      -> + ColorJitter   (this is the existing runs/resnet18_pretrained)

We do not re-train `full` here — it reuses runs/resnet18_pretrained/.
"""
import subprocess
import sys
import time
from pathlib import Path


ABLATION_RUNS = [
    {"name": "ablation_aug_none",
     "cmd": [sys.executable, "-m", "src.train",
             "--model", "resnet18", "--pretrained",
             "--name", "ablation_aug_none",
             "--epochs", "30", "--aug", "none"]},
    {"name": "ablation_aug_hflip",
     "cmd": [sys.executable, "-m", "src.train",
             "--model", "resnet18", "--pretrained",
             "--name", "ablation_aug_hflip",
             "--epochs", "30", "--aug", "hflip"]},
    {"name": "ablation_aug_hflip_rot",
     "cmd": [sys.executable, "-m", "src.train",
             "--model", "resnet18", "--pretrained",
             "--name", "ablation_aug_hflip_rot",
             "--epochs", "30", "--aug", "hflip_rot"]},
]


def main() -> None:
    Path("runs").mkdir(exist_ok=True)
    full_run = Path("runs/resnet18_pretrained/eval.json")
    if not full_run.exists():
        print(f"[warn] {full_run} missing — the 'full' baseline for ablation "
              f"comes from runs/resnet18_pretrained. Run scripts/run_all.py first.")

    summary: list[dict] = []
    for exp in ABLATION_RUNS:
        print(f"\n{'=' * 60}\n[run_ablation] starting: {exp['name']}\n{'=' * 60}")
        t0 = time.time()
        try:
            subprocess.run(exp["cmd"], check=True)
            status, err = "ok", ""
        except subprocess.CalledProcessError as e:
            status, err = "failed", str(e)
            print(f"[run_ablation] {exp['name']} FAILED: {err}")
        dt = time.time() - t0
        summary.append({"name": exp["name"], "status": status,
                        "elapsed_sec": round(dt, 1), "error": err})

    print("\n" + "=" * 60)
    print("[run_ablation] summary")
    for row in summary:
        print(f"  {row['name']:<28} {row['status']:<8} {row['elapsed_sec']:>8.1f}s "
              f"{row['error']}")


if __name__ == "__main__":
    main()
