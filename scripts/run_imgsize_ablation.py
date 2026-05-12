"""Input image size ablation runs.

Base model = ResNet18 pretrained (size-agnostic via adaptive avg pool).
Sizes:
    128  -> ~0.25x area
    192  -> ~0.56x area
    224  -> 0.77x area (ImageNet standard)
    256  -> baseline (reuses runs/resnet18_pretrained)
    384  -> 2.25x area

Same hyperparams (batch=32, lr=1e-4 AdamW, cosine+2ep warmup, AMP, 30 epoch)
as the main resnet18_pretrained run — only --img_size changes.
"""
import subprocess
import sys
import time
from pathlib import Path


ABLATION_RUNS = [
    {"name": "ablation_imgsize_128",
     "cmd": [sys.executable, "-m", "src.train",
             "--model", "resnet18", "--pretrained",
             "--name", "ablation_imgsize_128",
             "--epochs", "30", "--img_size", "128"]},
    {"name": "ablation_imgsize_192",
     "cmd": [sys.executable, "-m", "src.train",
             "--model", "resnet18", "--pretrained",
             "--name", "ablation_imgsize_192",
             "--epochs", "30", "--img_size", "192"]},
    {"name": "ablation_imgsize_224",
     "cmd": [sys.executable, "-m", "src.train",
             "--model", "resnet18", "--pretrained",
             "--name", "ablation_imgsize_224",
             "--epochs", "30", "--img_size", "224"]},
    {"name": "ablation_imgsize_384",
     "cmd": [sys.executable, "-m", "src.train",
             "--model", "resnet18", "--pretrained",
             "--name", "ablation_imgsize_384",
             "--epochs", "30", "--img_size", "384"]},
]


def main() -> None:
    Path("runs").mkdir(exist_ok=True)
    base = Path("runs/resnet18_pretrained/eval.json")
    if not base.exists():
        print(f"[warn] {base} missing — the 256-baseline for this ablation "
              f"comes from runs/resnet18_pretrained. Run scripts/run_all.py first.")

    summary: list[dict] = []
    for exp in ABLATION_RUNS:
        print(f"\n{'=' * 60}\n[run_imgsize_ablation] starting: {exp['name']}\n{'=' * 60}")
        t0 = time.time()
        try:
            subprocess.run(exp["cmd"], check=True)
            status, err = "ok", ""
        except subprocess.CalledProcessError as e:
            status, err = "failed", str(e)
            print(f"[run_imgsize_ablation] {exp['name']} FAILED: {err}")
        dt = time.time() - t0
        summary.append({"name": exp["name"], "status": status,
                        "elapsed_sec": round(dt, 1), "error": err})

    print("\n" + "=" * 60)
    print("[run_imgsize_ablation] summary")
    for row in summary:
        print(f"  {row['name']:<28} {row['status']:<8} {row['elapsed_sec']:>8.1f}s "
              f"{row['error']}")


if __name__ == "__main__":
    main()
