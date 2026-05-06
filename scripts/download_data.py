"""Download the Kaggle 'luisolazo/tomato-diseases' dataset and unzip into ./data.

Prereq: place your Kaggle API token at ~/.kaggle/kaggle.json. On Windows that
maps to %USERPROFILE%/.kaggle/kaggle.json. Get the token from
https://www.kaggle.com/settings -> Create New API Token.
"""
import sys
from pathlib import Path


KAGGLE_DATASET = "luisolazo/tomato-diseases"
DATA_DIR = Path("data")


def check_kaggle_credentials() -> None:
    home = Path.home()
    cred = home / ".kaggle" / "kaggle.json"
    if not cred.exists():
        print("ERROR: Kaggle credentials not found.")
        print(f"Expected at: {cred}")
        print("Steps:")
        print("  1. Log in to https://www.kaggle.com")
        print("  2. Settings -> Create New API Token (downloads kaggle.json)")
        print(f"  3. Move that file to {cred}")
        print("  4. (Linux/macOS) chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)
    print(f"[ok] kaggle credentials at {cred}")


def download() -> None:
    # Use the Python API directly — kaggle 2.x removed `python -m kaggle` support
    # and putting kaggle.exe on PATH varies by install.
    from kaggle.api.kaggle_api_extended import KaggleApi
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"[run] api.dataset_download_files({KAGGLE_DATASET!r}, path={DATA_DIR}, unzip=True)")
    api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_DIR), unzip=True, quiet=False)


def print_tree(root: Path, max_depth: int = 3) -> None:
    print(f"\n[tree] {root.resolve()}")
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        depth = len(rel.parts)
        if depth > max_depth:
            continue
        if path.is_dir():
            try:
                n = sum(1 for p in path.iterdir() if p.is_file())
            except PermissionError:
                n = -1
            indent = "  " * (depth - 1)
            print(f"{indent}{rel}/  ({n} files)")


def main() -> None:
    check_kaggle_credentials()
    download()
    print_tree(DATA_DIR, max_depth=3)
    print("\nNext: confirm that configs/default.yaml's data.train_subdir / data.test_subdir")
    print("match the actual folder names printed above. Edit the YAML if not.")


if __name__ == "__main__":
    main()
