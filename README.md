# Tomato Disease Classification — 4 Models 比較實驗

期末實驗：在 Kaggle [`luisolazo/tomato-diseases`](https://www.kaggle.com/datasets/luisolazo/tomato-diseases) 番茄病害圖像資料集上，公平比較 4 種模型（PCA+DT、ResNet18、ResNet50、ViT-base）的分類表現。詳細設計請見 `note.md` 與 `report.md`。

## 環境（一次性）

RTX 5070 Ti 是 Blackwell（sm_120），需要 PyTorch nightly cu130。

```powershell
# 1. 建 conda env
conda create -n ml_algorithm python=3.11 -y
conda activate ml_algorithm

# 2. 裝 PyTorch nightly cu130（不要放進 requirements.txt，否則會被解到穩定版而沒 CUDA kernel）
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130

# 3. 裝其他套件
pip install -r requirements.txt

# 4. 驗證 GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 預期：True NVIDIA GeForce RTX 5070 Ti
```

## 資料下載

需要 Kaggle API token。到 https://www.kaggle.com/settings → Create New API Token，把 `kaggle.json` 放到：

- Windows：`%USERPROFILE%\.kaggle\kaggle.json`
- Linux/macOS：`~/.kaggle/kaggle.json` (`chmod 600`)

```powershell
python scripts/download_data.py
```

腳本會印出解壓後的目錄結構。如果 `train` / `valid` 名稱與實際不符，編輯 [configs/default.yaml](configs/default.yaml) 的 `data.train_subdir` / `data.test_subdir`。

## 執行單一實驗

```powershell
# Deep learning 模型 (ResNet / ViT) - 統一 30 epoch
python -m src.train --model resnet18 --pretrained --name resnet18_pretrained --epochs 30
python -m src.train --model resnet50              --name resnet50_scratch    --epochs 30
python -m src.train --model vit_base --pretrained --name vit_pretrained      --epochs 30

# PCA + DecisionTree（無 epoch、自動掃 grid）
python -m src.train_pca_dt --name pca_dt
# smoke test 模式：只跑 1 個 grid 格
python -m src.train_pca_dt --name pca_dt_smoke --quick
```

每個實驗會在 `runs/<name>/` 產出：
- `config.yaml`：凍結的有效超參
- `log.csv`：每 epoch 的 train/val loss 與 acc（PCA+DT 改為 `grid.csv`）
- `best.pt` / `last.pt`：best val acc / 最後一個 epoch 的 ckpt
- `eval.json`：best epoch 的 val_acc 與 test_acc
- `curves.png`：訓練曲線（用 plot_curves.py 產）

> Note: note.md 原寫「保留所有 ckpt」。實作預設改為 best+last 兩個 ckpt，因為 7 個實驗的 per-epoch ckpt 合計 ~130 GB（ViT-base 單 ckpt ~1 GB）。要恢復原行為加 `--save_every_epoch`。

## 跑全部 7 個實驗

```powershell
python scripts/run_all.py
python scripts/plot_curves.py --all --comparison
python scripts/gradcam_viz.py            # 6 個 deep run 各產一張 gradcam.png
```

實測時間（RTX 5070 Ti, 統一 30 epoch）：
- pca_dt：~1.5 分鐘
- ResNet18 pretrained / scratch：~7.5 分鐘 each
- ResNet50 pretrained / scratch：~10.5 / ~11 分鐘
- ViT-base pretrained / scratch：~24 分鐘 each
- **合計：約 1.5 小時**

## 目錄結構

```
final/
├── README.md                    # 你正在看
├── report.md                    # 結果 + 分析（跑完實驗後填寫）
├── note.md                      # 原始實驗規劃
├── requirements.txt
├── configs/default.yaml         # 共用超參
├── src/
│   ├── data.py                  # ImageFolder + 85/15 split + augmentation
│   ├── utils.py                 # CSVLogger / set_seed / save_ckpt
│   ├── train.py                 # 通用 train loop（ResNet/ViT 共用）
│   ├── eval.py                  # 獨立 test eval
│   ├── train_pca_dt.py          # PCA + DT pipeline
│   ├── gradcam.py               # ResNet & ViT 通用 Grad-CAM
│   └── models/{resnet,vit}.py
├── scripts/
│   ├── download_data.py         # Kaggle API
│   ├── run_all.py               # 跑全部 7 實驗
│   ├── plot_curves.py           # 產 curves.png + comparison.png
│   └── gradcam_viz.py           # 為 6 個 deep run 產 gradcam.png
├── data/                        # ← Kaggle 下載到這
└── runs/<exp>/...               # ← 訓練輸出
```

## 共用超參數

| 項目          | 值              |
|---------------|-----------------|
| input size    | 256×256         |
| batch_size    | 32              |
| optimizer     | AdamW           |
| lr            | 1e-4            |
| weight_decay  | 1e-4            |
| scheduler     | cosine + 2 ep warmup |
| amp           | true (FP16)     |
| epochs        | 30 (所有 deep model 統一) |
| augment (train) | resize + hflip + rot15 + color_jitter |
| augment (val/test) | resize only |
| val_split     | 0.15 (from train) |
| seed          | 42              |
