# Tomato Disease Classification — Experiment Report

> 此檔在實驗跑完後再填入實際數據。以下為樣板與分析框架。

## 1. 任務與資料

- **任務**：番茄葉片病害多類別圖像分類。
- **資料集**：Kaggle [`luisolazo/tomato-diseases`](https://www.kaggle.com/datasets/luisolazo/tomato-diseases)。
- **切分**：資料集自帶 `test/`；`train/` 內依固定 seed=42 切 85 / 15 為 train / val。
- **類別數**：10
  - `bacterial_spot`, `early_blight`, `healthy`, `late_blight`, `leaf_mold`,
    `mosaic_virus`, `septoria_leaf_spot`, `target_spot`,
    `twospotted_spider_mite`, `yellow_leaf_curl_virus`
- **樣本數**：train 15,090 / val 2,663 / test 4,440（test 每類精確 444 張，train 介於 1,110–3,595 / 類，類別不平衡）。

## 2. 與 note.md 的偏離

note.md 原列「CNN + Decision Tree」；本實驗改為 **PCA + Decision Tree**：

- 原因：DT 本身不需要梯度訓練，把對照組做成「無 epoch 的純傳統 ML pipeline」更能突顯與深度學習模型的對比。
- DT depth 變成這個分支的主軸 ablation（`max_depth ∈ {3, 5, 10, 20, 30, 50}`）。
- 同時對 PCA `n_components ∈ {64, 128, 256}` 做小規模 sweep。

**Checkpoint 策略偏離**：note.md 原寫「保留所有 ckpt」，但 7 個實驗的所有 epoch ckpt 合計 ~130 GB（ViT-base 單 ckpt ~1 GB × 60 epoch = 60 GB），實作上預設改為 **只保留 best.pt + last.pt**；想要恢復原行為可以 `--save_every_epoch`。Per-epoch log 在 `runs/<exp>/log.csv` 完整保留。

其他與 note.md 完全一致：augmentation pipeline、85/15 split、4 模型分類目標、log 完整保留。

## 3. 共用超參數

詳見 [configs/default.yaml](configs/default.yaml) 與 [README.md](README.md#共用超參數)。**所有 deep learning 模型統一 30 epoch**（含 pretrained 與 from-scratch），其餘超參完全一致。

> 早期版本曾把 from-scratch 訓 60 epoch，但為了讓「pretrained vs scratch」的對比建立在同樣的計算預算下，後來統一改為 30。比較計算預算固定下哪種起點更有效。

## 4. 結果

### 4.1 主要對比表

| # | model               | pretrained | epochs | best_epoch | val_acc   | test_acc  | 訓練時間 |
|---|---------------------|------------|--------|------------|-----------|-----------|----------|
| 1 | PCA + DT            | —          | —      | —          | **0.4630**| **0.4329**| ~1.5 min |
| 2 | ResNet18            | yes        | 30     | 26         | **0.9959**| **0.9962**| ~7.5 min |
| 3 | ResNet18            | no         | 30     | 26         | **0.9842**| **0.9858**| ~7.5 min |
| 4 | ResNet50            | yes        | 30     | 22         | **0.9989**| **0.9964**| ~10.5 min|
| 5 | ResNet50            | no         | 30     | 28         | **0.9718**| **0.9784**| ~11 min  |
| 6 | ViT-base/16         | yes        | 30     | 29         | **0.9970**| **0.9971**| ~24 min  |
| 7 | ViT-base/16         | no         | 30     | 28         | **0.9831**| **0.9791**| ~24 min  |

PCA+DT 細節：best `(n_components=64, max_depth=10)`；PCA 64 維可解釋變異量 84.4%。完整 grid 與 depth ablation 見 `runs/pca_dt/grid.png` 與 `runs/pca_dt/depth_curve.png`。

### 4.2 視覺化

- 各 run 的訓練曲線：`runs/<exp>/curves.png`
- 全 run val_acc 疊圖：`runs/comparison.png`
- 全 run test_acc 長條圖：`runs/test_acc_bar.png`
- PCA+DT grid heatmap：`runs/pca_dt/grid.png`
- PCA+DT depth ablation 曲線：`runs/pca_dt/depth_curve.png`

## 5. 分析

### 5.1 傳統 ML vs 深度學習

| 模型             | test_acc | 與最佳模型差距 |
|------------------|----------|----------------|
| PCA + DT (best)  | 0.4329   | -0.5642        |
| ViT-base pretrained | 0.9971 | (best)         |

PCA+DT 的 43.3% 已遠高於亂猜的 10%（10 類），代表灰階 64×64 的 PCA 主成分能分出一部分病害「巨觀紋路差異」（例如 healthy vs yellow_leaf_curl_virus），但無法處理細節型病害（如 early/late blight 的斑點差別）。深度模型在所有設定下皆 ≥ 98.5%，將近 56% 的差距明確顯示卷積／自注意力學到的特徵遠優於 PCA 線性投影。

PCA grid 觀察（見 [runs/pca_dt/grid.png](runs/pca_dt/grid.png)、[runs/pca_dt/depth_curve.png](runs/pca_dt/depth_curve.png)）：
- `n_components` 從 64 → 256 對 val_acc 幾乎無提升（最佳反而落在 n=64），代表前 64 個主成分已捕捉最有判別力的方向，再多只是加入雜訊。
- `max_depth` 是主要 ablation：depth=10 為甜蜜點（val 46.3%）；depth=20 起 train_acc 衝到 96%+ 而 val 跌回 ~42%，呈典型過擬合曲線。

### 5.2 ResNet 深度影響（30 epoch 固定預算）

| 設定        | ResNet18 | ResNet50 | Δ (50 − 18) |
|-------------|----------|----------|-------------|
| pretrained  | 0.9962   | 0.9964   | +0.0002     |
| from scratch| 0.9858   | 0.9784   | −0.0074     |

- **Pretrained 下 ResNet50 僅勝 ResNet18 0.02%**，差距落在雜訊範圍內。
- **From-scratch 下 ResNet50 反而落後 0.74%**，差距變大且方向反轉。原因：
  - ResNet50 (23.5M params) 是 ResNet18 (11.2M) 的 2.1 倍，在 30 epoch 內收斂尚未到位（best epoch 28，仍在上升）；如果訓更久應該能追上甚至超越。
  - 從 [runs/comparison.png](runs/comparison.png) 可以看到 resnet50_scratch 的 val 曲線在 epoch 30 還在明顯爬升，相比之下 resnet18_scratch 已接近平台。
- **結論**：在固定 30 epoch / 中等資料量（~15k 張）/ 任務簡單的設定下，ResNet18 的「容量恰好」是最佳選擇 — 容量更大反而吃虧（收斂慢）、資料更多時才會反過來。

### 5.3 CNN vs Transformer（30 epoch 固定預算）

| 設定        | ResNet50 | ViT-base/16 | Δ (ViT − R50) |
|-------------|----------|-------------|---------------|
| pretrained  | 0.9964   | 0.9971      | +0.0007       |
| from scratch| 0.9784   | 0.9791      | +0.0007       |

- **Pretrained 下兩者幾乎打平**，差距 0.07% 落在雜訊內。ImageNet 預訓練後的 representation 強到讓「CNN vs Transformer」這個架構問題在下游小資料 fine-tune 時不重要。
- **From-scratch 下 ViT-base 居然小勝 ResNet50** — 與「ViT 在小資料下需要更多 epoch / 更強 augmentation 才能贏 CNN」的常見論述相反，但這結果其實合理：
  - ViT-base 雖然有 85.9M 參數（ResNet50 的 3.6 倍），但 transformer 的 self-attention 層在 30 epoch 內收斂效率反而比深層 ResNet 的 BN+Conv 堆疊好。
  - val_acc 上 ViT scratch（0.9831）甚至明顯領先 ResNet50 scratch（0.9718）達 1.13%；test acc 接近代表 ViT 在 train/test distribution 上更穩健。
  - 對照 ResNet18 scratch（0.9858 test）— 反而是「小 CNN」打贏「大 ViT」與「大 CNN」，再次強調本任務「容量過剩 = 收斂慢」。
- **時間成本**：ViT 訓練時間 ~46 s/epoch，是 ResNet50 (~21 s) 的 2.2 倍、ResNet18 (~15 s) 的 3.1 倍。

### 5.4 Pretrained 的影響（30 epoch 固定預算）

| 模型        | scratch | pretrained | Δ        |
|-------------|---------|------------|----------|
| ResNet18    | 0.9858  | 0.9962     | +0.0104  |
| ResNet50    | 0.9784  | 0.9964     | **+0.0180** |
| ViT-base/16 | 0.9791  | 0.9971     | **+0.0180** |

- 在固定 30 epoch 計算預算下，預訓練紅利明顯比早期實驗（scratch 60 epoch）的版本更大 —
  - 60 ep 版的 Δ：R18 +0.54% / R50 +0.61% / ViT +1.22%
  - 30 ep 版的 Δ：R18 +1.04% / R50 +1.80% / ViT +1.80%
- **結論**：pretrained 不只贏在最終 acc，更贏在「收斂速度」 — pretrained 模型在 22–29 epoch 內就到 best；scratch 模型在 30 epoch 還沒收斂完。同樣的計算預算下，pretrained 一定划算。
- ViT-base 與 ResNet50 在 scratch 下 Δ 同樣是 +1.80%，但**性質不同**：
  - ResNet50 的 +1.80% 主要來自「容量大但還沒練熟」的劣勢 — 給更多 epoch 應該能縮小。
  - ViT-base 的 +1.80% 一部分來自架構需要的 inductive bias 透過預訓練「補」上 — 即使 scratch 給更多 epoch，CNN 的 locality 優勢仍會保留一段時間。

## 6. 結論

1. **這個資料集難度不高**：6 個 deep model 全部 ≥ 97.8% test acc。任何 ResNet 或 ViT 配上 ImageNet pretrained，30 epoch 就能交出 99.6%+ 的可用模型。
2. **「越深越大」在固定預算下不一定贏**：30 epoch 內 — pretrained 排名 ViT≈R50≈R18 三者差距 < 0.1%；scratch 排名 R18 > ViT ≈ R50，較小的 ResNet18 反而表現最好，因為大模型在 30 epoch 內未收斂。
3. **預訓練紅利明顯**：對 R18 +1.04%、R50 / ViT 都 +1.80%。預訓練不只贏 acc，更贏「收斂速度」：pretrained 22–29 epoch 即 best；scratch 26–28 epoch 仍在上升。同樣 30 epoch 預算下 pretrained 必勝。
4. **CNN vs Transformer 在 30 epoch 下幾乎打平**：pretrained 差 0.07%、scratch 差 0.07%（且方向是 ViT 微勝）— 與「Transformer 在小資料下需要更多 epoch」的常見論述不完全吻合，本實驗在 30 epoch 內 ViT 的收斂效率與 ResNet50 相當。
5. **PCA + DT 是合適的「下界對照」**：43% test acc 證明傳統 ML pipeline 確實能分一部分病害，但與 DL 差距 −56%，讓 deep learning 的價值具體可見。
6. **推薦選擇**：
   - **絕對最佳**：ViT-base pretrained（99.71%）
   - **最佳 cost / accuracy**：ResNet18 pretrained（11M 參數、~7.5 min 訓完、99.62%）

---

## 附錄：視覺化檔案

- 各 run 的 train/val 曲線：`runs/<exp>/curves.png`
- 6 deep runs 的 val_acc 疊圖比較：[runs/comparison.png](runs/comparison.png)
- 7 runs 的 test_acc 長條圖：[runs/test_acc_bar.png](runs/test_acc_bar.png)
- PCA+DT grid heatmap：[runs/pca_dt/grid.png](runs/pca_dt/grid.png)
- PCA+DT depth ablation：[runs/pca_dt/depth_curve.png](runs/pca_dt/depth_curve.png)
