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

### 5.5 Augmentation 消融實驗

為了釐清「note.md 列出的 4 種 augmentation 各自貢獻多少」，以 **ResNet18 pretrained / 30 epoch** 為 base，逐層加 augmentation。`full` 一列直接重用主表中 [#2 resnet18_pretrained](runs/resnet18_pretrained/eval.json) 的數字，避免重複訓練。

| level         | resize | hflip | rotation(15°) | color_jitter | best_epoch | best_val_acc | test_acc   | Δ test vs none |
|---------------|--------|-------|---------------|--------------|-----------:|-------------:|-----------:|---------------:|
| none          | ✓      |       |               |              | 22         | 0.9944       | 0.9948     | (base)         |
| + hflip       | ✓      | ✓     |               |              | 30         | 0.9944       | 0.9973     | **+0.0025**    |
| + rotation    | ✓      | ✓     | ✓             |              | 27         | **0.9962**   | **0.9975** | **+0.0027**    |
| full          | ✓      | ✓     | ✓             | ✓            | 26         | 0.9959       | 0.9962     | +0.0014        |

完整曲線見 [runs/ablation_aug.png](runs/ablation_aug.png)、原始數字見 [runs/ablation_aug_table.csv](runs/ablation_aug_table.csv)。

**觀察**：

1. **`none` 已經有 99.48% test_acc**：pretrained ResNet18 + 15k 張訓練圖、僅 resize 就能拿到 99.48%。這呼應第 6 點的結論「這個資料集底子簡單」— augmentation 在這裡只是錦上添花，不是必要條件。

2. **hflip 是邊際貢獻最大的一層**：val_acc 沒動（都是 0.9944），但 test_acc +0.25%、test_loss 從 0.0178 降到 0.0092（幾乎砍半）。意思是 hflip 沒拉高 val 上限，但顯著縮小 train/val/test 之間的 generalization gap — 模型更不挑「左右朝向」這種偽特徵。

3. **rotation 帶來的邊際 ≈ 0**：加上 ±15° 旋轉後 val_acc 從 0.9944 → 0.9962（+0.18%）、test_acc 從 0.9973 → 0.9975（+0.02%）。val 上漲、test 幾乎不動，可解讀為 rotation 主要讓 val/train split 之間更公平，對「真實 test 分布」沒額外幫助。

4. **`full` 反而比 `hflip_rot` 略差**（test_acc: 0.9962 vs 0.9975, **−0.13%**）：加上 `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)` 之後居然倒退。最直接的解釋是 **本任務的判別特徵裡有「顏色」這條線索**（`healthy` 葉片飽和度與 `yellow_leaf_curl_virus`、`mosaic_virus` 不同），ColorJitter 把這條線索擾亂後反而抹掉資訊。`hue=0.1` 對葉片色相已經偏激進，是最可疑的一項。

5. **best_epoch 模式**：
   - `none` 在 epoch 22 就 best，之後 train_acc 直奔 1.0 但 val 停滯 — 沒 augmentation 的典型早飽和。
   - `hflip` best 落在 epoch 30（最後一個），val 一路爬升 — augmentation 在用「無限大」的訓練分布延緩收斂。
   - `hflip_rot` 與 `full` 在 epoch 26–27 收斂，介於兩者之間。

**結論**：對這個資料集而言，**最佳 augmentation 配置是 `hflip + rotation`**（test_acc 99.75%），不是 note.md 原列的「full」（99.62%）。`hflip` 是 must-have，`rotation` 是 nice-to-have，`color_jitter` 在這個顏色敏感的任務上是 **負貢獻** — 主流「越多 augmentation 越好」的直覺在此不適用，是這次 ablation 揭露的核心發現。

### 5.6 Input image size 消融實驗

ResNet 用 adaptive avg pool 收尾，本身 size-agnostic（任何輸入大小都可跑），但 input resolution 直接決定「能看到多細的紋路」與「每個 epoch 多貴」。以 **ResNet18 pretrained / 30 epoch / 其他超參完全固定** 為 base，掃 `img_size ∈ {128, 192, 224, 256, 384}`。256px 直接重用主表的 [#2 resnet18_pretrained](runs/resnet18_pretrained/eval.json)。

| img_size | best_epoch | best_val_acc | test_acc   | test_loss | avg epoch time | Δ test vs 256 |
|---------:|-----------:|-------------:|-----------:|----------:|---------------:|--------------:|
| 128      | 23         | 0.9944       | 0.9930     | 0.0245    | 7.1 s          | −0.0032       |
| 192      | 22         | 0.9940       | 0.9953     | 0.0142    | 10.9 s         | −0.0009       |
| 224      | 27         | **0.9970**   | 0.9955     | 0.0175    | 12.9 s         | −0.0007       |
| 256      | 26         | 0.9959       | 0.9962     | 0.0110    | 15.1 s         | (base)        |
| 384      | 26         | **0.9970**   | **0.9986** | **0.0065**| 31.8 s         | **+0.0024**   |

完整曲線見 [runs/ablation_imgsize.png](runs/ablation_imgsize.png)、原始數字見 [runs/ablation_imgsize_table.csv](runs/ablation_imgsize_table.csv)。

**觀察**：

1. **128px 明顯不夠**：test_acc 0.9930 是 5 個 size 裡最差的，比 256 退 0.32%。葉片病害的判別線索（小斑點、葉脈紋理）在 128×128 下被嚴重模糊化，模型必須從更粗的全域特徵硬猜，自然吃虧。

2. **192–256 形成一段平台**：test_acc 0.9953 / 0.9955 / 0.9962，三者差距 < 0.1% 落在雜訊內。224 與 384 並列 best_val_acc 0.9970，但 224 的 test_acc（0.9955）反而比 256（0.9962）略低 — 顯示「val 高」未必「test 高」，這個資料集 val/test 的相關性沒那麼緊。

3. **384px 拿到全 11 個深度 run 的最佳成績**（test_acc **0.9986**），且 test_loss 從 256 的 0.0110 降到 0.0065（**砍 41%**）。意思是 384 不只 acc 更高，連對預測的「信心」都顯著更穩，沒有靠運氣壓線。
   - 相對誤差解讀：256 的 test 錯誤率 0.38%（4440 張裡錯 17 張）→ 384 的 0.14%（錯 6 張），**錯誤率降 63%**。在 99%+ 的尾端，這個邊際很值錢。

4. **Cost ~ size² 成立**：epoch time 從 128 → 384 是 7.1 → 31.8s，比例 4.5×；面積比 9×；不到 9× 是因為 PyTorch / cuDNN 在 GPU 上有 launch overhead，小 size 利用率不滿。**384 比 256 貴 2.1×**，但換來 +0.24% test_acc + test_loss 砍半。

5. **「ImageNet pretrained 是 224 訓的，所以 224 最好」這個直覺在此不成立** — 224 在 val 上的確很強（0.9970），但 384 的 test 表現遠超。原因：本任務有大量「細粒度紋路」差異（splash 型 vs spot 型 vs leaf-curl 型），ImageNet pretrained 的 backbone 在 224 學的低層 conv kernel 雖然是 224-tuned，但 ResNet 的全卷積特性讓它在 384 直接拿到 2.25× 的「有效感受野細節密度」。

**結論**：
- **若以 test_acc 為唯一目標**：用 **384px**，可在 ResNet18 上拉到 **99.86%**，全 11 個深度 run 的最佳。
- **最佳 cost / accuracy**：仍是 **256px**（main 表中的 baseline），訓 15s/epoch、99.62%、堪用。
- **128px 是明確的「太小」**：在這個任務上不要往下走。

這個 ablation 與 §5.5（augmentation）配合看出一個共同主題：**「加大什麼」要看任務本身的判別線索在哪**。本資料集判別線索是 **顏色 + 細紋路**，所以：
- 提高解析度（384）→ 直接強化「細紋路」這條軸 → +0.24%。
- 加入 color_jitter → 削弱「顏色」這條軸 → −0.13%。

### 5.7 Grad-CAM 視覺化

對 6 個 deep learning run 都產出了 Grad-CAM，每個 run 一張 2×5 的 grid（10 類各挑一張被「正確分類」的 test 圖）。完整圖檔在 `runs/<exp>/gradcam.png`。

**實作細節**：
- ResNet：hook `model.layer4`（最後一個 conv stage 的輸出），標準 Grad-CAM。
- ViT-base/16：hook `model.blocks[-1].norm1`（最後 transformer block 的 LayerNorm 輸入），把 (B, N+1, C) 的 token 序列丟掉 CLS 後 reshape 成 (B, C, 16, 16) 再走同樣的 Grad-CAM 數學。
- 顏色：jet colormap 疊在原圖（55% 原圖 + 45% heatmap）。

**觀察**：
- **ResNet pretrained**（[runs/resnet18_pretrained/gradcam.png](runs/resnet18_pretrained/gradcam.png)、[runs/resnet50_pretrained/gradcam.png](runs/resnet50_pretrained/gradcam.png)）：注意力是**集中、平滑的橢圓型熱區**，幾乎都落在葉片本體中央，幾乎不看背景。pretrained 後對 leaf 的 saliency 很乾淨，符合「object-centric」直覺。
- **ResNet scratch**（[runs/resnet18_scratch/gradcam.png](runs/resnet18_scratch/gradcam.png)、[runs/resnet50_scratch/gradcam.png](runs/resnet50_scratch/gradcam.png)）：熱區較大、邊界較糊，偶爾會延伸到背景或包含整張圖的角落 — 顯示 scratch 模型雖然分類正確，但學到的 representation 比較「不集中」。
- **ViT pretrained**（[runs/vit_pretrained/gradcam.png](runs/vit_pretrained/gradcam.png)）：與 ResNet 完全不同的型態 — **稀疏、點狀的熱點**（一塊塊小斑點），常落在「斑點型病害」的具體斑點位置（如 `septoria_leaf_spot`、`twospotted_spider_mite`）。這呼應 ViT 的 self-attention 本質：每個 patch token 互相關注，最終可以選出少數關鍵 patches。
- **ViT scratch**（[runs/vit_scratch/gradcam.png](runs/vit_scratch/gradcam.png)）：熱點位置較雜，部分圖完全找不到清楚的關注區，與 ViT scratch test acc 較低（97.91%）一致。

> 解讀：Grad-CAM 不只是「看模型在看哪」的 sanity check，也讓我們看到**架構差異會反映在注意力的形狀**上：CNN 是「看一塊區域」，ViT 是「選幾個 patch」。兩者最後 acc 接近，但 representation 風格完全不同。

## 6. 結論

1. **這個資料集難度不高**：6 個 deep model 全部 ≥ 97.8% test acc。任何 ResNet 或 ViT 配上 ImageNet pretrained，30 epoch 就能交出 99.6%+ 的可用模型。
2. **「越深越大」在固定預算下不一定贏**：30 epoch 內 — pretrained 排名 ViT≈R50≈R18 三者差距 < 0.1%；scratch 排名 R18 > ViT ≈ R50，較小的 ResNet18 反而表現最好，因為大模型在 30 epoch 內未收斂。
3. **預訓練紅利明顯**：對 R18 +1.04%、R50 / ViT 都 +1.80%。預訓練不只贏 acc，更贏「收斂速度」：pretrained 22–29 epoch 即 best；scratch 26–28 epoch 仍在上升。同樣 30 epoch 預算下 pretrained 必勝。
4. **CNN vs Transformer 在 30 epoch 下幾乎打平**：pretrained 差 0.07%、scratch 差 0.07%（且方向是 ViT 微勝）— 與「Transformer 在小資料下需要更多 epoch」的常見論述不完全吻合，本實驗在 30 epoch 內 ViT 的收斂效率與 ResNet50 相當。
5. **PCA + DT 是合適的「下界對照」**：43% test acc 證明傳統 ML pipeline 確實能分一部分病害，但與 DL 差距 −56%，讓 deep learning 的價值具體可見。
6. **Augmentation 不是越多越好**（§5.5 ablation 結論）：對這個顏色敏感的任務，`hflip + rotation` 的 99.75% 反而打贏「全 augmentation」的 99.62%。`color_jitter`（特別是 hue 擾動）在這個資料集上是負貢獻 — 違反直覺但合理：判別線索本身就含顏色。
7. **Input resolution 仍有可拉空間**（§5.6 ablation 結論）：把 ResNet18 pretrained 的 input 從 256 提到 **384px**，test_acc 拉到 **99.86%**、test_loss 砍 41%，是全 11 個深度 run 的最佳成績。代價是 epoch 時間 2.1×。「ImageNet 是 224 訓的所以用 224 最好」在這個細紋路任務上不成立。
8. **推薦選擇**：
   - **絕對最佳（含 ablation）**：ResNet18 pretrained @ 384px（99.86%、~16 min 訓完）
   - **原主表最佳**：ViT-base pretrained @ 256px（99.71%）
   - **最佳 cost / accuracy**：ResNet18 pretrained @ 256px（11M 參數、~7.5 min 訓完、99.62%）
   - **若只在乎 ResNet18 @ 256 上限**：ResNet18 pretrained + `hflip+rotation`（無 color_jitter）可拉到 99.75%

---

## 附錄：視覺化檔案

- 各 run 的 train/val 曲線：`runs/<exp>/curves.png`
- 6 deep runs 的 val_acc 疊圖比較：[runs/comparison.png](runs/comparison.png)
- 7 runs 的 test_acc 長條圖：[runs/test_acc_bar.png](runs/test_acc_bar.png)
- PCA+DT grid heatmap：[runs/pca_dt/grid.png](runs/pca_dt/grid.png)
- PCA+DT depth ablation：[runs/pca_dt/depth_curve.png](runs/pca_dt/depth_curve.png)
- Augmentation 消融實驗：[runs/ablation_aug.png](runs/ablation_aug.png)、[runs/ablation_aug_table.csv](runs/ablation_aug_table.csv)
- Input image size 消融實驗：[runs/ablation_imgsize.png](runs/ablation_imgsize.png)、[runs/ablation_imgsize_table.csv](runs/ablation_imgsize_table.csv)
- 6 deep runs 的 Grad-CAM 視覺化：`runs/<exp>/gradcam.png`
