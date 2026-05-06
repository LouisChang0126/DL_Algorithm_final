## dataset:
https://www.kaggle.com/datasets/luisolazo/tomato-diseases

## train augmentation:
transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(15),      
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
    transforms.ToTensor(), 
])

## train-val spilt:
85%:15%

## 模型選用:
1. CNN+decision tree (讓大家知道決策樹也可以分圖片)
2. ResNet18
3. ResNet50 (比較resnet層數疊高的差異)
4. ViT-base (比較「CNN vs Transformer」)

## 注意事項
* 要把訓練的log與所有ckpt保留
* 超參數要盡量一樣，參考kaggle其他人的設定

## Deliverables
* README.md
* 所有模型的程式碼
* 所有模型的訓練曲線圖
* best val epoch模型的val acc跟test acc
* report.md