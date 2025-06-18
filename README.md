## 資料下載與解壓
1. 下載 FaceForensics++ C40 子集影格包：`wget http://.../dataset.zip`
2. 解壓：`unzip dataset.zip -d ./dataset`

## 訓練
```bash
bash run.sh
# 或者
python train.py   # LoRA + baseline
python evaluate.py   # 會輸出 experiment_log.json
python explain.py    # 顯示誤判的 原圖和Grad-CAM 圖

python explain.py    # 做完所有事情

## 預期執行時間
- LoRA 訓練 (5 epochs)：大約 3 分鐘 @ RTX3090 TI  
- 基線訓練      ：大約 3 分鐘 @ 同卡  
- 評估 & 可視化 ：< 3 分鐘
```

