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
### ROC_curve分析(report裡沒寫到來不及補，這裡補上)
AUC 0.98：極高的分類性能，模型區分「真／假」幾乎沒問題。

ROC 形狀：在 0–0.1 的 FPR 區間就能達到 ~0.9 的 TPR，代表只允許非常少的誤報，就能抓到大部分偽造。

可用來選擇最佳閾值：例如你想把 FPR 控制在 5% 以下，就可以看橫軸 0.05 對應的 TPR，大約能回收 90% 以上的「假」影格。


