import torch
from pathlib import Path

# ----------------- 全域參數設定 -----------------
DATA_DIR       = "./dataset"                       # 資料集資料夾
OUTPUT_DIR     = Path("./output")                  # 輸出資料夾
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"         # 檢查點資料夾

BATCH_SIZE = 32
EPOCHS     = 5
LR         = 5e-4
LORA_RANK  = 8
LORA_ALPHA = 16

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5
