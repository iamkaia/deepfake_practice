#!/usr/bin/env bash
# 1. 安裝依賴
pip install -r requirements.txt
# 2. 下載並解壓 dataset（如果需要）
#   wget ... && unzip ...
# 3. 執行完整流程
python main.py
