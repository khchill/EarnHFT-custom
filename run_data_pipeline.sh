#!/bin/bash
echo "=== BẮT ĐẦU CHẠY PIPELINE DỮ LIỆU TARDIS ==="
PYTHON_CMD=".venv/bin/python"

echo "[*] 1. Đang tải và tiền xử lý dữ liệu từ Tardis (Gộp file, tính Features, Split Data)..."
$PYTHON_CMD src/data_cleaning/tardis_pipeline.py

echo "[*] 2. Đang phân cụm thị trường bằng DTW (Slicing Model trên tập Valid)..."
$PYTHON_CMD src/EarnHFT_framwork/tool/slice_model.py --data_path data/cleaned_data/BTCUSDT/tardis/valid

echo "=== HOÀN TẤT PIPELINE DỮ LIỆU. BẠN CÓ THỂ BẮT ĐẦU TRAIN MODEL ==="
