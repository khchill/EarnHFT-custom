@echo off
chcp 65001 >nul
echo === BẮT ĐẦU CHẠY PIPELINE DỮ LIỆU TARDIS (WINDOWS) ===

set PYTHON_CMD=".venv\Scripts\python.exe"

echo [*] 1. Dang tai va tien xu ly du lieu tu Tardis (Gop file, tinh Features, Split Data)...
%PYTHON_CMD% src\data_cleaning\tardis_pipeline.py

echo [*] 2. Dang phan cum thi truong bang DTW (Slicing Model tren tap Valid)...
%PYTHON_CMD% src\EarnHFT_framwork\tool\slice_model.py --data_path data/cleaned_data/BTCUSDT/tardis/valid

echo === HOAN TAT PIPELINE DU LIEU. BAN CO THE BAT DAU TRAIN MODEL ===
pause
