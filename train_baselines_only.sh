#!/bin/bash

echo "=========================================================="
echo "    HUẤN LUYỆN LẠI TỪ ĐẦU 4 MODELS BASELINES (120 EPOCHS)"
echo "=========================================================="
echo "Mục đích: Chạy mới 100% từ Epoch 0 cho 4 con Baselines."

mkdir -p logs

echo "-> Đang phóng 4 Models Baselines (Chạy ngầm song song)..."
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/ppo_train.py --epochs 120 > logs/base_ppo.log 2>&1 &
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/dra_train.py --epochs 120 > logs/base_dra.log 2>&1 &
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/dqn_train.py --epochs 120 > logs/base_dqn.log 2>&1 &
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/cdqn_train.py --epochs 120 > logs/base_cdqn.log 2>&1 &

echo "[!] 4 model Baselines đang được huấn luyện ngầm từ đầu (Epoch 0 -> 120)."
echo "[!] Anh có thể dùng lệnh 'tail -f logs/base_ppo.log' để theo dõi tiến độ."
