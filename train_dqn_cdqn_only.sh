#!/bin/bash

echo "=========================================================="
echo "    HUẤN LUYỆN LẠI TỪ ĐẦU 2 MÔ HÌNH DQN & CDQN (120 EPOCHS)"
echo "=========================================================="

mkdir -p logs

echo "-> Đang phóng 2 Models (Chạy ngầm song song)..."
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/dqn_train.py --epochs 120 > logs/base_dqn.log 2>&1 &
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/cdqn_train.py --epochs 120 > logs/base_cdqn.log 2>&1 &

echo "[!] DQN và CDQN đang được huấn luyện ngầm từ đầu (Epoch 0 -> 120)."
echo "[!] Dùng lệnh 'tail -f logs/base_dqn.log' hoặc 'tail -f logs/base_cdqn.log' để xem tiến độ."
