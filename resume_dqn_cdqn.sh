#!/bin/bash

echo "=========================================================="
echo "    CHẠY TIẾP (RESUME) DQN VÀ CDQN TỪ EPOCH 50 -> 100"
echo "=========================================================="

mkdir -p logs

echo "-> Đang phóng 2 Models (Chạy ngầm song song)..."
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/dqn_train.py --resume_epoch 50 --epochs 100 >> logs/base_dqn.log 2>&1 &
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/cdqn_train.py --resume_epoch 50 --epochs 100 >> logs/base_cdqn.log 2>&1 &

echo "[!] DQN và CDQN đang được chạy tiếp từ Epoch 50 lên 100."
echo "[!] Dùng lệnh 'tail -f logs/base_dqn.log' hoặc 'tail -f logs/base_cdqn.log' để xem tiến độ."
