#!/bin/bash

echo "=========================================================="
echo "    CHẠY TIẾP (RESUME) CÁC MODEL BỊ BỎ LỠ"
echo "=========================================================="
echo "Mục đích: Chạy tiếp Baselines (từ 50 lên 200) và High-level (từ 50 lên 100)"

mkdir -p logs

echo "-> [1/2] Đang Resume 4 Models Baselines (từ Epoch 50 lên 200)..."
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/ppo_train.py --resume_epoch 50 --epochs 200 > logs/resume_base_ppo.log 2>&1 &
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/dra_train.py --resume_epoch 50 --epochs 200 > logs/resume_base_dra.log 2>&1 &
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/dqn_train.py --resume_epoch 50 --epochs 200 > logs/resume_base_dqn.log 2>&1 &
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/cdqn_train.py --resume_epoch 50 --epochs 200 > logs/resume_base_cdqn.log 2>&1 &

PID1=$!
PID2=$!
PID3=$!
PID4=$!

echo "-> [2/2] Đang Resume High-Level Router (từ Epoch 50 lên 100)..."
# Chạy ngầm High-Level luôn vì nó không phụ thuộc Baselines
.venv/bin/python src/EarnHFT_framwork/RL/agent/high_level/dqn_position.py --resume_epoch 50 --epochs 100 > logs/resume_high_level.log 2>&1 &

echo "[!] Các model đang được Resume chạy ngầm."
echo "[!] Anh có thể dùng lệnh 'tail -f logs/resume_base_ppo.log' để theo dõi."
