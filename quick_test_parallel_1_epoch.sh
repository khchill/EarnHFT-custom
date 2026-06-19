#!/bin/bash

echo "=========================================================="
echo "  CHẠY THỬ NGHIỆM (DRY-RUN) SONG SONG 10 MODELS - 1 EPOCH"
echo "=========================================================="
echo "Mục đích: Phóng toàn bộ 10 con Bot cùng 1 lúc để test lỗi siêu tốc."

mkdir -p logs
echo "[!] Tất cả log sẽ được lưu riêng lẻ vào thư mục 'logs/' (Màn hình sẽ gọn gàng hơn)."

# === 1. CHẠY TOÀN BỘ 10 MÔ HÌNH ĐỘC LẬP CÙNG LÚC ===
echo "-> [1/3] Đang train SONG SONG 10 Models (6 Low-level + 4 Baselines)..."

# Phóng 4 con Baselines (Chạy độc lập ngầm)
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/cdqn_train.py --epochs 1 > logs/base_cdqn.log 2>&1 &
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/dqn_train.py --epochs 1 > logs/base_dqn.log 2>&1 &
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/ppo_train.py --epochs 1 > logs/base_ppo.log 2>&1 &
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/dra_train.py --epochs 1 > logs/base_dra.log 2>&1 &

# Phóng 6 con Low-level (LƯU PID ĐỂ CHỜ)
.venv/bin/python src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py --beta 100.0 --epochs 1 > logs/low_100.log 2>&1 &
PID1=$!
.venv/bin/python src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py --beta 30.0 --epochs 1 > logs/low_30.log 2>&1 &
PID2=$!
.venv/bin/python src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py --beta -10.0 --epochs 1 > logs/low_minus10.log 2>&1 &
PID3=$!
.venv/bin/python src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py --beta -90.0 --epochs 1 > logs/low_minus90.log 2>&1 &
PID4=$!
.venv/bin/python src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_only_teacher.py --beta 30.0 --epochs 1 > logs/low_only_teacher.log 2>&1 &
PID5=$!
.venv/bin/python src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_only_td.py --beta 30.0 --epochs 1 > logs/low_only_td.log 2>&1 &
PID6=$!

# CHỈ CHỜ ĐÚNG 6 CON LOW-LEVEL
wait $PID1 $PID2 $PID3 $PID4 $PID5 $PID6
echo "[+] Hoàn tất train 6 Low-level Agents! Lập tức chạy Validation..."

# === 2. LỌC & CHỌN BOT NGON NHẤT ===
echo "-> [2/3] Đang test luồng Pick Agent..."
.venv/bin/python src/EarnHFT_framwork/tool/run_validation.py > logs/validation.log 2>&1
.venv/bin/python src/EarnHFT_framwork/analysis/pick_agent/pick_agent_position.py > logs/pick_agent.log 2>&1
.venv/bin/python src/EarnHFT_framwork/analysis/pick_agent/inspect_matrix.py > logs/inspect_matrix.log 2>&1

# === 3. CHẠY HIGH-LEVEL ROUTER ===
echo "-> [3/3] Đang test thử High-Level Router..."
.venv/bin/python src/EarnHFT_framwork/RL/agent/high_level/dqn_position.py --epochs 1 > logs/high_level.log 2>&1

echo "=========================================================="
echo "    TEST SONG SONG THÀNH CÔNG! KHÔNG CÓ LỖI."
echo "=========================================================="
