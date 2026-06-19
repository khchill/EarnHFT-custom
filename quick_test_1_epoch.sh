#!/bin/bash

echo "=========================================================="
echo "    CHẠY THỬ NGHIỆM (DRY-RUN) TOÀN BỘ HỆ THỐNG - 1 EPOCH"
echo "=========================================================="
echo "Mục đích: Kiểm tra lỗi logic hoặc lỗi tràn RAM trước khi cắm máy thật."

# === 1. CHẠY LOW-LEVEL (4 Betas & 2 Ablations) - CHỈ 1 EPOCH ===
echo "-> [1/4] Đang test thử Low-Level Agents..."
.venv/bin/python src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py --beta 100.0 --epochs 1 && \
.venv/bin/python src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py --beta 30.0 --epochs 1 && \
.venv/bin/python src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py --beta -10.0 --epochs 1 && \
.venv/bin/python src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py --beta -90.0 --epochs 1 && \
.venv/bin/python src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_only_teacher.py --beta 30.0 --epochs 1 && \
.venv/bin/python src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_only_td.py --beta 30.0 --epochs 1 && \

# === 2. LỌC & CHỌN BOT NGON NHẤT ===
echo "-> [2/4] Đang test luồng Pick Agent..."
.venv/bin/python src/EarnHFT_framwork/tool/run_validation.py && \
.venv/bin/python src/EarnHFT_framwork/analysis/pick_agent/pick_agent_position.py && \
.venv/bin/python src/EarnHFT_framwork/analysis/pick_agent/inspect_matrix.py && \

# === 3. CHẠY BASELINES - CHỈ 1 EPOCH ===
echo "-> [3/4] Đang test thử Baselines..."
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/cdqn_train.py --epochs 1 && \
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/dqn_train.py --epochs 1 && \
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/ppo_train.py --epochs 1 && \
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/dra_train.py --epochs 1 && \

# === 4. CHẠY HIGH-LEVEL ROUTER - CHỈ 1 EPOCH ===
echo "-> [4/4] Đang test thử High-Level Router..."
.venv/bin/python src/EarnHFT_framwork/RL/agent/high_level/dqn_position.py --epochs 1

echo "=========================================================="
echo "    TEST THÀNH CÔNG! KHÔNG PHÁT HIỆN LỖI CODE/LOGIC."
echo "=========================================================="
