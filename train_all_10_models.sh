#!/bin/bash

echo "=========================================================="
echo "      BẮT ĐẦU HUẤN LUYỆN TOÀN BỘ 10 MODELS EARNHFT"
echo "=========================================================="
echo "Cấu hình: Tận dụng GPU RTX 3090 Ti. Chạy nối tiếp để tránh OOM."

# === 1. CHẠY LOW-LEVEL (4 Betas & 2 Ablations) ===
echo "-> [1/4] Đang huấn luyện Low-Level Agents (Risk-aware & Ablations)..."
.venv/bin/python src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py --beta 100.0 --epochs 50 && \
.venv/bin/python src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py --beta 30.0 --epochs 50 && \
.venv/bin/python src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py --beta -10.0 --epochs 50 && \
.venv/bin/python src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py --beta -90.0 --epochs 50 && \
.venv/bin/python src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_only_teacher.py --beta 30.0 --epochs 50 && \
.venv/bin/python src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_only_td.py --beta 30.0 --epochs 50 && \

# === 2. LỌC & CHỌN BOT NGON NHẤT VÀO MA TRẬN 25 BOTS ===
echo "-> [2/4] Đang chạy Validation và Lọc Bot tốt nhất (Pick Agent)..."
.venv/bin/python src/EarnHFT_framwork/tool/run_validation.py && \
.venv/bin/python src/EarnHFT_framwork/analysis/pick_agent/pick_agent_position.py && \
.venv/bin/python src/EarnHFT_framwork/analysis/pick_agent/inspect_matrix.py && \

# === 3. CHẠY BASELINES ===
echo "-> [3/4] Đang huấn luyện các mô hình Baseline (CDQN, DQN, PPO, DRA)..."
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/cdqn_train.py --epochs 50 && \
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/dqn_train.py --epochs 50 && \
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/ppo_train.py --epochs 50 && \
.venv/bin/python src/EarnHFT_framwork/RL/agent/base/dra_train.py --epochs 50 && \

# === 4. CHẠY HIGH-LEVEL ROUTER ===
echo "-> [4/4] Đang huấn luyện High-Level Router (Mảnh ghép cuối cùng)..."
.venv/bin/python src/EarnHFT_framwork/RL/agent/high_level/dqn_position.py --epochs 100

echo "=========================================================="
echo "      HOÀN TẤT HUẤN LUYỆN TOÀN BỘ HỆ THỐNG!"
echo "=========================================================="
