#!/bin/bash
echo "=== STARTING PARALLEL TRAINING FOR 10 MODELS (4 LOW-LEVEL + 4 BASELINES + 2 ABLATION) ==="

PYTHON_CMD=".venv/bin/python"

# Create log directories
mkdir -p log/train/BTCUSDT/low_level
mkdir -p log/base/BTCUSDT

# -------------------------------------------------------------
# PART 1: TRAIN 4 LOW-LEVEL AGENTS
# -------------------------------------------------------------
echo "[*] Starting 4 Low-level Agents..."
CUDA_VISIBLE_DEVICES=0 nohup $PYTHON_CMD src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py \
    --beta 100 --train_data_path data/cleaned_data/BTCUSDT/tardis/train --dataset_name BTCUSDT \
    >log/train/BTCUSDT/low_level/beta_100.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup $PYTHON_CMD src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py \
    --beta -10 --train_data_path data/cleaned_data/BTCUSDT/tardis/train --dataset_name BTCUSDT \
    >log/train/BTCUSDT/low_level/beta_-10.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup $PYTHON_CMD src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py \
    --beta -90 --train_data_path data/cleaned_data/BTCUSDT/tardis/train --dataset_name BTCUSDT \
    >log/train/BTCUSDT/low_level/beta_-90.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup $PYTHON_CMD src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py \
    --beta 30 --train_data_path data/cleaned_data/BTCUSDT/tardis/train --dataset_name BTCUSDT \
    >log/train/BTCUSDT/low_level/beta_30.log 2>&1 &


# -------------------------------------------------------------
# PART 2: TRAIN 4 BASELINES AGENTS
# -------------------------------------------------------------
echo "[*] Starting 4 Baseline Agents..."
# Baseline 1: CDQN-RP
CUDA_VISIBLE_DEVICES=0 nohup $PYTHON_CMD src/EarnHFT_framwork/RL/agent/base/cdqn_train.py \
    --dataset_name BTCUSDT --train_data_path data/cleaned_data/BTCUSDT/tardis/train --transcation_cost 0.00015 --max_holding_number 0.01 \
    >log/base/BTCUSDT/train_cdqn_rp.log 2>&1 &

# Baseline 2: DQN
CUDA_VISIBLE_DEVICES=0 nohup $PYTHON_CMD src/EarnHFT_framwork/RL/agent/base/dqn_train.py \
    --dataset_name BTCUSDT --train_data_path data/cleaned_data/BTCUSDT/tardis/train --transcation_cost 0.00015 --max_holding_number 0.01 --ada_init 0 \
    >log/base/BTCUSDT/train_dqn_0.log 2>&1 &

# Baseline 3: DRA
CUDA_VISIBLE_DEVICES=0 nohup $PYTHON_CMD src/EarnHFT_framwork/RL/agent/base/dra_train.py \
    --dataset_name BTCUSDT --train_data_path data/cleaned_data/BTCUSDT/tardis/train --transcation_cost 0.00015 --max_holding_number 0.01 \
    >log/base/BTCUSDT/train_dra_short.log 2>&1 &

# Baseline 4: PPO
CUDA_VISIBLE_DEVICES=0 nohup $PYTHON_CMD src/EarnHFT_framwork/RL/agent/base/ppo_train.py \
    --dataset_name BTCUSDT --train_data_path data/cleaned_data/BTCUSDT/tardis/train --transcation_cost 0.00015 --max_holding_number 0.01 \
    >log/base/BTCUSDT/train_ppo.log 2>&1 &

# -------------------------------------------------------------
# PART 3: ABLATION STUDY (BETA 30)
# -------------------------------------------------------------
echo "[*] Starting 2 Ablation Study Agents..."

CUDA_VISIBLE_DEVICES=0 nohup $PYTHON_CMD src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_only_teacher.py \
    --beta 30 --train_data_path data/cleaned_data/BTCUSDT/tardis/train --dataset_name BTCUSDT \
    >log/train/BTCUSDT/low_level/beta_30_only_teacher.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup $PYTHON_CMD src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_only_td.py \
    --beta 30 --train_data_path data/cleaned_data/BTCUSDT/tardis/train --dataset_name BTCUSDT \
    >log/train/BTCUSDT/low_level/beta_30_only_td.log 2>&1 &

echo "[!] System is training 10 models simultaneously on GPU 0."
echo "[!] Use 'htop' to check CPU/RAM and 'nvidia-smi' to check VRAM."
wait
echo "[+] DONE TRAINING ALL 10 MODELS!"
