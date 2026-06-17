#!/bin/bash
echo "=== BẮT ĐẦU CHẠY SONG SONG 8 MÔ HÌNH CHO BTCUSDT (4 LOW-LEVEL + 4 BASELINES) ==="

PYTHON_CMD=".venv/bin/python"

# Tạo trước các thư mục log để tránh lỗi bash không tìm thấy đường dẫn ghi file
mkdir -p log/train/BTCUSDT/low_level
mkdir -p log/base/BTCUSDT

# -------------------------------------------------------------
# PHẦN 1: TRAIN 4 LOW-LEVEL AGENTS (Khẩu vị rủi ro khác nhau)
# -------------------------------------------------------------
echo "[*] Đang khởi động 4 Low-level Agents..."
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
# PHẦN 2: TRAIN 4 BASELINES AGENTS (Các đối thủ độc lập)
# -------------------------------------------------------------
echo "[*] Đang khởi động 4 Baseline Agents..."
# Baseline 1: CDQN-RP
CUDA_VISIBLE_DEVICES=0 nohup $PYTHON_CMD src/EarnHFT_framwork/RL/agent/base/cdqn_train.py \
    --dataset_name BTCUSDT --train_data_path data/cleaned_data/BTCUSDT/tardis/train --transcation_cost 0.00015 --max_holding_number 0.01 \
    >log/base/BTCUSDT/train_cdqn_rp.log 2>&1 &

# Baseline 2: DQN (Sử dụng ada_init 0)
CUDA_VISIBLE_DEVICES=0 nohup $PYTHON_CMD src/EarnHFT_framwork/RL/agent/base/dqn_train.py \
    --dataset_name BTCUSDT --train_data_path data/cleaned_data/BTCUSDT/tardis/train --transcation_cost 0.00015 --max_holding_number 0.01 --ada_init 0 \
    >log/base/BTCUSDT/train_dqn_0.log 2>&1 &

# Baseline 3: DRA (Dùng PPO + LSTM)
CUDA_VISIBLE_DEVICES=0 nohup $PYTHON_CMD src/EarnHFT_framwork/RL/agent/base/dra_train.py \
    --dataset_name BTCUSDT --train_data_path data/cleaned_data/BTCUSDT/tardis/train --transcation_cost 0.00015 --max_holding_number 0.01 \
    >log/base/BTCUSDT/train_dra_short.log 2>&1 &

# Baseline 4: PPO thuần
CUDA_VISIBLE_DEVICES=0 nohup $PYTHON_CMD src/EarnHFT_framwork/RL/agent/base/ppo_train.py \
    --dataset_name BTCUSDT --train_data_path data/cleaned_data/BTCUSDT/tardis/train --transcation_cost 0.00015 --max_holding_number 0.01 \
    >log/base/BTCUSDT/train_ppo.log 2>&1 &

echo "[!] Hệ thống đang huấn luyện đồng thời 8 mô hình trên GPU 0."
echo "[!] Vui lòng dùng lệnh 'htop' (để kiểm tra CPU/RAM) và 'nvidia-smi' (để xem VRAM)."
wait
echo "[+] HOÀN TẤT TRAIN TOÀN BỘ 8 MÔ HÌNH CHO ĐỒNG BTCUSDT!"
