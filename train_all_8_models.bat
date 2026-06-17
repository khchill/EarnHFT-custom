@echo off
chcp 65001 >nul
echo === BẮT ĐẦU CHẠY SONG SONG 8 MÔ HÌNH CHO BTCUSDT (WINDOWS) ===

set PYTHON_CMD=".venv\Scripts\python.exe"
set CUDA_VISIBLE_DEVICES=0

:: Tạo trước các thư mục log (bỏ qua nếu đã tồn tại)
if not exist "log\train\BTCUSDT\low_level" mkdir "log\train\BTCUSDT\low_level"
if not exist "log\base\BTCUSDT" mkdir "log\base\BTCUSDT"

:: -------------------------------------------------------------
:: PHẦN 1: TRAIN 4 LOW-LEVEL AGENTS
:: -------------------------------------------------------------
echo [*] Đang khởi động 4 Low-level Agents chạy ngầm...

start /b "" %PYTHON_CMD% src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py --beta 100 --train_data_path data/cleaned_data/BTCUSDT/tardis/train --dataset_name BTCUSDT > log\train\BTCUSDT\low_level\beta_100.log 2>&1

start /b "" %PYTHON_CMD% src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py --beta -10 --train_data_path data/cleaned_data/BTCUSDT/tardis/train --dataset_name BTCUSDT > log\train\BTCUSDT\low_level\beta_-10.log 2>&1

start /b "" %PYTHON_CMD% src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py --beta -90 --train_data_path data/cleaned_data/BTCUSDT/tardis/train --dataset_name BTCUSDT > log\train\BTCUSDT\low_level\beta_-90.log 2>&1

start /b "" %PYTHON_CMD% src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py --beta 30 --train_data_path data/cleaned_data/BTCUSDT/tardis/train --dataset_name BTCUSDT > log\train\BTCUSDT\low_level\beta_30.log 2>&1

:: -------------------------------------------------------------
:: PHẦN 2: TRAIN 4 BASELINES AGENTS
:: -------------------------------------------------------------
echo [*] Đang khởi động 4 Baseline Agents chạy ngầm...

:: Baseline 1: CDQN-RP
start /b "" %PYTHON_CMD% src/EarnHFT_framwork/RL/agent/base/cdqn_train.py --dataset_name BTCUSDT --train_data_path data/cleaned_data/BTCUSDT/tardis/train --transcation_cost 0.00015 --max_holding_number 0.01 > log\base\BTCUSDT\train_cdqn_rp.log 2>&1

:: Baseline 2: DQN
start /b "" %PYTHON_CMD% src/EarnHFT_framwork/RL/agent/base/dqn_train.py --dataset_name BTCUSDT --train_data_path data/cleaned_data/BTCUSDT/tardis/train --transcation_cost 0.00015 --max_holding_number 0.01 --ada_init 0 > log\base\BTCUSDT\train_dqn_0.log 2>&1

:: Baseline 3: DRA
start /b "" %PYTHON_CMD% src/EarnHFT_framwork/RL/agent/base/dra_train.py --dataset_name BTCUSDT --train_data_path data/cleaned_data/BTCUSDT/tardis/train --transcation_cost 0.00015 --max_holding_number 0.01 > log\base\BTCUSDT\train_dra_short.log 2>&1

:: Baseline 4: PPO
start /b "" %PYTHON_CMD% src/EarnHFT_framwork/RL/agent/base/ppo_train.py --dataset_name BTCUSDT --train_data_path data/cleaned_data/BTCUSDT/tardis/train --transcation_cost 0.00015 --max_holding_number 0.01 > log\base\BTCUSDT\train_ppo.log 2>&1

echo [!] Hệ thống đang huấn luyện đồng thời 8 mô hình trên Card đồ họa mặc định.
echo [!] Tiến trình đang chạy ngầm. Vui lòng mở Task Manager đê theo dõi CPU/RAM.
echo [!] Không được tắt cửa sổ này nếu muốn giữ tiến trình. Để tắt, bấm Ctrl+C nhiều lần.
