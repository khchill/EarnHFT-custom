@echo off
echo === STARTING QUICK 1-EPOCH TEST FOR 10 MODELS (WINDOWS) ===

set PYTHON_CMD=".venv\Scripts\python.exe"
set CUDA_VISIBLE_DEVICES=0
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Create log directories
if not exist "log\train\BTCUSDT\low_level" mkdir "log\train\BTCUSDT\low_level"
if not exist "log\base\BTCUSDT" mkdir "log\base\BTCUSDT"

echo [*] Starting 10 models in background on GPU...

REM 4 Low-level Agents
start /b "" %PYTHON_CMD% src\EarnHFT_framwork\RL\agent\low_level\ddqn_pes_risk_aware.py --beta 100 --train_data_path data\cleaned_data\BTCUSDT\tardis\train --dataset_name BTCUSDT --epochs 1 > log\train\BTCUSDT\low_level\beta_100_test_1ep.log 2>&1
start /b "" %PYTHON_CMD% src\EarnHFT_framwork\RL\agent\low_level\ddqn_pes_risk_aware.py --beta -10 --train_data_path data\cleaned_data\BTCUSDT\tardis\train --dataset_name BTCUSDT --epochs 1 > log\train\BTCUSDT\low_level\beta_-10_test_1ep.log 2>&1
start /b "" %PYTHON_CMD% src\EarnHFT_framwork\RL\agent\low_level\ddqn_pes_risk_aware.py --beta -90 --train_data_path data\cleaned_data\BTCUSDT\tardis\train --dataset_name BTCUSDT --epochs 1 > log\train\BTCUSDT\low_level\beta_-90_test_1ep.log 2>&1
start /b "" %PYTHON_CMD% src\EarnHFT_framwork\RL\agent\low_level\ddqn_pes_risk_aware.py --beta 30 --train_data_path data\cleaned_data\BTCUSDT\tardis\train --dataset_name BTCUSDT --epochs 1 > log\train\BTCUSDT\low_level\beta_30_test_1ep.log 2>&1

REM 4 Baseline Agents
start /b "" %PYTHON_CMD% src\EarnHFT_framwork\RL\agent\base\cdqn_train.py --dataset_name BTCUSDT --train_data_path data\cleaned_data\BTCUSDT\tardis\train --transcation_cost 0.00015 --max_holding_number 0.01 --epochs 1 > log\base\BTCUSDT\train_cdqn_rp_test_1ep.log 2>&1
start /b "" %PYTHON_CMD% src\EarnHFT_framwork\RL\agent\base\dqn_train.py --dataset_name BTCUSDT --train_data_path data\cleaned_data\BTCUSDT\tardis\train --transcation_cost 0.00015 --max_holding_number 0.01 --ada_init 0 --epochs 1 > log\base\BTCUSDT\train_dqn_0_test_1ep.log 2>&1
start /b "" %PYTHON_CMD% src\EarnHFT_framwork\RL\agent\base\dra_train.py --dataset_name BTCUSDT --train_data_path data\cleaned_data\BTCUSDT\tardis\train --transcation_cost 0.00015 --max_holding_number 0.01 --epochs 1 > log\base\BTCUSDT\train_dra_short_test_1ep.log 2>&1
start /b "" %PYTHON_CMD% src\EarnHFT_framwork\RL\agent\base\ppo_train.py --dataset_name BTCUSDT --train_data_path data\cleaned_data\BTCUSDT\tardis\train --transcation_cost 0.00015 --max_holding_number 0.01 --epochs 1 > log\base\BTCUSDT\train_ppo_test_1ep.log 2>&1

REM Ablation Study Agents (Beta 30)
start /b "" %PYTHON_CMD% src\EarnHFT_framwork\RL\agent\low_level\ddqn_pes_only_teacher.py --beta 30 --train_data_path data\cleaned_data\BTCUSDT\tardis\train --dataset_name BTCUSDT --epochs 1 > log\train\BTCUSDT\low_level\beta_30_only_teacher_test_1ep.log 2>&1
start /b "" %PYTHON_CMD% src\EarnHFT_framwork\RL\agent\low_level\ddqn_pes_only_td.py --beta 30 --train_data_path data\cleaned_data\BTCUSDT\tardis\train --dataset_name BTCUSDT --epochs 1 > log\train\BTCUSDT\low_level\beta_30_only_td_test_1ep.log 2>&1

echo [!] System is training 10 models simultaneously for 1 EPOCH.
echo [!] Logs are saved with "_test_1ep.log" suffix to not overwrite your main logs.
echo [!] QUAN TRONG: Vui long giu nguyen cua so CMD nay. Neu tat, tat ca model se bi dung.
pause
