@echo off
echo === STARTING BOT EVALUATION, ROUTER TRAINING AND REPORT GENERATION (WINDOWS) ===

:: Ensure using correct Python virtual environment
set PYTHON_CMD=".venv\Scripts\python.exe"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

:: -------------------------------------------------------------
:: PART 1: Low-level Bots Validation and High-level Router Training
:: -------------------------------------------------------------
echo [*] 1. Running Validation for all Models in Pool...
%PYTHON_CMD% src\EarnHFT_framwork\tool\run_validation.py

echo [*] 2. Filtering best Bots to create 5x5 Matrix...
%PYTHON_CMD% src\EarnHFT_framwork\analysis\pick_agent\pick_agent_position.py
%PYTHON_CMD% src\EarnHFT_framwork\analysis\pick_agent\inspect_matrix.py

echo [*] 3. Training High-Level Router...
%PYTHON_CMD% src\EarnHFT_framwork\RL\agent\high_level\dqn_position.py

:: -------------------------------------------------------------
:: PART 2: Report Generation and Chart Plotting
:: -------------------------------------------------------------
echo [*] 4. Calculating Metrics and generating comparison report...
%PYTHON_CMD% src\EarnHFT_framwork\analysis\calculate_metric\calculate_metric.py

echo [*] 5. Counting number of trades...
%PYTHON_CMD% count_trades.py

echo [*] 6. Plotting comprehensive Graph...
%PYTHON_CMD% src\EarnHFT_framwork\RL\util\graph.py



echo [*] 7. Plotting Training Convergence Graph...
%PYTHON_CMD% src\EarnHFT_framwork\analysis\visualize_convergence.py

echo === COMPLETELY FINISHED EVALUATION AND REPORTING ===
pause
