@echo off
echo === STARTING METRICS AND REPORT GENERATION ===

:: Ensure using correct Python virtual environment
set PYTHON_CMD=".venv\Scripts\python.exe"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo [*] 4. Calculating Metrics and generating comparison report...
%PYTHON_CMD% src\EarnHFT_framwork\analysis\calculate_metric\calculate_metric.py > log_reports.txt 2>&1

echo [*] 5. Counting number of trades...
%PYTHON_CMD% count_trades.py >> log_reports.txt 2>&1

echo [*] 6. Plotting comprehensive Graph...
%PYTHON_CMD% src\EarnHFT_framwork\RL\util\graph.py >> log_reports.txt 2>&1

echo [*] 7. Plotting Training Convergence Graph...
%PYTHON_CMD% src\EarnHFT_framwork\analysis\visualize_convergence.py >> log_reports.txt 2>&1

echo === COMPLETELY FINISHED REPORTING ===
echo Log duoc ghi vao file log_reports.txt
pause
