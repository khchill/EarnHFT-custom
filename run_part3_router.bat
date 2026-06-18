@echo off
echo === STARTING ROUTER TRAINING ===

:: Ensure using correct Python virtual environment
set PYTHON_CMD=".venv\Scripts\python.exe"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo [*] 2. Filtering best Bots to create 5x5 Matrix...
%PYTHON_CMD% src\EarnHFT_framwork\analysis\pick_agent\pick_agent_position.py > log_router_training.txt 2>&1

echo [*] 3. Inspecting Matrix...
%PYTHON_CMD% src\EarnHFT_framwork\analysis\pick_agent\inspect_matrix.py >> log_router_training.txt 2>&1

echo [*] 3. Training High-Level Router... 
%PYTHON_CMD% src\EarnHFT_framwork\RL\agent\high_level\dqn_position.py >> log_router_training.txt 2>&1

echo === COMPLETELY FINISHED ROUTER TRAINING ===
echo Log duoc ghi vao file log_router_training.txt
pause
