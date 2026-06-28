# EarnHFT: Efficient Hierarchical Reinforcement Learning for High Frequency Trading

**Final Project for CS106.Q21 (Academic Year 2025-2026, Semester II) - Artificial Intelligence**  
**University of Information Technology (UIT)**

**Team Members:**
- Nguyễn Khánh Hưng - 24520607
- Huỳnh Quang Hưng - 24520591

This repository is a customized and heavily extended implementation based on the AAAI 2024 paper: [EarnHFT: Efficient hierarchical reinforcement learning for high frequency trading](https://arxiv.org/pdf/2309.12891.pdf). 

While we utilize the core algorithmic concepts from the official source code, we have made several architectural changes, implemented a completely custom data pipeline from scratch, and introduced a robust multi-threaded backtesting & analysis suite.

## Key Customizations from the Original Paper

1. **Order Book Depth**: The original paper utilized a Limit Order Book (LOB) depth of 5. In our implementation, we customized the reinforcement learning environments and neural network architectures to process a deeper **LOB depth of 10**, providing the agent with more granular market microstructure information.
2. **Custom Data Pipeline**: We built a fully custom pipeline designed to handle raw tick-level and LOB data downloaded directly from [Tardis.dev](https://tardis.dev/).
3. **Bypassing Premium API Limits**: Our data pipeline is specifically engineered to take advantage of the **free historical datasets** that Tardis provides for the 1st day of every month. Our code automatically cleans, merges, and synchronizes these free slices to construct the RL environment without requiring a paid subscription.
4. **Automated Multi-Core Pipeline**: We designed custom shell scripts (`train_parallel.sh`) and multi-process Python evaluators (`batch_backtest.py`) to fully utilize high-end cloud compute (e.g., 3090 Ti + 16/32 CPU Threads) to drastically reduce training and backtesting times.

## Detailed Framework Architecture & Mechanics

The EarnHFT framework addresses extreme market noise and non-stationarity by decoupling the decision-making process into two layers: a microscopic **Low-Level Agent** and a macroscopic **High-Level Router**.

### 1. Low-Level Trading Bots (The Specialists)
The Low-Level agents are built using the **Double Deep Q-Network (DDQN)** architecture, operating at the **tick level (1-second intervals)**.
- **Action Space:** 5 discrete target positions: $A = \{0.0, 0.25, 0.5, 0.75, 1.0\}$.
- **Q-Teacher & Distillation:** Employs an oracle Q-Teacher to calculate theoretically optimal trajectories. Agents minimize **KL Divergence** against the Q-Teacher's action distribution alongside TD-Error to filter out high-frequency noise.
- **Prioritized Experience Sampling (PES) with Risk Awareness:** We train a pool of agents with different **Beta ($\beta$)** risk parameters using KDE (Kernel Density Estimation). Risk-seeking bots prioritize extreme volatility slices, while risk-averse bots focus on stable, sideways market segments.

### 2. Market Slicing & The Validation Tournament
- **Market Slicing & Clustering:** Validation data is smoothed (Butterworth filter), sliced, and clustered using **Dynamic Time Warping (DTW)** into 5 distinct "Market Conditions" (Labels 0 to 4).
- **The Matrix Construction:** We run all trained low-level bots across these 5 conditions. We select the top 5 performing bots for each condition, creating a elite $5 \times 5$ selection matrix.

### 3. High-Level Router (The Manager)
- **Minute-Level Decisions:** Operates every 60 seconds observing macroscopic technical indicators (MACD, RSI, Volatility).
- **Dynamic Routing:** Predicts which of the 25 elite Low-Level bots is best suited for the upcoming minute.
- **Authentic Backtesting:** Tick-by-tick simulation without mocked data to guarantee out-of-sample authenticity.

### 4. Evaluation (Valid vs Test Sets)
To maintain scientific integrity, the evaluation pipeline strictly separates environments:
- **Valid Set (Hyperparameter & Routing):** Used purely to rank the low-level bots and train the High-Level Router. It helps the system establish the $5 \times 5$ matrix and teaches the Router which bot works in which condition.
- **Test Set (Out-of-Sample):** The ultimate proving ground. The Test Set contains chronological data completely unseen by both the Low-Level bots and the Router. The Router observes the macro market, assigns a bot, and the bot executes trades tick-by-tick. The resulting `action.npy` logs and `final_balance.npy` represent the 100% authentic out-of-sample trading performance.

---

## Project Structure
This section explains the organization of the codebase, detailing what each directory and key file is responsible for within the EarnHFT framework.

```text
src_final_project/
├── data/                               # Dataset storage
│   ├── raw/                            # Raw data downloaded from Tardis.dev
│   └── cleaned_data/                   # Processed data, features, and target files
├── result_risk/                        # Output directory for trained models and backtest results
│   └── BTCUSDT/                        
│       ├── dqn_ada_0/                  # Standard DQN model outputs (Beta 0)
│       ├── dqn_ada_100/                # Risk-seeking Low-Level agent (Beta 100)
│       ├── dqn_ada_30/                 # Risk-seeking Low-Level agent (Beta 30)
│       ├── dqn_ada_-10/                # Risk-averse Low-Level agent (Beta -10)
│       ├── dqn_ada_-90/                # Risk-averse Low-Level agent (Beta -90)
│       ├── cdqn_rp/                    # CDQN-RP baseline model outputs
│       ├── dra_short/                  # DRA baseline model outputs
│       ├── ppo/                        # PPO baseline model outputs
│       ├── high_level/                 # High-level Router model outputs
│       └── rule_base/                  # MACD & Imbalance Volume baseline outputs
├── results/                            # Extracted metric tables, CSVs, and visualization images
├── src/                                # Core source code
│   ├── data_cleaning/                  # Custom data pipeline for Tardis data (LOB depth 10)
│   └── EarnHFT_framwork/               
│       ├── env/                        # RL environments (low_level_env.py, high_level_env.py)
│       ├── model/                      # Neural network architectures (Qnet, ActorCritic, etc.)
│       ├── RL/                         # RL training algorithms
│       │   ├── agent/base/             # Baseline training scripts (dqn_train.py, ppo_train.py, etc.)
│       │   ├── agent/low_level/        # Low-level agents training (DDQN with PES and Risk Awareness)
│       │   └── agent/high_level/       # High-level router training (dqn_position.py)
│       ├── tool/                       # Utilities for training and backtesting
│       │   ├── slice_model.py          # Market slicing and DTW clustering
│       │   ├── run_validation.py       # Validation tournament execution
│       │   └── batch_backtest.py       # Multi-core batch backtesting across all epochs
│       └── analysis/                   # Evaluation, metric computation, and visualizations
│           ├── pick_agent/             # Matrix construction logic
│           ├── calculate_metric/       # Metric calculations (Sharpe, Volatility, Max DD)
│           ├── check_action_epoch.py   # Analyzes action distribution and bot selection
│           ├── count_trades.py         # Counts the number of trades executed by each model
│           ├── visualize_kde.py        # Kernel Density Estimation plotting
│           └── visualize_segment_match.py # PnL Net Curve comparisons per market segment
├── train_parallel.sh                   # Bash script to parallelize low-level bot training (Tmux)
├── quick_test_1_epoch.sh               # Bash script for fast, single-epoch debugging
└── README.md                           # Main project documentation
```

---

## Hardware Specifications
This project was evaluated and optimized on the following hardware configuration:
- **CPU:** AMD Ryzen Threadripper PRO 3955WX 32-Cores (88GB RAM)
- **GPU:** 1x NVIDIA RTX 3090 Ti (24GB VRAM, 10752 CUDA Cores)
- **Storage:** 1600GB SSD
- **OS:** Ubuntu 24.04

### Low-Spec Machine Optimization
If you are running this on a weaker machine with limited CPU threads or GPU VRAM, you **must** adjust the parallelization parameters in `train_parallel.sh` and `quick_test_1_epoch.sh` to prevent system freezing or resource exhaustion. 

Specifically, you can restrict the thread limits by adding these environment variables at the top of the Python scripts (`dqn_position.py`, `ddqn_pes_only_selfplay_kl.py`, etc.) or exporting them in your terminal before running:
```python
import os
os.environ["MKL_NUM_THREADS"] = "2" # Adjust to match your CPU cores
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"
```

---

## Installation & Setup

To reproduce the environment and run the pipeline locally (NVIDIA GPU highly recommended):

> [!IMPORTANT]
> If your server runs a fresh OS and reports an environment error, ensure you have installed `python3-venv` using the command `sudo apt install python3-venv` before creating the `.venv`.

### Step 1: Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install CUDA-Optimized PyTorch & Dependencies
To ensure the system fully utilizes the RTX 3090 Ti (or similar GPUs), forcefully install the CUDA version of PyTorch instead of the CPU version:
```bash
sed '/torch/d' requirements.txt > req_no_torch.txt
pip install -r req_no_torch.txt
pip install prettytable  # Used for generating formatted tables
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Verify GPU Recognition
Run the following command to ensure the server successfully recognizes the RTX 3090 Ti:
```bash
.venv/bin/python -c "import torch; print('\n[+] PyTorch recognizes GPU:', torch.cuda.is_available()); print('[+] GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```
*(If it prints out RTX 3090 Ti, your setup is perfect!)*

---

## Full Execution Pipeline

To train and evaluate the complete EarnHFT framework, execute the following steps in order:

### 1. Data Preprocessing & Generation
Extract, clean, and engineer over 100 features from the raw Tardis datasets:
```bash
.venv/bin/python src/data_cleaning/tardis_pipeline.py
```

### 2. Dynamic Market Slicing
Segment the validation dataset into multi-label volatility clusters:
```bash
.venv/bin/python src/EarnHFT_framwork/tool/slice_model.py --data_path data/cleaned_data/BTCUSDT/tardis/valid
```

### 3. Agent Training (Parallel Execution)
Train the DDQN base agents (Risk-aware) and Baseline models (PPO, DRA, CDQN). We use Tmux & Shell scripts to parallelize this across multiple GPU streams:
```bash
# Grant execution permissions to shell scripts
chmod +x train_parallel.sh quick_test_1_epoch.sh

# Train Low-Level bots with Beta = [100, 30, -10, -90]
./train_parallel.sh
```

### 4. Validation Tournament & Matrix Construction
Evaluate all low-level agents across the validation slices to construct the 5x5 selection matrix:
```bash
.venv/bin/python src/EarnHFT_framwork/tool/run_validation.py
.venv/bin/python src/EarnHFT_framwork/analysis/pick_agent/pick_agent_position.py
.venv/bin/python src/EarnHFT_framwork/analysis/pick_agent/inspect_matrix.py
```

### 5. High-Level Router Training
Train the hierarchical Router agent to dynamically select the best low-level agent:
```bash
.venv/bin/python src/EarnHFT_framwork/RL/agent/high_level/dqn_position.py
```

---

## Evaluation & Analytics

Once all models are trained, we run the comprehensive analytics suite to generate metrics, charts, and tables. All outputs are automatically saved to the `results/` folder for easy extraction.

### 1. Multi-Core Batch Backtesting
To enforce authentic backtesting across Valid & Test datasets on Epoch 1, final, and those divisible by 5, run our parallel CPU-bound script:
```bash
.venv/bin/python src/EarnHFT_framwork/tool/batch_backtest.py
```
*(Automatically scales to utilize 100% of available CPU cores safely).*

### 2. Metric Computation & Table Generation
Extract all essential metrics (Total Return, Annualized Volatility, Sharpe Ratio, Max Drawdown) and export them into cleanly formatted text tables:
```bash
# Summary table for Trade counts & Bot selection
.venv/bin/python src/EarnHFT_framwork/analysis/check_action_epoch.py

# Calculate Metrics for all Epochs divisible by 5
.venv/bin/python src/EarnHFT_framwork/analysis/calculate_metric/calculate_metric_epoch.py

# Calculate Metrics and plot PnL (Net Curve) for the best Epochs
.venv/bin/python src/EarnHFT_framwork/analysis/calculate_metric/calculate_metric.py
```

### 3. Advanced Visualizations
Generate additional Deep Learning analytics charts:
```bash
# Visualize Segment Matches (Compare PnL Net Curves per segment)
.venv/bin/python src/EarnHFT_framwork/analysis/visualize_segment_match.py

# Visualize Kernel Density Estimation (Profit distribution analysis)
.venv/bin/python src/EarnHFT_framwork/analysis/visualize_kde.py

# Count total trades executed by the best models
.venv/bin/python src/EarnHFT_framwork/analysis/count_trades.py
```
