# EarnHFT: Efficient Hierarchical Reinforcement Learning for High Frequency Trading

**Final Project for CS106.Q21 2025 - Artificial Intelligence**  
**University of Information Technology (UIT)**

This repository is a customized implementation based on the AAAI 2024 paper: [EarnHFT: Efficient hierarchical reinforcement learning for high frequency trading](https://arxiv.org/pdf/2309.12891.pdf). 

While we utilize the core algorithmic concepts from the official source code, we have made several architectural changes and implemented a completely custom data pipeline from scratch.

## Key Customizations from the Original Paper

1. **Order Book Depth**: The original paper utilized a Limit Order Book (LOB) depth of 5. In our implementation, we customized the reinforcement learning environments and neural network architectures to process a deeper **LOB depth of 10**, providing the agent with more granular market microstructure information.
2. **Custom Data Pipeline**: We did not use the data preprocessing scripts provided in the original source code. Instead, we built a fully custom pipeline designed to handle raw tick-level and LOB data downloaded directly from [Tardis.dev](https://tardis.dev/).
3. **Bypassing Premium API Limits**: The original repository requires purchasing a Tardis API key to fully utilize their code. To resolve this, our data pipeline is specifically engineered to take advantage of the **free historical datasets** that Tardis provides for the 1st day of every month. Our code automatically cleans, merges, and synchronizes these free slices to construct the RL environment without requiring a paid subscription.

## Data Preprocessing

Our custom data pipeline handles the raw Tardis data extraction. We process the raw Limit Order Book and Trades data to construct a synchronized dataframe. This dataframe is then used to compute over 100 technical indicators, from which we extract the Top 54 features based on their Information Coefficient (IC) scores.

## Detailed Framework Architecture & Mechanics

The EarnHFT framework addresses the primary challenges in High-Frequency Trading (HFT): extreme market noise and constantly shifting distributions (non-stationarity). It does this by decoupling the decision-making process into two layers: a microscopic **Low-Level Agent** and a macroscopic **High-Level Router**.

### 1. Low-Level Trading Bots (The Specialists)
The Low-Level agents are built using the **Double Deep Q-Network (DDQN)** architecture. They operate at the **tick level (1-second intervals)**.
- **Action Space:** Instead of simple (Buy, Sell, Hold) signals, the agent's action space consists of 5 discrete target positions: $A = \{0.0, 0.25, 0.5, 0.75, 1.0\}$. This dictates the percentage of the maximum holding capacity the bot wants to hold, allowing for nuanced inventory management rather than binary trades.
- **Q-Teacher & Distillation:** High-frequency data is extremely noisy, causing standard RL rewards to be sparse and delayed. To guide the bot, we employ a "Q-Teacher". The Q-Teacher is an oracle that calculates the theoretically optimal Q-value $Q^*(s,a)$ by analyzing actual future price trajectories within a small window. During training, the agent minimizes the Mean Squared Error (MSE) against the Q-Teacher's values, alongside the traditional TD-Error:
  $$Loss = Loss_{TD} + \alpha_{teacher} \times MSE(Q(s, a), Q^*(s, a))$$
  Where $\alpha_{teacher}$ decays over time. This robust distillation process forces the agent to learn the underlying market physics rather than chasing random noise.
- **Prioritized Experience Sampling (PES) with Risk Awareness:** We don't train just one generalized agent. We train a pool of agents, each with a different **Beta ($\beta$)** parameter representing Risk Preference. 
  - Using KDE (Kernel Density Estimation), the framework analyzes the return distribution of market chunks to assign an inverse-density weight $\omega_i = \frac{1}{\text{KDE}(R_i)}$.
  - A Boltzmann-based transformation is then applied to calculate the final sampling priority $W_i$ for chunk $i$ with return $R_i$:
    $$W_i = \omega_i \cdot e^{\beta \cdot R_i}$$
  - Chunks with extreme returns (high volatility) are assigned higher sampling weights for risk-seeking bots ($\beta > 0$), while stable chunks are prioritized for risk-averse bots ($\beta < 0$). This guarantees the creation of a diverse pool of bots: some excel in chaotic bull/bear runs, while others are masters of quiet, sideways markets.

### 2. Market Slicing & The Validation Tournament
Once the low-level bots are trained, how do we map them to specific market trends? We use the **Validation Set**.
- **Dynamic Time Warping (DTW):** We slice the validation data into smaller chunks and use the DTW algorithm to cluster these chunks into 5 distinct "Market Conditions" (Labels 0 to 4).
- **The Matrix Construction:** We run all our trained low-level bots across these 5 market conditions. We record their PnL and Sharpe Ratios.
- **Selection:** For each market condition (0-4), we select the top 5 performing bots, creating a diverse $5 \times 5$ matrix of 25 elite bots (5 initial position states $\times$ 5 market conditions).

### 3. High-Level Router (The Manager)
The market changes constantly. A bot that makes profit at 9:00 AM might lose money at 10:00 AM. 
- **Minute-Level Decisions:** The High-Level Router operates at the **minute level (every 60 seconds)**. It looks at macroscopic technical indicators (MACD, RSI, Volatility).
- **Dynamic Routing:** Instead of predicting "Buy" or "Sell", the Router predicts which of the 25 elite Low-Level bots is best suited for the upcoming minute.
- **Authentic Backtesting:** In our pipeline, we enforce strict tick-by-tick simulation. Once the Router picks a bot, that bot controls the portfolio for the next 60 seconds, evaluating the Limit Order Book (LOB) tick-by-tick without any skipping or mocked data. 

### 4. Evaluation (Valid vs Test Sets)
To maintain scientific integrity, the evaluation pipeline strictly separates environments:
- **Valid Set (Hyperparameter & Routing):** Used purely to rank the low-level bots and train the High-Level Router. It helps the system establish the $5 \times 5$ matrix and teaches the Router which bot works in which condition.
- **Test Set (Out-of-Sample):** The ultimate proving ground. The Test Set contains chronological data completely unseen by both the Low-Level bots and the Router. The Router observes the macro market, assigns a bot, and the bot executes trades tick-by-tick. The resulting `action.npy` logs and `final_balance.npy` represent the 100% authentic out-of-sample trading performance.

## Installation & Setup

To reproduce the environment and run the pipeline locally (NVIDIA GPU required):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Note: Training the hierarchical pipeline requires significant GPU resources. Scripts should be executed sequentially to manage VRAM efficiently.)*

## Running the Pipeline

To train and evaluate the complete EarnHFT framework, please execute the scripts in the following sequential order:

### 1. Data Preprocessing
Extract, clean, and engineer features from the raw Tardis datasets:
```bash
python3 src/data_cleaning/tardis_pipeline.py
```

### 2. Dynamic Market Slicing
Segment the validation dataset into multi-label volatility clusters using DTW:
```bash
python3 src/EarnHFT_framwork/tool/slice_model.py --data_path data/cleaned_data/BTCUSDT/tardis/valid
```

### 3. Low-Level Agent Training
Train the DDQN base agents with distinct risk preferences (Beta parameters). Run these sequentially:
```bash
python3 src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py --beta 100 --dataset_name BTCUSDT
python3 src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py --beta 30 --dataset_name BTCUSDT
python3 src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py --beta -10 --dataset_name BTCUSDT
python3 src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py --beta -90 --dataset_name BTCUSDT
```

### 4. Agent Evaluation & Matrix Construction
Evaluate all low-level agents across the validation slices to construct the 5x5 selection matrix:
```bash
python3 src/EarnHFT_framwork/tool/run_validation.py
python3 src/EarnHFT_framwork/analysis/pick_agent/pick_agent_position.py
python3 src/EarnHFT_framwork/analysis/pick_agent/inspect_matrix.py
```

### 5. High-Level Router Training
Train the hierarchical Router agent to dynamically select the best low-level agent during live trading:
```bash
python3 src/EarnHFT_framwork/RL/agent/high_level/dqn_position.py
```

### 6. Baselines & Analytics (Optional)
Evaluate the baseline models to compare performance against the proposed EarnHFT framework:
```bash
# Rule-based baselines
python3 src/EarnHFT_framwork/RL/agent/base/rule_based_tune.py --technicail_indicator MACD
python3 src/EarnHFT_framwork/RL/agent/base/rule_based_tune.py --technicail_indicator Imbalance_Volume

# RL baselines
python3 src/EarnHFT_framwork/RL/agent/base/cdqn_train.py
python3 src/EarnHFT_framwork/RL/agent/base/dqn_train.py
python3 src/EarnHFT_framwork/RL/agent/base/ppo_train.py
python3 src/EarnHFT_framwork/RL/agent/base/dra_train.py

# Generate Charts and Metrics
python3 count_trades.py
python3 src/EarnHFT_framwork/analysis/calculate_metric/calculate_metric.py
python3 src/EarnHFT_framwork/analysis/visualize_segment_match.py
python3 src/EarnHFT_framwork/analysis/visualize_kde.py
```
