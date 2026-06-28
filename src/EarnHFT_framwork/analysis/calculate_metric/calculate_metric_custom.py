import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from collections import OrderedDict
import re

ALL_METRIC_TEXT = ""

sys.path.append(".")
sys.path.append("src")
sys.path.append("src/EarnHFT_framwork")
sys.path.append("./src/EarnHFT_framwork")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.calculate_metric.calculate_wsrc import print_metrics, evaluate_second

def sort_list(lst: list):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    lst.sort(key=alphanum_key)

models_to_check = {
    "IV": ("result_risk/BTCUSDT/rule_base/IV", None),
    "MACD": ("result_risk/BTCUSDT/rule_base/MACD", None),
    "DQN": ("result_risk/BTCUSDT/dqn_ada_0/seed_12345", "epoch_110"),
    "CDQN_RP": ("result_risk/BTCUSDT/cdqn_rp/seed_12345", "epoch_50"),
    "PPO": ("result_risk/BTCUSDT/ppo/seed_12345", "epoch_30"),
    "DRA": ("result_risk/BTCUSDT/dra_short/seed_12345", "epoch_10"),
    "EarnHFT (Router)": ("result_risk/BTCUSDT/high_level/seed_12345", "epoch_20")
}

def pick_agent_and_generate_result_table(root_path, name, custom_epoch=None, mode='test'):
    if not os.path.exists(root_path):
        print(f"Skipping: {root_path} not found")
        return
        
    if "rule_base" in root_path:
        test_path = os.path.join(root_path, mode)
        best_epoch_name = "rule_base"
    else:
        if custom_epoch is None:
            print(f"Skipping: custom_epoch is None for {name}")
            return
        test_path = os.path.join(root_path, custom_epoch, mode)
        best_epoch_name = custom_epoch
        
    reward_file = os.path.join(test_path, "reward.npy")
    require_money_file = os.path.join(test_path, "require_money.npy")
    if not os.path.exists(reward_file) or not os.path.exists(require_money_file):
        print(f"Missing test files for {name} in {test_path}")
        return
        
    test_dir = f"data/cleaned_data/BTCUSDT/tardis/{mode}"
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".feather")])
    lengths = [len(pd.read_feather(os.path.join(test_dir, f))) - 1 for f in test_files]

    reward_list = np.load(reward_file)
    require_money = np.load(require_money_file)
    
    # Check if require_money is an array, if not make it a list of same value
    if isinstance(require_money, float) or require_money.ndim == 0:
        require_money = [require_money] * len(lengths)

    original_lengths = lengths.copy()
    total_expected_len = sum(lengths)
    actual_len = len(reward_list)
    scale = 1
    if actual_len > 0 and actual_len < total_expected_len / 2:
        scale = int(np.round(total_expected_len / actual_len))
        if scale > 1:
            lengths = [int(np.ceil(l / scale)) for l in lengths]

    daily_metrics = []
    start = 0
    for i, length in enumerate(lengths):
        if start + length > len(reward_list):
            break
        daily_reward = reward_list[start:start+length]
        if len(daily_reward) == 0:
            break
        req_m = require_money[i] if i < len(require_money) else require_money[-1]
        
        # NOTE: if scaled, evaluate_second metrics using default 1s constants will be scaled
        # If the user wants precise annualization they might need to update evaluate_second.
        metric = list(evaluate_second(req_m, daily_reward))
        # Optional: adjust volatility and sharpe for 60s timescale if scale is ~60
        if scale > 1:
            # daily_vol (index 1), daily_dd (index 3), annual_sr (index 4), daily_cr (index 5)
            # daily_SoR (index 6), annual_cr (index 7), annual_SoR (index 8), annual_vol (index 9)
            # The formulas in evaluate_second multiply by sqrt(3600*24). If each step is `scale` seconds,
            # we should divide the annualized terms by sqrt(scale) to correct the time dimension.
            correction = np.sqrt(scale)
            metric[1] /= correction
            metric[3] /= correction
            metric[4] /= correction
            metric[6] /= correction
            metric[8] /= correction
            metric[9] /= correction
            # Calmar doesn't use sqrt, it uses linear scaling
            metric[5] /= scale
            metric[7] /= scale
        
        daily_metrics.append(metric)
        start += length

    if not daily_metrics:
        print(f"No valid segments for {name}")
        return

    avg_metrics = np.mean(daily_metrics, axis=0)
    (
        tr, daily_vol, mdd, daily_dd, annual_sr, daily_cr, 
        daily_SoR, annual_cr, annual_SoR, annual_vol,
    ) = avg_metrics
    
    stats = OrderedDict(
        {
            "Total Return (Avg)": ["{:04f}%".format(tr * 100)],
            "Annualized Sharp Ratio (Avg)": ["{:04f}".format(annual_sr)],
            "Annualized Calmar Ratio (Avg)": ["{:04f}".format(annual_cr)],
            "Annualized Sortino Ratio (Avg)": ["{:04f}".format(annual_SoR)],
            "Annualized Volatility (Avg)": ["{:04f}%".format(annual_vol * 100)],
            "Annualized Downside Deviation (Avg)": ["{:04f}%".format(daily_dd * 100 * np.sqrt(365))],
            "Max Drawdown (Avg)": ["{:04f}%".format(mdd * 100)],
        }
    )
    table = print_metrics(stats)
    
    global ALL_METRIC_TEXT
    output_str = f"Model: {name}\n"
    output_str += f"Custom Epoch: {best_epoch_name}\n"
    output_str += f"Mode: {mode.upper()}\n"
    output_str += str(table) + "\n"
    output_str += "-"*60 + "\n"
    
    print(output_str)
    ALL_METRIC_TEXT += output_str

def plot_all_baselines_curve(root_path_list, names, mode="test", save_path="result_risk/BTCUSDT/all_baselines_comparison.png"):
    save_path = save_path.replace(".png", f"_{mode}.png")
    plt.figure(figsize=(12, 7))
    colors = ['black', 'blue', 'purple', 'green', 'orange', 'red', 'brown']
    
    daily_curves_by_date = {}
    
    for root_path, name, color in zip(root_path_list, names, colors):
        if not os.path.exists(root_path):
            continue
            
        has_epochs = False
        subdirs = os.listdir(root_path)
        if any(d.startswith("epoch_") for d in subdirs):
            has_epochs = True
            
        if has_epochs:
            epoch_path_list = [d for d in os.listdir(root_path) if d.startswith("epoch_")]
            sort_list(epoch_path_list)
            
            # Tìm model tốt nhất trên tập valid
            valid_result_list = []
            for epoch in epoch_path_list:
                v_path = os.path.join(root_path, epoch, "valid")
                final_balance_file = os.path.join(v_path, "final_balance.npy")
                require_money_file = os.path.join(v_path, "require_money.npy")
                if not os.path.exists(final_balance_file) or not os.path.exists(require_money_file):
                    valid_result_list.append(-999999.0)
                    continue
                val = np.load(final_balance_file) / (np.load(require_money_file) + 1e-12)
                valid_result_list.append(float(np.mean(val)))
                
            if not valid_result_list or max(valid_result_list) == -999999.0:
                continue
                
            best_epoch = epoch_path_list[valid_result_list.index(max(valid_result_list))]
            test_path = os.path.join(root_path, best_epoch, mode)
        else:
            test_path = os.path.join(root_path, mode)
            
        reward_file = os.path.join(test_path, "reward.npy")
        require_money_file = os.path.join(test_path, "require_money.npy")
        if not os.path.exists(reward_file) or not os.path.exists(require_money_file):
            continue

        test_dir = f"data/cleaned_data/BTCUSDT/tardis/{mode}"
        test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".feather")])
        lengths = [len(pd.read_feather(os.path.join(test_dir, f))) - 1 for f in test_files]

        reward_list = np.load(reward_file)
        require_money = np.load(require_money_file)
        
        if isinstance(require_money, float) or require_money.ndim == 0:
            require_money = [require_money] * len(lengths)

        original_lengths = lengths.copy()
        total_expected_len = sum(lengths)
        actual_len = len(reward_list)
        scale = 1
        if actual_len > 0 and actual_len < total_expected_len / 2:
            scale = int(np.round(total_expected_len / actual_len))
            if scale > 1:
                lengths = [int(np.ceil(l / scale)) for l in lengths]

        daily_curves = []
        start = 0
        for i, length in enumerate(lengths):
            if start + length > len(reward_list):
                break
            daily_reward = reward_list[start:start+length]
            req_m = require_money[i] if i < len(require_money) else require_money[-1]
            
            acc_reward = np.cumsum(daily_reward)
            net_curve = (acc_reward) / (req_m + 1e-12) * 100
            
            # Upsample if needed so it aligns with others (for 86340 points)
            if scale > 1:
                x_old = np.linspace(0, 1, len(net_curve))
                x_new = np.linspace(0, 1, original_lengths[i])
                net_curve = np.interp(x_new, x_old, net_curve)
                
            daily_curves.append(net_curve)
            start += length

        if not daily_curves:
            continue
            
        min_len = min(len(c) for c in daily_curves)
        avg_net_curve = np.mean([c[:min_len] for c in daily_curves], axis=0)
        
        # Lưu vào dict để vẽ riêng từng ngày sau
        for j, test_file in enumerate(test_files):
            date_str = test_file.replace("df_", "").replace(".feather", "")
            if date_str not in daily_curves_by_date:
                daily_curves_by_date[date_str] = {}
            if j < len(daily_curves):
                daily_curves_by_date[date_str][name] = (daily_curves[j], color)
        
        plt.plot(range(len(avg_net_curve)), avg_net_curve, label=name.upper(), color=color, linewidth=1.5)

    # 1. Vẽ đồ thị Trung bình (Average)
    plt.xlabel("Trading Timestamp (s)", fontsize=14)
    plt.ylabel("Total Return (%)", fontsize=14)
    plt.title(f"Net Value Curve for BTCUSDT (Average All {mode.upper()} Days)", fontsize=16)
    plt.grid(linestyle='--')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Đã vẽ xong đồ thị Baselines TRUNG BÌNH tại: {save_path}")
    # vẽ riêng từng ngày test
    for date_str, curves in daily_curves_by_date.items():
        plt.figure(figsize=(10, 6))
        for bot_name, (curve, color) in curves.items():
            plt.plot(range(len(curve)), curve, label=bot_name.upper(), color=color, linewidth=1.5)
            
        plt.xlabel("Trading Timestamp (s)", fontsize=14)
        plt.ylabel("Total Return (%)", fontsize=14)
        plt.title(f"Net Value Curve for BTCUSDT ({mode.upper()} - {date_str})", fontsize=16)
        plt.grid(linestyle='--')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
        plt.tight_layout()
        
        day_save_path = save_path.replace(".png", f"_{date_str}.png")
        plt.savefig(day_save_path)
        plt.close()
        print(f"Đã vẽ xong đồ thị cho ngày {date_str} tại: {day_save_path}")


def get_custom_test_reward(root_path, custom_epoch, mode="test", model_name=None):
    if "rule_base" in root_path and model_name is not None:
        test_dir = os.path.join(root_path, mode)
        if os.path.exists(test_dir):
            return np.load(os.path.join(test_dir, "reward.npy")), \
                   np.load(os.path.join(test_dir, "require_money.npy"))
        return None, None
        
    if custom_epoch is None:
        return None, None
        
    test_path = os.path.join(root_path, custom_epoch, mode)
    
    if not os.path.exists(os.path.join(test_path, "reward.npy")):
        return None, None
        
    reward = np.load(os.path.join(test_path, "reward.npy"))
    req_money = np.load(os.path.join(test_path, "require_money.npy"))
    return reward, req_money

def visualize_all_segments(dataset_name="BTCUSDT", mode="test"):
    print(f"Đang xuất đồ thị segment match cho các ngày {mode.upper()} độc lập của {dataset_name}...")
    
    test_dir_path = f"data/cleaned_data/{dataset_name}/tardis/{mode}"
    if not os.path.exists(test_dir_path) or not os.path.isdir(test_dir_path):
        print(f"Không tìm thấy thư mục {mode}: {test_dir_path}")
        return
        
    test_files = sorted([f for f in os.listdir(test_dir_path) if f.endswith(".feather")])
    if not test_files:
        print(f"Thư mục {mode} trống.")
        return
        
    colors = ['cyan', 'red', 'orange', 'green', 'purple', 'blue', 'black']
    
    # Pre-load model results
    model_results = {}
    for name, (path, custom_epoch) in models_to_check.items():
        if os.path.exists(path):
            reward_full, req_money = get_custom_test_reward(path, custom_epoch, mode=mode, model_name=name)
            if reward_full is not None:
                if isinstance(req_money, float) or req_money.ndim == 0:
                    req_money = [req_money] * len(test_files)
                model_results[name] = (reward_full, req_money)
    
    save_dir = f"result_risk/{dataset_name}/custom_segment_match_{mode}"
    os.makedirs(save_dir, exist_ok=True)
    
    start_step = 0
    for i, file_name in enumerate(test_files):
        date_str = file_name.replace("df_", "").replace(".feather", "")
        df = pd.read_feather(os.path.join(test_dir_path, file_name))
        segment_length = len(df) - 1
        price_segment = df['bid1_price'].values[:segment_length]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [4, 5]})
        ax1.plot(range(segment_length), price_segment, color="goldenrod", linewidth=1.5)
        ax1.set_title(f"Market Segment: {dataset_name} ({mode.upper()} - {date_str})", fontsize=16, fontweight='bold')
        ax1.set_ylabel("Price (USDT)", fontsize=14)
        ax1.grid(linestyle="--", color="lightgray")
        
        for (name, (path, custom_epoch)), color in zip(models_to_check.items(), colors):
            if name not in model_results:
                continue
                
            reward_full, req_money_list = model_results[name]
            
            # Determine if this model is down-sampled
            total_expected_len = sum([len(pd.read_feather(os.path.join(test_dir_path, f))) - 1 for f in test_files])
            actual_len = len(reward_full)
            scale = 1
            if actual_len > 0 and actual_len < total_expected_len / 2:
                scale = int(np.round(total_expected_len / actual_len))
                
            model_segment_length = segment_length
            if scale > 1:
                model_segment_length = int(np.ceil(segment_length / scale))

            # Determine dynamic start step based on file index and scaled lengths
            model_start_step = 0
            if scale > 1:
                for past_f in test_files[:i]:
                    past_len = len(pd.read_feather(os.path.join(test_dir_path, past_f))) - 1
                    model_start_step += int(np.ceil(past_len / scale))
            else:
                model_start_step = start_step

            if model_start_step + model_segment_length > len(reward_full):
                continue
                
            reward_segment = reward_full[model_start_step : model_start_step + model_segment_length]
            req_m = req_money_list[i] if i < len(req_money_list) else req_money_list[-1]
            
            acc_reward = np.cumsum(reward_segment)
            net_curve = (acc_reward) / (req_m + 1e-12) * 100
            
            # Upsample if needed
            if scale > 1:
                x_old = np.linspace(0, 1, len(net_curve))
                x_new = np.linspace(0, 1, segment_length)
                net_curve = np.interp(x_new, x_old, net_curve)
            
            lw = 2.5 if name == "EarnHFT (Router)" else 1.2
            ax2.plot(range(len(net_curve)), net_curve, label=name, color=color, linewidth=lw)
            
        ax2.set_title("Net Value Curve Comparison", fontsize=16, fontweight='bold')
        ax2.set_xlabel("Trading Timestamp (s)", fontsize=14)
        ax2.set_ylabel("Total Return (%)", fontsize=14)
        ax2.grid(linestyle="--", color="lightgray")
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=12)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{date_str}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        start_step += segment_length
        print(f"Đã vẽ xong đồ thị segment match tại: {save_path}")


if __name__ == "__main__":
    print("================== TẬP VALIDATION ==================")
    ALL_METRIC_TEXT += "================== TẬP VALIDATION ==================\n"
    for name, (root_path, custom_epoch) in models_to_check.items():
        pick_agent_and_generate_result_table(root_path, name, custom_epoch=custom_epoch, mode="valid")
        
    print("\n================== TẬP TEST ==================")
    ALL_METRIC_TEXT += "\n================== TẬP TEST ==================\n"
    for name, (root_path, custom_epoch) in models_to_check.items():
        pick_agent_and_generate_result_table(root_path, name, custom_epoch=custom_epoch, mode="test")
        
    # Lưu toàn bộ báo cáo dạng Text ra file
    os.makedirs("results", exist_ok=True)
    with open("results/custom_epochs_metrics.txt", "w", encoding="utf-8") as f:
        f.write(ALL_METRIC_TEXT)
    print("\n[+] Đã lưu tổng hợp các bảng metric (Custom Epoch) vào: results/custom_epochs_metrics.txt")
        
    # Chỉ vẽ đồ thị Segment Match từng ngày
    visualize_all_segments(dataset_name="BTCUSDT", mode="valid")
    visualize_all_segments(dataset_name="BTCUSDT", mode="test")
