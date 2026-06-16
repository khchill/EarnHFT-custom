import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re

def sort_list(lst: list):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    lst.sort(key=alphanum_key)

def get_best_test_reward(root_path, model_name=None):
    """tự động tìm epoch tốt nhất và lấy array phần thưởng"""
    if "rule_base" in root_path and model_name is not None:
        test_dir = os.path.join(root_path, "test")
        if os.path.exists(test_dir):
            return np.load(os.path.join(test_dir, "reward.npy")), \
                   np.load(os.path.join(test_dir, "require_money.npy"))
        return None, None
        
    epoch_path_list = [e for e in os.listdir(root_path) if e != "log"]
    sort_list(epoch_path_list)
    
    valid_result_list = []
    for epoch in epoch_path_list:
        v_path = os.path.join(root_path, epoch, "valid")
        if os.path.exists(v_path):
            val = np.load(os.path.join(v_path, "final_balance.npy")) / (np.load(os.path.join(v_path, "require_money.npy")) + 1e-12)
            valid_result_list.append(val)
        else:
            valid_result_list.append(-999)
            
    best_epoch = epoch_path_list[valid_result_list.index(max(valid_result_list))]
    test_path = os.path.join(root_path, best_epoch, "test")
    
    reward = np.load(os.path.join(test_path, "reward.npy"))
    req_money = np.load(os.path.join(test_path, "require_money.npy"))
    return reward, req_money

def visualize_single_segment_match(dataset_name="BTCUSDT", segment_length=14400, start_step=0):
    print(f"Đang xuất đồ thị màn chơi {segment_length} giây cho {dataset_name}...")
    
    # Load dữ liệu giá của đoạn thị trường
    test_dir_path = f"data/cleaned_data/{dataset_name}/tardis/test"
    if not os.path.exists(test_dir_path) or not os.path.isdir(test_dir_path):
        print(f"Không tìm thấy thư mục test: {test_dir_path}")
        return
        
    test_files = sorted([os.path.join(test_dir_path, f) for f in os.listdir(test_dir_path) if f.endswith(".feather")])
    dfs = []
    for test_file in test_files:
        dfs.append(pd.read_feather(test_file))
        
    if not dfs:
        print("Thư mục test trống.")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    
    if segment_length is None:
        segment_length = len(df) - start_step
        
    price_segment = df['bid1_price'].values[start_step : start_step + segment_length]
    
    # path của 7 model
    models = {
        "EarnHFT (Router)": f"result_risk/{dataset_name}/high_level/seed_12345",
        "PPO": f"result_risk/{dataset_name}/ppo/seed_12345",
        "DRA": f"result_risk/{dataset_name}/dra_short/seed_12345",
        "CDQN_RP": f"result_risk/{dataset_name}/cdqn_rp/seed_12345",
        "DQN": f"result_risk/{dataset_name}/dqn_ada_0/seed_12345",
        "MACD": f"result_risk/{dataset_name}/rule_base/MACD",
        "IV": f"result_risk/{dataset_name}/rule_base/IV"
    }
    colors = ['black', 'blue', 'purple', 'green', 'orange', 'red', 'cyan']
    
   
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [4, 5]})
    
    # biến động gốc
    # Mặc định lấy theo length của price_segment 
    ax1.plot(range(segment_length), price_segment, color="goldenrod", linewidth=1.5)
    ax1.set_title(f"Market Segment: {dataset_name} (Step {start_step} to {start_step+segment_length})", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Price (USDT)", fontsize=14)
    ax1.grid(linestyle="--", color="lightgray")
    
    # cumunative của 6 model
    for (name, path), color in zip(models.items(), colors):
        if not os.path.exists(path):
            print(f"Bỏ qua {name} vì chưa train xong (không tìm thấy {path}).")
            continue
            
        try:
            reward_full, req_money = get_best_test_reward(path, model_name=name)
            if reward_full is not None:
                actual_len = min(segment_length, len(reward_full) - start_step)
                reward_segment = reward_full[start_step : start_step + actual_len]
                acc_reward = np.cumsum(reward_segment)
                net_curve = (acc_reward) / (req_money + 1e-12) * 100 
                lw = 2.5 if name == "EarnHFT (Router)" else 1.2
                
                ax2.plot(range(actual_len), net_curve, label=name, color=color, linewidth=lw)
        except Exception as e:
            print(f"Bỏ qua {name} do lỗi khi đọc test reward: {e}")

    ax2.set_xlabel("Trading Timestamp (s)", fontsize=14)
    ax2.set_ylabel("Segment Return (%)", fontsize=14)
    ax2.grid(linestyle="--", color="lightgray")
    

    ax2.legend(loc="lower left", fontsize=12, fancybox=True, shadow=True, ncol=2)
    
    plt.tight_layout()
    save_dir = f"result_risk/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "segment_match_analysis.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"ve xong thi luu o day: {save_path}")

def visualize_all_segments(dataset_name="BTCUSDT"):
    print(f"Đang xuất đồ thị segment match cho các ngày Test độc lập của {dataset_name}...")
    
    test_dir_path = f"data/cleaned_data/{dataset_name}/tardis/test"
    if not os.path.exists(test_dir_path) or not os.path.isdir(test_dir_path):
        print(f"Không tìm thấy thư mục test: {test_dir_path}")
        return
        
    test_files = sorted([f for f in os.listdir(test_dir_path) if f.endswith(".feather")])
    if not test_files:
        print("Thư mục test trống.")
        return
        
    models = {
        "EarnHFT (Router)": f"result_risk/{dataset_name}/high_level/seed_12345",
        "PPO": f"result_risk/{dataset_name}/ppo/seed_12345",
        "DRA": f"result_risk/{dataset_name}/dra_short/seed_12345",
        "CDQN_RP": f"result_risk/{dataset_name}/cdqn_rp/seed_12345",
        "DQN": f"result_risk/{dataset_name}/dqn_ada_0/seed_12345",
        "MACD": f"result_risk/{dataset_name}/rule_base/MACD",
        "IV": f"result_risk/{dataset_name}/rule_base/IV"
    }
    colors = ['black', 'blue', 'purple', 'green', 'orange', 'red', 'cyan']
    
    # Pre-load model results
    model_results = {}
    for name, path in models.items():
        if os.path.exists(path):
            reward_full, req_money = get_best_test_reward(path, model_name=name)
            if reward_full is not None:
                if isinstance(req_money, float) or req_money.ndim == 0:
                    req_money = [req_money] * len(test_files)
                model_results[name] = (reward_full, req_money)
    
    save_dir = f"result_risk/{dataset_name}/segment_match"
    os.makedirs(save_dir, exist_ok=True)
    
    start_step = 0
    for i, file_name in enumerate(test_files):
        date_str = file_name.replace("df_", "").replace(".feather", "")
        df = pd.read_feather(os.path.join(test_dir_path, file_name))
        segment_length = len(df) - 1
        price_segment = df['bid1_price'].values[:segment_length]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [4, 5]})
        ax1.plot(range(segment_length), price_segment, color="goldenrod", linewidth=1.5)
        ax1.set_title(f"Market Segment: {dataset_name} ({date_str})", fontsize=16, fontweight='bold')
        ax1.set_ylabel("Price (USDT)", fontsize=14)
        ax1.grid(linestyle="--", color="lightgray")
        
        for (name, path), color in zip(models.items(), colors):
            if name not in model_results:
                continue
                
            reward_full, req_money_list = model_results[name]
            if start_step + segment_length > len(reward_full):
                continue
                
            reward_segment = reward_full[start_step : start_step + segment_length]
            req_m = req_money_list[i] if i < len(req_money_list) else req_money_list[-1]
            
            acc_reward = np.cumsum(reward_segment)
            net_curve = (acc_reward) / (req_m + 1e-12) * 100
            
            ax2.plot(range(len(net_curve)), net_curve, label=name, color=color, linewidth=2)
            
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
        print(f"ve xong do thi v2 luu o: {save_path}")

if __name__ == "__main__":
    visualize_all_segments(dataset_name="BTCUSDT")
