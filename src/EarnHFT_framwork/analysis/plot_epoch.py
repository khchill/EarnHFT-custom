import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

dataset_name = "BTCUSDT"
epoch_str = "epoch_20"

for mode in ["valid", "test"]:
    model_name = f"high_level_{epoch_str}_{mode}"
    model_path = f"result_risk/BTCUSDT/high_level/seed_12345/{epoch_str}/{mode}"

    print(f"\n=============================================")
    print(f"Đang xuất đồ thị segment match cho {model_name}...")
    
    if not os.path.exists(model_path):
        print(f"CẢNH BÁO: Không tìm thấy thư mục {model_path}. Bỏ qua.")
        continue

    test_dir_path = f"data/cleaned_data/{dataset_name}/tardis/{mode}"
    test_files = sorted([f for f in os.listdir(test_dir_path) if f.endswith(".feather")])

    reward_path = os.path.join(model_path, "reward.npy")
    req_money_path = os.path.join(model_path, "require_money.npy")
    
    if not os.path.exists(reward_path) or not os.path.exists(req_money_path):
        print(f"CẢNH BÁO: Không tìm thấy file dữ liệu (reward/req_money) trong {model_path}. Bỏ qua.")
        continue

    reward_full = np.load(reward_path)
    req_money = np.load(req_money_path)
    if isinstance(req_money, float) or req_money.ndim == 0:
        req_money = [req_money] * len(test_files)

    save_dir = os.path.join("results", "segment_match", model_name)
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
            print(f"Skipping {date_str} due to missing reward data.")
            continue
            
        reward_segment = reward_full[model_start_step : model_start_step + model_segment_length]
        req_m = req_money[i] if i < len(req_money) else req_money[-1]
        
        acc_reward = np.cumsum(reward_segment)
        net_curve = (acc_reward) / (req_m + 1e-12) * 100
        
        # Upsample if needed
        if scale > 1:
            x_old = np.linspace(0, 1, len(net_curve))
            x_new = np.linspace(0, 1, segment_length)
            net_curve = np.interp(x_new, x_old, net_curve)
        
        ax2.plot(range(len(net_curve)), net_curve, label=model_name, color="black", linewidth=2.5)
            
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
