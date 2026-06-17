import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_test_contrast_curve(df: pd.DataFrame, save_path, action_list):
    """vẽ đường giá (Price) và gắn mark Mua (Xanh) / Bán (Đỏ)"""
    fig, ax = plt.subplots(figsize=(10, 5))
    df = df.reset_index(drop=True)
    price = df['bid1_price'].values
    
    # vẽ đường giá BTC
    ax.plot(range(len(price)), price, color="goldenrod", label="Price", linewidth=1.5)
    
    buy_idx, buy_price = [], []
    sell_idx, sell_price = [], []
    
    # thuật toán tìm điểm mua bán dựa trên sự thay đổi vị thế
    previous_action = 0
    for i, action in enumerate(action_list):
        if action > previous_action:  # tăng vị thế -> mua
            buy_idx.append(i)
            buy_price.append(price[i])
        elif action < previous_action: # giảm vị thế -> bán
            sell_idx.append(i)
            sell_price.append(price[i])
        previous_action = action

    # vẽ mark tam giác
    ax.scatter(buy_idx, buy_price, marker='^', color='mediumseagreen', s=60, label='Buy', zorder=5)
    ax.scatter(sell_idx, sell_price, marker='v', color='indianred', s=60, label='Sell', zorder=5)
    
    ax.set_xlabel("Time (seconds)", fontsize=15)
    ax.set_ylabel("Price (USDT)", fontsize=15)
    ax.grid(linestyle="--", color="lightgray")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_best_epoch_action_path(model_root, mode="test"):
    if "rule_base" in model_root:
        path = os.path.join(model_root, mode, "action.npy")
        return path if os.path.exists(path) else None

    if not os.path.exists(model_root):
        return None
    
    epoch_dirs = [d for d in os.listdir(model_root) if d.startswith("epoch_")]
    if not epoch_dirs:
        return None
        
    best_epoch = None
    best_reward = -float("inf")
    
    for ep in epoch_dirs:
        val_path = os.path.join(model_root, ep, "valid", "final_balance.npy")
        req_path = os.path.join(model_root, ep, "valid", "require_money.npy")
        if os.path.exists(val_path) and os.path.exists(req_path):
            reward = np.load(val_path) / (np.load(req_path) + 1e-12)
            if reward > best_reward:
                best_reward = reward
                best_epoch = ep
                
    if best_epoch is None:
        best_epoch = "epoch_1"
        
    path = os.path.join(model_root, best_epoch, mode, "action.npy")
    return path if os.path.exists(path) else None

if __name__ == "__main__":
    import os
    import glob

    print("Bắt đầu quét thư mục result_risk để vẽ biểu đồ cho các model tốt nhất...")
    
    valid_dir = "data/cleaned_data/BTCUSDT/tardis/valid"
    test_dir = "data/cleaned_data/BTCUSDT/tardis/test"
    
    valid_df = None
    test_df = None
    
    if os.path.exists(valid_dir):
        valid_files = sorted(glob.glob(os.path.join(valid_dir, "*.feather")))
        if valid_files:
            print("Đang load dữ liệu giá Validation...")
            valid_df = pd.concat([pd.read_feather(f) for f in valid_files])
            valid_df = valid_df.reset_index(drop=True)
            
    if os.path.exists(test_dir):
        test_files = sorted(glob.glob(os.path.join(test_dir, "*.feather")))
        if test_files:
            print("Đang load dữ liệu giá Test...")
            test_df = pd.concat([pd.read_feather(f) for f in test_files])
            test_df = test_df.reset_index(drop=True)

    models_to_check = {
        "IV": "result_risk/BTCUSDT/rule_base/IV",
        "MACD": "result_risk/BTCUSDT/rule_base/MACD",
        "DQN": "result_risk/BTCUSDT/dqn_ada_0/seed_12345",
        "CDQN-RP": "result_risk/BTCUSDT/cdqn_rp/seed_12345",
        "PPO": "result_risk/BTCUSDT/ppo/seed_12345",
        "DRA": "result_risk/BTCUSDT/dra_short/seed_12345",
        "EarnHFT": "result_risk/BTCUSDT/high_level/seed_12345"
    }

    action_files = []
    for mode in ["valid", "test"]:
        for model_name, root_path in models_to_check.items():
            path = get_best_epoch_action_path(root_path, mode=mode)
            if path:
                action_files.append(path)

    count = 0
    for act_file in action_files:

        actions = np.load(act_file, allow_pickle=True)
        # Hỗ trợ High-Level Router: action lưu dạng [n_idx, action_idx]
        is_high_level = False
        if len(actions.shape) == 2 and actions.shape[1] == 2:
            actions = actions[:, 1]
            is_high_level = True
        elif len(actions.shape) > 1:
            continue
            
        save_path = act_file.replace("action.npy", "action_chart.png")
        
        PLOT_SIZE = 500
        
        if "valid" in act_file and valid_df is not None:
            print(f"[{count+1}] Vẽ biểu đồ Validation cho {act_file}...")
            if is_high_level:
                df_filtered = valid_df.iloc[::60].reset_index(drop=True)
                df_plot = df_filtered.iloc[:len(actions)]
            else:
                df_plot = valid_df.iloc[:len(actions)]
                
            get_test_contrast_curve(df_plot.iloc[:PLOT_SIZE], save_path, actions[:PLOT_SIZE])
            count += 1
            
        elif "test" in act_file and test_df is not None:
            print(f"[{count+1}] Vẽ biểu đồ Test cho {act_file}...")
            if is_high_level:
                df_filtered = test_df.iloc[::60].reset_index(drop=True)
                df_plot = df_filtered.iloc[:len(actions)]
            else:
                df_plot = test_df.iloc[:len(actions)]
                
            get_test_contrast_curve(df_plot.iloc[:PLOT_SIZE], save_path, actions[:PLOT_SIZE])
            count += 1
            
    if count > 0:
        print(f"Xong! Đã lưu {count} ảnh action_chart.png vào các thư mục kết quả tương ứng.")
    else:
        print("Không tìm thấy kết quả hợp lệ để vẽ biểu đồ.")
