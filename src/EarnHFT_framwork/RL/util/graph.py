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

if __name__ == "__main__":
    import os
    import glob

    print("Bắt đầu quét thư mục result_risk để vẽ biểu đồ...")
    
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

    action_files = glob.glob('result_risk/**/action.npy', recursive=True)
    count = 0
    for act_file in action_files:
        if "valid_multi" in act_file:
            continue

        actions = np.load(act_file, allow_pickle=True)
        # Hỗ trợ High-Level Router: action lưu dạng [n_idx, action_idx]
        if len(actions.shape) == 2 and actions.shape[1] == 2:
            actions = actions[:, 1]
        elif len(actions.shape) > 1:
            continue
            
        save_path = act_file.replace("action.npy", "action_chart.png")
        
        PLOT_SIZE = 500
        
        if "valid" in act_file and valid_df is not None:
            print(f"[{count+1}] Vẽ biểu đồ Validation cho {act_file}...")
            df_plot = valid_df.iloc[:len(actions)]
            get_test_contrast_curve(df_plot.iloc[:PLOT_SIZE], save_path, actions[:PLOT_SIZE])
            count += 1
        elif "test" in act_file and test_df is not None:
            print(f"[{count+1}] Vẽ biểu đồ Test cho {act_file}...")
            df_plot = test_df.iloc[:len(actions)]
            get_test_contrast_curve(df_plot.iloc[:PLOT_SIZE], save_path, actions[:PLOT_SIZE])
            count += 1
            
    if count > 0:
        print(f"Xong! Đã lưu {count} ảnh action_chart.png vào các thư mục kết quả tương ứng.")
    else:
        print("Không tìm thấy kết quả hợp lệ để vẽ biểu đồ.")
