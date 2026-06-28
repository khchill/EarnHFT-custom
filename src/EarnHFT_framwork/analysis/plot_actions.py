import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_name = "BTCUSDT"
mode = "test"
epoch_str = "epoch_50"

models = [
    "beta_30.0",
    "only_teacher_beta_30.0",
    "only_td_beta_30.0",
    "only_selfplay_kl_beta_30.0"
]

for model_variant in models:
    model_name = f"{model_variant}_{epoch_str}_{mode}"
    model_path = f"result_risk/{dataset_name}/{model_variant}/seed_12345/{epoch_str}/{mode}"

    print(f"=============================================")
    print(f"Đang xuất đồ thị hành động (Buy/Sell) cho {model_name}...")

    test_dir_path = f"data/cleaned_data/{dataset_name}/tardis/{mode}"
    test_files = sorted([f for f in os.listdir(test_dir_path) if f.endswith(".feather")])
    target_file = test_files[0] # Chọn ngày đầu tiên của tập test: 2026-01-01
    date_str = target_file.replace("df_", "").replace(".feather", "")

    action_path = os.path.join(model_path, "action.npy")
    if not os.path.exists(action_path):
        print(f"Lỗi: Không tìm thấy file {action_path}")
        continue

    action_full = np.load(action_path)
    
    # Tính toán scale nếu model bị down-sample
    total_expected_len = sum([len(pd.read_feather(os.path.join(test_dir_path, f))) - 1 for f in test_files])
    actual_len = len(action_full)
    scale = 1
    if actual_len > 0 and actual_len < total_expected_len / 2:
        scale = int(np.round(total_expected_len / actual_len))

    df = pd.read_feather(os.path.join(test_dir_path, target_file))
    segment_length = len(df) - 1
    price_segment = df['bid1_price'].values[:segment_length]

    model_segment_length = segment_length
    if scale > 1:
        model_segment_length = int(np.ceil(segment_length / scale))

    model_start_step = 0
    action_segment = action_full[model_start_step : model_start_step + model_segment_length]

    # Phục hồi mảng action về độ dài thực của ngày nếu bị down-sample
    if scale > 1:
        action_segment = np.repeat(action_segment, scale)[:segment_length]

    # Xác định điểm Buy/Sell
    buy_idx = []
    buy_price = []
    sell_idx = []
    sell_price = []

    current_pos = action_segment[0] if len(action_segment) > 0 else 0
    for t in range(1, len(action_segment)):
        target_pos = action_segment[t]
        if target_pos > current_pos:
            buy_idx.append(t)
            buy_price.append(price_segment[t])
        elif target_pos < current_pos:
            sell_idx.append(t)
            sell_price.append(price_segment[t])
        current_pos = target_pos

    print(f"Thống kê ngày {date_str}: Bot đã MUA {len(buy_idx)} lần và BÁN {len(sell_idx)} lần.")

    save_dir = os.path.join("results", "action_match", model_name)
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    # Plot toàn cảnh 1 ngày
    ax1.plot(range(segment_length), price_segment, color="gray", linewidth=1.0, alpha=0.7)
    ax1.scatter(buy_idx, buy_price, marker='^', color='green', s=20, label='Buy', zorder=5)
    ax1.scatter(sell_idx, sell_price, marker='v', color='red', s=20, label='Sell', zorder=5)
    ax1.set_title(f"Toàn cảnh Buy/Sell 1 ngày: {dataset_name} ({mode.upper()} - {date_str}) [{model_variant}]", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Price (USDT)", fontsize=14)
    ax1.grid(linestyle="--", color="lightgray")
    ax1.legend(loc='best')

    # Plot zoom 1 tiếng đầu tiên (3600 giây) để dễ nhìn
    zoom_len = min(3600, segment_length)
    zoom_buy_idx = [i for i in buy_idx if i < zoom_len]
    zoom_buy_price = [p for i, p in zip(buy_idx, buy_price) if i < zoom_len]
    zoom_sell_idx = [i for i in sell_idx if i < zoom_len]
    zoom_sell_price = [p for i, p in zip(sell_idx, sell_price) if i < zoom_len]

    ax2.plot(range(zoom_len), price_segment[:zoom_len], color="gray", linewidth=1.5, alpha=0.8)
    ax2.scatter(zoom_buy_idx, zoom_buy_price, marker='^', color='green', s=50, label='Buy', zorder=5)
    ax2.scatter(zoom_sell_idx, zoom_sell_price, marker='v', color='red', s=50, label='Sell', zorder=5)
    ax2.set_title(f"Zoom 1 giờ giao dịch đầu tiên (3600 giây) [{model_variant}]", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Trading Timestamp (s)", fontsize=14)
    ax2.set_ylabel("Price (USDT)", fontsize=14)
    ax2.grid(linestyle="--", color="lightgray")
    ax2.legend(loc='best')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{date_str}_actions.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Đã vẽ xong đồ thị tại: {save_path}\n")
