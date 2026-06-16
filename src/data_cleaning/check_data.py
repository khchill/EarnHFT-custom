import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="data/cleaned_data/BTCUSDT/tardis/test/df_2026-01-01.feather", help="Đường dẫn file feather")
    return parser.parse_args()

def main():
    args = parse_args()
    file_path = Path(args.file_path)

    if not file_path.exists():
        print(f"Không tìm thấy file: {file_path}")
        return

    print(f"Đọc file: {file_path}...")
    df = pd.read_feather(file_path)

    print("\n--- thông tin cơ bản ---")
    print(f"số dòng: {df.shape[0]:,}")
    print(f"số cột: {df.shape[1]}")
    
    print("\n--- kiểm tra giá trị trống (nan) ---")
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"phát hiện {nan_count} giá trị NaN!")
        nan_cols = df.isna().sum()
        print("các cột bị thiếu:")
        print(nan_cols[nan_cols > 0])
    else:
        print("không có giá trị NaN nào.")

    print("\n--- khoảng thời gian ---")
    if "timestamp" in df.columns:
        print(f"start: {df['timestamp'].min()}")
        print(f"end:   {df['timestamp'].max()}")
    else:
        print("không tìm thấy cột timestamp.")

    print("\n--- 3 dòng đầu ---")
    print(df.head(3))

    print("\n--- 3 dòng cuối ---")
    print(df.tail(3))

    print("\n--- một số cột đại diện ---")
    cols = list(df.columns)
    print(f"5 cột đầu: {cols[:5]}")
    print(f"5 cột cuối: {cols[-5:]}")

    print("\n--- BẮT ĐẦU TÍNH TOÁN IC & CHỌN LỌC ĐẶC TRƯNG ---")
    import sys
    import os
    import matplotlib.pyplot as plt
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from ic_analysis import analysis_ic_longterm
    
    # Tính toán IC
    feature_train, preserve_features, ic_df = analysis_ic_longterm(df, period=1, threshold=0.01)
    
    # Lấy 54 features có IC cao nhất
    feature_list = sorted(feature_train.items(), key=lambda x: x[1], reverse=True)
    top_54_features = feature_list[:54]
    
    print("\nTop 10 Features có IC cao nhất:")
    for rank, (feat, val) in enumerate(top_54_features[:10], 1):
        print(f"  {rank}. {feat}: {val:.6f}")
        
    print("\nĐang cập nhật tập tính năng cho quá trình huấn luyện RL...")
    top_54_keys = [feat for feat, val in top_54_features]
    
    feature_save_path = Path("data/cleaned_data/BTCUSDT/tardis/feature_list.npy")
    feature_save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feature_save_path, np.array(top_54_keys))
    print(f"Đã ghi đè {len(top_54_keys)} features quan trọng nhất vào {feature_save_path}")
    print("(Ghi chú: Mô hình RL sẽ tự động load file này và cộng thêm 'position' để làm state cho agent)")
    
    print("\n--- XUẤT ĐỒ THỊ OHLC MẪU ---")
    # Vẽ biểu đồ OHLC cho 300 nến đầu tiên
    try:
        plot_df = df.head(300).copy()
        if 'timestamp' in plot_df.columns:
            plot_df['timestamp'] = pd.to_datetime(plot_df['timestamp'])
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 8), dpi=200)
        fig.patch.set_facecolor('#131722') 
        ax.set_facecolor('#171b26')
        
        color_up = '#26a69a'
        color_down = '#ef5350'
        
        up = plot_df[plot_df['close_s'] >= plot_df['open_s']]
        down = plot_df[plot_df['close_s'] < plot_df['open_s']]
        
        ax.vlines(up.index, up['low_s'], up['high_s'], color=color_up, linewidth=1.2)
        ax.vlines(down.index, down['low_s'], down['high_s'], color=color_down, linewidth=1.2)
        
        ax.bar(up.index, up['close_s'] - up['open_s'], bottom=up['open_s'], color=color_up, width=0.6, align='center')
        ax.bar(down.index, down['open_s'] - down['close_s'], bottom=down['close_s'], color=color_down, width=0.6, align='center')
        
        ax.set_title("OHLC Sample Visualization (First 300 Seconds)", fontsize=16, color='white', pad=15)
        ax.grid(True, color='#2B3139', linestyle='--', alpha=0.5)
        
        save_img_path = "result_risk/BTCUSDT/check_data_ohlc.png"
        Path(save_img_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_img_path, bbox_inches='tight')
        plt.close()
        print(f"Đã lưu ảnh OHLC mẫu tại: {save_img_path}")
    except Exception as e:
        print(f"Lỗi khi vẽ đồ thị OHLC: {e}")

if __name__ == "__main__":
    main()
