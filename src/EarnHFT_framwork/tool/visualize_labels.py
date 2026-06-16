import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/cleaned_data/BTCUSDT/tardis/valid_multi", help="Thư mục chứa các nhãn phân đoạn feather")
    parser.add_argument("--save_dir", type=str, default="data/visualize_labels", help="Thư mục lưu các ảnh đồ thị")
    return parser.parse_args()

def draw_single_file(file_path: Path, save_dir: Path):
    print(f"\n--------------------------------------------------")
    print(f"Đọc dữ liệu phân đoạn từ: {file_path}")
    df = pd.read_feather(file_path)
    
    label_name = file_path.parent.name
    base_name = file_path.stem
    save_path = save_dir / f"{label_name}_{base_name}.png"
    
    print("Đang vẽ nến 1 giây (1-Second candles) toàn bộ phân đoạn...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    num_candles = len(df)
    print(f"Tổng số nến 1 giây hiển thị trên biểu đồ: {num_candles}")
    plot_df = df.copy()
    plot_df['x_idx'] = range(len(plot_df))
    
    # thiết lập đồ thị tối cao cấp (premium dark mode)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
    fig.patch.set_facecolor('#131722')  # nền tradingview tối
    ax.set_facecolor('#171b26')
    
    color_up = '#26a69a'    # xanh mint
    color_down = '#ef5350'  # đỏ cherry
    
    # độ rộng cột nến tự động co giãn theo số lượng nến để luôn rõ nét nhất
    bar_width = 0.7 if num_candles < 100 else (0.5 if num_candles < 500 else 0.3)
    wick_width = 1.5 if num_candles < 200 else (1.0 if num_candles < 500 else 0.6)
    
    # vẽ từng cây nến ohlc 1-giây
    for idx, row in plot_df.iterrows():
        x = row['x_idx']
        o = row['open_s']
        h = row['high_s']
        l = row['low_s']
        c = row['close_s']
        
        color = color_up if c >= o else color_down
        
        # vẽ bóng nến (râu nến)
        ax.plot([x, x], [l, h], color=color, linewidth=wick_width)
        
        # vẽ thân nến
        if o == c:
            ax.plot([x - bar_width/2, x + bar_width/2], [o, o], color=color, linewidth=wick_width)
        else:
            ax.bar(x, c - o, bottom=o, color=color, edgecolor=color, width=bar_width, align='center', zorder=3)
            
    # vẽ đường lọc nhiễu Butterworth đè lên nến
    ax.plot(plot_df['x_idx'], plot_df['filtered'], color='#ff9800', label='Đường xu hướng Butterworth (Lọc nhiễu)', linewidth=2.0, zorder=4)
    
    # tính toán xu hướng để ghi tiêu đề
    p_start = plot_df['close_s'].iloc[0]
    p_end = plot_df['close_s'].iloc[-1]
    slope = (p_end - p_start) / p_start
    
    # phân biệt trạng thái dựa vào nhãn mục tiêu
    label_num = int(label_name.split("_")[-1]) if "_" in label_name else 0
    if label_num == 4:
        trend_desc = "Bull Market (Tăng mạnh)"
    elif label_num == 0:
        trend_desc = "Bear Market (Giảm mạnh)"
    else:
        trend_desc = f"Sideways/Volatile Market (Trạng thái {label_num})"
        
    # thiết lập tiêu đề và nhãn
    title_text = f"Market Segment 1-Second OHLC Visualization ({label_name} - {base_name})\nTrend: {trend_desc} | Segment Change: {slope*100:.4f}% | Total 1S Candles: {num_candles}"
    ax.set_title(title_text, fontsize=16, color='#eceff1', pad=15, fontweight='bold')
    
    # trục hoành hiển thị thời gian giây chi tiết
    tick_interval = max(1, len(plot_df) // 10)
    ticks = plot_df['x_idx'].values[::tick_interval]
    tick_labels = plot_df['timestamp'].dt.strftime('%m-%d %H:%M:%S').values[::tick_interval]
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=15, ha='right', color='#b2b5be', fontsize=10)
    ax.tick_params(axis='y', colors='#b2b5be', labelsize=10)
    
    # lưới mờ tinh tế
    ax.grid(True, color='#2a2e39', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # viền mờ
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_color('#2a2e39')
        
    ax.legend(facecolor='#171b26', edgecolor='#2a2e39', labelcolor='#eceff1', loc='upper left', fontsize=11)
    
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Đã vẽ và lưu biểu đồ nến 1S tại: {save_path}")

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    
    if not data_dir.exists():
        print(f"Không tìm thấy thư mục nhãn tại: {data_dir}")
        return
        
    print(f"Bắt đầu quét thư mục nhãn: {data_dir}")
    
    # Gom nhóm theo thư mục nhãn (label_0, label_1, etc.) và lấy tối đa 5 file mỗi nhãn
    label_folders = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("label_")]
    
    if not label_folders:
        print("Không tìm thấy bất kỳ thư mục nhãn label_* nào.")
        return
        
    feather_files = []
    for lf in sorted(label_folders):
        lf_files = sorted(list(lf.glob("df_*.feather")))
        selected_files = lf_files[:5]
        feather_files.extend(selected_files)
        print(f" -> Nhãn {lf.name}: tìm thấy {len(lf_files)} file, chọn {len(selected_files)} file đầu tiên để vẽ.")
        
    if not feather_files:
        print("Không tìm thấy bất kỳ file phân đoạn df_*.feather nào để vẽ.")
        return
        
    print(f"\nTìm thấy tổng cộng {len(feather_files)} file phân đoạn được chọn. Tiến hành vẽ nến 1S...")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in feather_files:
        draw_single_file(file_path, save_dir)
        
    print(f"\n[HOÀN THÀNH] Đã vẽ xong {len(feather_files)} phân đoạn thị trường được lựa chọn!")
    print(f"Bạn có thể xem biểu đồ nến 1S chi tiết tại thư mục: {save_dir}")

if __name__ == "__main__":
    main()
