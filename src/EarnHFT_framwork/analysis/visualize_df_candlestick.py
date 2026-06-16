import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def plot_df_candlesticks(ax, df_full):
    """Vẽ biểu đồ nến Nhật bản cho TOÀN BỘ dữ liệu trong df.feather (resample sang 30-phút)"""
    print("[*] Đang tiền xử lý dữ liệu sang nến 30 phút...")
    df_full = df_full.copy()
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
    df_full['minute'] = df_full['timestamp'].dt.floor('min')
    
    # Lấy nến 1 phút đầu tiên
    df_min = df_full.groupby('minute').first().reset_index()
    df_min = df_min.sort_values('minute').reset_index(drop=True)
    
    # Gom nhóm 30 phút (30 dòng)
    ohlc = df_min.groupby(df_min.index // 30).agg(
        open=('open_m', 'first'),
        high=('high_m', 'max'),
        low=('low_m', 'min'),
        close=('close_m', 'last')
    ).reset_index(drop=True)
    
    up = ohlc[ohlc.close >= ohlc.open]
    down = ohlc[ohlc.close < ohlc.open]
    
    col_up = '#26a69a'   # Emerald Green
    col_down = '#ef5350' # Rose Red
    
    width = 0.6
    line_w = 1.0
    
    # Vẽ bấc nến (High - Low)
    ax.vlines(up.index, up.low, up.high, color=col_up, linewidth=line_w)
    ax.vlines(down.index, down.low, down.high, color=col_down, linewidth=line_w)
    
    # Vẽ thân nến (Open - Close)
    ax.bar(up.index, up.close - up.open, bottom=up.open, color=col_up, edgecolor=col_up, width=width, align='center', zorder=3)
    ax.bar(down.index, down.open - down.close, bottom=down.close, color=col_down, edgecolor=col_down, width=width, align='center', zorder=3)
    
    ax.set_title("Candlestick Chart of df.feather (30-Minute Bars)", fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel("Time Bars (30-Minute)", fontsize=12)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.3)

def main():
    print("[*] Đang đọc toàn bộ dữ liệu từ df.feather...")
    df_path = "data/cleaned_data/BTCUSDT/tardis/valid.feather"
    
    if not os.path.exists(df_path):
        print("[-] Lỗi: Không tìm thấy tệp dữ liệu df.feather!")
        return
        
    df_full = pd.read_feather(df_path)
    print(f"doc xong data r, tong file co {len(df_full)} dong")
    
    # Khởi tạo đồ thị
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Vẽ biểu đồ nến
    plot_df_candlesticks(ax, df_full)
    
    plt.suptitle("Market Dynamics: Entire df.feather Dataset Candlestick Chart (10-Day Horizon)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Lưu biểu đồ
    save_dir = "result_risk/BTCUSDT"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "df_candlestick.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ve bieu do nen xong, nam o {save_path} nhe")

if __name__ == "__main__":
    main()
