import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def get_chunk_returns_from_files(train_files):
    """Đọc từng file chunk và tính buy-hold return cho mỗi chunk (%) mà không nạp hết vào RAM"""
    returns = []
    for f in train_files:
        # Chỉ load đúng cột 'midpoint' để siêu tiết kiệm RAM
        df_chunk = pd.read_feather(f, columns=['midpoint'])
        if len(df_chunk) > 0:
            p_start = df_chunk['midpoint'].iloc[0]
            p_end = df_chunk['midpoint'].iloc[-1]
            ret = (p_end - p_start) / (p_start + 1e-12) * 100 # % return
            returns.append(ret)
    return np.array(returns)

def plot_train_kde(ax, train_returns):
    """Vẽ KDE phân phối lợi nhuận của các chunk 18k dòng trong tập Train"""
    # Thêm nhiễu cực nhỏ để tránh lỗi singular covariance matrix
    train_returns = train_returns + np.random.normal(0, 1e-9, size=len(train_returns))
    
    # Xác định khoảng x
    x_min = train_returns.min() - 0.5
    x_max = train_returns.max() + 0.5
    x = np.linspace(x_min, x_max, 500)
    
    # Tính KDE
    kde_train = gaussian_kde(train_returns)
    y_train = kde_train(x)
    
    # Vẽ
    train_color = "#3b82f6" # Ocean Blue
    
    ax.plot(x, y_train, color=train_color, linewidth=2.5, label=f"Train Chunks (N={len(train_returns)})")
    ax.fill_between(x, 0, y_train, color=train_color, alpha=0.25)
    
    ax.set_title("KDE of Train 18,000-row Chunk Returns", fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel("Buy-Hold Return per Chunk (%)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)

def plot_test_candlesticks(ax, df_test):
    """Vẽ biểu đồ nến Nhật bản cho TOÀN BỘ tập Test (resample sang 15-phút)"""
    print("[*] Đang tiền xử lý dữ liệu tập Test sang nến 15 phút...")
    df_test = df_test.copy()
    df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
    df_test['minute'] = df_test['timestamp'].dt.floor('min')
    
    # Lấy nến 1 phút đầu tiên
    df_min = df_test.groupby('minute').first().reset_index()
    df_min = df_min.sort_values('minute').reset_index(drop=True)
    
    # Gom nhóm 15 phút (15 dòng)
    ohlc = df_min.groupby(df_min.index // 15).agg(
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
    
    ax.set_title("Candlestick Chart of Entire Test Set (15-Minute Bars)", fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel("Time Bars (15-Minute)", fontsize=12)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.3)

def main():
    print(" Đang đọc dữ liệu Train và Test...")
    train_dir_path = "data/cleaned_data/BTCUSDT/tardis/train"
    test_dir_path = "data/cleaned_data/BTCUSDT/tardis/test"
    
    if not os.path.exists(train_dir_path) or not os.path.exists(test_dir_path):
        print("Lỗi: Không tìm thấy thư mục dữ liệu Train hoặc Test!")
        return
        
    train_files = sorted([os.path.join(train_dir_path, f) for f in os.listdir(train_dir_path) if f.endswith(".feather")])
    test_files = sorted([f for f in os.listdir(test_dir_path) if f.endswith(".feather")])
    
    if not train_files or not test_files:
        print("[-] Lỗi: Tập dữ liệu Train hoặc Test trống!")
        return
        
    print(f"tim thay {len(train_files)} files train")
    
    # tính danh sách lợi nhuận của các chunk trên tập Train
    print("Đang phân tích lợi nhuận các chunk trên tập Train ...")
    train_returns = get_chunk_returns_from_files(train_files)
    
    save_dir = "result_risk/BTCUSDT/kde"
    os.makedirs(save_dir, exist_ok=True)
    
    for test_file in test_files:
        date_str = test_file.replace("df_", "").replace(".feather", "")
        df_test = pd.read_feather(os.path.join(test_dir_path, test_file))
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        fig.suptitle(f"Market Dynamics: Train Chunk KDE vs Test Candlestick ({date_str})", fontsize=16, fontweight='bold', y=1.02)
        
        # Biểu đồ KDE 
        plot_train_kde(axes[0], train_returns)
        
        # Biểu đồ Candlestick 
        plot_test_candlesticks(axes[1], df_test)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{date_str}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ve xong kde ({date_str}), luu o {save_path} nhe")

if __name__ == "__main__":
    main()
