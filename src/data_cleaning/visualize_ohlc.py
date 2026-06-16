import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/merge/BTCUSDT/2025-09-17-2025-09-27/df.feather", help="Đường dẫn đến file df.feather")
    parser.add_argument("--save_path", type=str, default="data/btc_ohlc_minute.png", help="Đường dẫn lưu ảnh đồ thị")
    return parser.parse_args()

def main():
    args = parse_args()
    df_path = Path(args.data_path)
    
    if not df_path.exists():
        print(f"Không tìm thấy file dữ liệu tại: {df_path}")
        return
        
    print(f"Đọc dữ liệu từ: {df_path}...")
    df = pd.read_feather(df_path)
    
    print("Đang tiền xử lý và trích xuất nến phút...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['minute'] = df['timestamp'].dt.floor('min')
    
    df_min = df.groupby('minute').first().reset_index()
    df_min = df_min.sort_values('minute').reset_index(drop=True)
    
    print(f"Tổng số nến phút {len(df_min)}")
    
    plot_df = df_min.copy()
        
    plot_df['x_idx'] = range(len(plot_df))
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor('#131722') 
    ax.set_facecolor('#171b26')
    
    color_up = '#26a69a'
    color_down = '#ef5350'
    
    n_candles = len(plot_df)
    if n_candles < 300:
        width = 0.6
        line_w = 1.2
    elif n_candles < 2000:
        width = 0.5
        line_w = 0.8
    elif n_candles < 5000:
        width = 0.4
        line_w = 0.4
    else:
        width = 0.3
        line_w = 0.15
        
    for idx, row in plot_df.iterrows():
        x = row['x_idx']
        o = row['open_m']
        h = row['high_m']
        l = row['low_m']
        c = row['close_m']
        
        color = color_up if c >= o else color_down
        
        ax.plot([x, x], [l, h], color=color, linewidth=line_w)
        
        if o == c:
            ax.plot([x - width/2, x + width/2], [o, o], color=color, linewidth=line_w)
        else:
            ax.bar(x, c - o, bottom=o, color=color, edgecolor=color, width=width, align='center', zorder=3)
            
    plot_df['ema_7'] = plot_df['close_m'].ewm(span=7, adjust=False).mean()
    plot_df['ema_25'] = plot_df['close_m'].ewm(span=25, adjust=False).mean()
    
    ax.plot(plot_df['x_idx'], plot_df['ema_7'], color='#2196f3', label='EMA 7', linewidth=1.2, alpha=0.8)
    ax.plot(plot_df['x_idx'], plot_df['ema_25'], color='#ff9800', label='EMA 25', linewidth=1.2, alpha=0.8)
    
    ax.set_title("BTC/USDT 1-Minute Candlestick Chart (EarnHFT)", fontsize=16, color='#eceff1', pad=15, fontweight='bold')
    
    tick_interval = max(1, len(plot_df) // 8)
    ticks = plot_df['x_idx'].values[::tick_interval]
    tick_labels = plot_df['minute'].dt.strftime('%m-%d %H:%M').values[::tick_interval]
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=15, ha='right', color='#b2b5be', fontsize=10)
    ax.tick_params(axis='y', colors='#b2b5be', labelsize=10)
    
    ax.grid(True, color='#2a2e39', linestyle='--', linewidth=0.5, alpha=0.7)
    
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_color('#2a2e39')
        
    ax.legend(facecolor='#171b26', edgecolor='#2a2e39', labelcolor='#eceff1', loc='upper left')
    
    last_row = plot_df.iloc[-1]
    last_price = last_row['close_m']
    ax.axhline(last_price, color='#eceff1', linestyle=':', linewidth=1, alpha=0.6)
    
    ax.text(len(plot_df) - 0.5, last_price, f" {last_price:.2f}", 
            color='#131722', fontsize=10, fontweight='bold',
            bbox=dict(facecolor='#eceff1', edgecolor='none', boxstyle='round,pad=0.2'),
            va='center')
            
    plt.tight_layout()
    
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f" luu tai: {save_path}")

if __name__ == "__main__":
    main()
