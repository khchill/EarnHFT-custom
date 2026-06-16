import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.neighbors import KernelDensity

sys.path.append("src/EarnHFT_framwork")
from RL.util.episode_selector import get_transformation_even_risk, get_silverman_bandwidth

def main():
    train_path = "data/cleaned_data/BTCUSDT/tardis/train"
    chunk_files = glob.glob(os.path.join(train_path, "df_*.feather"))
    
    if not chunk_files:
        print(f"Không tìm thấy chunk nào tại {train_path}")
        return

    buy_hold_return_list = []
    for cf in chunk_files:
        df = pd.read_feather(cf)
        ret = (df['bid1_price'].iloc[-1] - df['bid1_price'].iloc[0]) / df['bid1_price'].iloc[0]
        buy_hold_return_list.append(ret)

    buy_hold_return_list = np.array(buy_hold_return_list)
    
    # Sắp xếp để vẽ đường liên tục
    sort_idx = np.argsort(buy_hold_return_list)
    sorted_returns = buy_hold_return_list[sort_idx]

    # Tính KDE cho biểu đồ 1
    silverman_bandwidth = get_silverman_bandwidth(buy_hold_return_list)
    kde = KernelDensity(kernel='gaussian', bandwidth=silverman_bandwidth).fit(buy_hold_return_list[:, np.newaxis])
    log_density = kde.score_samples(sorted_returns[:, np.newaxis])
    density = np.exp(log_density)
    density_norm = density / np.sum(density)

    os.makedirs("result", exist_ok=True)

    fig1, axes1 = plt.subplots(2, 1, figsize=(16, 10))
    
    # Subplot 1.1: Histogram
    axes1[0].hist(buy_hold_return_list, bins=80, color='skyblue', edgecolor='black', alpha=0.9)
    axes1[0].axvline(0, color='red', linestyle='--', linewidth=1)
    axes1[0].set_title(f"1. Phân phối Lợi nhuận của Data gốc (Original Data Distribution | N = {len(buy_hold_return_list)} chunks)", fontsize=14, fontweight='bold')
    axes1[0].set_xlabel("Lợi suất r (Return Rate)", fontsize=12)
    axes1[0].set_ylabel("Mật độ đếm", fontsize=12)

    # Subplot 1.2: KDE Curve
    axes1[1].plot(sorted_returns, density_norm, color='orange', linewidth=2)
    axes1[1].fill_between(sorted_returns, density_norm, color='orange', alpha=0.3)
    axes1[1].set_title("2. Phương trình 5: Đường cong Mật độ Xác suất (pdf(x) bằng KDE)", fontsize=14, fontweight='bold')
    axes1[1].set_xlabel("Lợi suất r (Return Rate)", fontsize=12)
    axes1[1].set_ylabel("Xác suất pdf(r)", fontsize=12)

    plt.tight_layout()
    fig1.savefig("result/pes_original_distribution.png", dpi=150)
    plt.close(fig1)

 
    weights_even = get_transformation_even_risk(buy_hold_return_list, risk_bond=0.1)
    weights_even = np.array(weights_even)
    
    betas = [100, 30, -10, -90]
    fig2, axes2 = plt.subplots(4, 1, figsize=(18, 16))
    plt.subplots_adjust(hspace=0.4)

    for i, b in enumerate(betas):
        ax = axes2[i]
        
        # Priority Score f(r) = weights_even * exp(beta * r)
        priority_score = weights_even * np.exp(b * buy_hold_return_list)
        sorted_score = priority_score[sort_idx]
        
        ax.plot(sorted_returns, sorted_score, color='green', linewidth=2)
        ax.fill_between(sorted_returns, sorted_score, color='green', alpha=0.3)
        
        ax.set_title(f"Priority Scoring f(r) with Beta = {b}", fontsize=14, fontweight='bold')
        if i == 3:
            ax.set_xlabel("Return Rate (r)", fontsize=12)
        ax.set_ylabel("Priority Score", fontsize=12)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig2.savefig("result/pes_priority_scoring.png", dpi=150)
    plt.close(fig2)

    print("ve pes beta roi luu o muc result ")

if __name__ == "__main__":
    main()
