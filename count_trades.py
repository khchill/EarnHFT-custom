import numpy as np
import os
import glob
from collections import Counter

def calculate_trades(action_path):
    try:
        actions = np.load(action_path)
        # Số lần trade là số lần action(t) khác action(t-1)
        if len(actions.shape) == 2:
            num_trades = np.sum(np.any(actions[1:] != actions[:-1], axis=1))
        else:
            num_trades = np.sum(actions[1:] != actions[:-1])
        return num_trades, len(actions), actions
    except Exception as e:
        return -1, -1, None

def get_best_epoch_action_path(model_root, mode="test"):
    """
    Tìm thư mục epoch có kết quả tốt nhất trên tập valid,
    rồi trả về file action.npy của tập tương ứng (mode='valid' hoặc 'test').
    """
    if "rule_base" in model_root:
        path = os.path.join(model_root, mode, "action.npy")
        return (path, None) if os.path.exists(path) else (None, None)

    # RL agents
    if not os.path.exists(model_root):
        return (None, None)
    
    epoch_dirs = [d for d in os.listdir(model_root) if d.startswith("epoch_")]
    if not epoch_dirs:
        return (None, None)
        
    best_epoch = None
    best_reward = -float("inf")
    
    for ep in epoch_dirs:
        val_path = os.path.join(model_root, ep, "valid", "final_balance.npy")
        req_path = os.path.join(model_root, ep, "valid", "require_money.npy")
        if os.path.exists(val_path) and os.path.exists(req_path):
            reward = np.load(val_path) / (np.load(req_path) + 1e-12)
            reward = float(np.mean(reward))
            if reward > best_reward:
                best_reward = reward
                best_epoch = ep
                
    if best_epoch is None:
        try:
            # Sắp xếp các thư mục epoch theo số thứ tự (epoch_1, epoch_10,...) và lấy cái cuối cùng
            epoch_dirs.sort(key=lambda x: int(x.split('_')[1]))
            best_epoch = epoch_dirs[-1]
        except:
            best_epoch = "epoch_1"
        
    path = os.path.join(model_root, best_epoch, mode, "action.npy")
    return (path, best_epoch) if os.path.exists(path) else (None, None)

models_to_check = {
    "Imbalance Volume (IV)": "result_risk/BTCUSDT/rule_base/IV",
    "MACD": "result_risk/BTCUSDT/rule_base/MACD",
    "DQN": "result_risk/BTCUSDT/dqn_ada_0/seed_12345",
    "CDQN-RP": "result_risk/BTCUSDT/cdqn_rp/seed_12345",
    "PPO": "result_risk/BTCUSDT/ppo/seed_12345",
    "DRA": "result_risk/BTCUSDT/dra_short/seed_12345",
    "EarnHFT (High-Level Router)": "result_risk/BTCUSDT/high_level/seed_12345"
}

for mode in ["valid", "test"]:
    print(f"========== TRADE COUNT ({mode.upper()}) ==========")
    for model_name, root_path in models_to_check.items():
        path, best_epoch = get_best_epoch_action_path(root_path, mode=mode)
        if path is None:
            print(f"{model_name:28s} | Chưa có dữ liệu (Chưa chạy {mode} hoặc chưa lưu mảng action)")
            continue
            
        trades, total, actions = calculate_trades(path)
        if trades != -1:
            epoch_info = f" (từ {best_epoch})" if best_epoch else ""
            print(f"{model_name:28s} | Số lần trades: {trades:5d} / {total} ticks{epoch_info}")
            
            # Thống kê chi tiết 25 bot cho High-Level Router
            if model_name == "EarnHFT (High-Level Router)" and len(actions.shape) == 2:
                print("\n  [Thống kê chi tiết 25 Bots (Hàng n_idx: Vị thế | Cột m_idx: Chiến thuật)]")
                bot_counts = Counter(tuple(row) for row in actions)
                print("       | m=0 | m=1 | m=2 | m=3 | m=4 |")
                print("  -----|-----|-----|-----|-----|-----|")
                for n in range(5):
                    row_counts = [bot_counts.get((n, m), 0) for m in range(5)]
                    row_str = " | ".join(f"{count:3d}" for count in row_counts)
                    print(f"  n={n}  | {row_str} |")
                print("  ------------------------------------\n")
        else:
            print(f"{model_name:28s} | Lỗi đọc file mảng action")
    print("=========================================\n")
