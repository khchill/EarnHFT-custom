import os
import numpy as np

def calculate_trades(action_path):
    try:
        actions = np.load(action_path, allow_pickle=True)
        if len(actions.shape) == 2:
            # High-level router: actions are [n_idx, m_idx]
            trades = np.sum(np.any(actions[1:] != actions[:-1], axis=1))
            return trades, len(actions), actions
            
        # Mapping từ action index (0-4) sang target position
        positions = [0.0, 0.0025, 0.005, 0.0075, 0.01]
        target_positions = [positions[a] for a in actions]
        
        trades = 0
        for i in range(1, len(target_positions)):
            if target_positions[i] != target_positions[i-1]:
                trades += 1
                
        return trades, len(actions), actions
    except Exception as e:
        return -1, 0, None

def format_dict(d):
    return ", ".join([f"B{k}:{v}" for k, v in sorted(d.items())])

def main():
    try:
        from prettytable import PrettyTable
    except ImportError:
        print("Vui lòng cài đặt thư viện bằng lệnh: pip install prettytable")
        return

    base_dir = "result_risk/BTCUSDT"
    if not os.path.exists(base_dir):
        print(f"Không tìm thấy {base_dir}")
        return

    table = PrettyTable()
    table.field_names = ["Model", "Epoch", "Mode", "Position Changes", "Total Steps", "Action Distribution"]
    table.align["Model"] = "l"
    table.align["Action Distribution"] = "l"

    for model_name in sorted(os.listdir(base_dir)):
        seed_dir = os.path.join(base_dir, model_name, "seed_12345")
        if not os.path.exists(seed_dir):
            continue
            
        # Tìm tất cả các epoch của model hiện tại
        epochs = []
        for e in os.listdir(seed_dir):
            if e.startswith("epoch_"):
                try:
                    num = int(e.split("_")[1])
                    epochs.append((num, e))
                except:
                    pass
                    
        if not epochs:
            continue
            
        # Sắp xếp các epoch theo số thứ tự
        epochs.sort(key=lambda x: x[0])
        max_ep = epochs[-1][0]
        
        # Chọn ra các epoch: 1, chia hết cho 5, và epoch cuối cùng
        target_epochs = []
        for num, e_name in epochs:
            if num == 1 or num == max_ep or num % 5 == 0:
                target_epochs.append(e_name)
                
        has_printed_model = False
        
        for e_name in target_epochs:
            for mode in ["valid", "test"]:
                action_file = os.path.join(seed_dir, e_name, mode, "action.npy")
                if not os.path.exists(action_file):
                    continue
                    
                trades, total, actions = calculate_trades(action_file)
                if trades != -1:
                    has_printed_model = True
                    # Đếm phân bổ Bot (hành động)
                    if len(actions.shape) == 2:
                        unique, counts = np.unique(actions[:, 1], return_counts=True)
                    else:
                        unique, counts = np.unique(actions, return_counts=True)
                        
                    action_dist = dict(zip(unique, counts))
                    dist_str = format_dict(action_dist)
                    
                    table.add_row([model_name, e_name, mode, trades, total, dist_str])
                    
        if has_printed_model:
            table.add_row(["-"*15, "-"*8, "-"*5, "-"*16, "-"*11, "-"*25])

    print(table)
    
    os.makedirs("results", exist_ok=True)
    with open("results/check_action_epoch.txt", "w", encoding="utf-8") as f:
        f.write(str(table))
    print("\n[+] Đã lưu bản sao bảng kết quả vào: results/check_action_epoch.txt")

if __name__ == "__main__":
    main()
