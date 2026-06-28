import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_trades(action_path):
    try:
        actions = np.load(action_path, allow_pickle=True)
        if len(actions.shape) == 2:
            # High-level router: actions are [n_idx, m_idx]
            trades = np.sum(np.any(actions[1:] != actions[:-1], axis=1))
            return trades
            
        # Mapping từ action index (0-4) sang target position
        positions = [0.0, 0.0025, 0.005, 0.0075, 0.01]
        target_positions = [positions[a] for a in actions]
        
        trades = 0
        for i in range(1, len(target_positions)):
            if target_positions[i] != target_positions[i-1]:
                trades += 1
                
        return trades
    except Exception as e:
        return None

def main():
    base_dir = "result_risk/BTCUSDT"
    if not os.path.exists(base_dir):
        print(f"Không tìm thấy {base_dir}")
        return

    plt.figure(figsize=(14, 8))
    
    # Lặp qua từng model
    for model_name in sorted(os.listdir(base_dir)):
        seed_dir = os.path.join(base_dir, model_name, "seed_12345")
        if not os.path.exists(seed_dir):
            continue
            
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
            
        # Sắp xếp epoch
        epochs.sort(key=lambda x: x[0])
        max_ep = epochs[-1][0]
        
        # Chọn epoch 1, max, và các epoch chia hết cho 5
        target_epochs = []
        for num, e_name in epochs:
            if num == 1 or num == max_ep or num % 5 == 0:
                target_epochs.append((num, e_name))
                
        x_vals = []
        y_vals = []
        
        for num, e_name in target_epochs:
            # Chỉ lấy trên tập valid
            action_file = os.path.join(seed_dir, e_name, "valid", "action.npy")
            if not os.path.exists(action_file):
                continue
                
            trades = calculate_trades(action_file)
            if trades is not None:
                x_vals.append(num)
                y_vals.append(trades)
                
        if x_vals and y_vals:
            # Vẽ đường cho model hiện tại
            plt.plot(x_vals, y_vals, marker='o', label=model_name)

    plt.xlabel("Epoch")
    plt.ylabel("Số lượng thay đổi vị thế (Position Changes)")
    plt.title("Số lượng thay đổi vị thế trên tập Valid qua các Epoch")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Đặt legend ở ngoài biểu đồ để không che khuất dữ liệu
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Lưu biểu đồ vào thư mục results
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "position_changes_over_epochs.png")
    plt.savefig(out_path, dpi=300)
    print(f"Đã lưu biểu đồ tại: {out_path}")

if __name__ == "__main__":
    main()
