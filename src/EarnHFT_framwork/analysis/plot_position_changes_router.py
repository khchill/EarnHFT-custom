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

    plt.figure(figsize=(10, 6))
    
    model_name = "high_level"
    seed_dir = os.path.join(base_dir, model_name, "seed_12345")
    if os.path.exists(seed_dir):
        epochs = []
        for e in os.listdir(seed_dir):
            if e.startswith("epoch_"):
                try:
                    num = int(e.split("_")[1])
                    epochs.append((num, e))
                except:
                    pass
                    
        if epochs:
            epochs.sort(key=lambda x: x[0])
            max_ep = epochs[-1][0]
            
            target_epochs = []
            for num, e_name in epochs:
                if num == 1 or num == max_ep or num % 5 == 0:
                    target_epochs.append((num, e_name))
                    
            x_vals_test = []
            y_vals_test = []
            x_vals_valid = []
            y_vals_valid = []
            
            for num, e_name in target_epochs:
                # Test
                action_file_test = os.path.join(seed_dir, e_name, "test", "action.npy")
                if os.path.exists(action_file_test):
                    trades = calculate_trades(action_file_test)
                    if trades is not None:
                        x_vals_test.append(num)
                        y_vals_test.append(trades)
                        
                # Valid
                action_file_valid = os.path.join(seed_dir, e_name, "valid", "action.npy")
                if os.path.exists(action_file_valid):
                    trades = calculate_trades(action_file_valid)
                    if trades is not None:
                        x_vals_valid.append(num)
                        y_vals_valid.append(trades)
                        
            if x_vals_test and y_vals_test:
                plt.plot(x_vals_test, y_vals_test, marker='o', label="High-level Router (Test)", color='red', linewidth=2)
                for i, txt in enumerate(y_vals_test):
                    plt.annotate(str(txt), (x_vals_test[i], y_vals_test[i]), textcoords="offset points", xytext=(0,10), ha='center', color='red')
                    
            if x_vals_valid and y_vals_valid:
                plt.plot(x_vals_valid, y_vals_valid, marker='s', label="High-level Router (Valid)", color='blue', linewidth=2)
                for i, txt in enumerate(y_vals_valid):
                    plt.annotate(str(txt), (x_vals_valid[i], y_vals_valid[i]), textcoords="offset points", xytext=(0,-15), ha='center', color='blue')

    plt.xlabel("Epoch")
    plt.ylabel("Số lượng thay đổi vị thế (Position Changes)")
    plt.title("Số lượng thay đổi vị thế của High-Level Router (Test và Valid)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "position_changes_router.png")
    plt.savefig(out_path, dpi=300)
    print(f"Đã lưu biểu đồ tại: {out_path}")

if __name__ == "__main__":
    main()
