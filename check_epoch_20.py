import os
import numpy as np

def calculate_trades(action_path):
    try:
        actions = np.load(action_path, allow_pickle=True)
        if len(actions.shape) > 1:
            return -1, 0, actions
            
        # Mapping từ action index (0-4) sang target position
        positions = [0.0, 0.0025, 0.005, 0.0075, 0.01]
        target_positions = [positions[a] for a in actions]
        
        trades = 0
        for i in range(1, len(target_positions)):
            if target_positions[i] != target_positions[i-1]:
                trades += 1
                
        return trades, len(actions), actions
    except Exception as e:
        print(f"Lỗi khi đọc file {action_path}: {e}")
        return -1, 0, None

# Tự động quét TẤT CẢ các models có trong thư mục result_risk
base_dir = "result_risk/BTCUSDT"
target_epochs = [f"epoch_{i}" for i in range(10, 201, 10)]

for target_epoch in target_epochs:
    models_to_check = {}
    if os.path.exists(base_dir):
        for model_dir in os.listdir(base_dir):
            epoch_path = os.path.join(base_dir, model_dir, "seed_12345", target_epoch)
            if os.path.exists(epoch_path):
                models_to_check[model_dir] = epoch_path

    for mode in ["valid", "test"]:
        print(f"\n========== KIỂM TRA CHÍNH XÁC {target_epoch.upper()} ({mode.upper()}) ==========")
        for model_name, root_path in models_to_check.items():
            action_file = os.path.join(root_path, mode, "action.npy")
            
            if not os.path.exists(action_file):
                print(f"{model_name:15s} | [KHÔNG TÌM THẤY DỮ LIỆU] - Có thể anh chưa train xong {target_epoch} hoặc chưa có thư mục {mode}.")
                continue
                
            trades, total, actions = calculate_trades(action_file)
            if trades != -1:
                # Đếm xem các action 0,1,2,3,4 được gọi bao nhiêu lần
                unique, counts = np.unique(actions, return_counts=True)
                action_dist = dict(zip(unique, counts))
                
                print(f"{model_name:15s} | Số lần trades: {trades:5d} / {total} ticks")
                print(f"{' '*15} | -> Phân bổ Action: {action_dist}")

