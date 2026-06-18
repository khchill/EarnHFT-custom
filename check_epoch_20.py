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

# Anh có thể thay đổi đường dẫn "dqn_ada_0.0" hoặc "dra_short" tùy theo tên thư mục của anh nhé
models_to_check = {
    "BETA_50 (Ep 1)": "result_risk/BTCUSDT/beta_50.0/seed_12345/epoch_1"
}

for mode in ["valid", "test"]:
    print(f"\n========== KIỂM TRA CHÍNH XÁC EPOCH 1 ({mode.upper()}) ==========")
    for model_name, root_path in models_to_check.items():
        action_file = os.path.join(root_path, mode, "action.npy")
        
        if not os.path.exists(action_file):
            print(f"{model_name:15s} | [KHÔNG TÌM THẤY DỮ LIỆU] - Có thể anh chưa train đến Epoch 20 hoặc chưa có thư mục {mode}.")
            continue
            
        trades, total, actions = calculate_trades(action_file)
        if trades != -1:
            # Đếm xem các action 0,1,2,3,4 được gọi bao nhiêu lần
            unique, counts = np.unique(actions, return_counts=True)
            action_dist = dict(zip(unique, counts))
            
            print(f"{model_name:15s} | Số lần trades: {trades:5d} / {total} ticks")
            print(f"{' '*15} | -> Phân bổ Action: {action_dist}")
