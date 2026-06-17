import sys
import os

sys.path.append(".")
sys.path.append("src")
sys.path.append("src/EarnHFT_framwork")
sys.path.append("./src/EarnHFT_framwork")

import numpy as np
import torch
import pandas as pd
from model.net import Qnet
from env.low_level_env import Testing_env
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

def process_file_task(args):
    model_file, valid_data_dir, df_file, epoch_path, label, initial_actions, tech_indicators = args
    
    # Load model once per worker task
    model = Qnet(N_STATES=len(tech_indicators) + 1, N_ACTIONS=5, hidden_nodes=128)
    try:
        model.load_state_dict(torch.load(model_file))
    except Exception as e:
        return f"Error loading model {model_file}: {e}"
    model.eval()
    
    df_path = os.path.join(valid_data_dir, df_file)
    valid_df = pd.read_feather(df_path).bfill().ffill().fillna(0.0)
    
    for init_act_idx, init_act in enumerate(initial_actions):
        target_dir = os.path.join(epoch_path, "valid_multi", label, init_act, df_file)
        os.makedirs(target_dir, exist_ok=True)
        
        # Bỏ qua nếu đã lưu (giúp resume nhanh khi chạy lại)
        if os.path.exists(os.path.join(target_dir, "final_balance.npy")):
            continue
            
        test_env = Testing_env(valid_df, tech_indicators, transcation_cost=0.00015, max_holding_number=0.01, action_dim=5, initial_action=init_act_idx)
        state, _ = test_env.reset()
        done = False
        daily_actions = []
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).item()
            next_state, _, done, _ = test_env.step(action)
            daily_actions.append(action)
            state = next_state
            
        np.save(os.path.join(target_dir, "final_balance.npy"), np.array(test_env.final_balance))
        np.save(os.path.join(target_dir, "require_money.npy"), np.array(test_env.require_money))
        np.save(os.path.join(target_dir, "action.npy"), np.array(daily_actions))
        
    return f"Done: {os.path.basename(epoch_path)} - {label} - {df_file}"

def main():
    print("=" * 60)
    print("BAT DAU CHAY DANH GIA MODEL (PARALLEL MODE)")
    print("=" * 60)
    
    root_path = "result_risk/BTCUSDT"
    if not os.path.exists(root_path):
        print("chua co kq train dau, check lai nha")
        return

    betas = [-90.0, -10.0, 30.0, 100.0]
    initial_actions = ["initial_action_0", "initial_action_1", "initial_action_2", "initial_action_3", "initial_action_4"]
    labels = ["label_0", "label_1", "label_2", "label_3", "label_4"]
    
    try:
        feature_path_sec = "data/cleaned_data/BTCUSDT/tardis/second_feature.npy"
        if os.path.exists(feature_path_sec):
            tech_indicators = np.load(feature_path_sec, allow_pickle=True).tolist()
        else:
            print("cảnh báo: chưa có second_feature.npy, lấy tạm feature_list.npy")
            tech_indicators = np.load("data/cleaned_data/BTCUSDT/tardis/feature_list.npy", allow_pickle=True).tolist()
    except Exception:
        print("Khong tim thay feature nao ca")
        return

    # Tao danh sach tat ca cac tasks
    tasks = []
    
    for beta in betas:
        coffient_path = os.path.join(root_path, f"beta_{beta}", "seed_12345")
        if not os.path.exists(coffient_path):
            continue
            
        epoch_dirs = [d for d in os.listdir(coffient_path) if d.startswith("epoch_")]
        for epoch in epoch_dirs:
            epoch_path = os.path.join(coffient_path, epoch)
            model_file = os.path.join(epoch_path, "model.pth")
            
            if not os.path.exists(model_file):
                continue
                
            for label in labels:
                valid_data_dir = os.path.join("data/cleaned_data/BTCUSDT/tardis/valid_multi", label)
                if not os.path.exists(valid_data_dir):
                    continue
                df_files = [f for f in os.listdir(valid_data_dir) if f.endswith(".feather")]
                
                for df_file in df_files:
                    tasks.append((model_file, valid_data_dir, df_file, epoch_path, label, initial_actions, tech_indicators))

    total_tasks = len(tasks)
    print(f"Tổng số lượng task cần chạy: {total_tasks}")
    
    # Chay song song bang ProcessPoolExecutor
    num_workers = max(1, cpu_count() - 1)  # Giu lai 1 core cho OS
    print(f"Sử dụng {num_workers} tiến trình song song...")
    
    completed = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file_task, task): task for task in tasks}
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if completed % 50 == 0 or completed == total_tasks:
                print(f"Tiến độ: {completed}/{total_tasks} | {result}")

    print("\nXONG DANH GIA MODEL")

if __name__ == "__main__":
    main()
