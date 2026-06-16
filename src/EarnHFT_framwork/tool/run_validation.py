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

def main():
    print("=" * 60)
    print("bat dau chay danh gia model")
    print("=" * 60)
    
    root_path = "result_risk/BTCUSDT"
    if not os.path.exists(root_path):
        print("chua co kq train dau, check lai nha")
        return

    betas = [-90.0, -10.0, 30.0, 100.0]
    initial_actions = ["initial_action_0", "initial_action_1", "initial_action_2", "initial_action_3", "initial_action_4"]
    labels = ["label_0", "label_1", "label_2", "label_3", "label_4"]
    max_epochs = 40
    
    for beta in betas:
        coffient_path = os.path.join(root_path, f"beta_{beta}", "seed_12345")
        if not os.path.exists(coffient_path):
            os.makedirs(coffient_path, exist_ok=True)
            
        epoch_dirs = [f"epoch_{e}" for e in range(1, max_epochs + 1)]
        
        for epoch in epoch_dirs:
            epoch_path = os.path.join(coffient_path, epoch)
            os.makedirs(epoch_path, exist_ok=True)
            
            # Lấy danh sách feather files cho từng nhãn
            try:
                tech_indicators = np.load("data/cleaned_data/BTCUSDT/tardis/feature_list.npy", allow_pickle=True).tolist()
            except Exception:
                # Fallback nếu không có file feature riêng biệt
                tech_indicators = np.load("data/cleaned_data/BTCUSDT/tardis/feature_list.npy", allow_pickle=True).tolist()
            
            # Khởi tạo model và load trọng số
            model = Qnet(N_STATES=len(tech_indicators) + 1, N_ACTIONS=5, hidden_nodes=128)
            model_file = os.path.join(epoch_path, "model.pth")
            
            # Nếu model không tồn tại thì cảnh báo và skip
            if not os.path.exists(model_file):
                print(f"k thay {model_file}, bo qua ep nay")
                continue
            
            model.load_state_dict(torch.load(model_file))
            model.eval()
            
            # Sát hạch trên 5 market labels x 5 initial actions
            for label in labels:
                # Tìm các file feather trong thư mục label
                valid_data_dir = os.path.join("data/cleaned_data/BTCUSDT/tardis/valid_multi", label)
                if not os.path.exists(valid_data_dir):
                    continue
                df_files = [f for f in os.listdir(valid_data_dir) if f.endswith(".feather")]
                
                for df_file in df_files:
                    df_path = os.path.join(valid_data_dir, df_file)
                    valid_df = pd.read_feather(df_path).bfill().ffill().fillna(0.0)
                    
                    for init_act_idx, init_act in enumerate(initial_actions):
                        target_dir = os.path.join(epoch_path, "valid_multi", label, init_act, df_file)
                        os.makedirs(target_dir, exist_ok=True)
                        
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
                            
                        # Ghi nhận kết quả giao dịch thực
                        np.save(os.path.join(target_dir, "final_balance.npy"), np.array(test_env.final_balance))
                        np.save(os.path.join(target_dir, "require_money.npy"), np.array(test_env.require_money))
                        np.save(os.path.join(target_dir, "action.npy"), np.array(daily_actions))
                        
    print("\nxong danh gia model")

if __name__ == "__main__":
    main()
