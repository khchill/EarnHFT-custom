import sys
import os

# Thêm đường dẫn để import được code framework
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "EarnHFT_framwork"))

import torch
import numpy as np
import pandas as pd
from src.EarnHFT_framwork.model.net import Qnet
from src.EarnHFT_framwork.env.low_level_env import Testing_env

def test_chunk():
    print(">>> Khởi tạo Script Backtest 1 Chunk <<<")
    # 1. Đọc list features từ file .npy thay vì file yaml (bản gốc không dùng yaml)
    feature_path_sec = "data/cleaned_data/BTCUSDT/tardis/second_feature.npy"
    if not os.path.exists(feature_path_sec):
        print(f"[LỖI] Không tìm thấy file {feature_path_sec}")
        return
        
    tech_indicators = np.load(feature_path_sec, allow_pickle=True).tolist()
    
    # 2. Cấu hình đường dẫn
    model_path = "result_risk/BTCUSDT/beta_-10.0/seed_12345/epoch_1/model.pth"
    data_path = "data/cleaned_data/BTCUSDT/tardis/train/df_2022-01-01_0.feather"
    
    if not os.path.exists(model_path):
        print(f"[LỖI] Không tìm thấy file model: {model_path}")
        print("=> Anh hãy mở file test_low_level_chunk.py lên và sửa lại dòng 19 cho khớp với đường dẫn máy tính của anh nhé!")
        return

    print(f"Đang tải data từ {data_path}...")
    df = pd.read_feather(data_path).bfill().ffill().fillna(0.0)
    
    # Cắt lấy 200 dòng (chunk) để xem thử
    df_chunk = df.iloc[100:300].reset_index(drop=True)
    
    # 3. Load model
    print(f"Đang tải model từ {model_path}...")
    flat_tech = []
    for feat in tech_indicators:
        if isinstance(feat, list): flat_tech.extend(feat)
        else: flat_tech.append(feat)
        
    state_dim = len(flat_tech) + 1
    model = Qnet(state_dim, 5, hidden_nodes=128)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    # 4. Tạo môi trường
    env = Testing_env(df_chunk, tech_indicators, transcation_cost=0.00015)
    state, _ = env.reset()
    
    print("\n" + "="*50)
    print("BẮT ĐẦU BACKTEST 1 CHUNK (200 Ticks)")
    print("="*50)
    
    positions = [0.0, 0.0025, 0.005, 0.0075, 0.01]
    total_reward = 0
    trades = 0
    prev_pos = 0.0
    
    for i in range(len(df_chunk)):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()
            
        if i < 5:  # In 5 dòng đầu tiên để xem Q-value có bị NaN không
            print(f"Tick {i:3d} | Q-values: {q_values.numpy()[0]} | Action được chọn: {action}")
            
        next_state, reward, done, info = env.step(action)
        target_pos = positions[action]
        price = (env.ask_prices[i, 0] + env.bid_prices[i, 0]) / 2
        
        # In ra nếu có thay đổi vị thế (Trade) hoặc in tất cả
        if target_pos != prev_pos:
            trades += 1
            print(f"Tick {i:3d} | Giá: {price:8.2f} | Quyết định: ĐỔI VỊ THẾ ({prev_pos:.4f} -> {target_pos:.4f}) | Action: {action} | Phí+Lãi/Lỗ ròng: {reward:.6f}")
        else:
            # Comment dòng này lại nếu anh chỉ muốn xem lúc nó Trade, bỏ comment để xem từng giây
            print(f"Tick {i:3d} | Giá: {price:8.2f} | Quyết định: GIỮ NGUYÊN ({prev_pos:.4f}) | Action: {action} | Lãi/Lỗ ròng: {reward:.6f}")
            pass
            
        prev_pos = target_pos
        state = next_state
        total_reward += reward
        
        if done:
            break
            
    print("="*50)
    print(f"Tổng số lần thay đổi vị thế (Trades): {trades}")
    print(f"Tổng Lợi nhuận Ròng (Total Reward): {total_reward:.6f}")
    print("="*50)

if __name__ == "__main__":
    test_chunk()
