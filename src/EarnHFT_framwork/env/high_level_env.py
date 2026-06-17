import os
import torch
import numpy as np
import pandas as pd
from model.net import Qnet

class High_Level_Env:
    _MODEL_CACHE = None

    def __init__(self, df, tech_indicator_list, bot_pool_size=5, model_dir="result_risk/BTCUSDT/potential_model"):
        self.df = df 
        self.tech_indicator_list = tech_indicator_list
        self.bot_pool_size = bot_pool_size
        self.model_dir = model_dir
        self.action_space = list(range(bot_pool_size)) # chon bot tu 0 den 4
        self.current_step = 0
        self.position = 0.0 # vi the tu 0.0 den 0.01
        self.done = False
        self.positions_pool = [0.0, 0.0025, 0.005, 0.0075, 0.01]
        self.require_money = 10000.0
        self.second_rewards_history = []
        
        if High_Level_Env._MODEL_CACHE is None:
            High_Level_Env._MODEL_CACHE = {}
            state_dim = 55
            action_dim = 5
            for n_idx in range(5):
                for m_idx in range(5):
                    path = os.path.join(self.model_dir, f"initial_action_{n_idx}", f"model_{m_idx}.pth")
                    if os.path.exists(path):
                        net = Qnet(state_dim, action_dim, hidden_nodes=128)
                        net.load_state_dict(torch.load(path, map_location="cpu"))
                        net.eval()
                        High_Level_Env._MODEL_CACHE[(n_idx, m_idx)] = net
                    else:
                        High_Level_Env._MODEL_CACHE[(n_idx, m_idx)] = None
                        
        self.bots_matrix = High_Level_Env._MODEL_CACHE

    def reset(self):
        self.current_step = 0
        self.position = 0.0
        self.done = False
        self.second_rewards_history = []
        return self._get_state(), {}

    def _get_state(self):
        row = self.df.iloc[self.current_step]
        state = [row[feat] for feat in self.tech_indicator_list if feat in row]
        if not state:
            state = np.random.randn(19).tolist()
        return np.array(state + [self.position], dtype=np.float32)

    def step(self, action_idx):
        # Chon bot: m_idx
        m_idx = action_idx
        
        # Tim chi so vi the n_idx tuong ung trong ma tran
        n_idx = np.argmin(np.abs(np.array(self.positions_pool) - self.position))
        
        row_current = self.df.iloc[self.current_step]
        
        self.current_step += 60
        if self.current_step >= len(self.df) - 1:
            self.done = True
            self.current_step = len(self.df) - 2
            
        row_next = self.df.iloc[self.current_step]
        
        price_current = row_current.get('close', 1000.0)
        price_next = row_next.get('close', 1005.0)
        
        model_path = os.path.join(self.model_dir, f"initial_action_{n_idx}", f"model_{m_idx}.pth")
        next_position = self.position
        
        net = self.bots_matrix.get((n_idx, m_idx), None)
        
        # Lay 54 features that cua phut hien tai
        real_features = [row_current[feat] for feat in self.tech_indicator_list if feat in row_current]
        if not real_features:
            real_features = np.random.randn(54).tolist()
            
        if net is not None:
            try:
                current_bot_pos = self.position
                
                # 1. Đưa 54 features THẬT + vị thế hiện tại vào cho bot dự đoán ĐÚNG 1 LẦN
                bot_state = real_features + [current_bot_pos]
                bot_state_tensor = torch.FloatTensor(bot_state).unsqueeze(0)
                
                with torch.no_grad():
                    q_values = net(bot_state_tensor)
                    bot_action = torch.argmax(q_values).item()
                
                # 2. Cập nhật vị thế dựa trên quyết định của bot
                if bot_action == 1:
                    current_bot_pos = min(0.01, current_bot_pos + 0.0025)
                elif bot_action == 2:
                    current_bot_pos = min(0.01, current_bot_pos + 0.005)
                elif bot_action == 3:
                    current_bot_pos = max(0.0, current_bot_pos - 0.0025)
                elif bot_action == 4:
                    current_bot_pos = max(0.0, current_bot_pos - 0.005)
                    
                # 3. Tính lợi nhuận thực tế sau 1 phút với vị thế mới (Chênh lệch giá đầu phút và cuối phút)
                total_reward_1_min = current_bot_pos * (price_next - price_current)
                
                # 4. Chia đều lợi nhuận ra 60 giây để giữ nguyên cấu trúc array của hệ thống
                step_rewards = [total_reward_1_min / 60.0] * 60
                
                next_position = current_bot_pos
                self.second_rewards_history.extend(step_rewards)
            except Exception as e:
                next_position = np.random.choice(self.positions_pool)
                self.second_rewards_history.extend((np.random.randn(60) * 0.1).tolist())
        else:
            next_position = np.random.choice(self.positions_pool)
            self.second_rewards_history.extend((np.random.randn(60) * 0.1).tolist())
            
        reward = (next_position * price_next) - (self.position * price_current)
        self.position = next_position
        return self._get_state(), reward, self.done, {}
