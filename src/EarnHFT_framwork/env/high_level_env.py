import os
import torch
import numpy as np
import pandas as pd
from model.net import Qnet

class High_Level_Env:
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
        
        if os.path.exists(model_path):
            try:
                state_dim = 55
                action_dim = 5
                net = Qnet(state_dim, action_dim, hidden_nodes=128)
                net.load_state_dict(torch.load(model_path, map_location="cpu"))
                net.eval()
                
                # Gia lap 60 giay giao dich cua bot
                prices = np.linspace(price_current, price_next, 60)
                current_bot_pos = self.position
                step_rewards = []
                
                for step_sec in range(60):
                    bot_state = np.random.randn(54).tolist() + [current_bot_pos]
                    bot_state_tensor = torch.FloatTensor(bot_state).unsqueeze(0)
                    with torch.no_grad():
                        q_values = net(bot_state_tensor)
                        bot_action = torch.argmax(q_values).item()
                    
                    prev_bot_pos = current_bot_pos
                    if bot_action == 1:
                        current_bot_pos = min(0.01, current_bot_pos + 0.0025)
                    elif bot_action == 2:
                        current_bot_pos = min(0.01, current_bot_pos + 0.005)
                    elif bot_action == 3:
                        current_bot_pos = max(0.0, current_bot_pos - 0.0025)
                    elif bot_action == 4:
                        current_bot_pos = max(0.0, current_bot_pos - 0.005)
                        
                    p_curr = prices[step_sec]
                    p_nxt = prices[step_sec + 1] if step_sec + 1 < 60 else price_next
                    # Sửa lỗi: Phải tính cả sự thay đổi của tiền mặt (cash flow) khi mua/bán.
                    # Công thức: Vị_thế_mới*Giá_mới - Vị_thế_cũ*Giá_cũ + Dòng_tiền_mua_bán
                    # Dòng tiền mua/bán = -(Vị_thế_mới - Vị_thế_cũ)*Giá_cũ
                    # Rút gọn lại thành: Vị_thế_mới * (Giá_mới - Giá_cũ)
                    sec_reward = current_bot_pos * (p_nxt - p_curr)
                    step_rewards.append(sec_reward)
                
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
