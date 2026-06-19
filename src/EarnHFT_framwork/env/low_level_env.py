import numpy as np
import pandas as pd

class Training_Env:
    def __init__(self, df, tech_indicator_list, transcation_cost=0.00015, back_time_length=1, max_holding_number=0.01, action_dim=5, initial_action=0):
        self.df = df
        self.tech_indicator_list = tech_indicator_list
        self.transcation_cost = transcation_cost
        self.max_holding_number = max_holding_number
        self.action_dim = action_dim
        
        # Không gian Vị thế (Action Space)
        self.position_space = [i * (max_holding_number / (action_dim - 1)) for i in range(action_dim)]
        
        self.step_length = len(self.df)
        
        # Tối ưu hoá: Ép kiểu sang Numpy arrays
        flat_tech_list = []
        if len(self.tech_indicator_list) > 0 and isinstance(self.tech_indicator_list[0], list):
            for feature in self.tech_indicator_list:
                for feat in feature:
                    flat_tech_list.append(feat)
        else:
            flat_tech_list = self.tech_indicator_list
            
        for feat in flat_tech_list:
            if feat not in self.df.columns:
                self.df[feat] = 0.0
                
        self.tech_indicator_array = self.df[flat_tech_list].values
        
        # Pre-extract LOB columns
        self.ask_prices = np.array([self.df.get(f'ask{i}_price', pd.Series(0.0, index=self.df.index)).values for i in range(1, 11)]).T
        self.ask_sizes = np.array([self.df.get(f'ask{i}_size', pd.Series(0.0, index=self.df.index)).values for i in range(1, 11)]).T
        self.bid_prices = np.array([self.df.get(f'bid{i}_price', pd.Series(0.0, index=self.df.index)).values for i in range(1, 11)]).T
        self.bid_sizes = np.array([self.df.get(f'bid{i}_size', pd.Series(0.0, index=self.df.index)).values for i in range(1, 11)]).T
        
        self.initial_action = initial_action
        self.reset()

    def reset(self):
        self.current_step = 0
        self.previous_action = self.initial_action
        self.current_position = self.position_space[self.initial_action]
        self.holding_length = 1
        self.cash = 0.0
        
        state = self._get_state()
        info = {
            "previous_action": self.previous_action,
            "holding_length": self.holding_length
        }
        return state, info

    def _get_state(self):
        # Input cho mạng Nơ-ron (Loại bỏ overhead tolist và list concatenation)
        return np.append(self.tech_indicator_array[self.current_step], self.current_position).astype(np.float32)

    def _execute_order(self, target_position, step_idx):
        # Mô phỏng Khớp lệnh trên Sổ lệnh 5 mốc giá (Limit Order Book)
        M = target_position - self.current_position
        cash_flow = 0.0
        
        if M > 0: # LỆNH MUA (Mua thì tiền bị trừ cash_flow âm)
            remaining = M
            for i in range(10):
                price = self.ask_prices[step_idx, i]
                size = self.ask_sizes[step_idx, i]
                executed = min(remaining, size)
                
                # Tính tiền mua + phí giao dịch  
                cash_flow -= executed * price * (1 + self.transcation_cost)
                remaining -= executed
                if remaining <= 0:
                    break
                    
            # Nếu thanh khoản 10 mốc không đủ ép khớp nốt lượng dư ở mốc xấu nhất (mốc 10)
            if remaining > 0:
                cash_flow -= remaining * self.ask_prices[step_idx, 9] * (1 + self.transcation_cost)
                
        elif M < 0: # LỆNH BÁN (Bán thì thu tiền về ->cash_flow dương)
            remaining = -M
            for i in range(10):
                price = self.bid_prices[step_idx, i]
                size = self.bid_sizes[step_idx, i]
                executed = min(remaining, size)
                
                # Tính tiền bán + trừ phí 
                cash_flow += executed * price * (1 - self.transcation_cost)
                remaining -= executed
                if remaining <= 0:
                    break
                    
            if remaining > 0:
                cash_flow += remaining * self.bid_prices[step_idx, 9] * (1 - self.transcation_cost)
                
        return cash_flow

    def step(self, action):
        old_position = self.current_position
        target_position = self.position_space[action]
        
        # B1: Khớp lệnh thực tế trên LOB và cập nhật tiền mặt
        cash_flow = self._execute_order(target_position, self.current_step)
        self.cash += cash_flow
        self.current_position = target_position
        
        # B2: Lưu lại giá Bid ở giây hiện tại để tính reward
        current_bid1 = self.bid_prices[self.current_step, 0]
        
        # B3: Cập nhật thông tin
        if action == self.previous_action:
            self.holding_length += 1
        else:
            self.holding_length = 1
        self.previous_action = action
        
        # B4: Tiến sang giây tiếp theo
        self.current_step += 1
        done = (self.current_step >= self.step_length - 1)
        
        # B5: Tính toán reward
        if not done:
            next_bid1 = self.bid_prices[self.current_step, 0]
            # reward = [Giá trị Vị thế mới ở giây t+1] - [Giá trị Vị thế cũ ở giây t] + [Dòng tiền vừa thay đổi]
            reward = self.current_position * next_bid1 - old_position * current_bid1 + cash_flow
            next_state = self._get_state()
        else:
            reward = 0.0
            next_state = self._get_state()
            
        info = {
            "previous_action": self.previous_action,
            "holding_length": self.holding_length
        }
        
        return next_state, reward, done, info

class Testing_env(Training_Env):
    def __init__(self, df, tech_indicator_list, transcation_cost=0.00015, back_time_length=1, max_holding_number=0.01, action_dim=5, initial_action=0):
        super().__init__(df, tech_indicator_list, transcation_cost, back_time_length, max_holding_number, action_dim, initial_action)
        self.require_money = 0.0  
        self.final_balance = 0.0

    def reset(self):
        state, info = super().reset()
        
        # Lưu lại vị thế ban đầu
        target_initial_position = self.current_position
        self.current_position = 0.0
        
        # Nếu bắt đầu với vị thế > 0, ta phải mô phỏng việc "mua" vị thế này ở bước 0 
        # để ghi nhận chính xác dòng tiền âm (require_money) và không bị lãi ảo.
        if target_initial_position > 0:
            initial_cost = self._execute_order(target_initial_position, self.current_step)
            self.cash += initial_cost
            self.current_position = target_initial_position
            
        self.require_money = 0.0 if self.cash >= 0 else -self.cash
        self.final_balance = 0.0
        return state, info

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        
        # Maximum Drawdown của dòng tiền mặt
        if -self.cash > self.require_money:
            self.require_money = -self.cash
            
        if done:
            # Khi kết thúc phiên test, ép AI thanh lý toàn bộ vị thế
            final_liquidation_cash = self._execute_order(0.0, self.current_step)
            self.final_balance = self.cash + final_liquidation_cash
            
        return next_state, reward, done, info
