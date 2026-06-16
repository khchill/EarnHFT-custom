import pandas as pd
import numpy as np
import sys
import os
import argparse

sys.path.append(".")
sys.path.append("src")
sys.path.append("src/EarnHFT_framwork")
sys.path.append("./src/EarnHFT_framwork")
from env.low_level_env import Testing_env

# setting param cho baseline
parser = argparse.ArgumentParser()
parser.add_argument("--technicail_indicator", type=str, default="MACD", choices=["MACD", "Imbalance_Volume"], help="chon chi bao de trade")
parser.add_argument("--stop_win", type=float, default=0.1)
parser.add_argument("--stop_lose", type=float, default=0.1)
parser.add_argument("--long_term", type=int, default=1200, help="khung dai macd")
parser.add_argument("--mid_term", type=int, default=600, help="khung trung macd")
parser.add_argument("--short_term", type=int, default=60, help="khung ngan macd")
parser.add_argument("--upper_theshold", type=float, default=0.99, help="nguong tren iv")
parser.add_argument("--lower_theshold", type=float, default=-0.99, help="nguong duoi iv")
parser.add_argument("--result_path", type=str, default="result_risk")
parser.add_argument("--dataset_name", type=str, default="BTCUSDT")

class rule_base_trader:
    def __init__(self, args):
        self.args = args
        self.indicator = args.technicail_indicator
        
        # setup duong dan luu kq cho tung loai bot
        if self.indicator == "MACD":
            self.model_path = os.path.join(args.result_path, args.dataset_name, "rule_base", "MACD")
        elif self.indicator == "Imbalance_Volume":
            self.model_path = os.path.join(args.result_path, args.dataset_name, "rule_base", "IV")
            
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
        print(f"khoi tao bot thu cong: {self.indicator}")

    def test(self):
        print(f"bat dau backtest bot {self.indicator} nha...")
        
        dirs_to_save = [os.path.join("result_risk", "BTCUSDT", "rule_base", "epoch_1"), self.model_path]
        
        # test tren tap thuc te
        test_dir_path = "data/cleaned_data/BTCUSDT/tardis/test"
        if not os.path.exists(test_dir_path) or not os.path.isdir(test_dir_path):
            print(f"loi ko tim thay thu muc test o {test_dir_path}")
            return
            
        test_files = sorted([os.path.join(test_dir_path, f) for f in os.listdir(test_dir_path) if f.endswith(".feather")])
        
        # Load Tech Indicators
        tech_indicators = np.load("data/cleaned_data/BTCUSDT/tardis/feature_list.npy", allow_pickle=True).tolist()
        
        # tim vi tri cua indicator trong mang state
        indicator_idx = 0
        if self.indicator == "Imbalance_Volume":
            search_key = "imblance_volume_oe"
            for i, name in enumerate(tech_indicators):
                if search_key.lower() in name.lower():
                    indicator_idx = i
                    break
            print(f"xai feature '{tech_indicators[indicator_idx]}' o index [{indicator_idx}] de trade")
        else:
            print("macd se dc tinh dong tren tap test de ko lam hu state cua rl nha")
        
        all_rewards = []
        all_actions = []
        all_require_money = []
        
        for test_file_path in test_files:
            test_df = pd.read_feather(test_file_path).bfill().ffill().fillna(0.0)
            test_env = Testing_env(test_df, tech_indicators, transcation_cost=0.00015, back_time_length=1, max_holding_number=0.01, action_dim=5, initial_action=0)
            
            state, _ = test_env.reset()
            done = False
            step = 0
            daily_rewards = []
            daily_actions = []
            # tinh macd dong
            if self.indicator == "MACD":
                dif_series = test_df['midpoint'].ewm(span=self.args.short_term, adjust=False).mean() - \
                             test_df['midpoint'].ewm(span=self.args.long_term, adjust=False).mean()
                dea_series = dif_series.ewm(span=self.args.mid_term, adjust=False).mean()
                macd_series = dif_series - dea_series
                macd_values = macd_series.values
            
            entry_price = 0.0
            current_position = 0
            cooldown = False
            
            while not done:
                action = current_position # default giu nguyen vi the
                
                # lay tin hieu
                if self.indicator == "MACD":
                    current_val = macd_values[step]
                    signal_buy = (current_val > 0)
                    signal_sell = (current_val < 0)
                elif self.indicator == "Imbalance_Volume":
                    current_val = state[indicator_idx] if len(state) > indicator_idx else 0
                    signal_buy = (current_val > self.args.upper_theshold)
                    signal_sell = (current_val < self.args.lower_theshold)
                # logic vao hoac ra lenh
                if signal_sell:
                    action = 0 # xa hang
                    cooldown = False # reset chan mua
                elif signal_buy and not cooldown:
                    action = 4 # all in
                # stop win / stop lose
                current_price = test_df.iloc[step]['midpoint']
                if action > 0:
                    if current_position == 0:
                        entry_price = current_price # luu gia vao
                    else:
                        roi = (current_price - entry_price) / entry_price
                        if roi >= self.args.stop_win or roi <= -self.args.stop_lose:
                            action = 0 # force exit cat lo/chot loi
                            cooldown = True # cam mua cho den khi co tin hieu ban dao chieu
                            
                current_position = action
                    
                next_state, reward, done, _ = test_env.step(action)
                daily_rewards.append(reward)
                daily_actions.append(action)
                state = next_state
                step += 1
                
            all_rewards.extend(daily_rewards)
            all_actions.extend(daily_actions)
            all_require_money.append(test_env.require_money)
            
        for d in dirs_to_save:
            os.makedirs(d, exist_ok=True)
            t_dir = os.path.join(d, "test")
            os.makedirs(t_dir, exist_ok=True)
            np.save(os.path.join(t_dir, "reward.npy"), np.array(all_rewards))
            np.save(os.path.join(t_dir, "action.npy"), np.array(all_actions))
            np.save(os.path.join(t_dir, "require_money.npy"), np.array(all_require_money))
        
        print(f"xong bot rule-base ({self.indicator})")

if __name__ == "__main__":
    args = parser.parse_args()
    agent = rule_base_trader(args)
    agent.test()
