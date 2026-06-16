import pandas as pd
import numpy as np
import os
import re
import sys
from scipy import stats

sys.path.append(".")
sys.path.append("src")
sys.path.append("src/EarnHFT_framwork")
sys.path.append("./src/EarnHFT_framwork")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.ablation_analysis.util import calculate_holding_position_time

def sort_list(lst: list):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    lst.sort(key=alphanum_key)

def calculate_result(setting_path, rolling_window_length=10):
    seed_path = os.path.join(setting_path, "seed_12345")
    if not os.path.exists(seed_path):
        print(f"Skipping: {seed_path} not found")
        return
        
    update_conter_list = os.listdir(seed_path)
    sort_list(update_conter_list)
    if "log" in update_conter_list:
        update_conter_list.remove("log")
    if not update_conter_list:
        print(f"No epochs found in {seed_path}")
        return
        
    reward_list = []
    action_list_list = []
    
    for epoch in update_conter_list:
        epoch_path = os.path.join(seed_path, epoch)
        reward_file = os.path.join(epoch_path, "test", "reward.npy")
        action_file = os.path.join(epoch_path, "test", "action.npy")
        
        if not os.path.exists(reward_file) or not os.path.exists(action_file):
            continue
            
        reward = np.load(reward_file)
        reward_list.append(np.sum(reward))
        action_list = np.load(action_file)
        action_list_list.append(action_list)
        
    if len(reward_list) < rolling_window_length * 2:
        print(f"Too few epochs: {len(reward_list)}")
        return
        
    start = 0
    for start in range(rolling_window_length, len(reward_list) - rolling_window_length, 1):
        reward_list_1 = reward_list[start - rolling_window_length:start]
        reward_list_2 = reward_list[start:start + rolling_window_length]
        _, p_value = stats.ks_2samp(reward_list_1, reward_list_2)
        if p_value > 0.9:
            break
            
    converage_list = range(start - rolling_window_length, len(reward_list))
    print("CS:", (start - rolling_window_length) * 512)
    
    average_holding_time = []
    for idx in converage_list:
        converage_action_list = action_list_list[idx]
        holding_time = calculate_holding_position_time(converage_action_list)
        average_holding_time.append(holding_time)
        
    print("AHL:", np.mean(average_holding_time))
    print("RS:", np.mean(reward_list_1))

if __name__ =="__main__":
    calculate_result("result_risk/BTCUSDT/ablation/dqn_ada_0_trans_true")
