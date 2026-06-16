import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re

def sort_list(lst: list):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    lst.sort(key=alphanum_key)

def get_single_agent_contrast(reward_list_list, require_money_list, algorithm_names, colors, save_path):
    fig = plt.figure(figsize=(7.5, 5))
    for reward_list, require_money, algorithm_name, color in zip(reward_list_list, require_money_list, algorithm_names, colors):
        accummulative_reward_sum = [reward_list]
        for i in range(len(reward_list) - 1):
            accummulative_reward_sum.append(accummulative_reward_sum[-1] + reward_list[i + 1])
            
        our_net_curve = np.array(accummulative_reward_sum) / (require_money + 1e-12)
        plt.plot(range(len(our_net_curve)), our_net_curve * 100, color=color, label=algorithm_name)
        
    plt.xlabel("Trading Timestamp(s)", size=15)
    plt.ylabel("Total Return(%)", size=15)
    plt.legend()
    plt.grid(ls="--")
    
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
