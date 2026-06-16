import numpy as np

def evaluate_second(require_money, reward_list):
    """các metric evaluation tài chính"""
    reward_sum = np.cumsum(reward_list)
    total_asset_value_list = (require_money + reward_sum).tolist()
    
    second_return_rate_list = [
        total_asset_value_list[i + 1] / total_asset_value_list[i] - 1
        for i in range(len(total_asset_value_list) - 1)
    ]
    
    tr = total_asset_value_list[-1] / total_asset_value_list[0] - 1
    
    vol = np.std(second_return_rate_list)
    annual_vol = vol * np.sqrt(3600 * 24 * 365)
    
    mdd = 0
    peak = total_asset_value_list[0]
    for value in total_asset_value_list:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > mdd:
            mdd = dd
            
    negative_second_return_rate_list = [x for x in second_return_rate_list if x < 0]
    if len(negative_second_return_rate_list) > 0:
        downside_deviation = np.std(negative_second_return_rate_list)
    else:
        downside_deviation = 1e-8
        
    sr = np.mean(second_return_rate_list) / (np.std(second_return_rate_list) + 1e-8)
    annual_sr = sr * np.sqrt(24 * 60 * 60 * 365)
    
    cr = np.mean(second_return_rate_list) / (mdd + 1e-8)
    daily_cr = cr * 24 * 60 * 60
    annual_cr = daily_cr * 365
    
    SoR = np.mean(second_return_rate_list) / downside_deviation
    daily_SoR = SoR * np.sqrt(60 * 24 * 60)
    annual_SoR = daily_SoR * np.sqrt(365)
    
    daily_dd = downside_deviation * np.sqrt(3600 * 24)
    daily_vol = vol * np.sqrt(3600 * 24)
    
    return (tr, daily_vol, mdd, daily_dd, annual_sr, daily_cr, daily_SoR, annual_cr, annual_SoR, annual_vol)

from prettytable import PrettyTable
import matplotlib.pyplot as plt
import os

def print_metrics(stats):
    table = PrettyTable()
    for key, val in stats.items():
        table.add_column(key, val)
    return table

def calculate_portion_model(model_history: list, path):
    prob_list = []
    total_sample_number = len(model_history)
    if isinstance(model_history, np.ndarray):
        model_history = model_history.tolist()
    elif not isinstance(model_history, list):
        model_history = list(model_history)
    
    for i in range(25):
        prob = model_history.count(i) / total_sample_number
        prob_list.append(prob)
        
    xticks = np.arange(25)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.yaxis.grid(linestyle="--", color="lightgray")
    ax1.bar(
        xticks, np.array(prob_list) * 100,
        color=["moccasin", "aquamarine", "#dbc2ec", "orchid", "lightskyblue"] * 5,
        width=0.5,
    )
    ax1.set_ylabel("Probability(%)", fontsize=15)
    ax1.set_xlabel("Model ID", fontsize=15)
    
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, "probability.pdf"), bbox_inches="tight")
    plt.close()
