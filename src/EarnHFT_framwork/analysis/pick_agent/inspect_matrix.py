import os
import re
import numpy as np
import pandas as pd

try:
    from prettytable import PrettyTable
    HAS_PRETTYTABLE = True
except ImportError:
    HAS_PRETTYTABLE = False

def sort_list(lst: list):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    lst.sort(key=alphanum_key)

def get_best_model_details(label_name, initial_position, root_path="result_risk/BTCUSDT"):
    if not os.path.exists(root_path):
        return None
        
    coffient_list = os.listdir(root_path)
    if "potential_model" in coffient_list:
        coffient_list.remove("potential_model")
        
    best_return = -float('inf')
    best_details = {}
    
    for coffient in coffient_list:
        coffient_path = os.path.join(root_path, coffient, "seed_12345")
        if not os.path.exists(coffient_path): continue
        epoch_list = os.listdir(coffient_path)
        if "log" in epoch_list: epoch_list.remove("log")
        sort_list(epoch_list)
        
        for epoch in epoch_list:
            epoch_path = os.path.join(coffient_path, epoch)
            target_path = os.path.join(epoch_path, "valid_multi", label_name, initial_position)
            if not os.path.exists(target_path): continue
            
            df_list = os.listdir(target_path)
            epoch_return_rates = []
            epoch_pnls = []
            
            for df in df_list:
                result_path = os.path.join(target_path, df)
                balance_file = os.path.join(result_path, "final_balance.npy")
                req_money_file = os.path.join(result_path, "require_money.npy")
                if not os.path.exists(balance_file) or not os.path.exists(req_money_file):
                    continue
                balance = float(np.load(balance_file, allow_pickle=True))
                req_money = float(np.load(req_money_file, allow_pickle=True))
                
                return_rate = balance / (req_money + 1e-12)
                pnl_percent = (balance / (req_money + 1e-12)) * 100.0
                
                epoch_return_rates.append(return_rate)
                epoch_pnls.append(pnl_percent)
                
            if len(epoch_return_rates) > 0:
                epoch_mean_ret = sum(epoch_return_rates) / len(epoch_return_rates)
                epoch_mean_pnl = sum(epoch_pnls) / len(epoch_pnls)
                
                if epoch_mean_ret > best_return:
                    best_return = epoch_mean_ret
                    best_details = {
                        "beta": coffient.replace("beta_", ""),
                        "epoch": epoch.replace("epoch_", ""),
                        "pnl_percent": epoch_mean_pnl
                    }
                    
    return best_details

def main():
    print("inspect 25 bot (5x5) matrix and pnl on validation dataset...")
    initial_positions = ["initial_action_0", "initial_action_1", "initial_action_2", "initial_action_3", "initial_action_4"]
    pos_labels = ["0.0000", "0.0025", "0.0050", "0.0075", "0.0100"]
    
    matrix_info = [["" for _ in range(5)] for _ in range(5)]
    matrix_pnl = [["" for _ in range(5)] for _ in range(5)]
    
    all_pnls = []
    
    for r_idx, initial_position in enumerate(initial_positions):
        for c_idx in range(5):
            details = get_best_model_details(f"label_{c_idx}", initial_position)
            if details:
                matrix_info[r_idx][c_idx] = f"B:{details['beta']}/E:{details['epoch']}"
                matrix_pnl[r_idx][c_idx] = f"{details['pnl_percent']:+.3f}%"
                all_pnls.append(details['pnl_percent'])
            else:
                matrix_info[r_idx][c_idx] = "N/A"
                matrix_pnl[r_idx][c_idx] = "N/A"
                
    # matrix info
    print("\n" + "="*80)
    print("      matrix info (B: Beta | E: Epoch)")
    print("="*80)
    if HAS_PRETTYTABLE:
        t_info = PrettyTable()
        t_info.field_names = ["initial position \\ label", "Label 0", "Label 1", "Label 2", "Label 3", "Label 4"]
        for idx, pos_lbl in enumerate(pos_labels):
            t_info.add_row([f"{pos_lbl} ({initial_positions[idx]})", matrix_info[idx][0], matrix_info[idx][1], matrix_info[idx][2], matrix_info[idx][3], matrix_info[idx][4]])
        print(t_info)
    else:

        col_name = 'initial position \\ label'
        header_str = f"| {col_name:<25} | {'Label 0':^9} | {'Label 1':^9} | {'Label 2':^9} | {'Label 3':^9} | {'Label 4':^9} |"
        print("-" * len(header_str))
        print(header_str)
        print("-" * len(header_str))
        for idx, pos_lbl in enumerate(pos_labels):
            row_str = f"| {pos_lbl:<25} | {matrix_info[idx][0]:^9} | {matrix_info[idx][1]:^9} | {matrix_info[idx][2]:^9} | {matrix_info[idx][3]:^9} | {matrix_info[idx][4]:^9} |"
            print(row_str)
        print("-" * len(header_str))
    
    # matrix pnl
    print("\n" + "="*80)
    print("      matrix pnl")
    print("="*80)
    if HAS_PRETTYTABLE:
        t_pnl = PrettyTable()
        t_pnl.field_names = ["initial position \\ label", "Label 0", "Label 1", "Label 2", "Label 3", "Label 4"]
        for idx, pos_lbl in enumerate(pos_labels):
            t_pnl.add_row([f"{pos_lbl} ({initial_positions[idx]})", matrix_pnl[idx][0], matrix_pnl[idx][1], matrix_pnl[idx][2], matrix_pnl[idx][3], matrix_pnl[idx][4]])
        print(t_pnl)
    else:

        col_name = 'initial position \\ label'
        header_str = f"| {col_name:<25} | {'Label 0':^9} | {'Label 1':^9} | {'Label 2':^9} | {'Label 3':^9} | {'Label 4':^9} |"
        print("-" * len(header_str))
        print(header_str)
        print("-" * len(header_str))
        for idx, pos_lbl in enumerate(pos_labels):
            row_str = f"| {pos_lbl:<25} | {matrix_pnl[idx][0]:^9} | {matrix_pnl[idx][1]:^9} | {matrix_pnl[idx][2]:^9} | {matrix_pnl[idx][3]:^9} | {matrix_pnl[idx][4]:^9} |"
            print(row_str)
        print("-" * len(header_str))
    
    if all_pnls:
        print(f"number bot picked: {len(all_pnls)}/25")
        print(f"pnl mean: {np.mean(all_pnls):+.3f}%")
        print(f"pnl max: {np.max(all_pnls):+.3f}%")
        print(f"pnl min: {np.min(all_pnls):+.3f}%")
    else:
        print("no bot found check again")

if __name__ == "__main__":
    main()
