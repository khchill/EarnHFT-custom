import os
import re
import shutil
import numpy as np

def sort_list(lst: list):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    lst.sort(key=alphanum_key)

def get_best_model(label_name, initial_position, root_path="result_risk/BTCUSDT"):
    if not os.path.exists(root_path):
        return "", "", 0.0
        
    coffient_list = os.listdir(root_path)
    if "potential_model" in coffient_list:
        coffient_list.remove("potential_model")
        
    best_return = -float('inf')
    best_result_path = ""
    best_model_file = ""
    
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
            epoch_returns = []
            
            for df in df_list:
                result_path = os.path.join(target_path, df)
                # làm theo yêu cầu vốn
                balance_file = os.path.join(result_path, "final_balance.npy")
                req_money_file = os.path.join(result_path, "require_money.npy")
                if not os.path.exists(balance_file) or not os.path.exists(req_money_file):
                    continue
                balance = np.load(balance_file, allow_pickle=True)
                req_money = np.load(req_money_file, allow_pickle=True)
                return_rate = balance / (req_money + 1e-12)
                epoch_returns.append(float(np.mean(return_rate)))
                
            if len(epoch_returns) > 0:
                # Tính trung bình lợi nhuận của Epoch trên TẤT CẢ các file test
                epoch_mean_ret = sum(epoch_returns) / len(epoch_returns)
                if epoch_mean_ret > best_return:
                    best_return = epoch_mean_ret
                    best_result_path = target_path
                    # file model tương đương với epoch
                    best_model_file = os.path.join(epoch_path, "model.pth")
                    
    return best_model_file, best_result_path, best_return

if __name__ == "__main__":
    print("[*] Dang sang loc ma tran 25 Bot (5x5) tu pool...")
    initial_position_list = ["initial_action_0", "initial_action_1", "initial_action_2", "initial_action_3", "initial_action_4"]
    
    dest_potential_dir = "result_risk/BTCUSDT/potential_model"
    os.makedirs(dest_potential_dir, exist_ok=True)
    
    matrix_cells = [["" for _ in range(5)] for _ in range(5)]
    
    for n_idx, initial_position in enumerate(initial_position_list):
        pos_dir = os.path.join(dest_potential_dir, initial_position)
        os.makedirs(pos_dir, exist_ok=True)
        
        for m_idx in range(5):
            best_model, best_path, best_ret = get_best_model(
                label_name=f"label_{m_idx}", 
                initial_position=initial_position
            )
            
            if best_model and os.path.exists(best_model):
                dest_file = os.path.join(pos_dir, f"model_{m_idx}.pth")
                shutil.copy2(best_model, dest_file)
                match = re.search(r"beta_(-?[\d\.]+)/seed_12345/epoch_(\d+)", best_model)
                if match:
                    beta_val, epoch_val = match.group(1), match.group(2)
                    matrix_cells[n_idx][m_idx] = f"B:{beta_val}/E:{epoch_val}"
                else:
                    matrix_cells[n_idx][m_idx] = "Bot That"
            else:
                matrix_cells[n_idx][m_idx] = "None"
                
    # Hien thi Ma tran 5x5 ASCII
    print("\n" + "=" * 92)
    print("  MA TRAN POOL 25 BOT DUOC TUYEN CHON (5x5)")
    print("=" * 92)
    print(" Vi the (Row - n) \\ Trang thai (Col - m) |  Label 0  |  Label 1  |  Label 2  |  Label 3  |  Label 4  |")
    print("-" * 92)
    positions_label = [
        "0.0000 (initial_action_0)", 
        "0.0025 (initial_action_1)", 
        "0.0050 (initial_action_2)", 
        "0.0075 (initial_action_3)", 
        "0.0100 (initial_action_4)"
    ]
    for idx, pos_lbl in enumerate(positions_label):
        row_str = f" {pos_lbl:<39} |"
        for m_idx in range(5):
            cell_val = matrix_cells[idx][m_idx]
            row_str += f" {cell_val:^9} |"
        print(row_str)
    print("-" * 92)
    print("B:Beta | E:Epoch")
    print("=" * 92)
    print("Da sang loc va copy model vao potential_model.")
