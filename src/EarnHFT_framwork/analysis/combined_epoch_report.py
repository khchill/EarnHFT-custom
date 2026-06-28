import os
import sys
import numpy as np
import pandas as pd
import re

try:
    from prettytable import PrettyTable
except ImportError:
    print("Vui lòng cài đặt thư viện: pip install prettytable")
    sys.exit(1)

sys.path.append(".")
sys.path.append("src")
sys.path.append("src/EarnHFT_framwork")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.calculate_metric.calculate_wsrc import evaluate_second

def sort_list(lst: list):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    lst.sort(key=alphanum_key)

def calculate_metric_for_epoch(root_path, model_name, epoch_name, mode='test', lengths=[]):
    """Tính toán metric cho một epoch cụ thể"""
    test_path = os.path.join(root_path, epoch_name, mode)
    reward_file = os.path.join(test_path, "reward.npy")
    require_money_file = os.path.join(test_path, "require_money.npy")
    
    if not os.path.exists(reward_file):
        return None
        
    try:
        reward_list = np.load(reward_file)
        
        # Xử lý high_level ko có require_money list
        if os.path.exists(require_money_file):
            require_money = np.load(require_money_file)
        else:
            require_money = 10000.0
            
        if isinstance(require_money, float) or require_money.ndim == 0:
            require_money = [require_money] * len(lengths)

        original_lengths = lengths.copy()
        total_expected_len = sum(lengths)
        actual_len = len(reward_list)
        scale = 1
        if actual_len > 0 and actual_len < total_expected_len / 2:
            scale = int(np.round(total_expected_len / actual_len))
            if scale > 1:
                lengths = [int(np.ceil(l / scale)) for l in lengths]

        daily_metrics = []
        start = 0
        for i, length in enumerate(lengths):
            if start + length > len(reward_list):
                break
            daily_reward = reward_list[start:start+length]
            if len(daily_reward) == 0:
                break
            req_m = require_money[i] if i < len(require_money) else require_money[-1]
            
            metric = list(evaluate_second(req_m, daily_reward))
            if scale > 1:
                correction = np.sqrt(scale)
                metric[1] /= correction
                metric[3] /= correction
                metric[4] /= correction
                metric[6] /= correction
                metric[8] /= correction
                metric[9] /= correction
                metric[5] /= scale
                metric[7] /= scale
            
            daily_metrics.append(metric)
            start += length

        if not daily_metrics:
            return None

        avg_metrics = np.mean(daily_metrics, axis=0)
        return avg_metrics
    except Exception as e:
        return None

def calculate_trades(action_path):
    try:
        actions = np.load(action_path, allow_pickle=True)
        if len(actions.shape) == 2:
            # High-level router: actions are [n_idx, m_idx]
            trades = np.sum(np.any(actions[1:] != actions[:-1], axis=1))
            return trades, len(actions), actions
            
        # Mapping từ action index (0-4) sang target position
        positions = [0.0, 0.0025, 0.005, 0.0075, 0.01]
        target_positions = [positions[a] for a in actions]
        
        trades = 0
        for i in range(1, len(target_positions)):
            if target_positions[i] != target_positions[i-1]:
                trades += 1
                
        return trades, len(actions), actions
    except Exception as e:
        return -1, 0, None

def format_dict(d):
    return ", ".join([f"B{k}:{v}" for k, v in sorted(d.items())])

def main():
    base_dir = "result_risk/BTCUSDT"
    if not os.path.exists(base_dir):
        print(f"Không tìm thấy {base_dir}")
        return

    table = PrettyTable()
    table.field_names = [
        "Model", "Epoch", "Mode", 
        "Return(%)", "Ann. Ret(%)", "Ann. Vol", 
        "Sharpe", "Sortino", "Max DD(%)", "Calmar",
        "Position Changes", "Avg Return per Trade(%)", "Total Steps", "Action Distribution"
    ]
    table.align["Model"] = "l"
    table.align["Action Distribution"] = "l"
    
    csv_data = []

    for model_name in sorted(os.listdir(base_dir)):
        # Bỏ qua các thư mục không phải model
        if model_name in ["kde", "segment_match_test", "segment_match_valid", "potential_model"]:
            continue
        if "check_data" in model_name or model_name.endswith(".png"):
            continue

        seed_dir = os.path.join(base_dir, model_name, "seed_12345")
        
        # ==== RULE BASE MODELS ====
        if not os.path.exists(seed_dir):
            if model_name == "rule_base":
                for rb_model in os.listdir(os.path.join(base_dir, "rule_base")):
                    rb_dir = os.path.join(base_dir, "rule_base", rb_model)
                    if os.path.isdir(rb_dir):
                        for mode in ["valid", "test"]:
                            test_dir_data = f"data/cleaned_data/BTCUSDT/tardis/{mode}"
                            if not os.path.exists(test_dir_data):
                                continue
                            test_files = sorted([f for f in os.listdir(test_dir_data) if f.endswith(".feather")])
                            lengths = [len(pd.read_feather(os.path.join(test_dir_data, f))) - 1 for f in test_files]
                            
                            # Tính metric
                            avg_metrics = calculate_metric_for_epoch(os.path.join(base_dir, "rule_base"), rb_model, "", mode, lengths)
                            
                            # Tính trades & actions
                            action_file = os.path.join(rb_dir, mode, "action.npy")
                            trades, total, actions = -1, 0, None
                            if os.path.exists(action_file):
                                trades, total, actions = calculate_trades(action_file)
                                
                            if avg_metrics is not None:
                                total_return, ann_return, ann_vol, _, sharpe, sortino, max_dd, calmar = avg_metrics
                                
                                dist_str = "-"
                                if actions is not None:
                                    unique, counts = np.unique(actions, return_counts=True)
                                    dist_str = format_dict(dict(zip(unique, counts)))
                                
                                avg_cost = "0.00000"
                                if trades != -1 and trades > 0:
                                    avg_cost = f"{(total_return*100) / trades:.5f}"
                                
                                row_data = [
                                    f"Base: {rb_model}", "-", mode.upper(),
                                    f"{total_return*100:.2f}%", f"{ann_return*100:.2f}%", f"{ann_vol:.4f}",
                                    f"{sharpe:.4f}", f"{sortino:.4f}", f"{max_dd*100:.2f}%", f"{calmar:.4f}",
                                    trades if trades != -1 else "-", avg_cost, total if total > 0 else "-", dist_str
                                ]
                                table.add_row(row_data)
                                csv_data.append(row_data)
            continue

        # ==== RL MODELS ====
        epochs = []
        for e in os.listdir(seed_dir):
            if e.startswith("epoch_"):
                try:
                    num = int(e.split("_")[1])
                    epochs.append((num, e))
                except:
                    pass
                    
        if not epochs:
            continue
            
        epochs.sort(key=lambda x: x[0])
        max_ep = epochs[-1][0]
        
        target_epochs = []
        for num, e_name in epochs:
            if num == 1 or num == max_ep or num % 5 == 0:
                target_epochs.append(e_name)
                
        has_printed = False
        
        for e_name in target_epochs:
            for mode in ["valid", "test"]:
                test_dir_data = f"data/cleaned_data/BTCUSDT/tardis/{mode}"
                if not os.path.exists(test_dir_data):
                    continue
                test_files = sorted([f for f in os.listdir(test_dir_data) if f.endswith(".feather")])
                lengths = [len(pd.read_feather(os.path.join(test_dir_data, f))) - 1 for f in test_files]
                
                # 1. TÍNH METRICS
                avg_metrics = calculate_metric_for_epoch(seed_dir, model_name, e_name, mode, lengths)
                
                # 2. TÍNH ACTIONS/TRADES
                action_file = os.path.join(seed_dir, e_name, mode, "action.npy")
                trades, total, actions = -1, 0, None
                if os.path.exists(action_file):
                    trades, total, actions = calculate_trades(action_file)
                
                if avg_metrics is not None:
                    has_printed = True
                    tr, daily_vol, mdd, daily_dd, annual_sr, daily_cr, daily_SoR, annual_cr, annual_SoR, annual_vol = avg_metrics
                    ann_return = tr * (365 / len(lengths)) if len(lengths) > 0 else 0
                    
                    dist_str = "-"
                    if actions is not None:
                        if len(actions.shape) == 2:
                            unique, counts = np.unique(actions[:, 1], return_counts=True)
                        else:
                            unique, counts = np.unique(actions, return_counts=True)
                        dist_str = format_dict(dict(zip(unique, counts)))
                        
                    trades_val = trades if trades != -1 else "-"
                    total_val = total if total > 0 else "-"
                    
                    avg_cost = "0.00000"
                    if trades != -1 and trades > 0:
                        avg_cost = f"{(tr*100) / trades:.5f}"
                    
                    row_data = [
                        model_name, e_name, mode.upper(),
                        f"{tr*100:.2f}%", f"{ann_return*100:.2f}%", f"{annual_vol:.4f}",
                        f"{annual_sr:.4f}", f"{annual_SoR:.4f}", f"{mdd*100:.2f}%", f"{annual_cr:.4f}",
                        trades_val, avg_cost, total_val, dist_str
                    ]
                    table.add_row(row_data)
                    csv_data.append(row_data)
                    
        if has_printed:
            # Thêm đường phân cách giữa các model
            table.add_row(["-"*15, "-"*8, "-"*5, "-"*10, "-"*10, "-"*8, "-"*8, "-"*8, "-"*10, "-"*8, "-"*16, "-"*23, "-"*11, "-"*25])

    print("\n[+] KẾT QUẢ ĐÁNH GIÁ CHỈ SỐ METRIC VÀ ACTION (JOINED)")
    print(table)
    
    os.makedirs("results", exist_ok=True)
    out_path = "results/combined_epoch_report.txt"
    csv_path = "results/combined_epoch_report.csv"
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(str(table))
        
    df = pd.DataFrame(csv_data, columns=table.field_names)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    
    print(f"\n[+] Đã lưu bảng kết quả (TXT) vào: {out_path}")
    print(f"[+] Đã lưu bảng kết quả (CSV) vào: {csv_path}")

if __name__ == "__main__":
    main()
