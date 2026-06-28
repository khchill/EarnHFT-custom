import os
import numpy as np
import pandas as pd
import glob

class DummyArgs:
    def __init__(self):
        self.action_dim = 5
        self.reward_scale = 30.0

class DummyEnv:
    def __init__(self):
        self.transcation_cost = 0.00015
        self.position_space = [0.0, 0.0025, 0.005, 0.0075, 0.01]

def compute_q_teacher(df, args, env):
    N = len(df)
    Q_star = np.zeros((N, args.action_dim, args.action_dim))
    
    ask_prices, ask_sizes, bid_prices, bid_sizes = np.zeros((N, 10)), np.zeros((N, 10)), np.zeros((N, 10)), np.zeros((N, 10))
    for i in range(1, 11):
        ask_prices[:, i-1] = df[f'ask{i}_price'].values
        ask_sizes[:, i-1] = df[f'ask{i}_size'].values
        bid_prices[:, i-1] = df[f'bid{i}_price'].values
        bid_sizes[:, i-1] = df[f'bid{i}_size'].values
    bid1_prices = df['bid1_price'].values
    
    tc = env.transcation_cost
    
    for t in range(N - 2, -1, -1):
        pb1_t, pb1_next = bid1_prices[t], bid1_prices[t + 1]
        ask_p_t, ask_s_t = ask_prices[t], ask_sizes[t]
        bid_p_t, bid_s_t = bid_prices[t], bid_sizes[t]
        
        for p_idx, p_val in enumerate(env.position_space):
            for a_idx, a_val in enumerate(env.position_space):
                M = a_val - p_val
                cash_flow = 0.0
                if M > 0:
                    remaining = M
                    for i in range(10):
                        executed = min(remaining, ask_s_t[i])
                        cash_flow -= executed * ask_p_t[i] * (1 + tc)
                        remaining -= executed
                        if remaining <= 0: break
                    if remaining > 0: cash_flow -= remaining * ask_p_t[9] * (1 + tc)
                elif M < 0:
                    remaining = -M
                    for i in range(10):
                        executed = min(remaining, bid_s_t[i])
                        cash_flow += executed * bid_p_t[i] * (1 - tc)
                        remaining -= executed
                        if remaining <= 0: break
                    if remaining > 0: cash_flow += remaining * bid_p_t[9] * (1 - tc)
                    
                raw_reward = (a_val * pb1_next) - (p_val * pb1_t) + cash_flow
                scaled_reward = raw_reward * args.reward_scale
                
                Q_star[t, p_idx, a_idx] = np.max(Q_star[t+1, a_idx, :]) + scaled_reward
    return Q_star

def main():
    train_path = "data/cleaned_data/BTCUSDT/tardis/train"
    target_file = os.path.join(train_path, "df_2022-06-01_4.feather")
    if not os.path.exists(target_file):
        print(f"Không tìm thấy file: {target_file}")
        return
    
    output_lines = []
    def log_print(msg):
        print(msg)
        output_lines.append(msg)
        
    log_print(f"Phân tích file: {target_file}")
    
    df = pd.read_feather(target_file).bfill().ffill().fillna(0.0)
    
    args = DummyArgs()
    env = DummyEnv()
    
    log_print("Tính toán Q_star...")
    Q_star = compute_q_teacher(df, args, env)
    
    log_print("\n--- BẢNG Q-VALUE (LÁT CẮT) TỪ T=1 ĐẾN T=5 ---")
    for t in range(1, 6):
        log_print(f"\nLát cắt Q*[t={t}]:")
        header = f"{'Vị thế (p) | Hành động (a)':<30}"
        for a_idx in range(args.action_dim):
            header += f"a={a_idx:<15}"
        log_print(header)
        log_print("-" * (30 + 15 * args.action_dim))
        
        for p_idx in range(args.action_dim):
            pos_val = env.position_space[p_idx]
            row_str = f"p={p_idx} ({pos_val}):".ljust(30)
            for a_idx in range(args.action_dim):
                val = Q_star[t, p_idx, a_idx]
                row_str += f"{val:<15.4f}"
            log_print(row_str)
            
    N = len(df)
    current_p_idx = 0 
    
    changes = 0
    non_changes = 0
    
    log_print("\n--- THAY ĐỔI VỊ THẾ (10 LẦN ĐẦU) ---")
    print_count = 0
    
    for t in range(N - 1):
        best_a = np.argmax(Q_star[t, current_p_idx, :])
        
        if best_a != current_p_idx:
            changes += 1
            if print_count < 10:
                old_pos = env.position_space[current_p_idx]
                new_pos = env.position_space[best_a]
                log_print(f"t={t:05d}: {old_pos:.4f} BTC -> {new_pos:.4f} BTC")
                print_count += 1
        else:
            non_changes += 1
            
        current_p_idx = best_a
        
    total_steps = changes + non_changes
    log_print("\n--- THỐNG KÊ ---")
    log_print(f"Tổng số giây: {total_steps}")
    log_print(f"Giữ nguyên: {non_changes} ({non_changes/total_steps*100:.2f}%)")
    log_print(f"Thay đổi: {changes} ({changes/total_steps*100:.2f}%)")
    
    # Save to file
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "q_teacher_actions_log.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
        
    print(f"\n[+] Đã lưu log vào: {out_path}")

if __name__ == "__main__":
    main()
