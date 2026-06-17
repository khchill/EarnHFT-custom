import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_epoch_balance(epoch_path, is_multi=False):
    """Lấy tổng reward (hoặc final_balance) trên tập validation cho một epoch cụ thể"""
    if not is_multi:
        valid_dir = os.path.join(epoch_path, "valid")
        balance_path = os.path.join(valid_dir, "final_balance.npy")
        if os.path.exists(balance_path):
            return np.load(balance_path).item()
        
        # Thử đọc từ reward.npy nếu không có final_balance.npy
        reward_path = os.path.join(valid_dir, "reward.npy")
        if os.path.exists(reward_path):
            rewards = np.load(reward_path)
            return np.sum(rewards)
        return None
    else:
        # Nếu là multi-label (Low-level agent)
        valid_dir = os.path.join(epoch_path, "valid_multi")
        if not os.path.exists(valid_dir):
            return None
            
        total_balance = 0
        count = 0
        for label_dir in os.listdir(valid_dir):
            lbl_path = os.path.join(valid_dir, label_dir)
            if not os.path.isdir(lbl_path): continue
            
            balance_path = os.path.join(lbl_path, "final_balance.npy")
            if os.path.exists(balance_path):
                total_balance += np.load(balance_path).item()
                count += 1
            else:
                reward_path = os.path.join(lbl_path, "reward.npy")
                if os.path.exists(reward_path):
                    total_balance += np.sum(np.load(reward_path))
                    count += 1
        return total_balance / count if count > 0 else None

def get_learning_curve(agent_root, is_multi=False):
    """Trả về list các giá trị reward theo từng epoch"""
    epochs_data = []
    
    if not os.path.exists(agent_root):
        return []
        
    for epoch_dir in os.listdir(agent_root):
        if not epoch_dir.startswith("epoch_"): continue
        
        try:
            ep_num = int(epoch_dir.split("_")[1])
        except:
            continue
            
        ep_path = os.path.join(agent_root, epoch_dir)
        balance = get_epoch_balance(ep_path, is_multi)
        
        if balance is not None:
            epochs_data.append((ep_num, balance))
            
    # Sort theo epoch number
    epochs_data.sort(key=lambda x: x[0])
    return [x[1] for x in epochs_data]

def plot_convergence():
    base_result_dir = "result_risk/BTCUSDT"
    
    # 1. DQN (bây giờ là DDQN Baseline)
    dqn_path = os.path.join(base_result_dir, "dqn_ada_0", "seed_12345")
    dqn_curve = get_learning_curve(dqn_path, is_multi=False)
    
    # 2. DDQN PES Risk Aware (thử nghiệm với một beta ví dụ như beta_30.0)
    beta_path = os.path.join(base_result_dir, "beta_30.0", "seed_12345")
    ddqn_pes_curve = get_learning_curve(beta_path, is_multi=True)

    plt.figure(figsize=(10, 6))
    
    if dqn_curve:
        plt.plot(range(1, len(dqn_curve) + 1), dqn_curve, label='Baseline DQN/DDQN (ada_0)', marker='o')
    if ddqn_pes_curve:
        plt.plot(range(1, len(ddqn_pes_curve) + 1), ddqn_pes_curve, label='Low-level DDQN PES (beta_30.0)', marker='s')

    plt.title('Hội tụ Validation Reward qua các Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Total Reward / Final Balance')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    output_path = os.path.join(base_result_dir, "convergence_plot.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Đã lưu biểu đồ hội tụ tại: {output_path}")

if __name__ == "__main__":
    plot_convergence()
