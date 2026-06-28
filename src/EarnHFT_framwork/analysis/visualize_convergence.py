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
    epochs = [x[0] for x in epochs_data]
    balances = [x[1] for x in epochs_data]
    return epochs, balances

def plot_convergence():
    base_result_dir = "result_risk/BTCUSDT"
    
    # 1. Full Model (TD + Q-Teacher)
    full_path = os.path.join(base_result_dir, "beta_30.0", "seed_12345")
    full_epochs, full_curve = get_learning_curve(full_path, is_multi=False)
    
    # 2. Only Q-Teacher
    teacher_path = os.path.join(base_result_dir, "only_teacher_beta_30.0", "seed_12345")
    teacher_epochs, teacher_curve = get_learning_curve(teacher_path, is_multi=False)
    
    # 3. Only TD Error
    td_path = os.path.join(base_result_dir, "only_td_beta_30.0", "seed_12345")
    td_epochs, td_curve = get_learning_curve(td_path, is_multi=False)
    
    # 4. Only Selfplay (KL)
    selfplay_path = os.path.join(base_result_dir, "only_selfplay_kl_beta_30.0", "seed_12345")
    selfplay_epochs, selfplay_curve = get_learning_curve(selfplay_path, is_multi=False)

    plt.figure(figsize=(10, 6))
    
    if full_curve:
        plt.plot(full_epochs, full_curve, label='Full (TD + Teacher)', marker='o', linewidth=2)
    if teacher_curve:
        plt.plot(teacher_epochs, teacher_curve, label='Only Q-Teacher', marker='s', linewidth=2)
    if td_curve:
        plt.plot(td_epochs, td_curve, label='Only TD Error', marker='^', linewidth=2)
    if selfplay_curve:
        plt.plot(selfplay_epochs, selfplay_curve, label='Only Selfplay (KL)', marker='x', linewidth=2)

    plt.title('Hội tụ Validation Reward qua các Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Total Reward / Final Balance')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "convergence_plot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Đã lưu biểu đồ hội tụ tại: {output_path}")

if __name__ == "__main__":
    plot_convergence()
