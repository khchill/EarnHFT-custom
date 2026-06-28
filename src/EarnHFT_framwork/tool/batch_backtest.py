import sys
import os
import shutil
import glob
import torch
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Bắt buộc giới hạn luồng của mỗi Process để tránh nổ CPU khi chạy Multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

sys.path.append(".")
sys.path.append("src")
sys.path.append("src/EarnHFT_framwork")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.net import Qnet, Qnet_high_level_position
from RL.agent.base.cdqn_train import CQnet
from RL.agent.base.ppo_train import Actor
from RL.agent.base.dra_train import ActorCriticLSTM
from env.low_level_env import Testing_env
from env.high_level_env import High_Level_Env

def process_backtest(args):
    model_name, epoch_folder, mode = args
    
    # 1. Xóa thư mục cũ nếu có (để ép backtest lại từ đầu)
    target_dir = os.path.join(epoch_folder, mode)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    # 2. Định nghĩa đường dẫn file
    data_dir = f"data/cleaned_data/BTCUSDT/tardis/{mode}"
    if not os.path.exists(data_dir):
        return f"[SKIP] Không tìm thấy dữ liệu {mode}"
        
    data_files = sorted(glob.glob(os.path.join(data_dir, "df_*.feather")))
    if len(data_files) == 0:
        data_files = sorted(glob.glob(os.path.join(data_dir, "*.feather")))
    
    model_path = os.path.join(epoch_folder, "model.pth")
    if not os.path.exists(model_path):
        return f"[SKIP] Không có model.pth ở {epoch_folder}"
        
    # Load feature
    low_level_features = np.load("data/cleaned_data/BTCUSDT/tardis/second_feature.npy", allow_pickle=True).tolist()
    high_level_features = np.load("data/cleaned_data/BTCUSDT/tardis/minitue_feature.npy", allow_pickle=True).tolist()
    
    device = torch.device("cpu") # Ép chạy CPU
    
    # Nhận diện Model
    is_high_level = ("high_level" in model_name)
    is_dra = ("dra" in model_name)
    is_ppo = ("ppo" in model_name)
    is_cdqn = ("cdqn" in model_name)
    is_dqn = ("dqn" in model_name or "beta" in model_name or "only" in model_name) and not is_cdqn and not is_high_level
    
    if is_high_level:
        model = Qnet_high_level_position(len(high_level_features) + 1, 5, 128)
    elif is_dra:
        model = ActorCriticLSTM(len(low_level_features) + 1, 5, 128)
    elif is_ppo:
        model = Actor(len(low_level_features) + 1, 5, 128)
    elif is_cdqn:
        model = CQnet(len(low_level_features) + 1, 5, 128)
    else:
        model = Qnet(len(low_level_features) + 1, 5, 128)
        
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if is_ppo and "actor" in checkpoint:
            model.load_state_dict(checkpoint["actor"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
    except Exception as e:
        return f"[LỖI] Load model {model_path} thất bại: {e}"

    all_rewards, all_actions, all_require_money = [], [], []
    
    for file_path in data_files:
        df = pd.read_feather(file_path).bfill().ffill().fillna(0.0)
        
        if is_high_level:
            env = High_Level_Env(df, high_level_features, low_level_features)
        else:
            env = Testing_env(df, low_level_features, transcation_cost=0.00015, max_holding_number=0.01, action_dim=5)
            
        state, _ = env.reset()
        done = False
        daily_actions = []
        daily_rewards = []
        
        if is_dra:
            h_eval_v = torch.zeros(1, 1, 128)
            c_eval_v = torch.zeros(1, 1, 128)
            
        while not done:
            if is_dra:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    action_logits, _, (h_eval_v, c_eval_v) = model(state_tensor, (h_eval_v, c_eval_v))
                    action = torch.argmax(action_logits.squeeze(0).squeeze(0), dim=-1).item()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_vals = model(state_tensor)
                    action = torch.argmax(q_vals, dim=-1).item()
                        
            next_state, reward, done, _ = env.step(action)
            daily_actions.append(action)
            daily_rewards.append(reward)
            state = next_state
            
        all_actions.extend(daily_actions)
        all_rewards.extend(daily_rewards)
        all_require_money.append(env.require_money)
        
    np.save(os.path.join(target_dir, "action.npy"), np.array(all_actions))
    np.save(os.path.join(target_dir, "reward.npy"), np.array(all_rewards))
    np.save(os.path.join(target_dir, "require_money.npy"), np.array(all_require_money))
    np.save(os.path.join(target_dir, "final_balance.npy"), np.array(np.sum(all_rewards)))
    
    return f"[XONG] {model_name:25s} | {os.path.basename(epoch_folder):10s} | {mode.upper()}"

def main():
    base_dir = "result_risk/BTCUSDT"
    if not os.path.exists(base_dir):
        print("Không tìm thấy result_risk/BTCUSDT")
        return
        
    tasks = []
    
    for model_name in os.listdir(base_dir):
        # Bỏ qua các thư mục rác / baselines 
        if model_name in ["rule_base", "kde", "segment_match_test", "segment_match_valid", "potential_model"]:
            continue
        if "check_data" in model_name or model_name.endswith(".png"):
            continue
            
        seed_dir = os.path.join(base_dir, model_name, "seed_12345")
        if not os.path.exists(seed_dir):
            continue
            
        epochs = []
        for e in os.listdir(seed_dir):
            if e.startswith("epoch_"):
                try:
                    num = int(e.split("_")[1])
                    epochs.append((num, e))
                except:
                    pass
                    
        if len(epochs) == 0:
            continue
            
        epochs.sort(key=lambda x: x[0])
        max_ep = epochs[-1][0]
        
        target_epoch_dirs = []
        for num, e_name in epochs:
            # Chọn epoch chia hết cho 5, epoch 1, hoặc epoch cuối cùng
            if num == 1 or num == max_ep or num % 5 == 0:
                target_epoch_dirs.append(os.path.join(seed_dir, e_name))
                
        for ep_dir in target_epoch_dirs:
            tasks.append((model_name, ep_dir, "valid"))
            tasks.append((model_name, ep_dir, "test"))
            
    print(f"Tổng số tasks: {len(tasks)}")
    print(f"Bắt đầu chạy bằng tất cả nhân CPU")
    
    # Chạy song song
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_backtest, t) for t in tasks]
        for future in as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    main()
