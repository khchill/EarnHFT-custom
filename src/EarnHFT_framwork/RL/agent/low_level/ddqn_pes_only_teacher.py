import sys
import os
import random
import argparse
import numpy as np
import pandas as pd
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(".")
sys.path.append("src")
sys.path.append("src/EarnHFT_framwork")
sys.path.append("./src/EarnHFT_framwork")

from model.net import Qnet
from env.low_level_env import Testing_env, Training_Env
from RL.util.replay_buffer_DQN import Multi_step_Prioritized_ReplayBuffer_multi_info
from RL.util.episode_selector import (
    start_selector, 
    get_transformation_even_risk, 
    get_transformation_even_based_boltzmann_risk, 
    get_transformation_even_based_sigmoid_risk
)

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--buffer_size", type=int, default=1000000)
parser.add_argument("--n_step", type=int, default=1)
parser.add_argument("--action_dim", type=int, default=5)
parser.add_argument("--back_time_length", type=int, default=1)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--reward_scale", type=float, default=30.0)
parser.add_argument("--hidden_nodes", type=int, default=128)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--update_times", type=int, default=1)
parser.add_argument("--gamma", type=float, default=1)
parser.add_argument("--epsilon_init", type=float, default=0.9)
parser.add_argument("--epsilon_min", type=float, default=0.01)
parser.add_argument("--epsilon_step", type=float, default=5e5)
parser.add_argument("--lr_init", type=float, default=1e-3)

parser.add_argument("--train_data_path", type=str, default="data/cleaned_data/BTCUSDT/tardis/train")
parser.add_argument("--dataset_name", type=str, default="BTCUSDT")
parser.add_argument("--transcation_cost", type=float, default=0.00015)
parser.add_argument("--max_holding_number", type=float, default=0.01)
parser.add_argument("--result_path", type=str, default="result_risk")

# Tham số của Q-Teacher và PES Selector 
parser.add_argument("--ada_init", type=float, default=256)
parser.add_argument("--ada_min", type=float, default=128)
parser.add_argument("--ada_step", type=float, default=5e6)
parser.add_argument("--beta", type=float, default=100) 
parser.add_argument("--type", type=str, default="boltzmann")
parser.add_argument("--risk_bond", type=float, default=0.1)

def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

class DQN(object):
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        seed_torch(self.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.beta = args.beta
        self.type = args.type
        
        if self.type == "even":
            self.priority_transformation = get_transformation_even_risk
        elif self.type == "sigmoid":
            self.priority_transformation = get_transformation_even_based_sigmoid_risk
        elif self.type == "boltzmann":
            self.priority_transformation = get_transformation_even_based_boltzmann_risk
            
        self.alpha_teacher = args.ada_init 
        
        feature_path = "data/cleaned_data/BTCUSDT/tardis/feature_list.npy"
        if not os.path.exists(feature_path):
            print("loi ko thay feature_list.npy")
            sys.exit(1)
            
        self.state_dim = len(np.load(feature_path, allow_pickle=True)) + 1 
        
        self.eval_net = Qnet(self.state_dim, args.action_dim, args.hidden_nodes).to(self.device)
        self.target_net = Qnet(self.state_dim, args.action_dim, args.hidden_nodes).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=args.lr_init)
        
        print(f"khoi tao low-level ddqn voi beta {self.beta} risk_bond {args.risk_bond}")

    def compute_q_teacher(self, df, env):
        N = len(df)
        Q_star = np.zeros((N, self.args.action_dim, self.args.action_dim))
        
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
                    scaled_reward = raw_reward * self.args.reward_scale
                    
                    Q_star[t, p_idx, a_idx] = np.max(Q_star[t+1, a_idx, :]) + scaled_reward
        return Q_star

    def train(self):
        train_path = self.args.train_data_path
        chunk_files = glob.glob(os.path.join(train_path, "df_*.feather"))
        if not chunk_files:
            return
            
        tech_indicators = np.load("data/cleaned_data/BTCUSDT/tardis/feature_list.npy", allow_pickle=True).tolist()
        
        chunk_returns = []
        for cf in chunk_files:
            temp_df = pd.read_feather(cf)
            ret = (temp_df['bid1_price'].iloc[-1] - temp_df['bid1_price'].iloc[0]) / temp_df['bid1_price'].iloc[0]
            chunk_returns.append(ret)
            
        weights = self.priority_transformation(chunk_returns, beta=self.beta, risk_bond=self.args.risk_bond)
        weights = np.array(weights) / np.sum(weights) 
        chunk_selector = start_selector(chunk_files, weights)
        
        replay_buffer = Multi_step_Prioritized_ReplayBuffer_multi_info(
            alpha=0.6, beta=0.4, batch_size=self.args.batch_size, n_steps=self.args.n_step,
            state_dim=self.state_dim, action_dim=self.args.action_dim, 
            buffer_capacity=self.args.buffer_size, gamma=self.args.gamma
        )
        
        epochs = self.args.epochs
        total_steps = 0
        
        # Thư mục lưu kết quả ứng với Bot Beta
        root_dir = f"{self.args.result_path}/BTCUSDT/only_teacher_beta_{self.beta}/seed_{self.seed}"
        
        for epoch in range(epochs):
            sampled_chunk_file, _, _ = chunk_selector.sample()
            print(f"Low-level Agent đang nạp chunk: {os.path.basename(sampled_chunk_file)}")
            df = pd.read_feather(sampled_chunk_file).bfill().ffill().fillna(0.0)
            env = Training_Env(df, tech_indicators, transcation_cost=self.args.transcation_cost, max_holding_number=self.args.max_holding_number)
            
            print(f"ep {epoch+1}/{epochs} | dang tinh q-teacher cho {os.path.basename(sampled_chunk_file)}...")
            Q_teacher = self.compute_q_teacher(df, env)
            
            state, _ = env.reset()
            done = False
            total_loss = 0
            
            while not done:
                total_steps += 1
                # alpha cố định không giảm (đối với Only Q-Teacher)
                self.alpha_teacher = self.args.ada_init                
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_eval = self.eval_net(state_tensor)
                
                current_step_idx = env.current_step
                current_position_idx = env.position_space.index(env.current_position)
                teacher_q_values = Q_teacher[current_step_idx, current_position_idx, :]
                
                epsilon = max(self.args.epsilon_min, self.args.epsilon_init - (total_steps / self.args.epsilon_step) * (self.args.epsilon_init - self.args.epsilon_min))
                
                if epoch % 2 == 1:
                    action_idx = int(np.argmax(teacher_q_values))
                else:
                    if np.random.rand() < epsilon:
                        action_idx = np.random.randint(0, self.args.action_dim)
                    else:
                        with torch.no_grad():
                            action_idx = torch.argmax(q_eval).item()
                
                next_state, reward, done, _ = env.step(action_idx)
                
                scaled_reward = reward * self.args.reward_scale
                
                replay_buffer.add(
                    market_state=state, previous_action=0, avaliable_action=np.zeros(self.args.action_dim), 
                    q_action=teacher_q_values, action=action_idx, reward=scaled_reward, 
                    next_market_state=next_state, next_previous_action=0, 
                    next_avaliable_action=np.zeros(self.args.action_dim), next_q_action=np.zeros(self.args.action_dim), terminal=done
                )
                
                state = next_state
                
                if replay_buffer.current_size > self.args.batch_size and total_steps % self.args.update_times == 0:
                    batch_tree_idx, b_states, b_actions, b_rewards, b_next_states, b_dones, b_IS_weights, b_q_action = replay_buffer.sample()
                    
                    b_states = torch.FloatTensor(b_states).to(self.device)
                    b_actions = torch.LongTensor(b_actions).view(-1, 1).to(self.device)
                    b_rewards = torch.FloatTensor(b_rewards).unsqueeze(1).to(self.device)
                    b_next_states = torch.FloatTensor(b_next_states).to(self.device)
                    b_dones = torch.FloatTensor(b_dones).unsqueeze(1).to(self.device)
                    b_IS_weights = torch.FloatTensor(b_IS_weights).unsqueeze(1).to(self.device)
                    b_q_action = torch.FloatTensor(b_q_action).to(self.device) 
                    
                    q_eval_action = self.eval_net(b_states).gather(1, b_actions)
                    
                    with torch.no_grad():
                        best_next_actions = self.eval_net(b_next_states).argmax(1).unsqueeze(1)
                        q_next_target = self.target_net(b_next_states).gather(1, best_next_actions)
                        q_target = b_rewards + self.args.gamma * q_next_target * (1 - b_dones)
                    
                    td_errors = torch.abs(q_eval_action - q_target).detach().cpu().numpy().flatten()
                    replay_buffer.update_priorities(batch_tree_idx, td_errors)
                    
                    loss_td = (b_IS_weights * F.mse_loss(q_eval_action, q_target, reduction='none')).mean()
                    # Loss KL tính với Q-Teacher
                    loss_kl = F.kl_div(F.log_softmax(self.eval_net(b_states), dim=1), F.softmax(b_q_action, dim=1), reduction='batchmean')
                    
                    # Only Q-Teacher loss, no TD Error loss
                    loss = self.alpha_teacher * loss_kl
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    for target_param, eval_param in zip(self.target_net.parameters(), self.eval_net.parameters()):
                        target_param.data.copy_(self.args.tau * eval_param.data + (1.0 - self.args.tau) * target_param.data)
                    
                    total_loss += loss.item()
                    
            print(f" -> loss: {total_loss:.4f} | eps: {epsilon:.4f} | alpha: {self.alpha_teacher:.2f}")
            
            epoch_dir = os.path.join(root_dir, f"epoch_{epoch+1}")
            os.makedirs(epoch_dir, exist_ok=True)
            torch.save(self.eval_net.state_dict(), os.path.join(epoch_dir, "model.pth"))
            
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"[*] Epoch {epoch+1}: Đang chạy Valid...")
                test_dir_path = "data/cleaned_data/BTCUSDT/tardis/valid"
                if os.path.exists(test_dir_path) and os.path.isdir(test_dir_path):
                    test_files = sorted([os.path.join(test_dir_path, f) for f in os.listdir(test_dir_path) if f.endswith(".feather")])
                    all_rewards, all_actions, all_require_money = [], [], []
                    
                    for test_file_path in test_files:
                        test_df = pd.read_feather(test_file_path).bfill().ffill().fillna(0.0)
                        test_env = Testing_env(test_df, tech_indicators, transcation_cost=self.args.transcation_cost, max_holding_number=self.args.max_holding_number, action_dim=self.args.action_dim)
                        test_state, _ = test_env.reset()
                        test_done = False
                        daily_rewards, daily_actions = [], []
                        
                        while not test_done:
                            state_tensor = torch.FloatTensor(test_state).unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                action_idx = torch.argmax(self.eval_net(state_tensor)).item()
                            next_state, reward, test_done, _ = test_env.step(action_idx)
                            daily_rewards.append(reward)
                            daily_actions.append(action_idx)
                            test_state = next_state
                            
                        all_rewards.extend(daily_rewards)
                        all_actions.extend(daily_actions)
                        all_require_money.append(test_env.require_money)
                        
                    t_dir = os.path.join(epoch_dir, "test")
                    os.makedirs(t_dir, exist_ok=True)
                    np.save(os.path.join(t_dir, "reward.npy"), np.array(all_rewards))
                    np.save(os.path.join(t_dir, "action.npy"), np.array(all_actions))
                    np.save(os.path.join(t_dir, "require_money.npy"), np.array(all_require_money))
            else:
                print(f"[*] Epoch {epoch+1} hoàn tất Train (Bỏ qua Test để tiết kiệm thời gian)")
                
        print("xong low-level")

if __name__ == "__main__":
    args = parser.parse_args()
    agent = DQN(args)
    agent.train()
